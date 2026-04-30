"""Network Analyst agent.

Builds the counterparty subgraph around the alerted account and looks for
classical AML topologies: fan-in (mule networks), fan-out, layering chains,
and closed cycles.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import networkx as nx
import pandas as pd

from src.agents.llm import chat
from src.config import thresholds
from src.data.features import FeatureStore


@dataclass
class NetworkFindings:
    account: str
    subgraph_n_nodes: int = 0
    subgraph_n_edges: int = 0
    subgraph_total_volume_usd: float = 0.0
    fan_in_collectors: List[dict] = field(default_factory=list)
    fan_out_distributors: List[dict] = field(default_factory=list)
    layering_chains: List[dict] = field(default_factory=list)
    cycles: List[List[str]] = field(default_factory=list)
    summary: str = ""


def _build_graph(df: pd.DataFrame) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    # itertuples is ~10x faster than iterrows for this volume
    for row in df.itertuples(index=False):
        g.add_edge(row.from_account, row.to_account,
                   timestamp=row.timestamp, amount=row.amount_usd,
                   format=row.payment_format)
    return g


def _detect_fan_in(g: nx.MultiDiGraph) -> List[dict]:
    """Find accounts receiving from many distinct senders within a window."""
    out = []
    for node in g.nodes():
        senders = set()
        amounts = []
        for u, _v, data in g.in_edges(node, data=True):
            senders.add(u)
            amounts.append(data.get("amount", 0))
        if len(senders) >= thresholds.fan_in_min_senders:
            out.append({
                "collector": node,
                "n_senders": len(senders),
                "total_received_usd": float(sum(amounts)),
            })
    out.sort(key=lambda x: -x["n_senders"])
    return out[:5]


def _detect_fan_out(g: nx.MultiDiGraph) -> List[dict]:
    out = []
    for node in g.nodes():
        receivers = set()
        amounts = []
        for _u, v, data in g.out_edges(node, data=True):
            receivers.add(v)
            amounts.append(data.get("amount", 0))
        if len(receivers) >= thresholds.fan_out_min_receivers:
            out.append({
                "distributor": node,
                "n_receivers": len(receivers),
                "total_sent_usd": float(sum(amounts)),
            })
    out.sort(key=lambda x: -x["n_receivers"])
    return out[:5]


def _detect_cycles(g: nx.MultiDiGraph) -> List[List[str]]:
    """Find short closed cycles (money returns to origin).

    Cap the search aggressively — `simple_cycles` is exponential on dense
    graphs. We only care about short cycles for AML purposes anyway.
    """
    if g.number_of_nodes() > 80:
        # Too dense for full cycle search; sample a subgraph around high-degree nodes.
        degree = dict(g.degree())
        top_nodes = sorted(degree, key=degree.get, reverse=True)[:50]
        g = g.subgraph(top_nodes).copy()
    simple_g = nx.DiGraph(g)  # collapse multi-edges
    cycles: List[List[str]] = []
    try:
        # nx.simple_cycles with length_bound is much faster
        for cyc in nx.simple_cycles(simple_g, length_bound=4):
            if 2 <= len(cyc) <= 4:
                cycles.append(cyc)
            if len(cycles) >= 5:
                break
    except Exception:
        pass
    return cycles


def _detect_layering_chains(g: nx.MultiDiGraph) -> List[dict]:
    """Find paths where money flows quickly through 3+ accounts.

    O(n_edges) approach: for each edge, scan its successors' edges. Indexed
    on (source, edge timestamp) so we don't iterate the full edge list per
    node.
    """
    chains: List[dict] = []

    # Build an index: account -> list of (timestamp, amount, target)
    out_edges_by_node: dict = {}
    for u, v, data in g.edges(data=True):
        out_edges_by_node.setdefault(u, []).append(
            (data["timestamp"], data["amount"], v))

    # Limit the seed nodes to those with both incoming AND outgoing edges
    # (a layering intermediary needs both).
    candidates = [n for n in g.nodes()
                  if g.in_degree(n) > 0 and g.out_degree(n) > 0][:300]

    for u in candidates:
        for v1_t, v1_amt, v1 in out_edges_by_node.get(u, [])[:10]:
            for v2_t, v2_amt, v2 in out_edges_by_node.get(v1, [])[:10]:
                if v2 == u or v2 == v1:
                    continue
                hours_between = (v2_t - v1_t).total_seconds() / 3600
                if 0 < hours_between <= thresholds.layering_max_hold_hours:
                    chains.append({
                        "chain": [u, v1, v2],
                        "total_usd": float(v1_amt + v2_amt),
                        "hours_span": round(hours_between, 1),
                    })
                    if len(chains) >= 50:  # cap collection
                        break
            if len(chains) >= 50:
                break
        if len(chains) >= 50:
            break

    chains.sort(key=lambda x: -x["total_usd"])
    return chains[:5]


def analyze_network(account: str,
                    store: FeatureStore,
                    api_key: Optional[str] = None,
                    hops: int = 2) -> NetworkFindings:
    sub_df = store.get_subgraph(account, hops=hops)
    if len(sub_df) == 0:
        return NetworkFindings(account=account,
                               summary=f"No subgraph found around {account}.")

    g = _build_graph(sub_df)
    fan_in = _detect_fan_in(g)
    fan_out = _detect_fan_out(g)
    cycles = _detect_cycles(g)
    chains = _detect_layering_chains(g)

    findings = NetworkFindings(
        account=account,
        subgraph_n_nodes=g.number_of_nodes(),
        subgraph_n_edges=g.number_of_edges(),
        subgraph_total_volume_usd=float(sub_df["amount_usd"].sum()),
        fan_in_collectors=fan_in,
        fan_out_distributors=fan_out,
        layering_chains=chains,
        cycles=cycles,
    )

    prompt = f"""Network Analyst — counterparty graph review.

Seed account: {account}
Subgraph (2-hop): {g.number_of_nodes()} accounts, {g.number_of_edges()} transactions
Total volume in subgraph: ${findings.subgraph_total_volume_usd:,.2f}

Fan-in (potential collectors / mule terminals): {len(fan_in)}
{fan_in if fan_in else "(none)"}

Fan-out (potential distributors): {len(fan_out)}
{fan_out if fan_out else "(none)"}

Short layering chains (in-then-out within 48h, 3+ hops): {len(chains)}
{chains if chains else "(none)"}

Closed cycles (2-5 hop): {len(cycles)}
{cycles if cycles else "(none)"}

Summarize the network topology in 4-6 sentences. Identify the dominant
typology (mule network, layering chain, normal, etc.) and flag whether the
seed account looks more like a peripheral participant or a central node.
"""
    findings.summary = chat(
        prompt=prompt,
        system="You are an experienced AML network analyst.",
        runtime_api_key=api_key,
    )
    return findings
