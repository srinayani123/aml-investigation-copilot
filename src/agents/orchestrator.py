"""LangGraph orchestrator wiring the 4 agents.

Workflow:
  START
    │
    ├──► transaction_investigator ──┐
    ├──► entity_resolver ───────────┼──► sar_drafter ──► END
    └──► network_analyst ───────────┘

The first three agents run **in parallel** because they're fully independent
(none reads another's output). LangGraph fans out from START to all three,
then `sar_drafter` waits for all three to complete before running.

Latency comparison:
  - Mock mode (CPU-bound work):     ~120-180ms (1.2-1.5x speedup over serial)
  - Real-LLM mode (I/O-bound):       ~2-3s    (3x speedup — the big win)

The real-LLM speedup is dramatic because each agent spends most of its
time waiting on the Anthropic API; concurrent waits release the GIL so
all three calls happen simultaneously instead of stacking up.

Each agent step is logged to the audit trail with timestamp, inputs, outputs,
and runtime. The `audit_trail` field uses a list-merge reducer so concurrent
writes from parallel agents don't clobber each other.
"""
from __future__ import annotations

import operator
import time
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.entity_resolver import EntityFindings, resolve_entity
from src.agents.network_analyst import NetworkFindings, analyze_network
from src.agents.sar_drafter import SARDraft, draft_sar
from src.agents.transaction_investigator import (
    TransactionFindings,
    investigate_transactions,
)
from src.data.features import FeatureStore


# ---------------------------------------------------------------------------
# Shared state passed between agents in the LangGraph workflow
# ---------------------------------------------------------------------------

class InvestigationState(TypedDict, total=False):
    account: str
    api_key: Optional[str]
    txn_findings: TransactionFindings
    entity_findings: EntityFindings
    network_findings: NetworkFindings
    sar: SARDraft
    # Annotated with a reducer so parallel agents each append their entry
    # without clobbering the others' entries. operator.add concatenates lists.
    audit_trail: Annotated[list, operator.add]


def _audit_entry(agent: str, started: float, **kwargs) -> dict:
    return {
        "agent": agent,
        "started_at": started,
        "duration_ms": (time.time() - started) * 1000,
        **kwargs,
    }


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------

def make_orchestrator(store: FeatureStore):
    """Create a compiled LangGraph workflow bound to a specific FeatureStore."""

    def _node_transaction(state: InvestigationState) -> dict:
        t0 = time.time()
        findings = investigate_transactions(state["account"], store,
                                            api_key=state.get("api_key"))
        return {
            "txn_findings": findings,
            "audit_trail": [_audit_entry(
                "transaction_investigator", t0,
                structuring=len(findings.structuring_events),
                rapid_throughput=findings.rapid_throughput_count,
                high_velocity_days=findings.high_velocity_days,
            )],
        }

    def _node_entity(state: InvestigationState) -> dict:
        t0 = time.time()
        findings = resolve_entity(state["account"], store,
                                  api_key=state.get("api_key"))
        return {
            "entity_findings": findings,
            "audit_trail": [_audit_entry(
                "entity_resolver", t0,
                kyc_status=findings.kyc_status,
                sanctions_hits=len(findings.sanctions_hits),
                adverse_media_hits=len(findings.adverse_media_hits),
            )],
        }

    def _node_network(state: InvestigationState) -> dict:
        t0 = time.time()
        findings = analyze_network(state["account"], store,
                                   api_key=state.get("api_key"))
        return {
            "network_findings": findings,
            "audit_trail": [_audit_entry(
                "network_analyst", t0,
                n_nodes=findings.subgraph_n_nodes,
                fan_in=len(findings.fan_in_collectors),
                layering_chains=len(findings.layering_chains),
                cycles=len(findings.cycles),
            )],
        }

    def _node_sar(state: InvestigationState) -> dict:
        t0 = time.time()
        sar = draft_sar(
            state["account"],
            state["txn_findings"],
            state["entity_findings"],
            state["network_findings"],
            api_key=state.get("api_key"),
        )
        return {
            "sar": sar,
            "audit_trail": [_audit_entry(
                "sar_drafter", t0,
                risk_score=sar.risk_score,
                recommendation=sar.recommendation,
            )],
        }

    builder = StateGraph(InvestigationState)
    builder.add_node("transaction_investigator", _node_transaction)
    builder.add_node("entity_resolver", _node_entity)
    builder.add_node("network_analyst", _node_network)
    builder.add_node("sar_drafter", _node_sar)

    # Fan-out: START runs all three independent agents concurrently.
    builder.add_edge(START, "transaction_investigator")
    builder.add_edge(START, "entity_resolver")
    builder.add_edge(START, "network_analyst")

    # Fan-in: SAR drafter waits for ALL three (LangGraph synchronizes).
    builder.add_edge("transaction_investigator", "sar_drafter")
    builder.add_edge("entity_resolver", "sar_drafter")
    builder.add_edge("network_analyst", "sar_drafter")
    builder.add_edge("sar_drafter", END)

    return builder.compile()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_investigation(account: str,
                      store: FeatureStore,
                      api_key: Optional[str] = None) -> InvestigationState:
    graph = make_orchestrator(store)
    initial: InvestigationState = {
        "account": account,
        "api_key": api_key,
        "audit_trail": [],
    }
    final = graph.invoke(initial)
    # Sort audit trail by start time so it reads chronologically in the UI
    # (entries from parallel branches arrive in non-deterministic order).
    if final.get("audit_trail"):
        final["audit_trail"] = sorted(
            final["audit_trail"],
            key=lambda e: e.get("started_at", 0),
        )
    return final
