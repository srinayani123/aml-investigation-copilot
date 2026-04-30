"""SAR Drafter agent.

Takes outputs from the other three agents and drafts a Suspicious Activity
Report following FinCEN-style structure:

1. Subject identification
2. Date range and account info
3. Description of suspicious activity (the narrative)
4. Specific indicators / typologies cited
5. Supporting evidence summary
6. Filing recommendation (with explicit human-review caveat)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.agents.entity_resolver import EntityFindings
from src.agents.llm import chat
from src.agents.network_analyst import NetworkFindings
from src.agents.transaction_investigator import TransactionFindings
from src.config import thresholds


@dataclass
class SARDraft:
    account: str
    risk_score: float            # 0..1, computed deterministically from findings
    recommendation: str          # "FILE_SAR" | "MONITOR" | "NO_ACTION"
    narrative: str = ""
    indicators_cited: list = None
    # New: per-agent score contributions (for the "Why this decision" panel)
    score_breakdown: dict = None        # {"transaction": 0.45, "entity": 0.10, ...}
    contributing_factors: list = None   # [{"factor": str, "contribution": float}, ...]
    headline: str = ""                  # one-sentence reasoning


def _compute_risk_score(txn: TransactionFindings,
                        ent: EntityFindings,
                        net: NetworkFindings) -> tuple[float, dict, list]:
    """Heuristic 0..1 risk score plus per-agent breakdown and contributing factors.

    The breakdown is the deterministic decision component — humans need a
    number AND its provenance they can defend in audit, not an LLM opinion.

    Returns:
        (total_score, agent_contributions, contributing_factors)
    """
    txn_score = 0.0
    ent_score = 0.0
    net_score = 0.0
    factors: list = []

    # ---- Transaction patterns ----
    if txn.structuring_events:
        txn_score += 0.30
        factors.append({
            "factor": f"{len(txn.structuring_events)} structuring event(s) detected (sub-CTR cash deposits)",
            "contribution": 0.30,
            "agent": "transaction",
        })
    if txn.rapid_throughput_count >= 3:
        txn_score += 0.15
        factors.append({
            "factor": f"Rapid throughput: {txn.rapid_throughput_count} in-then-out within 48h",
            "contribution": 0.15,
            "agent": "transaction",
        })
    if txn.high_velocity_days >= 2:
        txn_score += 0.05
        factors.append({
            "factor": f"High-velocity activity on {txn.high_velocity_days} days",
            "contribution": 0.05,
            "agent": "transaction",
        })
    if txn.round_amount_count >= 5:
        txn_score += 0.05
        factors.append({
            "factor": f"{txn.round_amount_count} round-dollar amounts (clustering)",
            "contribution": 0.05,
            "agent": "transaction",
        })

    # ---- Entity factors ----
    if ent.kyc_status in {"incomplete", "expired"}:
        ent_score += 0.10
        factors.append({
            "factor": f"KYC status: {ent.kyc_status}",
            "contribution": 0.10,
            "agent": "entity",
        })
    if ent.sanctions_hits:
        ent_score += 0.20
        factors.append({
            "factor": f"{len(ent.sanctions_hits)} sanctions fuzzy match(es) require manual review",
            "contribution": 0.20,
            "agent": "entity",
        })
    if ent.adverse_media_hits:
        ent_score += 0.05
        factors.append({
            "factor": f"{len(ent.adverse_media_hits)} adverse media hit(s)",
            "contribution": 0.05,
            "agent": "entity",
        })
    if ent.geo_risk_score >= 0.3:
        ent_score += 0.05
        factors.append({
            "factor": f"Elevated geographic risk (score={ent.geo_risk_score:.2f})",
            "contribution": 0.05,
            "agent": "entity",
        })

    # ---- Network factors ----
    if net.fan_in_collectors:
        net_score += 0.15
        factors.append({
            "factor": f"Fan-in pattern: {len(net.fan_in_collectors)} collector(s) detected (mule-network signal)",
            "contribution": 0.15,
            "agent": "network",
        })
    if net.layering_chains:
        net_score += 0.15
        factors.append({
            "factor": f"{len(net.layering_chains)} layering chain(s) found",
            "contribution": 0.15,
            "agent": "network",
        })
    if net.cycles:
        net_score += 0.10
        factors.append({
            "factor": f"{len(net.cycles)} closed cycle(s) detected",
            "contribution": 0.10,
            "agent": "network",
        })

    total = min(1.0, txn_score + ent_score + net_score)
    breakdown = {
        "transaction": round(txn_score, 3),
        "entity": round(ent_score, 3),
        "network": round(net_score, 3),
    }
    # Sort factors by contribution descending
    factors.sort(key=lambda f: -f["contribution"])
    return total, breakdown, factors


def _build_headline(score: float,
                    breakdown: dict,
                    factors: list,
                    recommendation: str) -> str:
    """One-sentence reasoning for the top of the decision card."""
    if not factors:
        if recommendation == "NO_ACTION":
            return ("No significant risk indicators triggered. Account activity "
                    "appears consistent with legitimate use.")
        return "Insufficient signal — recommend continued monitoring."

    # Determine the dominant agent
    dominant_agent = max(breakdown, key=breakdown.get)
    dominant_score = breakdown[dominant_agent]
    agent_label = {
        "transaction": "transaction patterns",
        "entity": "entity-level risk indicators",
        "network": "network topology",
    }[dominant_agent]

    top_factor = factors[0]["factor"]

    if recommendation == "FILE_SAR":
        verb = "Flagged for"
    elif recommendation == "MONITOR":
        verb = "Elevated risk from"
    else:
        verb = "Minor signals from"

    second_part = ""
    if len(factors) >= 2 and factors[1]["contribution"] >= 0.1:
        second_part = f" Secondary indicator: {factors[1]['factor'].lower()}."

    return (f"{verb} {top_factor.lower()}. "
            f"Dominant signal: {agent_label} "
            f"({dominant_score:.2f}/{score:.2f} of total score).{second_part}")


def _recommendation(score: float) -> str:
    if score >= thresholds.sar_file_threshold:
        return "FILE_SAR"
    if score >= thresholds.sar_monitor_threshold:
        return "MONITOR"
    return "NO_ACTION"


def _gather_indicators(txn: TransactionFindings,
                       ent: EntityFindings,
                       net: NetworkFindings) -> list:
    indicators = []
    if txn.structuring_events:
        indicators.append("31 CFR 1010.314 — structuring (sub-CTR deposits)")
    if txn.rapid_throughput_count >= 3:
        indicators.append("FFIEC BSA/AML — funds layering (rapid in-and-out)")
    if ent.sanctions_hits:
        indicators.append("OFAC sanctions screening — fuzzy match requires review")
    if ent.kyc_status in {"incomplete", "expired"}:
        indicators.append("CDD Rule (31 CFR 1020.230) — KYC gaps")
    if net.fan_in_collectors:
        indicators.append("FinCEN typology — mule/funnel account (fan-in)")
    if net.layering_chains:
        indicators.append("FinCEN typology — layering chain")
    if ent.geo_risk_score >= 0.3:
        indicators.append("FATF high-risk-jurisdiction exposure")
    return indicators


def draft_sar(account: str,
              txn: TransactionFindings,
              ent: EntityFindings,
              net: NetworkFindings,
              api_key: Optional[str] = None) -> SARDraft:
    score, breakdown, factors = _compute_risk_score(txn, ent, net)
    recommendation = _recommendation(score)
    indicators = _gather_indicators(txn, ent, net)
    headline = _build_headline(score, breakdown, factors, recommendation)

    prompt = f"""Draft a Suspicious Activity Report (SAR) for the following alert.

==== SUBJECT ACCOUNT ====
{account}

==== TRANSACTION INVESTIGATOR FINDINGS ====
{txn.summary}

Structuring events: {len(txn.structuring_events)}
Rapid throughput count: {txn.rapid_throughput_count}
High-velocity days: {txn.high_velocity_days}
Round-amount transactions: {txn.round_amount_count}

==== ENTITY RESOLVER FINDINGS ====
{ent.summary}

KYC status: {ent.kyc_status}
Declared occupation: {ent.declared_occupation}
Sanctions fuzzy hits: {len(ent.sanctions_hits)}
Adverse media: {len(ent.adverse_media_hits)}

==== NETWORK ANALYST FINDINGS ====
{net.summary}

Subgraph: {net.subgraph_n_nodes} nodes, {net.subgraph_n_edges} edges
Fan-in collectors: {len(net.fan_in_collectors)}
Layering chains: {len(net.layering_chains)}
Cycles: {len(net.cycles)}

==== HEURISTIC RISK SCORE ====
Score: {score:.2f}/1.00
Recommendation: {recommendation}

==== TYPOLOGIES / REGULATORY INDICATORS CITED ====
{chr(10).join(f"- {ind}" for ind in indicators) or "(none)"}

----

Task: write the SAR NARRATIVE section (the prose explaining the suspicious
activity to a regulator). Use FinCEN-style language. Be factual and specific.
DO NOT make up details that aren't supported by the findings above. End with
a one-sentence note that this draft requires human investigator review
before filing.

The narrative should cover, in order:
1. Subject and time period
2. Description of the patterns observed
3. Why the activity is considered suspicious
4. Supporting evidence
5. Note on human review requirement
"""

    narrative = chat(
        prompt=prompt,
        system=("You are a senior BSA/AML analyst drafting a SAR narrative. "
                "Be precise, factual, and use FinCEN terminology."),
        runtime_api_key=api_key,
        max_tokens=1024,
    )

    return SARDraft(
        account=account,
        risk_score=score,
        recommendation=recommendation,
        narrative=narrative,
        indicators_cited=indicators,
        score_breakdown=breakdown,
        contributing_factors=factors,
        headline=headline,
    )
