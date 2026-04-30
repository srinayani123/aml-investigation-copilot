"""Transaction Investigator agent.

Pulls the account's transaction history and surfaces deterministic patterns:
- Structuring (sub-CTR deposits)
- Round-amount clustering
- Velocity spikes
- Short hold times (rapid onward transfer)

Then asks the LLM to summarize the findings as an analyst would.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd

from src.agents.llm import chat
from src.config import thresholds
from src.data.features import FeatureStore


@dataclass
class TransactionFindings:
    account: str
    txn_count_total: int
    total_volume_usd: float
    structuring_events: List[dict] = field(default_factory=list)
    round_amount_count: int = 0
    high_velocity_days: int = 0
    rapid_throughput_count: int = 0   # in-then-out within 48h
    summary: str = ""


def _detect_structuring(history: pd.DataFrame) -> List[dict]:
    """Find clusters of sub-CTR deposits in a 7-day window."""
    deposits = history[
        (history["amount_usd"] >= thresholds.structuring_amount_band_low)
        & (history["amount_usd"] <= thresholds.structuring_amount_band_high)
    ].sort_values("timestamp")
    if len(deposits) < thresholds.structuring_min_count:
        return []

    events: List[dict] = []
    timestamps = deposits["timestamp"].values
    for i in range(len(deposits) - thresholds.structuring_min_count + 1):
        window_end_idx = i + thresholds.structuring_min_count - 1
        window_span = timestamps[window_end_idx] - timestamps[i]
        if window_span <= np.timedelta64(thresholds.structuring_window_days, "D"):
            cluster = deposits.iloc[i:window_end_idx + 1]
            events.append({
                "window_start": str(cluster["timestamp"].iloc[0]),
                "window_end": str(cluster["timestamp"].iloc[-1]),
                "n_deposits": len(cluster),
                "total_usd": float(cluster["amount_usd"].sum()),
                "amounts": [round(a, 2) for a in cluster["amount_usd"].tolist()],
            })
    # Dedupe overlapping windows — keep the largest
    if events:
        events.sort(key=lambda e: -e["total_usd"])
        events = events[:3]
    return events


def _detect_rapid_throughput(history: pd.DataFrame, account: str) -> int:
    """Count instances where money came in and left within 48h."""
    incoming = history[history["to_account"] == account].sort_values("timestamp")
    outgoing = history[history["from_account"] == account].sort_values("timestamp")
    if incoming.empty or outgoing.empty:
        return 0

    count = 0
    out_times = outgoing["timestamp"].values
    for in_time in incoming["timestamp"].values:
        delta_hours = (out_times - in_time) / np.timedelta64(1, "h")
        if np.any((delta_hours > 0) & (delta_hours <= thresholds.layering_max_hold_hours)):
            count += 1
    return count


def investigate_transactions(account: str,
                             store: FeatureStore,
                             api_key: Optional[str] = None) -> TransactionFindings:
    history = store.get_account_history(account, limit=500)
    feats = store.account_features.get(account)

    if feats is None or len(history) == 0:
        return TransactionFindings(
            account=account, txn_count_total=0, total_volume_usd=0.0,
            summary=f"Account {account} has no transaction history.",
        )

    structuring = _detect_structuring(history)
    rapid = _detect_rapid_throughput(history, account)

    # Velocity: count days with >threshold transactions
    daily = history.groupby(history["timestamp"].dt.date).size()
    high_velocity_days = int((daily > thresholds.high_velocity_txns_per_day).sum())

    findings = TransactionFindings(
        account=account,
        txn_count_total=feats.txn_count,
        total_volume_usd=feats.total_volume_usd,
        structuring_events=structuring,
        round_amount_count=feats.n_round_amounts,
        high_velocity_days=high_velocity_days,
        rapid_throughput_count=rapid,
    )

    # Build LLM prompt with the deterministic findings
    prompt = f"""Transaction Investigator — analyse this account.

Account: {account}
Transaction count: {feats.txn_count}
Total volume: ${feats.total_volume_usd:,.2f}
Average transaction: ${feats.avg_txn_usd:,.2f}
Counterparties: {feats.n_unique_counterparties}
Round-dollar amounts: {feats.n_round_amounts}
High-risk payment formats (cash, credit-card): {feats.n_high_risk_format}
High-velocity days (> {thresholds.high_velocity_txns_per_day} txns/day): {high_velocity_days}
Rapid throughput (in-then-out within 48h): {rapid}
Structuring events detected: {len(structuring)}

Detected structuring events:
{structuring if structuring else "(none)"}

Summarize the transaction patterns in 4-6 sentences. Highlight anything that
warrants escalation. Do not draft a SAR yet — just summarize the patterns
factually.
"""
    findings.summary = chat(
        prompt=prompt,
        system="You are an experienced AML transaction investigator.",
        runtime_api_key=api_key,
    )
    return findings
