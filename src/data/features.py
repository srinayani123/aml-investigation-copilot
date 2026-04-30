"""Account-level feature aggregations.

Pre-computes the lookups each agent needs so we don't rescan the full DataFrame
on every tool call. Persisted to a single object the agents share.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass
class AccountFeatures:
    """Pre-computed per-account aggregates."""
    txn_count: int = 0
    total_volume_usd: float = 0.0
    n_unique_counterparties: int = 0
    avg_txn_usd: float = 0.0
    n_round_amounts: int = 0
    n_high_risk_format: int = 0   # cash + credit-card
    first_seen: pd.Timestamp = field(default_factory=lambda: pd.NaT)
    last_seen: pd.Timestamp = field(default_factory=lambda: pd.NaT)
    n_laundering_in_history: int = 0  # ground truth — used only for evaluation


@dataclass
class FeatureStore:
    """All pre-computed features needed by agents."""
    account_features: Dict[str, AccountFeatures]
    transactions_df: pd.DataFrame    # the full trimmed table
    counterparty_index: Dict[str, List[str]]  # account -> list of counterparties

    def get_account_history(self, account: str,
                            limit: int = 100) -> pd.DataFrame:
        """Return all transactions touching this account, most recent first."""
        df = self.transactions_df
        mask = (df["from_account"] == account) | (df["to_account"] == account)
        sub = df.loc[mask].sort_values("timestamp", ascending=False).head(limit)
        return sub

    def get_counterparties(self, account: str) -> List[str]:
        return self.counterparty_index.get(account, [])

    def get_subgraph(self, seed_account: str,
                     hops: int = 2,
                     max_per_hop: int = 25) -> pd.DataFrame:
        """Return all transactions in the N-hop neighborhood of an account."""
        seen = {seed_account}
        frontier = {seed_account}
        for _ in range(hops):
            next_frontier = set()
            for acct in frontier:
                neighbors = self.counterparty_index.get(acct, [])[:max_per_hop]
                for n in neighbors:
                    if n not in seen:
                        next_frontier.add(n)
                        seen.add(n)
            frontier = next_frontier
        df = self.transactions_df
        mask = df["from_account"].isin(seen) | df["to_account"].isin(seen)
        return df.loc[mask].copy()


def build_feature_store(df: pd.DataFrame) -> FeatureStore:
    """Build the feature store from a transactions DataFrame."""
    accounts = pd.concat([df["from_account"], df["to_account"]]).unique()
    feats: Dict[str, AccountFeatures] = {}
    counterparty_index: Dict[str, List[str]] = {}

    df_from = df.groupby("from_account")
    df_to = df.groupby("to_account")

    high_risk_formats = {"Cash", "Credit Card"}

    for acct in accounts:
        outgoing = df_from.get_group(acct) if acct in df_from.groups else pd.DataFrame()
        incoming = df_to.get_group(acct) if acct in df_to.groups else pd.DataFrame()
        all_txns = pd.concat([outgoing, incoming])

        if len(all_txns) == 0:
            continue

        partners = set()
        if not outgoing.empty:
            partners |= set(outgoing["to_account"].unique())
        if not incoming.empty:
            partners |= set(incoming["from_account"].unique())

        n_round = int((all_txns["amount_usd"] % 1000 == 0).sum())
        n_high_risk = int(all_txns["payment_format"].isin(high_risk_formats).sum())

        feats[acct] = AccountFeatures(
            txn_count=len(all_txns),
            total_volume_usd=float(all_txns["amount_usd"].sum()),
            n_unique_counterparties=len(partners),
            avg_txn_usd=float(all_txns["amount_usd"].mean()),
            n_round_amounts=n_round,
            n_high_risk_format=n_high_risk,
            first_seen=all_txns["timestamp"].min(),
            last_seen=all_txns["timestamp"].max(),
            n_laundering_in_history=int(all_txns["is_laundering"].sum()),
        )
        counterparty_index[acct] = list(partners)

    return FeatureStore(
        account_features=feats,
        transactions_df=df,
        counterparty_index=counterparty_index,
    )
