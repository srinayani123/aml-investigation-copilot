"""Tests for the data layer."""
from __future__ import annotations

from src.data.features import build_feature_store
from src.data.loader import generate_synthetic


def test_synthetic_has_expected_planted_typologies(synthetic_df):
    df = synthetic_df
    # We planted 3 structuring + 3 layering + 6 fan-in = 12 laundering txns
    assert df["is_laundering"].sum() >= 10

    # Structuring planted on acct_0001
    structuring = df[
        (df["from_account"] == "acct_0001")
        & (df["amount_usd"] >= 9000)
        & (df["amount_usd"] < 10000)
    ]
    assert len(structuring) == 3

    # Fan-in planted on acct_0050
    fan_in_to_collector = df[df["to_account"] == "acct_0050"]
    assert len(fan_in_to_collector) >= 6


def test_feature_store_basic(feature_store):
    store = feature_store
    assert len(store.account_features) > 0
    # acct_0001 (structuring sender) should be present
    assert "acct_0001" in store.account_features
    # acct_0050 (fan-in collector) should be present
    assert "acct_0050" in store.account_features


def test_subgraph_returns_neighbors(feature_store):
    sub = feature_store.get_subgraph("acct_0050", hops=1)
    # Should include acct_0050 and its 6 senders, with their transactions
    accounts_in_sub = set(sub["from_account"]).union(set(sub["to_account"]))
    assert "acct_0050" in accounts_in_sub
    assert len(accounts_in_sub) >= 6
