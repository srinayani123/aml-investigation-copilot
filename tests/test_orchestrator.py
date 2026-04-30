"""End-to-end test of the LangGraph orchestrator."""
from __future__ import annotations

from src.agents.orchestrator import run_investigation


def test_orchestrator_full_flow_planted_account(feature_store):
    final = run_investigation("acct_0050", feature_store)

    # All four agent outputs should be present
    assert "txn_findings" in final
    assert "entity_findings" in final
    assert "network_findings" in final
    assert "sar" in final

    # Audit trail should have one entry per agent. Order is now non-strict
    # because the first three agents run in parallel — only sar_drafter is
    # guaranteed to come last (it depends on the others' outputs).
    trail = final["audit_trail"]
    assert len(trail) == 4
    agents_in_trail = [t["agent"] for t in trail]
    assert set(agents_in_trail) == {
        "transaction_investigator",
        "entity_resolver",
        "network_analyst",
        "sar_drafter",
    }
    # SAR drafter must come last (it depends on the others)
    assert agents_in_trail[-1] == "sar_drafter"

    # SAR should have a numeric score and a valid recommendation
    sar = final["sar"]
    assert 0.0 <= sar.risk_score <= 1.0
    assert sar.recommendation in {"FILE_SAR", "MONITOR", "NO_ACTION"}
    assert sar.narrative  # non-empty


def test_orchestrator_returns_for_unknown_account(feature_store):
    """Should not crash on an account it doesn't know about."""
    final = run_investigation("acct_doesnt_exist", feature_store)
    assert "sar" in final
    # Risk score should be low for unknown account
    assert final["sar"].risk_score < 0.3
