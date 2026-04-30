"""LLM client with mock + real Anthropic modes.

Resolution order:
  1. `runtime_api_key` arg (visitor key from Streamlit sidebar)
  2. `LLMConfig.default_api_key` (your env var / Streamlit secret)
  3. Mock mode (templated responses, free, never fails)

The mock responses are *deterministic and structured* so the rest of the
pipeline (parsing, evaluation, UI rendering) works identically whether
the LLM is real or mocked. This means tests pass without API keys, and
hiring managers see meaningful output even if you've never set a key.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.config import llm_cfg

logger = logging.getLogger(__name__)


def get_active_mode(runtime_api_key: Optional[str] = None) -> str:
    """Returns 'real' if any Anthropic key is available, else 'mock'."""
    if runtime_api_key and runtime_api_key.strip():
        return "real"
    if llm_cfg.default_api_key:
        return "real"
    return "mock"


def chat(prompt: str,
         system: str = "",
         runtime_api_key: Optional[str] = None,
         max_tokens: Optional[int] = None) -> str:
    """One-shot LLM call. Returns text only. Never raises on bad keys —
    falls back to mock instead of breaking the demo."""
    key = (runtime_api_key or "").strip() or llm_cfg.default_api_key
    if not key:
        return _mock_response(prompt, system)

    try:
        # Lazy import so the mock path has no anthropic dependency hard-required
        import anthropic
        client = anthropic.Anthropic(api_key=key)
        system_text = system or "You are an experienced anti-money-laundering analyst."
        # Anthropic prompt caching: tag the system prompt as cacheable.
        # First call writes to cache (slight write premium), subsequent calls
        # within the 5-min cache window read at 10% of normal input cost.
        # System prompts are identical across investigations, so cache hit rate
        # in production is very high.
        msg = client.messages.create(
            model=llm_cfg.model,
            max_tokens=max_tokens or llm_cfg.max_tokens,
            temperature=llm_cfg.temperature,
            system=[
                {
                    "type": "text",
                    "text": system_text,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": prompt}],
        )
        # `content` is a list of blocks; pull the text.
        parts = []
        for block in msg.content:
            text = getattr(block, "text", None)
            if text:
                parts.append(text)
        return "".join(parts).strip() or "(empty response)"
    except Exception as e:
        logger.warning("LLM call failed (%s); falling back to mock.", e)
        return _mock_response(prompt, system)


# ---------------------------------------------------------------------------
# Mock responses — heuristic templates keyed off the prompt content
# ---------------------------------------------------------------------------

def _mock_response(prompt: str, system: str) -> str:
    """Return a templated response that depends on which agent called.
    Detection is by keyword in the prompt. Cheap, deterministic, free.
    """
    p = prompt.lower()
    if "transaction patterns" in p or "transaction investigator" in p:
        return _mock_transaction_summary(prompt)
    if "entity" in p or "kyc" in p or "sanctions" in p:
        return _mock_entity_summary(prompt)
    if "network" in p or "subgraph" in p or "counterparty graph" in p:
        return _mock_network_summary(prompt)
    if "draft a sar" in p or "suspicious activity report" in p:
        return _mock_sar_draft(prompt)
    return ("[mock LLM] No specialised template matched. The real LLM would "
            "produce a longer, more contextual analysis here.")


def _mock_transaction_summary(prompt: str) -> str:
    return (
        "Based on the transaction history I reviewed, three patterns stand out:\n\n"
        "1. **Multiple sub-CTR-threshold cash deposits** — the account received "
        "several deposits in the $9,000–$9,999 range within a 7-day window, "
        "consistent with classic structuring to evade Currency Transaction "
        "Report filing.\n"
        "2. **Round-dollar wire amounts** — outbound wires are predominantly "
        "in clean $5,000 / $10,000 increments, which is unusual for "
        "legitimate business spending.\n"
        "3. **Short hold time** — funds typically leave the account within "
        "24–48 hours of arrival, suggesting layering rather than ordinary "
        "deposit behaviour.\n\n"
        "Recommend escalation to network-level analysis to determine whether "
        "this account is part of a broader chain or fan-in pattern."
    )


def _mock_entity_summary(prompt: str) -> str:
    return (
        "Entity resolution complete:\n\n"
        "- **KYC status**: account opened 14 months ago with limited supporting "
        "documentation; declared occupation and observed transaction profile "
        "show notable mismatch (declared: 'consultant'; profile: high-volume "
        "incoming wires from multiple unrelated parties).\n"
        "- **Sanctions screening**: no direct hit against OFAC SDN; one "
        "counterparty has a fuzzy-match alert (60% similarity) requiring "
        "manual review.\n"
        "- **Adverse media**: nothing surfaced in the past 12 months.\n"
        "- **Geographic risk**: counterparties span 4 jurisdictions, including "
        "one FATF grey-list country, which elevates the residual risk score."
    )


def _mock_network_summary(prompt: str) -> str:
    return (
        "Network analysis findings:\n\n"
        "- **Subgraph size**: 14 accounts within 2 hops, 47 transactions over "
        "the past 30 days.\n"
        "- **Suspicious topology**: a fan-in pattern is present — five distinct "
        "originating accounts route funds through 2 intermediaries to a single "
        "collector. Classic mule-network signature.\n"
        "- **Cycle detection**: no closed cycles within the lookback window.\n"
        "- **Velocity**: median hold time at intermediaries is ~18 hours, "
        "well below the 48-hour layering threshold.\n"
        "- **Aggregate volume through the collector**: $186,400 in 30 days, "
        "originating from accounts with no apparent commercial relationship."
    )


def _mock_sar_draft(prompt: str) -> str:
    return (
        "**SUSPICIOUS ACTIVITY REPORT (DRAFT — for human review)**\n\n"
        "**Subject account**: as identified in the alert.\n\n"
        "**Summary of suspicious activity**:\n"
        "Between [start] and [end], the subject account exhibited several "
        "patterns consistent with money laundering. Specifically: (a) repeated "
        "structured cash deposits below the $10,000 CTR threshold; "
        "(b) rapid onward transfer of received funds to multiple counterparties "
        "within 24–48 hours of receipt; and (c) participation in a fan-in "
        "topology suggesting a mule-network role.\n\n"
        "**Supporting observations**:\n"
        "- Deposit pattern statistically inconsistent with declared occupation.\n"
        "- Counterparty network includes one fuzzy sanctions match requiring "
        "secondary review.\n"
        "- Aggregate volume through the network exceeds $180k over 30 days "
        "with no documented commercial purpose.\n\n"
        "**Recommended actions**:\n"
        "- Escalate for human investigator review prior to filing.\n"
        "- Hold pending outcome; do not close until investigator clears.\n"
        "- Consider 314(b) information sharing with peer institutions if mule "
        "network spans multiple banks.\n\n"
        "**Filing recommendation**: Suspicious — file SAR within statutory "
        "30-day window. Final determination requires qualified human "
        "investigator (BSA Officer) sign-off."
    )
