"""Streamlit dashboard for the AML Investigation Copilot.

Design: dense, clean, banking-tool aesthetic. Mirrors how real compliance
software (Actimize, Mantas, Bloomberg) is laid out — toolbar at top, decision
strip prominent, supporting evidence in tight tables, status pills not emoji.

Three LLM modes (resolved at request time):
  1. Visitor key from sidebar  (per-session, never stored, never logged)
  2. ANTHROPIC_API_KEY env var  (deployment default / Streamlit secret)
  3. Mock fallback              (no keys, free, templated responses)
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import hashlib
import re
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.agents.llm import get_active_mode  # noqa: E402
from src.agents.orchestrator import run_investigation  # noqa: E402
from src.config import data_cfg, llm_cfg  # noqa: E402
from src.data.features import build_feature_store  # noqa: E402
from src.data.loader import load_data, load_patterns_doc  # noqa: E402


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AML Investigation Copilot",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _safe_text(text: str) -> str:
    """Prepare LLM-generated narrative text for safe markdown rendering.

    1. Escapes `$` so Streamlit/KaTeX doesn't interpret amounts as math.
    2. Strips redundant top-level markdown headers — the LLMs sometimes
       prepend `# Title` even when we ask for prose; we render the section
       label ourselves so the prose stays clean.
    """
    if not text:
        return ""
    # Escape $ to prevent LaTeX math-mode rendering ($4.5B → \$4.5B)
    out = text.replace("$", "\\$")
    # Strip a leading top-level markdown header line if the LLM added one
    out = re.sub(r"^\s*#\s+[^\n]*\n", "", out, count=1)
    return out.strip()


def _format_agent_name(raw: str) -> str:
    """Format an agent name like 'sar_drafter' → 'SAR Drafter'.

    Python's .title() naively lowercases letters after the first, so
    'sar_drafter'.replace('_', ' ').title() → 'Sar Drafter' (wrong).
    We patch known acronyms after title-casing.
    """
    base = raw.replace('_', ' ').title()
    # Acronym fixups — extend this map if new agents are added.
    return (base
            .replace('Sar', 'SAR')
            .replace('Kyc', 'KYC')
            .replace('Kyb', 'KYB')
            .replace('Pep', 'PEP'))


def _score_severity(score: float) -> str:
    """Map a 0..1 risk score to a severity bucket name."""
    if score >= 0.6:
        return "high"
    if score >= 0.35:
        return "med"
    return "low"


# ---------------------------------------------------------------------------
# CSS — dense banking-tool aesthetic, polished for product-feel
# ---------------------------------------------------------------------------

st.markdown("""
<style>
:root {
    --c-bg: #fafaf9;
    --c-surface: #ffffff;
    --c-surface-2: #f7f7f5;
    --c-border: #e5e5e3;
    --c-border-strong: #d4d4d2;
    --c-text: #18181a;
    --c-text-muted: #6b6b6b;
    --c-text-faint: #9b9b9b;
    --c-mono: ui-monospace, 'SF Mono', Menlo, Monaco, Consolas, monospace;
    /* Severity tokens */
    --c-sev-high: #a32d2d;
    --c-sev-high-bg: #fcebeb;
    --c-sev-med: #c4882b;
    --c-sev-med-bg: #faeeda;
    --c-sev-low: #1d7050;
    --c-sev-low-bg: #eaf3de;
}

.block-container { padding-top: 1.25rem; padding-bottom: 2rem; max-width: 1440px; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stStatusWidget"] { display: none; }

/* Type ramp */
.aml-label {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.7px;
    color: var(--c-text-muted); font-weight: 500; margin-bottom: 6px;
}
.aml-h2 {
    font-size: 13px; font-weight: 600; color: var(--c-text);
    margin: 0 0 4px 0;
}

/* Top brand strip — small but signals "real product" */
.aml-brand {
    display: flex; align-items: center; gap: 10px;
    padding-bottom: 10px; margin-bottom: 10px;
    border-bottom: 0.5px solid var(--c-border);
}
.aml-brand-mark {
    width: 22px; height: 22px; border-radius: 4px;
    background: linear-gradient(135deg, #1a1a1a 0%, #3a3a3a 100%);
    display: flex; align-items: center; justify-content: center;
    color: white; font-family: var(--c-mono); font-size: 11px; font-weight: 600;
}
.aml-brand-name {
    font-size: 13px; font-weight: 600; color: var(--c-text);
    letter-spacing: -0.1px;
}
.aml-brand-tag {
    font-size: 10px; color: var(--c-text-muted);
    text-transform: uppercase; letter-spacing: 0.6px;
    padding: 2px 7px; border: 0.5px solid var(--c-border-strong);
    border-radius: 3px; margin-left: 4px;
}
.aml-brand-spacer { flex: 1; }
.aml-brand-meta {
    font-family: var(--c-mono); font-size: 11px;
    color: var(--c-text-muted);
}

/* Toolbar (account picker / key stats) */
.aml-toolbar {
    display: flex; align-items: center; justify-content: space-between;
    padding: 10px 14px; background: var(--c-surface);
    border: 0.5px solid var(--c-border); border-radius: 6px;
    margin-bottom: 12px; font-size: 13px;
}
.aml-toolbar-left { display: flex; align-items: center; gap: 14px; }
.aml-toolbar-title { font-weight: 600; }
.aml-toolbar-divider { width: 1px; height: 14px; background: var(--c-border); }
.aml-toolbar-meta { font-family: var(--c-mono); font-size: 12px; color: var(--c-text-muted); }
.aml-toolbar-right { display: flex; align-items: center; gap: 8px; font-size: 11px; color: var(--c-text-muted); }

/* Pills */
.aml-pill {
    display: inline-block; padding: 3px 11px; border-radius: 999px;
    font-size: 11px; font-weight: 600; letter-spacing: 0.4px;
    text-transform: uppercase;
}
.aml-pill-red { background: var(--c-sev-high-bg); color: var(--c-sev-high); }
.aml-pill-amber { background: var(--c-sev-med-bg); color: var(--c-sev-med); }
.aml-pill-green { background: var(--c-sev-low-bg); color: var(--c-sev-low); }
.aml-pill-gray { background: #f1efe8; color: #444441; }

/* Dots */
.aml-dot { display: inline-block; width: 7px; height: 7px; border-radius: 50%; vertical-align: middle; }
.aml-dot-green { background: #1d9e75; }
.aml-dot-amber { background: var(--c-sev-med); }
.aml-dot-red { background: var(--c-sev-high); }
.aml-dot-gray { background: #888780; }

/* Cards */
.aml-card {
    background: var(--c-surface); border: 0.5px solid var(--c-border);
    border-radius: 6px; padding: 14px 16px; margin-bottom: 12px;
}
.aml-card-tight { padding: 12px 14px; }

/* Decision strip */
.aml-decision-row {
    display: flex; align-items: center; gap: 14px; margin-bottom: 12px;
}
.aml-score-big {
    font-family: var(--c-mono); font-size: 28px; font-weight: 600;
    color: var(--c-text); line-height: 1;
}
.aml-score-big.sev-high { color: var(--c-sev-high); }
.aml-score-big.sev-med { color: var(--c-sev-med); }
.aml-score-big.sev-low { color: var(--c-sev-low); }
.aml-score-label {
    font-size: 10px; color: var(--c-text-muted); text-transform: uppercase;
    letter-spacing: 0.5px;
}
.aml-decision-meta {
    margin-left: auto; display: flex; gap: 18px; font-size: 12px;
    color: var(--c-text-muted);
}

.aml-reasoning {
    font-size: 13px; line-height: 1.6; color: var(--c-text);
    padding: 11px 13px; background: var(--c-surface-2); border-radius: 4px;
    border-left: 2px solid var(--c-border-strong);
}
.aml-reasoning.sev-high { border-left-color: var(--c-sev-high); }
.aml-reasoning.sev-med { border-left-color: var(--c-sev-med); }
.aml-reasoning.sev-low { border-left-color: var(--c-sev-low); }

/* Score breakdown bars */
.aml-bar-row { margin-top: 6px; }
.aml-bar-label {
    display: flex; justify-content: space-between; font-size: 11px;
    color: var(--c-text-muted); margin-bottom: 4px;
}
.aml-bar-track {
    height: 4px; background: #ececea; border-radius: 2px; overflow: hidden;
}
.aml-bar-fill { height: 100%; background: #4a4a4a; }

/* Tables */
.aml-table { width: 100%; font-size: 12px; border-collapse: collapse; }
.aml-table td { padding: 7px 0; border-bottom: 0.5px solid var(--c-border); }
.aml-table tr:last-child td { border-bottom: none; }
.aml-mono { font-family: var(--c-mono); font-size: 12px; }
.aml-tag {
    font-size: 10px; color: var(--c-text-muted); text-transform: uppercase;
    letter-spacing: 0.5px; font-weight: 500;
}

/* Provenance pills (refined) */
.aml-prov-row {
    display: flex; flex-wrap: wrap; gap: 6px; align-items: center;
}
.aml-prov-pill {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 11px; padding: 3px 9px; border-radius: 4px;
    background: var(--c-surface-2); border: 0.5px solid var(--c-border);
    color: var(--c-text-muted);
}
.aml-prov-pill.real { color: var(--c-text); }
.aml-prov-pill.real .aml-prov-name { color: var(--c-text); font-weight: 500; }
.aml-prov-pill.mock .aml-prov-name { color: var(--c-text-faint); font-style: italic; }
.aml-prov-pill.sim .aml-prov-name { color: var(--c-text); }
.aml-prov-key {
    font-size: 10px; text-transform: uppercase; letter-spacing: 0.5px;
    color: var(--c-text-muted);
}
.aml-prov-name { font-size: 11px; }

/* Footer governance bar */
.aml-footer {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 14px; background: var(--c-surface-2);
    border-radius: 4px; font-size: 11px; color: var(--c-text-muted);
    margin-top: 8px;
}

/* Streamlit metric overrides */
[data-testid="stMetric"] { background: transparent; padding: 0; }
[data-testid="stMetricLabel"] {
    font-size: 10px !important; text-transform: uppercase;
    letter-spacing: 0.5px; color: var(--c-text-muted) !important;
    font-weight: 500 !important;
}
[data-testid="stMetricValue"] {
    font-family: var(--c-mono) !important; font-size: 18px !important;
    font-weight: 500 !important; color: var(--c-text) !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 0.5px solid var(--c-border);
}
.stTabs [data-baseweb="tab"] {
    height: 38px; padding: 0 18px; font-size: 12px;
    color: var(--c-text-muted); font-weight: 500;
}
.stTabs [aria-selected="true"] {
    color: var(--c-text) !important;
    border-bottom-color: var(--c-text) !important;
    font-weight: 600 !important;
}

/* Sidebar */
section[data-testid="stSidebar"] .block-container { padding-top: 1.5rem; }
section[data-testid="stSidebar"] {
    background: var(--c-surface) !important;
    border-right: 0.5px solid var(--c-border);
}

/* Buttons */
.stButton > button {
    border-radius: 4px; font-size: 13px; font-weight: 500;
    border: 0.5px solid var(--c-border-strong);
    transition: all 0.15s ease;
}
.stButton > button[kind="primary"] {
    background: #18181a; color: white; border-color: #18181a;
}
.stButton > button[kind="primary"]:hover {
    background: #2c2c2e; border-color: #2c2c2e; color: white;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08);
}

/* Narrative prose */
.aml-prose {
    font-size: 13px; line-height: 1.65; color: var(--c-text);
}
.aml-prose strong { color: var(--c-text); font-weight: 600; }
.aml-prose p { margin: 0 0 10px 0; }

/* SAR draft regulatory list */
.aml-reg-row {
    display: flex; align-items: center; gap: 8px;
    font-size: 12px; padding: 5px 0;
    color: var(--c-text);
}
.aml-reg-bullet {
    width: 4px; height: 4px; border-radius: 50%;
    background: var(--c-text-muted); flex-shrink: 0;
}

code { font-size: 11.5px; background: var(--c-surface-2); padding: 1px 5px; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data loading
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading transaction data...")
def get_store():
    df = load_data()
    return build_feature_store(df), df


@st.cache_data
def get_patterns_doc():
    return load_patterns_doc()


@st.cache_data(ttl=300, show_spinner=False)
def cached_run_investigation(account: str, api_key_hash: str, _store_id: str):
    return run_investigation(
        account=account,
        store=st.session_state["_aml_store"],
        api_key=st.session_state.get("_aml_api_key") or None,
    )


def run_with_cache(account: str, store, api_key: Optional[str]):
    st.session_state["_aml_store"] = store
    st.session_state["_aml_api_key"] = api_key
    api_key_hash = hashlib.sha256((api_key or "mock").encode()).hexdigest()[:8]
    store_id = str(id(store))
    return cached_run_investigation(account, api_key_hash, store_id)


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------

def _prov_pill(key: str, name: str, kind: str) -> str:
    """Render one provenance pill. kind ∈ {'real', 'mock', 'sim'}."""
    return (f'<span class="aml-prov-pill {kind}">'
            f'<span class="aml-prov-key">{key}</span>'
            f'<span class="aml-prov-name">{name}</span>'
            f'</span>')


def _format_provenance(ds: dict) -> str:
    """Build the per-source provenance row. Each source becomes a small pill
    with the API key (Sanctions/PEP/Adverse/KYB/KYC) and provider name."""
    pills = []

    # Sanctions
    if ds.get("sanctions") == "real":
        pills.append(_prov_pill("Sanctions", "OpenSanctions", "real"))
    else:
        pills.append(_prov_pill("Sanctions", "mock", "mock"))

    # PEP
    if ds.get("pep") == "real":
        pills.append(_prov_pill("PEP", "OpenSanctions", "real"))
    else:
        pills.append(_prov_pill("PEP", "mock", "mock"))

    # Adverse media
    adverse_state = ds.get("adverse_media", "mock")
    if adverse_state == "tavily":
        pills.append(_prov_pill("Adverse", "Tavily", "real"))
    elif adverse_state == "gdelt":
        pills.append(_prov_pill("Adverse", "GDELT", "real"))
    else:
        pills.append(_prov_pill("Adverse", "mock", "mock"))

    # KYB / LEI
    lei_state = ds.get("lei", "n/a")
    if lei_state == "real":
        pills.append(_prov_pill("KYB", "GLEIF", "real"))
    elif lei_state == "n/a":
        pills.append(_prov_pill("KYB", "n/a", "mock"))
    else:
        pills.append(_prov_pill("KYB", "mock", "mock"))

    # KYC — always simulated CIF
    pills.append(_prov_pill("KYC", "CIF (sim)", "sim"))

    return f'<div class="aml-prov-row">{"".join(pills)}</div>'


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    # Brand strip in sidebar
    st.markdown("""
<div class="aml-brand">
  <div class="aml-brand-mark">A</div>
  <div>
    <div class="aml-brand-name">AML Copilot</div>
    <div style="font-size: 10px; color: var(--c-text-muted); margin-top: 2px;">v1.0 · investigation suite</div>
  </div>
</div>
""", unsafe_allow_html=True)

    st.markdown('<div class="aml-label">Configuration</div>', unsafe_allow_html=True)

    visitor_key = st.text_input(
        "Anthropic API key",
        type="password",
        placeholder="sk-ant-... (optional)",
        help="Per-session only. Not stored, not logged.",
        key="visitor_api_key",
    )

    active_mode = get_active_mode(runtime_api_key=visitor_key)
    if active_mode == "real" and visitor_key:
        mode_pill = '<span class="aml-pill aml-pill-green">live · session key</span>'
    elif active_mode == "real":
        mode_pill = '<span class="aml-pill aml-pill-green">live · default key</span>'
    else:
        mode_pill = '<span class="aml-pill aml-pill-gray">mock</span>'
    st.markdown(f"**Mode** &nbsp; {mode_pill}", unsafe_allow_html=True)
    st.markdown(
        f'<div class="aml-mono" style="color: var(--c-text-muted); margin-top: 4px;">{llm_cfg.model}</div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown('<div class="aml-label">External integrations</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="font-size: 11px; line-height: 1.7; color: var(--c-text-muted);">
<div><strong style="color: var(--c-text);">OpenSanctions</strong> &middot; sanctions + PEP</div>
<div><strong style="color: var(--c-text);">Tavily</strong> &middot; adverse media (primary)</div>
<div><strong style="color: var(--c-text);">GDELT</strong> &middot; adverse media (fallback)</div>
<div><strong style="color: var(--c-text);">GLEIF</strong> &middot; legal entity / KYB</div>
<div style="color: var(--c-text-faint);">CIF &middot; bank-internal (simulated)</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="aml-label">About</div>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size: 11.5px; line-height: 1.6; color: var(--c-text-muted);">'
        'Multi-agent investigation copilot. Four specialized agents coordinate '
        'via LangGraph to investigate flagged accounts and draft Suspicious '
        'Activity Reports for human review.'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div style="font-size: 11px; color: var(--c-text-faint); margin-top: 8px;">'
        'Data: <a href="https://www.kaggle.com/datasets/ealtman2019/'
        'ibm-transactions-for-anti-money-laundering-aml" '
        'style="color: var(--c-text-muted);">IBM AML (NeurIPS 2023)</a>'
        '</div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

try:
    store, df = get_store()
except FileNotFoundError as e:
    st.error(f"{e}")
    st.stop()


# ---------------------------------------------------------------------------
# Top brand strip — gives the page a "this is a real product" header
# ---------------------------------------------------------------------------

mode_dot_color = "#1d9e75" if active_mode == "real" else "#888780"
mode_label = "Live LLM" if active_mode == "real" else "Mock mode"

st.markdown(f"""
<div class="aml-brand">
  <div class="aml-brand-mark">A</div>
  <div class="aml-brand-name">AML Investigation Copilot</div>
  <div class="aml-brand-tag">Compliance · BSA / AML</div>
  <div class="aml-brand-spacer"></div>
  <div class="aml-brand-meta">{len(df):,} txns &middot; {len(store.account_features):,} accounts &middot; laundering rate {100*df['is_laundering'].mean():.3f}%</div>
  <span style="font-size: 11px; color: var(--c-text-muted); margin-left: 14px;">{mode_label}</span>
  <span class="aml-dot" style="background: {mode_dot_color};"></span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Account picker row
# ---------------------------------------------------------------------------

laundering_accounts = df[df["is_laundering"] == 1].groupby(
    "from_account").size().sort_values(ascending=False).head(8).index.tolist()
legit_high_vol = df[df["is_laundering"] == 0].groupby(
    "from_account").size().sort_values(ascending=False).head(4).index.tolist()
legit_high_vol = [a for a in legit_high_vol if a not in set(laundering_accounts)]

picker_options = ["—"] + laundering_accounts + ["———"] + legit_high_vol

col_pick, col_manual, col_btn = st.columns([3, 2, 1])
with col_pick:
    picked = st.selectbox(
        "Account",
        picker_options,
        index=1 if len(picker_options) > 1 else 0,
        label_visibility="collapsed",
    )
with col_manual:
    manual = st.text_input(
        "Manual account ID",
        placeholder="or paste account ID",
        label_visibility="collapsed",
    )
with col_btn:
    run_clicked = st.button("Run investigation", type="primary",
                            use_container_width=True)

target_account = manual.strip() or (picked if picked not in {"—", "———"} else "")


# ---------------------------------------------------------------------------
# Investigation results
# ---------------------------------------------------------------------------

if not target_account:
    st.markdown(
        '<div class="aml-card" style="text-align: center; color: var(--c-text-muted); '
        'padding: 40px 32px; font-size: 13px;">'
        'Select an account from the dropdown or paste an account ID, then run the investigation.'
        '<div style="font-size: 11px; color: var(--c-text-faint); margin-top: 6px;">'
        'Top items in the dropdown are accounts with confirmed laundering ground truth.</div>'
        '</div>',
        unsafe_allow_html=True,
    )
elif run_clicked or st.session_state.get("_last_account") == target_account:
    st.session_state["_last_account"] = target_account

    with st.spinner(f"Investigating {target_account}..."):
        final_state = run_with_cache(
            account=target_account,
            store=store,
            api_key=visitor_key or None,
        )

    sar = final_state["sar"]
    txn = final_state["txn_findings"]
    ent = final_state["entity_findings"]
    net = final_state["network_findings"]
    trail = final_state["audit_trail"]

    # ---- Decision strip --------------------------------------------------
    pill_class = {
        "FILE_SAR": "aml-pill-red",
        "MONITOR": "aml-pill-amber",
        "NO_ACTION": "aml-pill-green",
    }[sar.recommendation]
    pill_label = sar.recommendation.replace("_", " ")

    sev = _score_severity(sar.risk_score)
    sev_class_map = {"high": "sev-high", "med": "sev-med", "low": "sev-low"}
    sev_class = sev_class_map[sev]

    total_runtime = sum(t["duration_ms"] for t in trail) / 1000
    n_indicators = len(sar.indicators_cited or [])

    headline_safe = _safe_text(sar.headline) or "No reasoning generated."

    st.markdown(f"""
<div class="aml-card">
  <div class="aml-decision-row">
    <span class="aml-pill {pill_class}">{pill_label}</span>
    <span class="aml-score-big {sev_class}">{sar.risk_score:.2f}</span>
    <span class="aml-score-label">risk score</span>
    <div class="aml-decision-meta">
      <span><span class="aml-mono">{n_indicators}</span> indicators</span>
      <span><span class="aml-mono">{total_runtime:.2f}s</span> runtime</span>
      <span><span class="aml-mono">4</span> agents</span>
      <span><span class="aml-mono">{target_account}</span></span>
    </div>
  </div>
  <div class="aml-reasoning {sev_class}">{headline_safe}</div>
""", unsafe_allow_html=True)

    bd = sar.score_breakdown or {}
    max_per_agent = 0.5

    def _bar(label: str, val: float) -> str:
        pct = min(100, (val / max_per_agent) * 100)
        return f"""
<div class="aml-bar-row">
  <div class="aml-bar-label">
    <span>{label}</span>
    <span class="aml-mono" style="color: var(--c-text);">+{val:.2f}</span>
  </div>
  <div class="aml-bar-track">
    <div class="aml-bar-fill" style="width: {pct}%;"></div>
  </div>
</div>"""

    st.markdown(f"""
<div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 18px; margin-top: 14px;">
  {_bar("Transaction patterns", bd.get("transaction", 0))}
  {_bar("Entity factors", bd.get("entity", 0))}
  {_bar("Network topology", bd.get("network", 0))}
</div>
</div>
""", unsafe_allow_html=True)

    # ---- Two columns: factors + execution -------------------------------
    col_factors, col_exec = st.columns(2)

    with col_factors:
        rows = []
        for f in (sar.contributing_factors or [])[:5]:
            agent_tag = {
                "transaction": "TXN",
                "entity": "ENT",
                "network": "NET",
            }.get(f["agent"], "—")
            rows.append(f"""
<tr>
  <td class="aml-mono" style="color: var(--c-text-muted); width: 44px;">+{f['contribution']:.2f}</td>
  <td>{f['factor']}</td>
  <td class="aml-tag" style="text-align: right; width: 36px;">{agent_tag}</td>
</tr>""")
        if not rows:
            rows = ['<tr><td colspan="3" style="color: var(--c-text-muted); padding: 12px 0;">No factors triggered.</td></tr>']

        st.markdown(f"""
<div class="aml-card">
  <div class="aml-label">Top contributing factors</div>
  <table class="aml-table">{"".join(rows)}</table>
</div>
""", unsafe_allow_html=True)

    with col_exec:
        exec_rows = []
        for t in trail:
            agent_display = _format_agent_name(t['agent'])
            exec_rows.append(f"""
<tr>
  <td>{agent_display}</td>
  <td class="aml-mono" style="text-align: right; color: var(--c-text-muted);">{t['duration_ms']:.0f} ms</td>
  <td style="text-align: right; width: 28px;"><span class="aml-dot aml-dot-green"></span></td>
</tr>""")

        provenance_html = _format_provenance(ent.data_sources or {})

        st.markdown(f"""
<div class="aml-card">
  <div class="aml-label">Agent execution</div>
  <table class="aml-table">{"".join(exec_rows)}</table>
  <div style="margin-top: 12px; padding-top: 12px; border-top: 0.5px solid var(--c-border);">
    <div class="aml-label" style="margin-bottom: 8px;">Data sources</div>
    {provenance_html}
    <div style="display: flex; justify-content: flex-end; margin-top: 10px;
                font-size: 11px; color: var(--c-text-muted);">
      Total <span class="aml-mono" style="color: var(--c-text); margin-left: 6px;">{total_runtime:.2f}s</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ---- Agent findings tabs --------------------------------------------
    tab_txn, tab_ent, tab_net, tab_sar = st.tabs([
        "Transaction", "Entity", "Network", "SAR draft",
    ])

    with tab_txn:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Txn count", txn.txn_count_total)
        c2.metric("Volume", f"${txn.total_volume_usd:,.0f}")
        c3.metric("Structuring", len(txn.structuring_events))
        c4.metric("Throughput", txn.rapid_throughput_count)

        st.markdown('<div class="aml-label" style="margin-top: 18px;">Pattern summary</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="aml-prose">{_safe_text(txn.summary) or "(no summary generated)"}</div>',
            unsafe_allow_html=True,
        )
        if txn.structuring_events:
            st.markdown('<div class="aml-label" style="margin-top: 16px;">Structuring detail</div>',
                        unsafe_allow_html=True)
            st.json(txn.structuring_events, expanded=False)

    with tab_ent:
        # 5 metrics: KYC | Sanctions | PEP | Adverse Media | Geo Risk
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("KYC", ent.kyc_status)
        c2.metric("Sanctions", len(ent.sanctions_hits))
        c3.metric("PEP", len(ent.pep_hits))
        c4.metric("Adverse media", len(ent.adverse_media_hits))
        c5.metric("Geo risk", f"{ent.geo_risk_score:.2f}")

        # Entity-type subline
        entity_type = getattr(ent, "entity_type", "individual")
        lei_records = getattr(ent, "lei_records", []) or []
        if entity_type == "corporate":
            lei_text = (f'LEI / KYB matches: <span class="aml-mono" style="color: var(--c-text);">'
                        f'{len(lei_records)}</span>')
        else:
            lei_text = 'LEI / KYB: <span style="color: var(--c-text-faint);">n/a (individual)</span>'

        st.markdown(
            f'<div style="font-size: 12px; color: var(--c-text-muted); margin-top: 10px;">'
            f'Entity type: <span style="color: var(--c-text); font-weight: 500;">{entity_type}</span> '
            f'&nbsp;&middot;&nbsp; '
            f'Declared: <span style="color: var(--c-text);">{ent.declared_occupation}</span> '
            f'&nbsp;&middot;&nbsp; '
            f'Account age: <span class="aml-mono" style="color: var(--c-text);">{ent.account_age_days} days</span> '
            f'&nbsp;&middot;&nbsp; '
            f'{lei_text}'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown('<div class="aml-label" style="margin-top: 18px;">Entity risk assessment</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="aml-prose">{_safe_text(ent.summary) or "(no summary generated)"}</div>',
            unsafe_allow_html=True,
        )

        if ent.sanctions_hits:
            st.markdown('<div class="aml-label" style="margin-top: 16px;">Sanctions matches</div>',
                        unsafe_allow_html=True)
            st.json(ent.sanctions_hits, expanded=False)
        if getattr(ent, "pep_hits", None):
            st.markdown('<div class="aml-label" style="margin-top: 16px;">PEP matches</div>',
                        unsafe_allow_html=True)
            st.json(ent.pep_hits, expanded=False)
        if ent.adverse_media_hits:
            st.markdown('<div class="aml-label" style="margin-top: 16px;">Adverse media</div>',
                        unsafe_allow_html=True)
            st.json(ent.adverse_media_hits, expanded=False)
        if lei_records:
            st.markdown('<div class="aml-label" style="margin-top: 16px;">Legal entity records (GLEIF)</div>',
                        unsafe_allow_html=True)
            st.json(lei_records, expanded=False)

    with tab_net:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Subgraph nodes", net.subgraph_n_nodes)
        c2.metric("Subgraph edges", net.subgraph_n_edges)
        c3.metric("Fan-in", len(net.fan_in_collectors))
        c4.metric("Layering chains", len(net.layering_chains))

        st.markdown('<div class="aml-label" style="margin-top: 18px;">Network topology summary</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="aml-prose">{_safe_text(net.summary) or "(no summary generated)"}</div>',
            unsafe_allow_html=True,
        )
        if net.fan_in_collectors:
            st.markdown('<div class="aml-label" style="margin-top: 16px;">Fan-in patterns</div>',
                        unsafe_allow_html=True)
            st.json(net.fan_in_collectors, expanded=False)
        if net.layering_chains:
            st.markdown('<div class="aml-label" style="margin-top: 16px;">Layering chains</div>',
                        unsafe_allow_html=True)
            st.json(net.layering_chains, expanded=False)

    with tab_sar:
        st.markdown(
            '<div style="background: var(--c-sev-med-bg); color: var(--c-sev-med); '
            'font-size: 12px; padding: 9px 12px; border-radius: 4px; '
            'margin-bottom: 14px; display: flex; align-items: center; gap: 8px;">'
            '<span class="aml-dot aml-dot-amber"></span>'
            '<span><strong>Draft for human review.</strong> '
            'Filing requires BSA Officer sign-off per 31 CFR 1020.320.</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        if sar.indicators_cited:
            st.markdown('<div class="aml-label">Regulatory indicators cited</div>',
                        unsafe_allow_html=True)
            for ind in sar.indicators_cited:
                st.markdown(
                    f'<div class="aml-reg-row">'
                    f'<span class="aml-reg-bullet"></span>'
                    f'<span>{ind}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        st.markdown('<div class="aml-label" style="margin-top: 16px;">Narrative</div>',
                    unsafe_allow_html=True)
        st.markdown(
            f'<div class="aml-prose">{_safe_text(sar.narrative)}</div>',
            unsafe_allow_html=True,
        )

    # ---- Footer governance bar ------------------------------------------
    st.markdown("""
<div class="aml-footer">
  <span class="aml-dot aml-dot-amber"></span>
  <span><strong style="color: var(--c-text-muted);">Decision support tool.</strong>
    Filing decisions require BSA Officer review per 31 CFR 1020.320.
    Full audit trail captured in the agent execution panel.</span>
</div>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Bottom: collapsed dataset and how-it-works panels
# ---------------------------------------------------------------------------

st.markdown("<br>", unsafe_allow_html=True)

with st.expander("Dataset overview"):
    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="aml-label">Payment formats</div>',
                    unsafe_allow_html=True)
        st.bar_chart(df["payment_format"].value_counts())
    with c2:
        st.markdown('<div class="aml-label">Laundering by payment format</div>',
                    unsafe_allow_html=True)
        st.bar_chart(df[df["is_laundering"] == 1]["payment_format"].value_counts())
    st.markdown('<div class="aml-label" style="margin-top: 12px;">Sample transactions</div>',
                unsafe_allow_html=True)
    st.dataframe(df.head(50), use_container_width=True, height=240)

with st.expander("How it works"):
    st.markdown("""
**Architecture.** Four specialized agents coordinate via LangGraph in a
fan-out / fan-in topology. The Transaction Investigator, Entity Resolver,
and Network Analyst run in parallel; the SAR Drafter waits for all three
to complete before synthesizing the findings.

**Risk scoring.** Deterministic — each detected pattern adds a fixed
contribution to the score. The LLM only writes the narrative; it does not
move the score. Auditable by design.

**External data sources.** The Entity Resolver integrates four production-grade
public APIs with graceful mock-fallback:

- **Sanctions + PEP screening** — [OpenSanctions /match/default](https://www.opensanctions.org/docs/api/matching/),
  the open-source equivalent of ComplyAdvantage and Refinitiv World-Check.
  Aggregates 320+ watchlists (OFAC SDN, EU FSF, UN Security Council, UK HMT,
  national PEP databases). A single call returns scored matches with
  topic-tagged classifications (`role.pep`, `sanction`, `crime`, etc.) which
  the agent splits into separate sanctions and PEP findings.
- **Adverse media** — [Tavily AI Search](https://app.tavily.com/) primary,
  [GDELT DOC 2.0](https://blog.gdeltproject.org/gdelt-doc-2-0-api-debuts/) fallback.
  Tavily provides AI-optimized news search; GDELT covers 65 languages of
  online news with no key required.
- **Legal entity verification (KYB)** — [GLEIF](https://www.gleif.org/) Legal
  Entity Identifier API. The G20/Financial Stability Board-mandated standard
  for corporate counterparty identification, required under MiFID II /
  Dodd-Frank derivatives reporting. Free, no key, public reference data.
  Called only when the CIF lookup returns `entity_type=corporate`.

**KYC architecture.** Individual KYC is intentionally simulated as a bank-internal
CIF (Customer Information File) lookup, not an external vendor call. In production,
KYC at investigation time is a CIF read of previously-collected onboarding data —
not a fresh document/selfie verification. Public KYC verification APIs (Onfido,
Sumsub, Didit) operate at onboarding time on real customer documents and would
not apply to existing-customer investigation workflows. The mock simulates the
CIF response shape (status, occupation, account age, document quality, entity
type) so the agent code is identical to production. Replacing the mock means
swapping in the bank's own internal CIF service over mTLS; no agent logic changes.

**Governance.** No SAR is filed automatically. Every recommendation is a
draft requiring human review per 31 CFR 1020.320. Full audit trail captured
in the agent execution panel above.
""")
    st.markdown('<div class="aml-label" style="margin-top: 12px;">Planted typologies in IBM AML</div>',
                unsafe_allow_html=True)
    st.text(get_patterns_doc())
    