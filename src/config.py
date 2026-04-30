"""Centralized configuration."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

for _d in (DATA_DIR, RAW_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


@dataclass
class DataConfig:
    # IBM AML (NeurIPS 2023, ealtman2019) — HI-Small variant
    transactions_file: str = "HI-Small_Trans.csv"
    patterns_file: str = "HI-Small_Patterns.txt"

    # For the in-memory subset shown to agents (Streamlit Cloud has 1 GB cap)
    max_rows_in_memory: int = 500_000

    # Default investigation window
    investigation_lookback_days: int = 30


@dataclass
class LLMConfig:
    """LLM mode resolution priority:

    1. Visitor key passed via the Streamlit sidebar (per-session, never stored)
    2. ANTHROPIC_API_KEY env var (your key for local testing or Streamlit secret)
    3. Mock fallback (no keys, free, templated responses)
    """
    model: str = os.getenv("AML_LLM_MODEL", "claude-haiku-4-5-20251001")
    max_tokens: int = 1024
    temperature: float = 0.0
    # Default key from env (your testing key, or Streamlit secret)
    default_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")


@dataclass
class DetectionThresholds:
    """Thresholds for the heuristic checks each agent uses.

    These are the levers a real AML team would tune based on tolerance for
    false positives. Defaults are conservative — the evaluation harness
    measures TPR/FPR so you can see the trade-off.
    """
    # Structuring (sub-CTR-threshold deposits)
    ctr_threshold_usd: float = 10_000.0
    structuring_window_days: int = 7
    structuring_min_count: int = 3
    structuring_amount_band_low: float = 9_000.0
    structuring_amount_band_high: float = 9_999.99

    # Layering (money flowing through quickly)
    layering_max_hold_hours: float = 48.0
    layering_min_chain_length: int = 3
    layering_min_total_usd: float = 50_000.0

    # Velocity (unusually high txn count)
    high_velocity_txns_per_day: int = 15

    # Round-amount detection
    round_amount_modulus: float = 1000.0
    round_amount_min_count: int = 3

    # Fan-in / fan-out (mule networks)
    fan_in_min_senders: int = 5
    fan_in_window_days: int = 7
    fan_out_min_receivers: int = 5

    # SAR recommendation thresholds (policy weights — set by BSA Officer
    # based on the bank's risk appetite, NOT learned from data, per
    # OCC SR 11-7 explainability guidance for compliance models).
    sar_file_threshold: float = 0.55      # >= this score → FILE_SAR
    sar_monitor_threshold: float = 0.25   # >= this score → MONITOR
                                          # below → NO_ACTION


data_cfg = DataConfig()
llm_cfg = LLMConfig()
thresholds = DetectionThresholds()
