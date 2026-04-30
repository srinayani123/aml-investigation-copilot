"""IBM AML (NeurIPS 2023) dataset loader, with HF auto-download + synthetic fallback.

Loading strategy (in order):
  1. Local file present at `data/raw/HI-Small_Trans.csv` → read directly
  2. Hugging Face Datasets mirror → download once, cache in HF cache dir, read
  3. Synthetic fallback → tiny fake dataset for tests / offline demo

The HF mirror lives at `srinayani123/ibm-aml-mirror` and is a verbatim copy of
the IBM HI-Small dataset (Altman et al., NeurIPS 2023). It exists only to make
Streamlit Cloud deployment work — the original dataset is too large for git.

Schema (canonical, used everywhere downstream):
  - timestamp     : datetime
  - from_bank     : str
  - from_account  : str
  - to_bank       : str
  - to_account    : str
  - amount_usd    : float (always converted to USD even if currency was foreign)
  - currency      : str
  - payment_format: str (Wire, Cash, ACH, Cheque, Credit Card, ...)
  - is_laundering : int (0/1)  ← ground truth label
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.config import RAW_DIR, data_cfg

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hugging Face mirror configuration
# ---------------------------------------------------------------------------

# Set HF_REPO_ID = None (or empty string) to disable HF download attempts —
# useful for fully-offline development.
HF_REPO_ID = "srinayani123/ibm-aml-mirror"
HF_REPO_TYPE = "dataset"


# ---------------------------------------------------------------------------
# IBM AML dataset — currency normalization
# ---------------------------------------------------------------------------

USD_RATES = {
    "US Dollar": 1.0, "Euro": 1.05, "UK Pound": 1.20, "Yuan": 0.14,
    "Yen": 0.0070, "Bitcoin": 30_000.0, "Australian Dollar": 0.66,
    "Canadian Dollar": 0.74, "Swiss Franc": 1.10, "Mexican Peso": 0.05,
    "Brazil Real": 0.18, "Saudi Riyal": 0.27, "Ruble": 0.014,
    "Rupee": 0.012, "Shekel": 0.27,
}


# ---------------------------------------------------------------------------
# Resolve the file path: local first, then HF download, then None
# ---------------------------------------------------------------------------

def _try_hf_download(filename: str) -> Optional[Path]:
    """Download `filename` from the HF mirror. Returns the cached local path
    on success, or None if HF is unreachable / disabled / the file is missing.

    huggingface_hub caches downloads in ~/.cache/huggingface/ by default, so
    subsequent calls with the same filename are instant (no re-download).
    """
    if not HF_REPO_ID:
        return None
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logger.warning(
            "huggingface_hub not installed; cannot download dataset. "
            "Add `huggingface_hub` to requirements.txt to enable."
        )
        return None

    try:
        logger.info("Downloading %s from HF mirror %s ...",
                    filename, HF_REPO_ID)
        path_str = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            repo_type=HF_REPO_TYPE,
        )
        logger.info("Downloaded %s → %s", filename, path_str)
        return Path(path_str)
    except Exception as e:
        logger.warning("HF download failed for %s: %s", filename, e)
        return None


def _resolve_transactions_path() -> Optional[Path]:
    """Return a usable path to the transactions CSV, or None if unavailable."""
    local = RAW_DIR / data_cfg.transactions_file
    if local.exists():
        return local
    return _try_hf_download(data_cfg.transactions_file)


def _resolve_patterns_path() -> Optional[Path]:
    local = RAW_DIR / data_cfg.patterns_file
    if local.exists():
        return local
    return _try_hf_download(data_cfg.patterns_file)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ibm_aml_available() -> bool:
    """True if the dataset can be loaded — either locally or via HF download."""
    return _resolve_transactions_path() is not None


def load_ibm_aml(max_rows: Optional[int] = None) -> pd.DataFrame:
    """Load the IBM AML CSV. Optionally subsample to fit memory.

    For demo purposes we keep at most `max_rows` (default 500k) — but we
    intentionally keep ALL laundering rows so the agents have signal.
    """
    txn_path = _resolve_transactions_path()
    if txn_path is None:
        raise FileNotFoundError(
            f"IBM AML file not found locally at {RAW_DIR / data_cfg.transactions_file} "
            f"and HF mirror {HF_REPO_ID} is unreachable. "
            f"See data/raw/PUT_DATA_HERE.md for instructions."
        )

    logger.info("Loading %s ...", txn_path.name)
    df = pd.read_csv(txn_path)
    logger.info("Raw shape: %s", df.shape)

    # The dataset's column names have spaces; standardize.
    df = df.rename(columns={
        "Timestamp": "timestamp",
        "From Bank": "from_bank",
        "Account": "from_account",      # sometimes the from-account column
        "To Bank": "to_bank",
        "Account.1": "to_account",      # sometimes the to-account column
        "Amount Received": "amount_received",
        "Receiving Currency": "receiving_currency",
        "Amount Paid": "amount_paid",
        "Payment Currency": "payment_currency",
        "Payment Format": "payment_format",
        "Is Laundering": "is_laundering",
    })

    # Some Kaggle exports use slightly different headers. Normalise.
    if "from_account" not in df.columns and "From Account" in df.columns:
        df = df.rename(columns={"From Account": "from_account"})
    if "to_account" not in df.columns and "To Account" in df.columns:
        df = df.rename(columns={"To Account": "to_account"})

    # Build the canonical USD amount.
    df["amount_usd"] = (
        df["amount_paid"].astype(float)
        * df["payment_currency"].map(USD_RATES).fillna(1.0)
    )

    # Stringify accounts (they're hex in the source) and bank ids.
    df["from_account"] = df["from_account"].astype(str)
    df["to_account"] = df["to_account"].astype(str)
    df["from_bank"] = df["from_bank"].astype(str)
    df["to_bank"] = df["to_bank"].astype(str)
    df["currency"] = df["payment_currency"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["is_laundering"] = df["is_laundering"].astype(int)

    keep = ["timestamp", "from_bank", "from_account", "to_bank", "to_account",
            "amount_usd", "currency", "payment_format", "is_laundering"]
    df = df[keep].dropna(subset=["timestamp"])

    # Optional subsample — but always keep all positives.
    if max_rows is not None and len(df) > max_rows:
        pos = df[df["is_laundering"] == 1]
        neg = df[df["is_laundering"] == 0].sample(
            n=max(max_rows - len(pos), 0), random_state=42)
        df = pd.concat([pos, neg]).sort_values("timestamp").reset_index(drop=True)
        logger.info("Subsampled to %d rows (%d positives kept)",
                    len(df), int(df["is_laundering"].sum()))

    return df


def load_patterns_doc() -> str:
    """Load the patterns description file (typology ground truth)."""
    p = _resolve_patterns_path()
    if p is not None and p.exists():
        return p.read_text(errors="replace")
    return "(Patterns file not available — see data/raw/PUT_DATA_HERE.md.)"


# ---------------------------------------------------------------------------
# Synthetic fallback (tiny, fast — for tests + offline demo)
# ---------------------------------------------------------------------------

def generate_synthetic(n_accounts: int = 200,
                       n_transactions: int = 5000,
                       seed: int = 42) -> pd.DataFrame:
    """Generate a small AML-style dataset with planted typologies.

    Plants:
      - 1 structuring case  (3 sub-CTR deposits in a week)
      - 1 layering chain    (A -> B -> C -> D -> E within 24h)
      - 1 fan-in pattern    (5 senders all to the same collector)
    """
    rng = np.random.default_rng(seed)
    accounts = [f"acct_{i:04x}" for i in range(n_accounts)]
    banks = [f"bank_{i}" for i in range(8)]
    formats = ["Wire", "ACH", "Cash", "Cheque", "Credit Card"]
    base_date = pd.Timestamp("2024-01-01")

    rows = []
    # Background legitimate traffic
    for _ in range(n_transactions):
        rows.append({
            "timestamp": base_date + pd.Timedelta(seconds=int(rng.integers(0, 90 * 86400))),
            "from_bank": str(rng.choice(banks)),
            "from_account": str(rng.choice(accounts)),
            "to_bank": str(rng.choice(banks)),
            "to_account": str(rng.choice(accounts)),
            "amount_usd": float(rng.lognormal(4, 1) * 10),
            "currency": "US Dollar",
            "payment_format": str(rng.choice(formats)),
            "is_laundering": 0,
        })

    # Planted: structuring (acct_0001 deposits 3x sub-$10k in a week)
    for i in range(3):
        rows.append({
            "timestamp": base_date + pd.Timedelta(days=10 + i),
            "from_bank": "bank_0", "from_account": "acct_0001",
            "to_bank": "bank_1", "to_account": "acct_0002",
            "amount_usd": float(9000 + rng.uniform(0, 999)),
            "currency": "US Dollar", "payment_format": "Cash",
            "is_laundering": 1,
        })

    # Planted: layering (A -> B -> C -> D within 24h)
    chain = ["acct_0010", "acct_0011", "acct_0012", "acct_0013"]
    for i in range(len(chain) - 1):
        rows.append({
            "timestamp": base_date + pd.Timedelta(days=20, hours=i * 4),
            "from_bank": "bank_2", "from_account": chain[i],
            "to_bank": "bank_3", "to_account": chain[i + 1],
            "amount_usd": 75_000.0,
            "currency": "US Dollar", "payment_format": "Wire",
            "is_laundering": 1,
        })

    # Planted: fan-in (acct_0050 receives from 6 senders in a week)
    for i in range(6):
        rows.append({
            "timestamp": base_date + pd.Timedelta(days=30 + i // 2),
            "from_bank": "bank_4", "from_account": f"acct_{0x60+i:04x}",
            "to_bank": "bank_5", "to_account": "acct_0050",
            "amount_usd": float(rng.uniform(2000, 4500)),
            "currency": "US Dollar", "payment_format": "ACH",
            "is_laundering": 1,
        })

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)
    return df


def load_data(use_synthetic_if_missing: bool = True,
              max_rows: Optional[int] = None) -> pd.DataFrame:
    """Top-level loader.

    Tries (in order): local file → HF download → synthetic fallback.
    """
    max_rows = max_rows or data_cfg.max_rows_in_memory
    if ibm_aml_available():
        return load_ibm_aml(max_rows=max_rows)
    if use_synthetic_if_missing:
        logger.warning("IBM AML files not available (local or HF) — using synthetic fallback.")
        return generate_synthetic()
    raise FileNotFoundError("IBM AML files not available locally or via HF mirror.")
