"""Evaluation harness.

Runs the multi-agent investigation on a sample of accounts (some with known
laundering, some legit) and measures:

- True positive rate (correctly flags laundering accounts)
- False positive rate (incorrectly flags clean accounts)
- Mean time-to-resolution (total agent runtime per investigation)
- Indicator citation distribution

This is the part that makes the project defensible. Anyone can ship a
multi-agent demo. Few measure whether it actually works.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import pandas as pd

from src.agents.orchestrator import run_investigation
from src.config import ARTIFACTS_DIR
from src.data.features import FeatureStore, build_feature_store
from src.data.loader import load_data

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    account: str
    ground_truth_laundering: bool
    risk_score: float
    recommendation: str
    runtime_seconds: float
    indicators: List[str] = field(default_factory=list)


@dataclass
class EvalReport:
    n_total: int
    n_positive: int
    n_negative: int
    tp: int
    fp: int
    fn: int
    tn: int
    tpr: float
    fpr: float
    precision: float
    mean_runtime_seconds: float
    per_account: List[EvalResult]


def evaluate(store: FeatureStore,
             n_positive: int = 10,
             n_negative: int = 10,
             api_key: Optional[str] = None,
             seed: int = 42) -> EvalReport:
    df = store.transactions_df
    rng = pd.Series(range(len(df))).sample(frac=1, random_state=seed)

    # Pick accounts with known laundering activity
    laundering_accounts = (
        df[df["is_laundering"] == 1].groupby("from_account").size()
        .sort_values(ascending=False).head(n_positive).index.tolist()
    )
    # Pick legitimate but high-volume accounts (FP test)
    legit_accounts = (
        df[df["is_laundering"] == 0].groupby("from_account").size()
        .sort_values(ascending=False).head(n_negative * 3)
        .index.tolist()
    )
    # Filter out any legit accounts that ARE in the laundering set (defensive)
    legit_accounts = [a for a in legit_accounts if a not in set(laundering_accounts)]
    legit_accounts = legit_accounts[:n_negative]

    results: List[EvalResult] = []

    for acct in laundering_accounts + legit_accounts:
        is_pos = acct in set(laundering_accounts)
        t0 = time.time()
        final = run_investigation(account=acct, store=store, api_key=api_key)
        runtime = time.time() - t0
        sar = final["sar"]
        results.append(EvalResult(
            account=acct,
            ground_truth_laundering=is_pos,
            risk_score=sar.risk_score,
            recommendation=sar.recommendation,
            runtime_seconds=runtime,
            indicators=sar.indicators_cited or [],
        ))
        logger.info("[%s] %s -> %s (score=%.2f, %.1fs)",
                    "POS" if is_pos else "NEG", acct,
                    sar.recommendation, sar.risk_score, runtime)

    # Compute confusion matrix using FILE_SAR as positive prediction
    tp = sum(1 for r in results if r.ground_truth_laundering and r.recommendation == "FILE_SAR")
    fn = sum(1 for r in results if r.ground_truth_laundering and r.recommendation != "FILE_SAR")
    fp = sum(1 for r in results if not r.ground_truth_laundering and r.recommendation == "FILE_SAR")
    tn = sum(1 for r in results if not r.ground_truth_laundering and r.recommendation != "FILE_SAR")

    n_pos = sum(1 for r in results if r.ground_truth_laundering)
    n_neg = sum(1 for r in results if not r.ground_truth_laundering)

    tpr = tp / max(n_pos, 1)
    fpr = fp / max(n_neg, 1)
    precision = tp / max(tp + fp, 1)
    mean_runtime = sum(r.runtime_seconds for r in results) / max(len(results), 1)

    return EvalReport(
        n_total=len(results),
        n_positive=n_pos,
        n_negative=n_neg,
        tp=tp, fp=fp, fn=fn, tn=tn,
        tpr=tpr, fpr=fpr, precision=precision,
        mean_runtime_seconds=mean_runtime,
        per_account=results,
    )


def save_report(report: EvalReport,
                path: Path = ARTIFACTS_DIR / "eval_report.json") -> None:
    serializable = {
        "n_total": report.n_total,
        "n_positive": report.n_positive,
        "n_negative": report.n_negative,
        "tp": report.tp, "fp": report.fp, "fn": report.fn, "tn": report.tn,
        "tpr": report.tpr, "fpr": report.fpr, "precision": report.precision,
        "mean_runtime_seconds": report.mean_runtime_seconds,
        "per_account": [asdict(r) for r in report.per_account],
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2, default=str)
