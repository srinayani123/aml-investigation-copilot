"""Run the evaluation harness on a sample of accounts.

Usage:
    python scripts/run_evaluation.py
    python scripts/run_evaluation.py --n-positive 5 --n-negative 5

Uses mock LLM by default (so it runs without API keys). Set
ANTHROPIC_API_KEY in your environment to use real Claude calls.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ARTIFACTS_DIR  # noqa: E402
from src.data.features import build_feature_store  # noqa: E402
from src.data.loader import load_data  # noqa: E402
from src.evaluation.harness import evaluate, save_report  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s | %(message)s")
logger = logging.getLogger("evaluate")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-positive", type=int, default=10)
    parser.add_argument("--n-negative", type=int, default=10)
    args = parser.parse_args()

    logger.info("Loading data...")
    df = load_data()
    store = build_feature_store(df)
    logger.info("Loaded %d transactions, %d accounts", len(df), len(store.account_features))

    logger.info("Running evaluation: %d positive + %d negative accounts",
                args.n_positive, args.n_negative)
    report = evaluate(store, n_positive=args.n_positive, n_negative=args.n_negative)

    print("\n" + "=" * 60)
    print("=== EVALUATION REPORT ===")
    print("=" * 60)
    print(f"N total       : {report.n_total}")
    print(f"  positives   : {report.n_positive}")
    print(f"  negatives   : {report.n_negative}")
    print(f"True positive : {report.tp}")
    print(f"False positive: {report.fp}")
    print(f"False negative: {report.fn}")
    print(f"True negative : {report.tn}")
    print(f"TPR (recall)  : {report.tpr:.2%}")
    print(f"FPR           : {report.fpr:.2%}")
    print(f"Precision     : {report.precision:.2%}")
    print(f"Mean runtime  : {report.mean_runtime_seconds:.2f}s per investigation")

    save_report(report)
    print(f"\nFull report saved to {ARTIFACTS_DIR / 'eval_report.json'}")


if __name__ == "__main__":
    main()
