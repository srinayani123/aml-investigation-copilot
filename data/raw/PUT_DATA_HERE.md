# Place IBM AML data files HERE

## What you need

The IBM "Transactions for Anti-Money Laundering" dataset (NeurIPS 2023):
- Kaggle: https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml

After you download the Kaggle zip, extract these **TWO** files into this folder:

    HI-Small_Trans.csv       (~500 MB — required)
    HI-Small_Patterns.txt    (~1 KB — optional, but used in the "How it works" tab)

You can ignore the other variants (HI-Medium, HI-Large, LI-Small, LI-Medium,
LI-Large) — we use HI-Small because:
- Smallest file that fits in the Streamlit Cloud free tier (1 GB RAM)
- "High illicit ratio" (HI) gives the agents more fraud signal to work with
- Comparable to what the original NeurIPS paper used for benchmarking

After dropping the files here, run from the project root:

    python scripts/run_evaluation.py

Or just launch the demo:

    streamlit run src/ui/app.py

If the files aren't here, the app falls back to a small synthetic dataset
with three planted typologies (structuring, layering, fan-in). The synthetic
fallback is fine for testing but the real headlines come from IBM AML.
