"""Find real corporate accounts that exist in the IBM data and pass the
mock-CIF corporate filter, so we can verify GLEIF flows real LEI data
through the live UI."""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data.loader import load_data
from src.data.features import build_feature_store


def is_corporate(account: str) -> bool:
    """Mirrors the rule in entity_resolver._mock_cif."""
    h = int(hashlib.md5(account.encode()).hexdigest(), 16)
    return h % 5 == 0


print("Loading IBM AML data...")
df = load_data()
store = build_feature_store(df)

# Pick accounts that have meaningful activity (>50 txns)
print("Scanning for corporate accounts with real activity...")
candidates = []
for acct in list(store.account_features.keys())[:5000]:
    feats = store.account_features[acct]
    if feats.txn_count >= 50 and is_corporate(acct):
        candidates.append((acct, feats.txn_count, feats.total_volume_usd))
    if len(candidates) >= 10:
        break

print("\nFirst 10 corporate accounts with >=50 txns:")
print(f"{'Account':<14} {'Txns':>8} {'Volume USD':>16}")
print("-" * 42)
for acct, n, vol in candidates:
    print(f"{acct:<14} {n:>8,} ${vol:>14,.0f}")