"""Probe GDELT to see if it's reachable and what it returns."""
import requests
import urllib.parse
import time

# Same query the production code builds
ADVERSE_KEYWORDS = (
    '(fraud OR "money laundering" OR investigation OR indicted '
    'OR sanctions OR "financial crime" OR scandal)'
)
NAME = "FTX"

query = f'"{NAME}" {ADVERSE_KEYWORDS}'
params = {
    "query": query,
    "mode": "artlist",
    "format": "json",
    "maxrecords": "5",
    "sort": "datedesc",
    "timespan": "365d",
}
url = f"https://api.gdeltproject.org/api/v2/doc/doc?{urllib.parse.urlencode(params)}"

print(f"Final URL:\n{url}\n")

t = time.time()
try:
    r = requests.get(url, timeout=30)
    elapsed = round(time.time() - t, 2)
    print(f"Elapsed: {elapsed}s")
    print(f"Status: {r.status_code}")
    print(f"Bytes: {len(r.content)}")
    print(f"Content-Type: {r.headers.get('content-type', '?')}")
    print(f"\nFirst 800 chars of body:")
    print(r.text[:800] if r.text else "(empty body)")
except requests.exceptions.Timeout:
    print(f"TIMEOUT after {round(time.time()-t,2)}s")
except requests.exceptions.ConnectionError as e:
    print(f"CONNECTION ERROR after {round(time.time()-t,2)}s: {e}")
except Exception as e:
    print(f"OTHER ERROR after {round(time.time()-t,2)}s: {type(e).__name__}: {e}")

# Also try the bare API URL with no query — that should always work
print("\n--- Bare API connectivity test ---")
try:
    r2 = requests.get("https://api.gdeltproject.org/api/v2/doc/doc?query=test&mode=artlist&format=json&maxrecords=1", timeout=15)
    print(f"Status: {r2.status_code}")
    print(f"Bytes: {len(r2.content)}")
    print(f"First 300: {r2.text[:300]}")
except Exception as e:
    print(f"Error: {type(e).__name__}: {e}")
    