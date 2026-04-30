"""Probe OpenSanctions /match/default to confirm PEP topics are returned."""
import os
import requests

# Try env var first; if not set, paste key here for one-off test then delete it.
key = os.environ.get("OPENSANCTIONS_API_KEY") or "PASTE_KEY_HERE"

if key == "PASTE_KEY_HERE":
    raise SystemExit("Set OPENSANCTIONS_API_KEY env var or paste key into the file.")

URL = "https://api.opensanctions.org/match/default?algorithm=logic-v2"
body = {
    "queries": {
        "q1": {
            "schema": "Person",
            "properties": {"name": ["Volodymyr Zelensky"]},
        }
    }
}

r = requests.post(URL, headers={"Authorization": f"ApiKey {key}"}, json=body)
print(f"Status: {r.status_code}\n")

if r.status_code != 200:
    print(r.text[:500])
    raise SystemExit(1)

data = r.json()
results = data.get("responses", {}).get("q1", {}).get("results", [])
print(f"Hits: {len(results)}\n")

for h in results[:3]:
    caption = h.get("caption", "(unknown)")
    score = h.get("score", 0.0)
    topics = h.get("properties", {}).get("topics", [])
    datasets = h.get("datasets", [])[:3]
    print(f"- {caption}")
    print(f"    score:    {score:.3f}")
    print(f"    topics:   {topics}")
    print(f"    datasets: {datasets}")
    print()
    