"""Probe Tavily search to confirm it works for adverse-media queries."""
import os
import requests
import json

key = os.environ.get("TAVILY_API_KEY") or "PASTE_KEY_HERE"
if key == "PASTE_KEY_HERE":
    raise SystemExit("Set TAVILY_API_KEY env var or paste key into the file.")

r = requests.post(
    "https://api.tavily.com/search",
    json={
        "api_key": key,
        "query": "FTX fraud money laundering investigation",
        "max_results": 3,
        "topic": "news",
    },
)
print(f"Status: {r.status_code}\n")
if r.status_code != 200:
    print(r.text[:500])
    raise SystemExit(1)

data = r.json()
results = data.get("results", [])
print(f"Hits: {len(results)}\n")
for art in results[:3]:
    print(f"- {art.get('title')}")
    print(f"    url:    {art.get('url')}")
    print(f"    score:  {art.get('score', 0):.3f}")
    print(f"    date:   {art.get('published_date', '?')}")
    print()
    