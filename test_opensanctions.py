"""Probe OpenSanctions /search endpoint."""
import requests

BASE = "https://api.opensanctions.org/search/default"

names = ["Putin", "Vladimir Putin", "Maduro", "Kim Jong"]

for n in names:
    print(f"\n=== Searching: {n} ===")
    try:
        r = requests.get(BASE, params={"q": n, "limit": 3}, timeout=8)
        print(f"Status: {r.status_code}")
        if r.status_code == 200:
            data = r.json()
            results = data.get("results", [])
            print(f"Total hits: {data.get('total', {}).get('value', 'unknown')}")
            print(f"Top {len(results)} results:")
            for hit in results:
                caption = hit.get("caption", "(unnamed)")
                schema = hit.get("schema", "?")
                topics = hit.get("properties", {}).get("topics", [])
                datasets = hit.get("datasets", [])[:3]
                print(f"  - {caption} [{schema}]")
                print(f"      topics: {topics}")
                print(f"      datasets: {datasets}")
        else:
            print(f"Body: {r.text[:300]}")
    except Exception as e:
        print(f"Error: {e}")
        