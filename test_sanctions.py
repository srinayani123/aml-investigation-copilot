"""Probe sanctions.network to find the correct query format."""
import requests
import json

BASE = "https://api.sanctions.network/sanctions"
NAME = "Putin"

# Format 1: original PostgREST ilike with asterisks (current code)
attempts = [
    ("PostgREST ilike with *", {"name": f"ilike.*{NAME}*", "limit": "3"}),
    ("PostgREST ilike with %", {"name": f"ilike.%{NAME}%", "limit": "3"}),
    ("Simple eq match", {"name": f"eq.{NAME}", "limit": "3"}),
    ("Simple q param", {"q": NAME, "limit": "3"}),
    ("Search param", {"search": NAME, "limit": "3"}),
    ("No params", {"limit": "3"}),
]

for label, params in attempts:
    print(f"\n=== {label} ===")
    print(f"Params: {params}")
    try:
        r = requests.get(BASE, params=params, timeout=8)
        print(f"URL sent: {r.url}")
        print(f"Status: {r.status_code}")
        body = r.text[:400]
        print(f"Body: {body}")
    except Exception as e:
        print(f"Error: {e}")

# Also try the root to see if the API has self-documentation
print("\n=== Root endpoint ===")
try:
    r = requests.get("https://api.sanctions.network/", timeout=8)
    print(f"Status: {r.status_code}")
    print(f"Body: {r.text[:600]}")
except Exception as e:
    print(f"Error: {e}")
    