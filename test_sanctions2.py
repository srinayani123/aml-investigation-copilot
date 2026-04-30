"""Probe to find correct array filter syntax for sanctions.names."""
import requests

BASE = "https://api.sanctions.network/sanctions"
NAME = "Putin"

attempts = [
    ("contains array (cs)", {"names": f"cs.{{{NAME}}}", "limit": "3"}),
    ("overlap array (ov)", {"names": f"ov.{{{NAME}}}", "limit": "3"}),
    ("any-element ilike via or", {"or": f"(names.cs.{{{NAME}}})", "limit": "3"}),
    ("plfts (full text)", {"names": f"plfts.{NAME}", "limit": "3"}),
    ("phfts (phrase fts)", {"names": f"phfts.{NAME}", "limit": "3"}),
    ("fts (full text default)", {"names": f"fts.{NAME}", "limit": "3"}),
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
        