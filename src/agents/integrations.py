"""Real external API integrations for the Entity Resolver agent.

Four integrations, each with graceful fallback to None on any failure:
  - OpenSanctions /match/default — combined sanctions + PEP screening
  - Tavily AI Search             — adverse-media news search (primary)
  - GDELT DOC 2.0                — adverse-media news search (fallback)
  - GLEIF /fuzzycompletions      — Legal Entity Identifier (KYB enrichment)

KYC for individuals is intentionally mocked elsewhere — see query_kyc_real
for the architectural reasoning. (Short version: in production, KYC at
investigation time is a CIF lookup, not an external vendor call.)

Design pattern:
  Every function returns either:
    - List[dict] of normalized findings (success or empty)
    - None (real-API attempted but failed; caller falls back to mock)
  This means the agent code is identical regardless of which path was
  taken, and the UI provenance bar reports real/mock per source.
"""
from __future__ import annotations

import logging
import os
import time
import urllib.parse
from typing import List, Optional

import requests

logger = logging.getLogger(__name__)

# Per-call HTTP timeout. Generous because GDELT can be slow on first hit
# from a new IP and Tavily's news index sometimes takes 2-3s.
HTTP_TIMEOUT = 10.0


# ===========================================================================
# OpenSanctions /match/default — combined sanctions + PEP screening
# ===========================================================================

OPENSANCTIONS_URL = "https://api.opensanctions.org/match/default"

# Topic taxonomy from OpenSanctions used to classify a hit:
#   role.pep, role.pol, role.rca   → PEP / political exposure
#   sanction, sanction.linked      → sanctioned or linked to sanctioned
#   crime, poi, wanted             → criminal / person of interest
#   debarment                      → barred from public contracts
PEP_TOPICS = {"role.pep", "role.pol", "role.rca"}
SANCTION_TOPICS = {"sanction", "sanction.linked"}
CRIME_TOPICS = {"crime", "poi", "wanted", "debarment"}


def _opensanctions_key() -> Optional[str]:
    """Read key from env at call-time so tests/UI can patch dynamically."""
    return os.environ.get("OPENSANCTIONS_API_KEY")


def query_screening_real(name: str,
                         max_results: int = 5) -> Optional[dict]:
    """Combined sanctions + PEP screening via OpenSanctions /match/default.

    Returns a dict with two finding lists:
        {
          "sanctions": [...],   # entries with sanction-related topics
          "pep":       [...],   # entries with PEP-related topics
        }
    or None on failure (no key, network, auth, rate limit, schema issue).

    A single hit can appear in both lists if its topics span both
    categories (e.g. a sanctioned former minister has both 'sanction' and
    'role.pep' in topics) — the UI handles that as two separate signals.
    """
    if not name or len(name.strip()) < 3:
        return {"sanctions": [], "pep": []}

    api_key = _opensanctions_key()
    if not api_key:
        return None  # No key set → caller falls back to mock

    body = {
        "queries": {
            "q1": {
                "schema": "Person",
                "properties": {"name": [name.strip()]},
            }
        }
    }
    try:
        r = requests.post(
            f"{OPENSANCTIONS_URL}?algorithm=logic-v2",
            headers={"Authorization": f"ApiKey {api_key}"},
            json=body,
            timeout=HTTP_TIMEOUT,
        )
        if r.status_code == 401:
            logger.warning("OpenSanctions: invalid API key")
            return None
        if r.status_code == 429:
            logger.warning("OpenSanctions: rate-limited (HTTP 429)")
            return None
        if r.status_code != 200:
            logger.warning("OpenSanctions: status %d", r.status_code)
            return None
        data = r.json()
        results = (data.get("responses", {})
                       .get("q1", {})
                       .get("results", []))
    except (requests.exceptions.Timeout,
            requests.exceptions.ConnectionError) as e:
        logger.warning("OpenSanctions unreachable (%s)", type(e).__name__)
        return None
    except Exception as e:
        logger.warning("OpenSanctions query failed: %s", e)
        return None

    sanctions_hits = []
    pep_hits = []
    for item in results[:max_results]:
        props = item.get("properties", {}) or {}
        topics = set(props.get("topics", []) or [])
        datasets = item.get("datasets", []) or []
        names = props.get("name") or [item.get("caption", "(unknown)")]
        primary_name = next(
            (n for n in names if all(ord(c) < 128 for c in n)),
            names[0],
        )
        base = {
            "name": primary_name,
            "source": "OpenSanctions",
            "source_id": item.get("id"),
            "match_type": "real_api",
            "score": item.get("score", 0.0),
            "datasets": datasets,
            "topics": sorted(list(topics)),
            "schema": item.get("schema", "Person"),
            "requires_manual_review": True,
        }
        if topics & SANCTION_TOPICS:
            sanctions_hits.append({
                **base,
                "category": "sanctions",
                "list": ", ".join(datasets[:3]) if datasets else "OpenSanctions",
            })
        if topics & PEP_TOPICS:
            pep_hits.append({
                **base,
                "category": "pep",
                "pep_role": ", ".join(sorted(topics & (PEP_TOPICS | CRIME_TOPICS))),
            })
    return {"sanctions": sanctions_hits, "pep": pep_hits}


# ===========================================================================
# Adverse media — Tavily primary + GDELT fallback
# ===========================================================================

TAVILY_URL = "https://api.tavily.com/search"
GDELT_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

# GDELT enforces ~1 request per 5 seconds per IP. Module-level throttle.
GDELT_MIN_INTERVAL_SEC = 5.5
_gdelt_last_call_ts: float = 0.0

ADVERSE_KEYWORDS_GDELT = (
    '(fraud OR "money laundering" OR investigation OR indicted '
    'OR sanctions OR "financial crime" OR scandal)'
)


def _tavily_key() -> Optional[str]:
    return os.environ.get("TAVILY_API_KEY")


def _gdelt_throttle() -> None:
    """Sleep just long enough since the last GDELT call to be polite."""
    global _gdelt_last_call_ts
    now = time.time()
    delta = now - _gdelt_last_call_ts
    if delta < GDELT_MIN_INTERVAL_SEC:
        time.sleep(GDELT_MIN_INTERVAL_SEC - delta)
    _gdelt_last_call_ts = time.time()


def _query_adverse_media_tavily(name: str, max_results: int) -> Optional[List[dict]]:
    """Tavily AI Search with finance-crime keywords. None on failure."""
    api_key = _tavily_key()
    if not api_key:
        return None
    try:
        r = requests.post(
            TAVILY_URL,
            json={
                "api_key": api_key,
                "query": (f'"{name.strip()}" '
                          f'(fraud OR "money laundering" OR indictment '
                          f'OR sanctions OR "financial crime")'),
                "max_results": max_results,
                "topic": "news",
                "search_depth": "basic",
            },
            timeout=HTTP_TIMEOUT,
        )
        if r.status_code != 200:
            logger.warning("Tavily: status %d", r.status_code)
            return None
        data = r.json()
        results = data.get("results", [])
        return [{
            "source": _domain_from_url(art.get("url", "")),
            "headline": art.get("title", "(no title)"),
            "url": art.get("url"),
            "date": art.get("published_date"),
            "score": art.get("score", 0.0),
            "language": "English",
            "provider": "Tavily",
            "relevance": "needs_human_review",
        } for art in results[:max_results]]
    except (requests.exceptions.Timeout,
            requests.exceptions.ConnectionError) as e:
        logger.warning("Tavily unreachable (%s)", type(e).__name__)
        return None
    except Exception as e:
        logger.warning("Tavily query failed: %s", e)
        return None


def _query_adverse_media_gdelt(name: str, max_results: int,
                               timespan_days: int) -> Optional[List[dict]]:
    """GDELT DOC 2.0 fallback. Throttled to GDELT's ~1 req/5sec rate limit."""
    try:
        _gdelt_throttle()
        params = {
            "query": f'"{name.strip()}" {ADVERSE_KEYWORDS_GDELT}',
            "mode": "artlist",
            "format": "json",
            "maxrecords": str(max_results),
            "sort": "datedesc",
            "timespan": f"{timespan_days}d",
        }
        url = f"{GDELT_URL}?{urllib.parse.urlencode(params)}"
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        if r.status_code == 429:
            logger.warning("GDELT rate-limited; not retrying")
            return None
        if r.status_code != 200:
            logger.warning("GDELT: status %d", r.status_code)
            return None
        text = r.text.strip()
        if not text or text == "{}":
            return []
        try:
            data = r.json()
        except ValueError:
            logger.warning("GDELT returned non-JSON (~error)")
            return None
        articles = data.get("articles", [])
        return [{
            "source": art.get("domain", "(unknown)"),
            "headline": art.get("title", "(no title)"),
            "url": art.get("url"),
            "date": art.get("seendate"),
            "score": None,
            "language": art.get("language", "English"),
            "provider": "GDELT",
            "relevance": "needs_human_review",
        } for art in articles[:max_results]]
    except (requests.exceptions.Timeout,
            requests.exceptions.ConnectionError) as e:
        logger.warning("GDELT unreachable (%s)", type(e).__name__)
        return None
    except Exception as e:
        logger.warning("GDELT query failed: %s", e)
        return None


def query_adverse_media_real(name: str,
                             max_results: int = 5,
                             timespan_days: int = 365) -> Optional[List[dict]]:
    """Adverse-media query with Tavily primary + GDELT fallback cascade.

    Returns:
      - list of articles (may be empty if nothing found, both APIs OK)
      - None only if BOTH Tavily and GDELT failed to reach
    The provider field on each article tells the UI which source produced it.
    """
    if not name or len(name.strip()) < 3:
        return []
    # Try Tavily first
    tavily = _query_adverse_media_tavily(name, max_results)
    if tavily is not None:
        return tavily
    # Fall back to GDELT
    gdelt = _query_adverse_media_gdelt(name, max_results, timespan_days)
    if gdelt is not None:
        return gdelt
    return None


def _domain_from_url(url: str) -> str:
    """Extract bare domain from a URL for display."""
    if not url:
        return "(unknown)"
    try:
        netloc = urllib.parse.urlparse(url).netloc
        return netloc.lstrip("www.") if netloc else "(unknown)"
    except Exception:
        return "(unknown)"


# ===========================================================================
# GLEIF — Legal Entity Identifier (KYB enrichment for corporate accounts)
# ===========================================================================

GLEIF_URL = "https://api.gleif.org/api/v1/fuzzycompletions"


def query_lei_real(entity_name: str,
                   max_results: int = 3) -> Optional[List[dict]]:
    """GLEIF fuzzy entity-name lookup. Free, no key, public reference data.

    Returns:
      - list of matches with LEI codes and entity names (may be empty)
      - None on network failure

    GLEIF (Global Legal Entity Identifier Foundation) is the G20/FSB-mandated
    standard for corporate counterparty identification. LEIs are required for
    derivatives reporting under MiFID II / Dodd-Frank. This is the proper
    KYB enrichment path: when CIF returns entity_type=corporate, look up
    the LEI to get authoritative jurisdiction, parent entity, registration
    status, and BIC code mapping.
    """
    if not entity_name or len(entity_name.strip()) < 3:
        return []
    try:
        r = requests.get(
            GLEIF_URL,
            params={"field": "entity.legalName", "q": entity_name.strip()},
            timeout=HTTP_TIMEOUT,
        )
        if r.status_code != 200:
            logger.warning("GLEIF: status %d", r.status_code)
            return None
        data = r.json()
        items = data.get("data", [])[:max_results]
        return [{
            "name": item.get("attributes", {}).get("value", "(unknown)"),
            "lei": (item.get("relationships", {})
                       .get("lei-records", {})
                       .get("data", {})
                       .get("id")),
            "source": "GLEIF",
            "match_type": "real_api",
            "details_url": (item.get("relationships", {})
                                .get("lei-records", {})
                                .get("links", {})
                                .get("related")),
        } for item in items]
    except (requests.exceptions.Timeout,
            requests.exceptions.ConnectionError) as e:
        logger.warning("GLEIF unreachable (%s)", type(e).__name__)
        return None
    except Exception as e:
        logger.warning("GLEIF query failed: %s", e)
        return None


# ===========================================================================
# KYC (individual) — INTENTIONAL MOCK with banking-architecture comment
# ===========================================================================
#
# In production, KYC for an existing customer (account-holder being
# investigated) is NOT obtained via an external KYC verification vendor
# at investigation time. Instead, the bank's investigation system queries
# its own internal Customer Information File (CIF) — a record populated
# during onboarding by vendors like Onfido / Sumsub / Jumio, then stored
# permanently in the bank's customer database.
#
# A real production replacement here would call the bank's internal CIF
# service (typically SOAP or REST with mTLS auth) and return whatever
# structured KYC record was previously collected. Public KYC verification
# APIs (Didit, Sumsub, Onfido) only operate at onboarding-time on real
# documents, which makes them inappropriate for synthetic IBM dataset
# accounts. Mocking the CIF response shape is the architecturally correct
# approach here.
#
# The mock lives in entity_resolver.py alongside the other mock fallbacks.

