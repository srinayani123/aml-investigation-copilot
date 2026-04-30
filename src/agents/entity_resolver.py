"""Entity Resolver agent.

Five external lookups, with real-API-first / mock-fallback pattern:

  1. KYC (individual)     — INTENTIONAL MOCK simulating bank-internal CIF.
                             In production: bank's internal CIF service via
                             SOAP/REST with mTLS. There is no public KYC API
                             at investigation time — KYC was collected at
                             onboarding, stored in CIF, retrieved here.
  2. Sanctions screening  — REAL via OpenSanctions /match/default. Combined
                             with PEP screening in a single call.
  3. PEP screening        — REAL, same OpenSanctions call as sanctions.
                             Distinguished by 'topics' field (role.pep etc.)
  4. Adverse media        — REAL via Tavily (primary) + GDELT (fallback).
  5. Legal entity (KYB)   — REAL via GLEIF /fuzzycompletions, called only
                             when the mock CIF returns entity_type=corporate.

Mock responses are deterministic functions of the account ID — same account
returns the same fake data. Tests are reproducible; demo works offline.

To inspect which path was taken, look at `EntityFindings.data_sources` —
it maps each lookup to 'real', 'mock', 'tavily', 'gdelt', etc.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional

from src.agents.integrations import (
    query_adverse_media_real,
    query_lei_real,
    query_screening_real,
)
from src.agents.llm import chat
from src.data.features import FeatureStore

logger = logging.getLogger(__name__)


@dataclass
class EntityFindings:
    account: str
    kyc_status: str = "unknown"
    declared_occupation: str = "unknown"
    account_age_days: int = 0
    entity_type: str = "individual"  # "individual" or "corporate"
    sanctions_hits: List[dict] = field(default_factory=list)
    pep_hits: List[dict] = field(default_factory=list)
    adverse_media_hits: List[dict] = field(default_factory=list)
    lei_records: List[dict] = field(default_factory=list)
    geo_risk_score: float = 0.0
    countries_seen: List[str] = field(default_factory=list)
    summary: str = ""
    # Provenance map. Values: 'real', 'mock', 'tavily', 'gdelt',
    # 'cif_sim' (KYC mock simulating bank-internal CIF), 'n/a' (skipped).
    data_sources: dict = field(default_factory=lambda: {
        "kyc": "cif_sim",
        "sanctions": "mock",
        "pep": "mock",
        "adverse_media": "mock",
        "lei": "n/a",
    })


def _deterministic_hash(account: str) -> int:
    """Deterministic int from an account id, used to seed mock CIF."""
    return int(hashlib.md5(account.encode()).hexdigest(), 16)


def _mock_cif(account: str) -> dict:
    """Simulated bank-internal CIF (Customer Information File) lookup.

    NOT a public KYC API call. This represents the bank's existing internal
    customer record retrieved at investigation time. Production replacement
    would query the bank's CIF service over mTLS — same response shape,
    different transport.

    ~20% of accounts are flagged as corporate (eligible for GLEIF lookup).
    """
    h = _deterministic_hash(account)
    is_corporate = (h % 5 == 0)

    if is_corporate:
        # Corporate entity record
        legal_names = [
            "Acme Holdings LLC", "Apex Trading Co", "Crescent Logistics",
            "Vanguard Imports Inc", "Sterling Industries", "Pacific Ventures",
            "Goldman Sachs Group", "JPMorgan Chase & Co",
        ]
        return {
            "entity_type": "corporate",
            "legal_name": legal_names[h % len(legal_names)],
            "occupation": "registered legal entity",
            "kyc_status": "complete" if h % 4 != 0 else "review_due",
            "account_age_days": 120 + (h % 2000),
            "documentation_quality": "standard" if h % 3 != 0 else "limited",
        }

    # Individual entity record
    occupations = ["consultant", "retail business", "restaurant owner",
                   "import/export", "crypto trader", "salaried employee",
                   "real estate broker", "freelance designer"]
    statuses = ["complete", "complete", "complete", "incomplete", "expired"]
    return {
        "entity_type": "individual",
        "legal_name": f"(individual a/c {account})",
        "occupation": occupations[h % len(occupations)],
        "kyc_status": statuses[h % len(statuses)],
        "account_age_days": 60 + (h % 1500),
        "documentation_quality": "limited" if h % 4 == 0 else "standard",
    }


def _mock_sanctions(counterparties: List[str]) -> List[dict]:
    """Deterministic mock OFAC fuzzy-match. ~3% rate."""
    hits: List[dict] = []
    for cp in counterparties[:25]:
        h = _deterministic_hash(cp)
        if h % 33 == 0:
            hits.append({
                "counterparty": cp,
                "name": cp,
                "match_type": "fuzzy",
                "similarity": round(0.55 + (h % 30) / 100, 2),
                "list": "OFAC SDN (mock)",
                "source": "mock",
                "category": "sanctions",
                "requires_manual_review": True,
            })
    return hits


def _mock_pep(account: str) -> List[dict]:
    """Deterministic mock PEP. ~2% of accounts flagged."""
    h = _deterministic_hash(account)
    if h % 50 == 0:
        return [{
            "name": f"(simulated PEP for a/c {account})",
            "match_type": "mock",
            "pep_role": "role.pep",
            "source": "mock",
            "category": "pep",
            "requires_manual_review": True,
        }]
    return []


def _mock_adverse_media(account: str) -> List[dict]:
    """Deterministic mock adverse-media. ~2% of accounts."""
    h = _deterministic_hash(account)
    if h % 50 == 0:
        return [{
            "source": "Regional News Wire (mock)",
            "headline": "Local business named in regulatory inquiry",
            "date": "2024-03-15",
            "provider": "mock",
            "relevance": "medium",
        }]
    return []


def _geo_risk(countries: List[str]) -> float:
    """Higher score if counterparty banks span FATF grey-list buckets.
    For the IBM AML dataset we use bank-id endings as a proxy."""
    risky_buckets = {"7", "8", "9"}
    risky = sum(1 for c in countries if any(c.endswith(b) for b in risky_buckets))
    return min(1.0, risky / max(len(countries), 1))


def resolve_entity(account: str,
                   store: FeatureStore,
                   api_key: Optional[str] = None) -> EntityFindings:
    feats = store.account_features.get(account)
    counterparties = store.get_counterparties(account)

    if feats is None:
        return EntityFindings(account=account, summary="Account not found.")

    # ---- KYC: simulated bank-internal CIF lookup -------------------------
    cif = _mock_cif(account)
    data_sources = {"kyc": "cif_sim"}

    # ---- Combined sanctions + PEP screening (one OpenSanctions call) ----
    # In a real bank, the screening name comes from CIF (cif["legal_name"]).
    # For IBM-AML we use the legal_name from the simulated CIF record.
    screening_name = cif["legal_name"]
    screening = query_screening_real(screening_name, max_results=5)
    if screening is not None:
        sanctions = screening["sanctions"]
        pep = screening["pep"]
        data_sources["sanctions"] = "real"
        data_sources["pep"] = "real"
    else:
        sanctions = _mock_sanctions(counterparties)
        pep = _mock_pep(account)
        data_sources["sanctions"] = "mock"
        data_sources["pep"] = "mock"

    # ---- Adverse media: Tavily primary, GDELT fallback, mock last -------
    real_adverse = query_adverse_media_real(screening_name, max_results=5)
    if real_adverse is not None:
        adverse = real_adverse
        # The provider on each article tells us which API succeeded
        if real_adverse and real_adverse[0].get("provider") == "GDELT":
            data_sources["adverse_media"] = "gdelt"
        else:
            data_sources["adverse_media"] = "tavily"
    else:
        adverse = _mock_adverse_media(account)
        data_sources["adverse_media"] = "mock"

    # ---- LEI lookup (KYB enrichment, corporate accounts only) -----------
    lei_records: List[dict] = []
    if cif["entity_type"] == "corporate":
        real_lei = query_lei_real(cif["legal_name"], max_results=3)
        if real_lei is not None:
            lei_records = real_lei
            data_sources["lei"] = "real"
        else:
            data_sources["lei"] = "mock"  # GLEIF unreachable; no mock data
    else:
        data_sources["lei"] = "n/a"  # individual; not applicable

    # ---- Geographic risk via counterparty banks --------------------------
    history = store.get_account_history(account, limit=200)
    countries = list(set(history["from_bank"].tolist() + history["to_bank"].tolist()))
    geo_risk = _geo_risk(countries)

    findings = EntityFindings(
        account=account,
        kyc_status=cif["kyc_status"],
        declared_occupation=cif["occupation"],
        account_age_days=cif["account_age_days"],
        entity_type=cif["entity_type"],
        sanctions_hits=sanctions,
        pep_hits=pep,
        adverse_media_hits=adverse,
        lei_records=lei_records,
        geo_risk_score=geo_risk,
        countries_seen=countries[:10],
        data_sources=data_sources,
    )

    prompt = f"""Entity Resolver — assess this account holder.

Account: {account}
Entity type: {cif["entity_type"]}
Legal name: {cif["legal_name"]}
Declared occupation: {cif["occupation"]}
KYC status: {cif["kyc_status"]}
Account age (days): {cif["account_age_days"]}
Documentation quality: {cif["documentation_quality"]}

Sanctions screening: {len(sanctions)} hit(s)
{sanctions if sanctions else "(no sanctions hits)"}

PEP screening: {len(pep)} hit(s)
{pep if pep else "(no PEP hits)"}

Adverse media: {len(adverse)} item(s)
{adverse if adverse else "(nothing surfaced)"}

LEI / KYB records: {len(lei_records)} match(es)
{lei_records if lei_records else "(not applicable or no match)"}

Geographic risk score: {geo_risk:.2f} (counterparty banks span {len(countries)} buckets)

Observed activity profile:
- Transaction count: {feats.txn_count}
- Total volume: ${feats.total_volume_usd:,.2f}
- Counterparties: {feats.n_unique_counterparties}

Summarize entity-level risk findings in 4-6 sentences. Note any KYC gaps,
mismatches between declared profile and observed activity, sanctions/PEP
exposure, and unresolved screening hits.
"""
    findings.summary = chat(
        prompt=prompt,
        system="You are an experienced KYC / financial-crime analyst.",
        runtime_api_key=api_key,
    )
    return findings
