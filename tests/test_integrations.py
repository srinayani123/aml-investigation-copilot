"""Tests for the integrations module.

We don't hit real APIs in tests; we monkeypatch requests.post / requests.get
to simulate failure modes. This verifies the graceful-fallback contract:
integrations either return a list/dict (success) or None (graceful failure),
never raise.

GDELT's throttle uses module-level state plus time.sleep — both are
patched out so the suite stays fast and deterministic.
"""
from __future__ import annotations

import os
from unittest.mock import patch

import pytest
import requests

import src.agents.integrations as integrations_module
from src.agents.integrations import (
    query_adverse_media_real,
    query_lei_real,
    query_screening_real,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int = 200, json_data=None, text: str = ""):
        self.status_code = status_code
        self._json = json_data
        self.text = text or (str(json_data) if json_data is not None else "")
        self.headers = {}

    def json(self):
        if self._json is None:
            raise ValueError("no json")
        return self._json


@pytest.fixture(autouse=True)
def _reset_gdelt_and_skip_sleep():
    integrations_module._gdelt_last_call_ts = 0.0
    with patch("src.agents.integrations.time.sleep"):
        yield
    integrations_module._gdelt_last_call_ts = 0.0


@pytest.fixture
def with_keys():
    """Inject fake API keys into env for tests that need them."""
    with patch.dict(os.environ, {
        "OPENSANCTIONS_API_KEY": "fake_os_key",
        "TAVILY_API_KEY": "fake_tvly_key",
    }):
        yield


# ---------------------------------------------------------------------------
# OpenSanctions screening tests
# ---------------------------------------------------------------------------

def _opensanctions_response(results):
    return {"responses": {"q1": {"status": 200, "results": results}}}


def test_screening_returns_none_when_no_key():
    with patch.dict(os.environ, {}, clear=True):
        with patch("src.agents.integrations.requests.post") as p:
            assert query_screening_real("Vladimir Putin") is None
            p.assert_not_called()


def test_screening_returns_empty_dict_on_short_name(with_keys):
    out = query_screening_real("ab")
    assert out == {"sanctions": [], "pep": []}


def test_screening_returns_none_on_timeout(with_keys):
    with patch("src.agents.integrations.requests.post",
               side_effect=requests.exceptions.Timeout()):
        assert query_screening_real("Vladimir Putin") is None


def test_screening_returns_none_on_connection_error(with_keys):
    with patch("src.agents.integrations.requests.post",
               side_effect=requests.exceptions.ConnectionError()):
        assert query_screening_real("Vladimir Putin") is None


def test_screening_returns_none_on_401(with_keys):
    with patch("src.agents.integrations.requests.post",
               return_value=_FakeResponse(status_code=401)):
        assert query_screening_real("Vladimir Putin") is None


def test_screening_returns_none_on_429(with_keys):
    with patch("src.agents.integrations.requests.post",
               return_value=_FakeResponse(status_code=429)):
        assert query_screening_real("Vladimir Putin") is None


def test_screening_classifies_pep_and_sanctions_separately(with_keys):
    fake = _opensanctions_response([
        {  # Sanctioned + PEP (e.g. Putin)
            "id": "Q7747",
            "caption": "Vladimir Putin",
            "schema": "Person",
            "score": 0.99,
            "datasets": ["us_ofac_sdn", "eu_fsf"],
            "properties": {
                "name": ["Vladimir Putin"],
                "topics": ["sanction", "role.pep", "role.pol"],
            },
        },
        {  # Pure PEP (e.g. Zelenskyy)
            "id": "Q123",
            "caption": "Volodymyr Zelenskyy",
            "schema": "Person",
            "score": 1.0,
            "datasets": ["us_cia_world_leaders"],
            "properties": {
                "name": ["Volodymyr Zelenskyy"],
                "topics": ["role.pep", "role.pol"],
            },
        },
    ])
    with patch("src.agents.integrations.requests.post",
               return_value=_FakeResponse(status_code=200, json_data=fake)):
        out = query_screening_real("Vladimir Putin")
    assert out is not None
    assert len(out["sanctions"]) == 1  # only Putin
    assert out["sanctions"][0]["name"] == "Vladimir Putin"
    assert len(out["pep"]) == 2        # both Putin and Zelenskyy
    assert {h["name"] for h in out["pep"]} == {"Vladimir Putin", "Volodymyr Zelenskyy"}


def test_screening_returns_empty_lists_on_no_match(with_keys):
    fake = _opensanctions_response([])
    with patch("src.agents.integrations.requests.post",
               return_value=_FakeResponse(status_code=200, json_data=fake)):
        out = query_screening_real("ZZZ_no_such_person")
    assert out == {"sanctions": [], "pep": []}


def test_screening_picks_ascii_name_over_other_scripts(with_keys):
    fake = _opensanctions_response([{
        "id": "Q7747",
        "caption": "Vladimir Putin",
        "schema": "Person",
        "score": 0.95,
        "datasets": ["us_ofac_sdn"],
        "properties": {
            "name": ["Владимир Путин", "Vladimir Putin", "ولاديمير بوتين"],
            "topics": ["sanction"],
        },
    }])
    with patch("src.agents.integrations.requests.post",
               return_value=_FakeResponse(status_code=200, json_data=fake)):
        out = query_screening_real("Vladimir Putin")
    assert out["sanctions"][0]["name"] == "Vladimir Putin"


# ---------------------------------------------------------------------------
# Adverse media tests (Tavily primary + GDELT fallback)
# ---------------------------------------------------------------------------

def test_adverse_returns_empty_on_short_name():
    assert query_adverse_media_real("ab") == []


def test_adverse_uses_tavily_when_available(with_keys):
    tavily_data = {"results": [
        {"title": "Headline 1", "url": "https://news.com/a",
         "score": 0.8, "published_date": "2025-01-01"}
    ]}
    with patch("src.agents.integrations.requests.post",
               return_value=_FakeResponse(status_code=200, json_data=tavily_data)) as p_post, \
         patch("src.agents.integrations.requests.get") as p_get:
        result = query_adverse_media_real("Acme Corp")
    assert result is not None
    assert len(result) == 1
    assert result[0]["provider"] == "Tavily"
    assert result[0]["headline"] == "Headline 1"
    p_get.assert_not_called()  # GDELT shouldn't be hit when Tavily succeeds


def test_adverse_falls_back_to_gdelt_when_tavily_fails(with_keys):
    gdelt_data = {"articles": [
        {"title": "Smith indicted", "url": "https://x.com/y",
         "domain": "x.com", "seendate": "20250101T120000Z"}
    ]}
    with patch("src.agents.integrations.requests.post",
               return_value=_FakeResponse(status_code=500)), \
         patch("src.agents.integrations.requests.get",
               return_value=_FakeResponse(status_code=200,
                                          json_data=gdelt_data,
                                          text='{"articles":[{}]}')):
        result = query_adverse_media_real("Smith")
    assert result is not None
    assert result[0]["provider"] == "GDELT"


def test_adverse_falls_back_to_gdelt_when_no_tavily_key():
    """No Tavily key set, but GDELT works."""
    with patch.dict(os.environ, {}, clear=True):
        gdelt_data = {"articles": [
            {"title": "X", "domain": "y.com", "url": "https://y.com/z"}
        ]}
        with patch("src.agents.integrations.requests.get",
                   return_value=_FakeResponse(status_code=200,
                                              json_data=gdelt_data,
                                              text='{"articles":[{}]}')) as p_get:
            result = query_adverse_media_real("Smith")
    assert result is not None
    assert result[0]["provider"] == "GDELT"
    p_get.assert_called_once()


def test_adverse_returns_none_when_both_apis_fail(with_keys):
    with patch("src.agents.integrations.requests.post",
               side_effect=requests.exceptions.Timeout()), \
         patch("src.agents.integrations.requests.get",
               return_value=_FakeResponse(status_code=429)):
        result = query_adverse_media_real("Smith")
    assert result is None


def test_adverse_handles_empty_gdelt_response(with_keys):
    with patch("src.agents.integrations.requests.post",
               return_value=_FakeResponse(status_code=500)), \
         patch("src.agents.integrations.requests.get",
               return_value=_FakeResponse(status_code=200,
                                          json_data={}, text="{}")):
        result = query_adverse_media_real("Smith")
    assert result == []


# ---------------------------------------------------------------------------
# GLEIF KYB tests
# ---------------------------------------------------------------------------

def test_lei_returns_empty_on_short_name():
    assert query_lei_real("a") == []


def test_lei_returns_records_on_success():
    gleif_data = {"data": [
        {
            "type": "fuzzycompletions",
            "attributes": {"value": "Goldman Sachs LLC"},
            "relationships": {
                "lei-records": {
                    "data": {"type": "lei-records", "id": "549300CWUTEDC3CFJ739"},
                    "links": {"related": "https://api.gleif.org/api/v1/lei-records/549300CWUTEDC3CFJ739"},
                }
            },
        }
    ]}
    with patch("src.agents.integrations.requests.get",
               return_value=_FakeResponse(status_code=200, json_data=gleif_data)):
        result = query_lei_real("Goldman Sachs")
    assert result is not None
    assert len(result) == 1
    assert result[0]["name"] == "Goldman Sachs LLC"
    assert result[0]["lei"] == "549300CWUTEDC3CFJ739"
    assert result[0]["source"] == "GLEIF"


def test_lei_returns_none_on_timeout():
    with patch("src.agents.integrations.requests.get",
               side_effect=requests.exceptions.Timeout()):
        assert query_lei_real("Goldman Sachs") is None


def test_lei_returns_none_on_500():
    with patch("src.agents.integrations.requests.get",
               return_value=_FakeResponse(status_code=500)):
        assert query_lei_real("Goldman Sachs") is None


def test_lei_returns_empty_when_no_matches():
    with patch("src.agents.integrations.requests.get",
               return_value=_FakeResponse(status_code=200,
                                          json_data={"data": []})):
        assert query_lei_real("ZZZ_no_company") == []
        