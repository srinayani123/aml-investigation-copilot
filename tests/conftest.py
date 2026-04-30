"""Shared test fixtures."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.features import build_feature_store  # noqa: E402
from src.data.loader import generate_synthetic  # noqa: E402


@pytest.fixture(scope="session")
def synthetic_df():
    """Small synthetic dataset with planted typologies."""
    return generate_synthetic()


@pytest.fixture(scope="session")
def feature_store(synthetic_df):
    return build_feature_store(synthetic_df)
