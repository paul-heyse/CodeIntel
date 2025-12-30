"""Import-time schema safety checks."""

from __future__ import annotations

import importlib

import pytest

from codeintel.build.schemas import iter_contracts_by_table_key, reset_contract_service_state
from codeintel.build.target_metadata import (
    is_target_metadata_loaded,
    reset_target_metadata_state,
)


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_history_helper_does_not_resolve_contracts_on_import() -> None:
    """Verify history helpers do not resolve contracts at import time."""
    history_module = importlib.import_module("tests._helpers.orchestration.history")
    importlib.reload(history_module)
    cache_initialized = history_module.contracts_cache_initialized()
    _require(condition=not cache_initialized, message="Contracts resolved during import")


def test_contract_enumeration_initializes_targets() -> None:
    """Ensure contract enumeration initializes the Hamilton DAG."""
    reset_contract_service_state()
    reset_target_metadata_state()
    contracts = iter_contracts_by_table_key()
    _ = next(iter(contracts), None)
    _require(condition=is_target_metadata_loaded(), message="Target metadata not loaded")
