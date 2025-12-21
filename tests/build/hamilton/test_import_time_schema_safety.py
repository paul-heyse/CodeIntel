"""Import-time schema safety checks."""

from __future__ import annotations

import importlib

import pytest

from codeintel.build.schemas import (
    ContractResolutionMode,
    ContractResolutionSettings,
    iter_contracts_by_table_key,
)
from codeintel.build.target_metadata import (
    clear_target_metadata_cache,
    is_target_metadata_loaded,
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


def test_contract_enumeration_does_not_initialize_targets() -> None:
    """Ensure schema-only contract enumeration avoids the Hamilton DAG."""
    clear_target_metadata_cache()
    contracts = iter_contracts_by_table_key(
        settings=ContractResolutionSettings(mode=ContractResolutionMode.DECLARED_ONLY)
    )
    _ = next(iter(contracts), None)
    _require(condition=not is_target_metadata_loaded(), message="Target metadata loaded")
