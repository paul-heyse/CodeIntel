"""Import-time schema safety checks."""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas import (
    configure_contract_service,
    configure_schema_service,
    iter_contracts_by_table_key,
    reset_contract_service_state,
)
from codeintel.build.target_metadata import (
    is_target_metadata_loaded,
    reset_target_metadata_state,
)

if TYPE_CHECKING:
    from codeintel.runtime.runtime_bundle import RuntimeBundle


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_history_helper_does_not_resolve_contracts_on_import() -> None:
    """Verify history helpers do not resolve contracts at import time."""
    history_module = importlib.import_module("tests._helpers.orchestration.history")
    importlib.reload(history_module)
    cache_initialized = history_module.contracts_cache_initialized()
    _require(condition=not cache_initialized, message="Contracts resolved during import")


def test_contract_enumeration_initializes_targets(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Ensure contract enumeration initializes the Hamilton DAG."""
    reset_contract_service_state()
    reset_target_metadata_state()
    configure_schema_service(runtime=hamilton_runtime)
    configure_contract_service(runtime=hamilton_runtime)
    contracts = iter_contracts_by_table_key()
    _ = next(iter(contracts), None)
    _require(condition=is_target_metadata_loaded(), message="Target metadata not loaded")
