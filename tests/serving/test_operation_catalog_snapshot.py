"""Snapshot test to guard against accidental operation loss during refactoring."""

from __future__ import annotations

import pytest

from codeintel.serving.backend.operations import OPERATION_CONTRACTS
from codeintel.serving.registry import iter_operation_specs

# Expected counts for snapshot verification
EXPECTED_OPERATION_COUNT = 26


def test_operation_specs_count_snapshot() -> None:
    """Lock in the current OperationSpec count to detect accidental removal."""
    specs = iter_operation_specs()
    if len(specs) != EXPECTED_OPERATION_COUNT:
        pytest.fail(f"Expected {EXPECTED_OPERATION_COUNT} OperationSpecs, got {len(specs)}")


def test_operation_contracts_count_snapshot() -> None:
    """Lock in the current OperationContract count to detect accidental removal.

    After the OperationCatalog unification, contracts are derived from the canonical
    catalog, so they now match the OperationSpec count.
    """
    contracts = OPERATION_CONTRACTS
    if len(contracts) != EXPECTED_OPERATION_COUNT:
        pytest.fail(f"Expected {EXPECTED_OPERATION_COUNT} OperationContracts, got {len(contracts)}")


def test_operation_spec_ids_snapshot() -> None:
    """Lock in the exact set of OperationSpec IDs."""
    expected_ids = {
        "architecture.function",
        "architecture.module",
        "datasets.list",
        "datasets.rows",
        "datasets.schema",
        "datasets.specs",
        "file.summary",
        "function.summary",
        "functions.high_risk",
        "functions.tests",
        "graph.call_neighborhood",
        "graph.call_neighbors",
        "graph.import_boundary",
        "graph.plugins.plan",
        "health.status",
        "ide.hints",
        "profiles.file",
        "profiles.function",
        "profiles.module",
        "subsystems.coverage",
        "subsystems.detail",
        "subsystems.list",
        "subsystems.module_memberships",
        "subsystems.profiles",
        "subsystems.search",
        "subsystems.summarize",
    }
    actual_ids = {spec.id for spec in iter_operation_specs()}
    # Validate expected count
    if len(expected_ids) != EXPECTED_OPERATION_COUNT:
        pytest.fail(f"Expected IDs count mismatch: {len(expected_ids)}")
    if actual_ids != expected_ids:
        pytest.fail(f"Mismatch: {actual_ids.symmetric_difference(expected_ids)}")


def test_operation_contract_names_snapshot() -> None:
    """Lock in the exact set of OperationContract names.

    After the OperationCatalog unification, contract names match OperationSpec IDs
    since they're derived from the canonical catalog.
    """
    # Contract names now match OperationSpec IDs
    expected_names = {
        "architecture.function",
        "architecture.module",
        "datasets.list",
        "datasets.rows",
        "datasets.schema",
        "datasets.specs",
        "file.summary",
        "function.summary",
        "functions.high_risk",
        "functions.tests",
        "graph.call_neighborhood",
        "graph.call_neighbors",
        "graph.import_boundary",
        "graph.plugins.plan",
        "health.status",
        "ide.hints",
        "profiles.file",
        "profiles.function",
        "profiles.module",
        "subsystems.coverage",
        "subsystems.detail",
        "subsystems.list",
        "subsystems.module_memberships",
        "subsystems.profiles",
        "subsystems.search",
        "subsystems.summarize",
    }
    actual_names = set(OPERATION_CONTRACTS.keys())
    # Validate expected count
    if len(expected_names) != EXPECTED_OPERATION_COUNT:
        pytest.fail(f"Expected names count mismatch: {len(expected_names)}")
    if actual_names != expected_names:
        pytest.fail(f"Mismatch: {actual_names.symmetric_difference(expected_names)}")
