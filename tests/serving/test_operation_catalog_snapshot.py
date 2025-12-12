"""Snapshot test to guard against accidental operation loss during refactoring."""

from __future__ import annotations

import pytest

from codeintel.serving.operations import iter_operations
from codeintel.serving.operations.catalog import iter_registry_operations

EXPECTED_OPERATION_COUNT = 26


def test_operations_count_snapshot() -> None:
    """Lock in the current Operation count to detect accidental removal."""
    operations = iter_registry_operations()
    if len(operations) != EXPECTED_OPERATION_COUNT:
        pytest.fail(f"Expected {EXPECTED_OPERATION_COUNT} Operations, got {len(operations)}")


def test_catalog_count_snapshot() -> None:
    """Lock in the catalog Operation count to detect accidental removal."""
    ops = list(iter_operations())
    if len(ops) != EXPECTED_OPERATION_COUNT:
        pytest.fail(f"Expected {EXPECTED_OPERATION_COUNT} catalog Operations, got {len(ops)}")


def test_operation_ids_snapshot() -> None:
    """Lock in the exact set of Operation IDs."""
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
    actual_ids = {op.id for op in iter_registry_operations()}

    if len(expected_ids) != EXPECTED_OPERATION_COUNT:
        pytest.fail(f"Expected IDs count mismatch: {len(expected_ids)}")
    if actual_ids != expected_ids:
        pytest.fail(f"Mismatch: {actual_ids.symmetric_difference(expected_ids)}")
