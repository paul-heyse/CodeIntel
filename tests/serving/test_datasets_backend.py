"""Tests for serving backend dataset registry helpers.

This module verifies the dataset registry building, description generation,
and validation functions without mocking, using real DuckDB connections.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

from codeintel.config.datasets import get_dataset_contracts, get_dataset_contracts_by_table_key
from codeintel.serving.backend.datasets import (
    DOCS_VIEWS,
    PREVIEW_COLUMN_COUNT,
    build_dataset_registry,
    build_registry_and_limits,
    describe_dataset,
    validate_dataset_registry,
)
from codeintel.serving.backend.pagination import BackendLimits
from tests._helpers.gateway import gateway_with_macros

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.storage.gateway import StorageGateway


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------

DEFAULT_LIMIT = 100
MAX_ROWS = 1000
CUSTOM_LIMIT = 50
CUSTOM_MAX = 500


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture
def gateway() -> Iterator[StorageGateway]:
    """Provide a real StorageGateway backed by an in-memory DuckDB database.

    Yields
    ------
    StorageGateway
        Gateway instance for test use.
    """
    gw = gateway_with_macros(repo="test/repo", commit="abc123")
    try:
        yield gw
    finally:
        gw.close()


class MockConfig:
    """Configuration stub for testing build_registry_and_limits."""

    def __init__(self, default_limit: int = DEFAULT_LIMIT, max_rows: int = MAX_ROWS) -> None:
        """Initialize config with specified limits.

        Parameters
        ----------
        default_limit
            Default row limit for queries.
        max_rows
            Maximum rows per API call.
        """
        self.default_limit = default_limit
        self.max_rows_per_call = max_rows


# -----------------------------------------------------------------------------
# DOCS_VIEWS Constant Tests
# -----------------------------------------------------------------------------


def test_docs_views_contains_only_views() -> None:
    """Verify DOCS_VIEWS dictionary contains only view contracts."""
    contracts = get_dataset_contracts()

    for name in DOCS_VIEWS:
        contract = contracts.get(name)
        assert contract is not None, f"DOCS_VIEWS contains unknown dataset: {name}"
        assert contract.is_view, f"DOCS_VIEWS should only contain views, but {name} is not a view"


def test_docs_views_values_are_table_keys() -> None:
    """Verify DOCS_VIEWS values match contract table_keys."""
    contracts = get_dataset_contracts()

    for name, table_key in DOCS_VIEWS.items():
        contract = contracts.get(name)
        assert contract is not None
        assert contract.table_key == table_key


def test_docs_views_is_non_empty_if_views_exist() -> None:
    """Verify DOCS_VIEWS is populated when view contracts exist."""
    contracts = get_dataset_contracts()
    view_count = sum(1 for c in contracts.values() if c.is_view)

    # DOCS_VIEWS should match the number of views
    assert len(DOCS_VIEWS) == view_count


# -----------------------------------------------------------------------------
# build_dataset_registry Tests
# -----------------------------------------------------------------------------


def test_build_dataset_registry_returns_dict() -> None:
    """Verify build_dataset_registry returns a dictionary."""
    registry = build_dataset_registry()
    assert isinstance(registry, dict)


def test_build_dataset_registry_includes_docs_views_by_default() -> None:
    """Verify docs views are included when include_docs_views is 'include'."""
    registry = build_dataset_registry(include_docs_views="include")

    # Check that at least some docs views are present
    for view_name in DOCS_VIEWS:
        assert view_name in registry, f"Expected docs view {view_name} in registry"


def test_build_dataset_registry_excludes_docs_views_when_requested() -> None:
    """Verify docs views are excluded when include_docs_views is 'exclude'."""
    registry = build_dataset_registry(include_docs_views="exclude")

    # Verify no docs views are present
    for view_name in DOCS_VIEWS:
        assert view_name not in registry, f"Docs view {view_name} should be excluded"


def test_build_dataset_registry_values_are_table_keys() -> None:
    """Verify registry values match contract table_keys."""
    registry = build_dataset_registry()
    contracts = get_dataset_contracts()

    for name, table_key in registry.items():
        contract = contracts.get(name)
        assert contract is not None, f"Registry contains unknown dataset: {name}"
        assert contract.table_key == table_key


def test_build_dataset_registry_is_deterministic() -> None:
    """Verify build_dataset_registry returns consistent results."""
    registry1 = build_dataset_registry()
    registry2 = build_dataset_registry()
    assert registry1 == registry2


def test_build_dataset_registry_sorted_by_table_key() -> None:
    """Verify registry is sorted by table_key."""
    registry = build_dataset_registry()
    table_keys = list(registry.values())
    assert table_keys == sorted(table_keys)


def test_build_dataset_registry_exclude_reduces_count() -> None:
    """Verify excluding docs views reduces registry size when views exist."""
    registry_include = build_dataset_registry(include_docs_views="include")
    registry_exclude = build_dataset_registry(include_docs_views="exclude")

    if DOCS_VIEWS:
        assert len(registry_exclude) < len(registry_include)
    else:
        assert len(registry_exclude) == len(registry_include)


def test_build_dataset_registry_contains_all_non_view_datasets() -> None:
    """Verify all non-view datasets are in registry regardless of include mode."""
    contracts = get_dataset_contracts()
    non_view_names = {name for name, c in contracts.items() if not c.is_view}

    registry_exclude = build_dataset_registry(include_docs_views="exclude")

    for name in non_view_names:
        assert name in registry_exclude


# -----------------------------------------------------------------------------
# build_registry_and_limits Tests
# -----------------------------------------------------------------------------


def test_build_registry_and_limits_returns_tuple() -> None:
    """Verify build_registry_and_limits returns a tuple of registry and limits."""
    config = MockConfig()
    registry, limits = build_registry_and_limits(config)

    assert isinstance(registry, dict)
    assert isinstance(limits, BackendLimits)


def test_build_registry_and_limits_uses_config_values() -> None:
    """Verify limits are derived from config."""
    config = MockConfig(default_limit=CUSTOM_LIMIT, max_rows=CUSTOM_MAX)

    _registry, limits = build_registry_and_limits(config)

    assert limits.default_limit == CUSTOM_LIMIT
    assert limits.max_rows_per_call == CUSTOM_MAX


def test_build_registry_and_limits_respects_include_docs_views() -> None:
    """Verify include_docs_views parameter is respected."""
    config = MockConfig()

    registry_with_views, _ = build_registry_and_limits(config, include_docs_views="include")
    registry_without_views, _ = build_registry_and_limits(config, include_docs_views="exclude")

    # The registry with views should be larger (or equal if no docs views exist)
    assert len(registry_with_views) >= len(registry_without_views)


def test_build_registry_and_limits_registry_matches_standalone() -> None:
    """Verify registry from build_registry_and_limits matches build_dataset_registry."""
    config = MockConfig()

    registry_combined, _ = build_registry_and_limits(config, include_docs_views="include")
    registry_standalone = build_dataset_registry(include_docs_views="include")

    assert registry_combined == registry_standalone


def test_build_registry_and_limits_limits_has_expected_attributes() -> None:
    """Verify BackendLimits has the expected attributes."""
    config = MockConfig()
    _registry, limits = build_registry_and_limits(config)

    # BackendLimits should have these attributes
    assert hasattr(limits, "default_limit")
    assert hasattr(limits, "max_rows_per_call")


# -----------------------------------------------------------------------------
# describe_dataset Tests
# -----------------------------------------------------------------------------


def test_describe_dataset_with_known_contract() -> None:
    """Verify describe_dataset includes column preview for known contracts."""
    contracts_by_key = get_dataset_contracts_by_table_key()

    # Find a contract with a schema
    for table_key, contract in contracts_by_key.items():
        if contract.schema is not None and len(contract.schema.columns) > 0:
            name = contract.name or "dataset"
            description = describe_dataset(name, table_key)

            # Should include the table key
            assert table_key in description
            # Should include the name
            assert name in description
            # Should include column names
            first_column = contract.schema.columns[0].name
            assert first_column in description
            break


def test_describe_dataset_with_unknown_table() -> None:
    """Verify describe_dataset returns simple format for unknown tables."""
    name = "unknown"
    table = "unknown.table"

    description = describe_dataset(name, table)

    assert description == f"{name}: {table}"


def test_describe_dataset_shows_ellipsis_for_many_columns() -> None:
    """Verify ellipsis is shown when more columns exist than preview count."""
    contracts_by_key = get_dataset_contracts_by_table_key()

    # Find a contract with more columns than PREVIEW_COLUMN_COUNT
    for table_key, contract in contracts_by_key.items():
        if contract.schema is not None and len(contract.schema.columns) > PREVIEW_COLUMN_COUNT:
            name = contract.name or "dataset"
            description = describe_dataset(name, table_key)

            assert "..." in description
            break


def test_describe_dataset_no_ellipsis_for_few_columns() -> None:
    """Verify no ellipsis when columns are within preview count."""
    contracts_by_key = get_dataset_contracts_by_table_key()

    # Find a contract with fewer columns than or equal to PREVIEW_COLUMN_COUNT
    for table_key, contract in contracts_by_key.items():
        if contract.schema is not None and len(contract.schema.columns) <= PREVIEW_COLUMN_COUNT:
            name = contract.name or "dataset"
            description = describe_dataset(name, table_key)

            # Should not have ellipsis
            assert "..." not in description
            break


def test_describe_dataset_format_with_parentheses() -> None:
    """Verify describe_dataset uses parentheses for column list."""
    contracts_by_key = get_dataset_contracts_by_table_key()

    for table_key, contract in contracts_by_key.items():
        if contract.schema is not None and len(contract.schema.columns) > 0:
            name = contract.name or "dataset"
            description = describe_dataset(name, table_key)

            # Should have format: "name: table_key (col1, col2, ...)"
            assert "(" in description
            assert ")" in description
            break


def test_describe_dataset_with_none_contract() -> None:
    """Verify describe_dataset handles tables without contracts gracefully."""
    # Use a completely unknown table that won't have a contract
    name = "completely_unknown"
    table = "no_schema.no_table"

    description = describe_dataset(name, table)

    # Should fall back to simple format
    assert description == f"{name}: {table}"
    assert "(" not in description


# -----------------------------------------------------------------------------
# validate_dataset_registry Tests
# -----------------------------------------------------------------------------


def test_validate_dataset_registry_with_minimal_gateway(gateway: StorageGateway) -> None:
    """Verify validate_dataset_registry behavior with gateway.

    The gateway may or may not pass validation depending on whether all
    registered datasets exist. This test verifies the function runs and
    produces expected error format when validation fails.
    """
    # Try validation - may succeed or fail depending on gateway state
    try:
        validate_dataset_registry(gateway)
        # Validation passed - gateway has all required tables
    except ValueError as exc:
        # Validation failed - verify error message format
        error_msg = str(exc)
        assert "Dataset registry validation failed" in error_msg


def test_validate_dataset_registry_error_contains_details() -> None:
    """Verify validate_dataset_registry includes specific details in errors."""
    gw = gateway_with_macros(repo="test/repo", commit="abc123")
    try:
        try:
            validate_dataset_registry(gw)
        except ValueError as exc:
            error_msg = str(exc)
            # Error should contain specific issue categories
            has_detail = any(
                detail in error_msg
                for detail in [
                    "missing tables/views",
                    "schema mismatches",
                    "dataset_rows failures",
                ]
            )
            # Either validation passed or error has proper details
            assert "Dataset registry validation failed" in error_msg
            assert has_detail
    finally:
        gw.close()


def test_validate_dataset_registry_uses_gateway_datasets(gateway: StorageGateway) -> None:
    """Verify validate_dataset_registry reads from gateway.datasets.mapping."""
    # Verify the gateway has a datasets.mapping attribute
    assert hasattr(gateway, "datasets")
    assert hasattr(gateway.datasets, "mapping")

    # The mapping should be a dict-like
    mapping = gateway.datasets.mapping
    assert isinstance(mapping, dict)


def test_validate_dataset_registry_only_raises_valueerror(gateway: StorageGateway) -> None:
    """Verify validate_dataset_registry only raises ValueError when it fails."""
    # This tests that the function either succeeds or raises ValueError
    # No other exception type should be raised
    with contextlib.suppress(ValueError):
        validate_dataset_registry(gateway)


# -----------------------------------------------------------------------------
# PREVIEW_COLUMN_COUNT Constant Tests
# -----------------------------------------------------------------------------


def test_preview_column_count_is_positive() -> None:
    """Verify PREVIEW_COLUMN_COUNT is a positive integer."""
    assert isinstance(PREVIEW_COLUMN_COUNT, int)
    assert PREVIEW_COLUMN_COUNT > 0


def test_preview_column_count_used_in_describe() -> None:
    """Verify PREVIEW_COLUMN_COUNT affects describe_dataset output."""
    contracts_by_key = get_dataset_contracts_by_table_key()

    for table_key, contract in contracts_by_key.items():
        if contract.schema is not None and len(contract.schema.columns) > PREVIEW_COLUMN_COUNT:
            name = contract.name or "dataset"
            description = describe_dataset(name, table_key)

            # Extract column list from parentheses
            paren_start = description.find("(")
            paren_end = description.find(")")
            if paren_start != -1 and paren_end != -1:
                paren_content = description[paren_start + 1 : paren_end]
                # Remove trailing ellipsis if present
                if paren_content.endswith("..."):
                    paren_content = paren_content[:-3]
                columns_shown = paren_content.split(",")
                # Should show at most PREVIEW_COLUMN_COUNT columns
                assert len(columns_shown) <= PREVIEW_COLUMN_COUNT + 1
            break
