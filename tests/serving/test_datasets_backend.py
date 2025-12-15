"""Tests for serving backend dataset registry helpers.

This module verifies the dataset registry building, description generation,
and validation functions without mocking, using real DuckDB connections.
"""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import pytest

from codeintel.build.schemas import iter_contracts, iter_contracts_by_table_key
from codeintel.serving.backend.datasets import (
    DOCS_VIEWS,
    PREVIEW_COLUMN_COUNT,
    build_dataset_registry,
    build_registry_and_limits,
    describe_dataset,
    validate_dataset_registry,
)
from codeintel.serving.backend.pagination import BackendLimits
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_is_not_none,
    expect_not_in,
    expect_true,
)
from tests._helpers.gateway import GatewayFactory

if TYPE_CHECKING:
    from collections.abc import Iterator

    from codeintel.storage.gateway import StorageGateway


DEFAULT_LIMIT = 100
MAX_ROWS = 1000
CUSTOM_LIMIT = 50
CUSTOM_MAX = 500


@pytest.fixture
def gateway() -> Iterator[StorageGateway]:
    """Provide a real StorageGateway backed by an in-memory DuckDB database.

    Yields
    ------
    StorageGateway
        Gateway instance for test use.
    """
    gw = GatewayFactory().with_snapshot("test/repo", "abc123").open()
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


def test_docs_views_contains_only_views() -> None:
    """Verify DOCS_VIEWS dictionary contains only view contracts."""
    contracts = {c.name: c for c in iter_contracts()}

    for name in DOCS_VIEWS:
        contract = expect_is_not_none(
            contracts.get(name),
            message=f"DOCS_VIEWS contains unknown dataset: {name}",
        )
        expect_true(
            contract.is_view,
            message=f"DOCS_VIEWS should only contain views, but {name} is not a view",
        )


def test_docs_views_values_are_table_keys() -> None:
    """Verify DOCS_VIEWS values match contract table_keys."""
    contracts = {c.name: c for c in iter_contracts()}

    for name, table_key in DOCS_VIEWS.items():
        contract = expect_is_not_none(contracts.get(name))
        expect_equal(contract.table_key, table_key)


def test_docs_views_is_non_empty_if_views_exist() -> None:
    """Verify DOCS_VIEWS is populated when view contracts exist."""
    contracts = {c.name: c for c in iter_contracts()}
    view_count = sum(1 for c in contracts.values() if c.is_view)

    expect_equal(len(DOCS_VIEWS), view_count)


def test_build_dataset_registry_returns_dict() -> None:
    """Verify build_dataset_registry returns a dictionary."""
    registry = build_dataset_registry()
    expect_is_instance(registry, dict)


def test_build_dataset_registry_includes_docs_views_by_default() -> None:
    """Verify docs views are included when include_docs_views is 'include'."""
    registry = build_dataset_registry(include_docs_views="include")

    for view_name in DOCS_VIEWS:
        expect_in(view_name, registry, label=f"Expected docs view {view_name} in registry")


def test_build_dataset_registry_excludes_docs_views_when_requested() -> None:
    """Verify docs views are excluded when include_docs_views is 'exclude'."""
    registry = build_dataset_registry(include_docs_views="exclude")

    for view_name in DOCS_VIEWS:
        expect_not_in(view_name, registry, label=f"Docs view {view_name} should be excluded")


def test_build_dataset_registry_values_are_table_keys() -> None:
    """Verify registry values match contract table_keys."""
    registry = build_dataset_registry()
    contracts = {c.name: c for c in iter_contracts()}

    for name, table_key in registry.items():
        contract = expect_is_not_none(
            contracts.get(name),
            message=f"Registry contains unknown dataset: {name}",
        )
        expect_equal(contract.table_key, table_key)


def test_build_dataset_registry_is_deterministic() -> None:
    """Verify build_dataset_registry returns consistent results."""
    registry1 = build_dataset_registry()
    registry2 = build_dataset_registry()
    expect_equal(registry1, registry2)


def test_build_dataset_registry_sorted_by_table_key() -> None:
    """Verify registry is sorted by table_key."""
    registry = build_dataset_registry()
    table_keys = list(registry.values())
    expect_equal(table_keys, sorted(table_keys))


def test_build_dataset_registry_exclude_reduces_count() -> None:
    """Verify excluding docs views reduces registry size when views exist."""
    registry_include = build_dataset_registry(include_docs_views="include")
    registry_exclude = build_dataset_registry(include_docs_views="exclude")

    if DOCS_VIEWS:
        expect_true(len(registry_exclude) < len(registry_include))
    else:
        expect_equal(len(registry_exclude), len(registry_include))


def test_build_dataset_registry_contains_all_non_view_datasets() -> None:
    """Verify all non-view datasets are in registry regardless of include mode."""
    contracts = {c.name: c for c in iter_contracts()}
    non_view_names = {name for name, c in contracts.items() if not c.is_view}

    registry_exclude = build_dataset_registry(include_docs_views="exclude")

    for name in non_view_names:
        expect_in(name, registry_exclude)


def test_build_registry_and_limits_returns_tuple() -> None:
    """Verify build_registry_and_limits returns a tuple of registry and limits."""
    config = MockConfig()
    registry, limits = build_registry_and_limits(config)

    expect_is_instance(registry, dict)
    expect_is_instance(limits, BackendLimits)


def test_build_registry_and_limits_uses_config_values() -> None:
    """Verify limits are derived from config."""
    config = MockConfig(default_limit=CUSTOM_LIMIT, max_rows=CUSTOM_MAX)

    _registry, limits = build_registry_and_limits(config)

    expect_equal(limits.default_limit, CUSTOM_LIMIT)
    expect_equal(limits.max_rows_per_call, CUSTOM_MAX)


def test_build_registry_and_limits_respects_include_docs_views() -> None:
    """Verify include_docs_views parameter is respected."""
    config = MockConfig()

    registry_with_views, _ = build_registry_and_limits(config, include_docs_views="include")
    registry_without_views, _ = build_registry_and_limits(config, include_docs_views="exclude")

    expect_true(len(registry_with_views) >= len(registry_without_views))


def test_build_registry_and_limits_registry_matches_standalone() -> None:
    """Verify registry from build_registry_and_limits matches build_dataset_registry."""
    config = MockConfig()

    registry_combined, _ = build_registry_and_limits(config, include_docs_views="include")
    registry_standalone = build_dataset_registry(include_docs_views="include")

    expect_equal(registry_combined, registry_standalone)


def test_build_registry_and_limits_limits_has_expected_attributes() -> None:
    """Verify BackendLimits has the expected attributes."""
    config = MockConfig()
    _registry, limits = build_registry_and_limits(config)

    expect_true(hasattr(limits, "default_limit"))
    expect_true(hasattr(limits, "max_rows_per_call"))


def test_describe_dataset_with_known_contract() -> None:
    """Verify describe_dataset includes column preview for known contracts."""
    contracts_by_key = dict(iter_contracts_by_table_key())

    for table_key, contract in contracts_by_key.items():
        if contract.schema is not None and len(contract.schema.columns) > 0:
            name = contract.name or "dataset"
            description = describe_dataset(name, table_key)

            expect_in(table_key, description)

            expect_in(name, description)

            first_column = contract.schema.columns[0].name
            expect_in(first_column, description)
            break


def test_describe_dataset_with_unknown_table() -> None:
    """Verify describe_dataset returns simple format for unknown tables."""
    name = "unknown"
    table = "unknown.table"

    description = describe_dataset(name, table)

    expect_equal(description, f"{name}: {table}")


def test_describe_dataset_shows_ellipsis_for_many_columns() -> None:
    """Verify ellipsis is shown when more columns exist than preview count."""
    contracts_by_key = dict(iter_contracts_by_table_key())

    for table_key, contract in contracts_by_key.items():
        if contract.schema is not None and len(contract.schema.columns) > PREVIEW_COLUMN_COUNT:
            name = contract.name or "dataset"
            description = describe_dataset(name, table_key)

            expect_in("...", description)
            break


def test_describe_dataset_no_ellipsis_for_few_columns() -> None:
    """Verify no ellipsis when columns are within preview count."""
    contracts_by_key = dict(iter_contracts_by_table_key())

    for table_key, contract in contracts_by_key.items():
        if contract.schema is not None and len(contract.schema.columns) <= PREVIEW_COLUMN_COUNT:
            name = contract.name or "dataset"
            description = describe_dataset(name, table_key)

            expect_not_in("...", description)
            break


def test_describe_dataset_format_with_parentheses() -> None:
    """Verify describe_dataset uses parentheses for column list."""
    contracts_by_key = dict(iter_contracts_by_table_key())

    for table_key, contract in contracts_by_key.items():
        if contract.schema is not None and len(contract.schema.columns) > 0:
            name = contract.name or "dataset"
            description = describe_dataset(name, table_key)

            expect_in("(", description)
            expect_in(")", description)
            break


def test_describe_dataset_with_none_contract() -> None:
    """Verify describe_dataset handles tables without contracts gracefully."""
    name = "completely_unknown"
    table = "no_schema.no_table"

    description = describe_dataset(name, table)

    expect_equal(description, f"{name}: {table}")
    expect_not_in("(", description)


def test_validate_dataset_registry_with_minimal_gateway(gateway: StorageGateway) -> None:
    """Verify validate_dataset_registry behavior with gateway.

    The gateway may or may not pass validation depending on whether all
    registered datasets exist. This test verifies the function runs and
    produces expected error format when validation fails.
    """
    try:
        validate_dataset_registry(gateway)

    except ValueError as exc:
        error_msg = str(exc)
        expect_in("Dataset registry validation failed", error_msg)


def test_validate_dataset_registry_error_contains_details() -> None:
    """Verify validate_dataset_registry includes specific details in errors."""
    gw = GatewayFactory().with_snapshot("test/repo", "abc123").open()
    try:
        try:
            validate_dataset_registry(gw)
        except ValueError as exc:
            error_msg = str(exc)

            has_detail = any(
                detail in error_msg
                for detail in [
                    "missing tables/views",
                    "schema mismatches",
                    "dataset_rows failures",
                ]
            )

            expect_in("Dataset registry validation failed", error_msg)
            expect_true(has_detail)
    finally:
        gw.close()


def test_validate_dataset_registry_uses_gateway_datasets(gateway: StorageGateway) -> None:
    """Verify validate_dataset_registry reads from gateway.datasets.mapping."""
    expect_true(hasattr(gateway, "datasets"))
    expect_true(hasattr(gateway.datasets, "mapping"))

    mapping = gateway.datasets.mapping
    expect_is_instance(mapping, dict)


def test_validate_dataset_registry_only_raises_valueerror(gateway: StorageGateway) -> None:
    """Verify validate_dataset_registry only raises ValueError when it fails."""
    with contextlib.suppress(ValueError):
        validate_dataset_registry(gateway)


def test_preview_column_count_is_positive() -> None:
    """Verify PREVIEW_COLUMN_COUNT is a positive integer."""
    expect_is_instance(PREVIEW_COLUMN_COUNT, int)
    expect_true(PREVIEW_COLUMN_COUNT > 0)


def test_preview_column_count_used_in_describe() -> None:
    """Verify PREVIEW_COLUMN_COUNT affects describe_dataset output."""
    contracts_by_key = dict(iter_contracts_by_table_key())

    for table_key, contract in contracts_by_key.items():
        if contract.schema is not None and len(contract.schema.columns) > PREVIEW_COLUMN_COUNT:
            name = contract.name or "dataset"
            description = describe_dataset(name, table_key)

            paren_start = description.find("(")
            paren_end = description.find(")")
            if paren_start != -1 and paren_end != -1:
                paren_content = description[paren_start + 1 : paren_end]

                if paren_content.endswith("..."):
                    paren_content = paren_content[:-3]
                columns_shown = paren_content.split(",")

                expect_true(len(columns_shown) <= PREVIEW_COLUMN_COUNT + 1)
            break
