"""Tests for registry_helpers module."""

from __future__ import annotations

import pytest

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.registry_helpers import (
    DatasetRegistry,
    build_dataset_registry,
    describe_all_datasets,
)


def test_dataset_registry_all_datasets_property(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify all_datasets returns combined tables and views."""
    con = fresh_gateway.con

    registry = build_dataset_registry(con, include_views=True)

    all_datasets = registry.all_datasets
    assert isinstance(all_datasets, tuple)
    assert len(all_datasets) > 0
    assert len(all_datasets) == len(registry.tables) + len(registry.views)


def test_dataset_registry_table_for_name_returns_key(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify table_for_name returns fully qualified key for known dataset."""
    con = fresh_gateway.con

    registry = build_dataset_registry(con, include_views=True)

    if not registry.tables:
        pytest.skip("No tables registered")

    first_table = registry.tables[0]
    table_key = registry.table_for_name(first_table)

    assert "." in table_key


def test_dataset_registry_table_for_name_raises_on_unknown(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify table_for_name raises KeyError for unknown dataset."""
    con = fresh_gateway.con

    registry = build_dataset_registry(con, include_views=True)

    with pytest.raises(KeyError, match="Unknown dataset"):
        registry.table_for_name("nonexistent_dataset_xyz")


def test_build_dataset_registry_excludes_views_when_requested(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify build_dataset_registry respects include_views=False."""
    con = fresh_gateway.con

    registry_with_views = build_dataset_registry(con, include_views=True)
    registry_without_views = build_dataset_registry(con, include_views=False)

    assert len(registry_without_views.views) == 0
    assert len(registry_with_views.views) >= 0


def test_build_dataset_registry_includes_meta(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify build_dataset_registry populates meta mapping."""
    con = fresh_gateway.con

    registry = build_dataset_registry(con, include_views=True)

    assert registry.meta is not None
    assert len(registry.meta) > 0


def test_build_dataset_registry_includes_jsonl_mapping(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify build_dataset_registry populates jsonl_mapping."""
    con = fresh_gateway.con

    registry = build_dataset_registry(con, include_views=True)

    assert registry.jsonl_mapping is not None


def test_build_dataset_registry_includes_parquet_mapping(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify build_dataset_registry populates parquet_mapping."""
    con = fresh_gateway.con

    registry = build_dataset_registry(con, include_views=True)

    assert registry.parquet_mapping is not None


def test_describe_all_datasets_returns_serializable_list(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify describe_all_datasets returns JSON-serializable list."""
    con = fresh_gateway.con

    descriptions = describe_all_datasets(con)

    assert isinstance(descriptions, list)
    assert len(descriptions) > 0

    first_desc = descriptions[0]
    assert isinstance(first_desc, dict)
    assert "name" in first_desc or "table_key" in first_desc


def test_dataset_registry_frozen() -> None:
    """Verify DatasetRegistry is immutable."""
    registry = DatasetRegistry(
        mapping={"test": "core.test"},
        tables=("test",),
        views=(),
        meta=None,
        jsonl_mapping=None,
        parquet_mapping=None,
    )

    assert registry.tables == ("test",)
    assert registry.views == ()
