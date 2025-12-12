"""Tests verifying alignment between canonical catalog and registry."""

from __future__ import annotations

import re

import pytest

from codeintel.config.datasets import get_dataset_contracts_by_table_key
from codeintel.serving.operations import (
    DataSourceType,
    get_operation,
    iter_operations,
)
from codeintel.serving.operations.catalog import (
    get_registry_operation,
    iter_registry_operations,
)


def test_registry_and_catalog_agree_on_ids() -> None:
    """Registry operation IDs must match catalog operation IDs."""
    catalog_ids = {op.id for op in iter_operations()}
    registry_ids = {op.id for op in iter_registry_operations()}
    if catalog_ids != registry_ids:
        diff = catalog_ids.symmetric_difference(registry_ids)
        pytest.fail(f"Catalog and registry IDs differ: {diff}")


def test_registry_returns_catalog_objects() -> None:
    """Get_registry_operation should return objects with same attributes as catalog."""
    for op in iter_operations():
        registry_op = get_registry_operation(op.id)
        catalog_op = get_operation(op.id)

        if registry_op is None:
            pytest.fail(f"get_registry_operation returned None for {op.id}")
        if catalog_op is None:
            pytest.fail(f"get_operation returned None for {op.id}")
        if registry_op.id != catalog_op.id:
            pytest.fail(f"ID mismatch for {op.id}")
        if registry_op.category != catalog_op.category:
            pytest.fail(f"Category mismatch for {op.id}")
        if registry_op.backend_method != catalog_op.backend_method:
            pytest.fail(f"backend_method mismatch for {op.id}")


def test_all_operation_ids_are_unique() -> None:
    """Every operation ID must be unique."""
    ids = [op.id for op in iter_operations()]
    if len(ids) != len(set(ids)):
        pytest.fail("Duplicate operation IDs found")


def test_view_operations_have_source_names() -> None:
    """Operations with VIEW data_source should have a source_name."""
    for op in iter_operations():
        if op.data_source == DataSourceType.VIEW:
            if not op.source_name:
                pytest.fail(f"VIEW operation {op.id} missing source_name")
            if not op.source_name.startswith(("docs.", "analytics.")):
                pytest.fail(
                    f"VIEW operation {op.id} has unexpected source_name prefix: {op.source_name}"
                )


def test_table_operations_have_source_names() -> None:
    """Operations with TABLE data_source should have a source_name."""
    for op in iter_operations():
        if op.data_source == DataSourceType.TABLE and not op.source_name:
            pytest.fail(f"TABLE operation {op.id} missing source_name")


def test_graph_engine_operations_have_required_graphs() -> None:
    """Operations with GRAPH_ENGINE data_source should have required_graphs."""
    for op in iter_operations():
        if op.data_source == DataSourceType.GRAPH_ENGINE and not op.required_graphs:
            pytest.fail(f"GRAPH_ENGINE operation {op.id} should have required_graphs")


def test_required_datasets_are_valid() -> None:
    """All required_datasets should exist in DATASET_CONTRACTS_BY_TABLE_KEY."""
    valid_keys = set(get_dataset_contracts_by_table_key().keys())

    for op in iter_operations():
        for dataset_key in op.required_datasets:
            if "." in dataset_key and dataset_key not in valid_keys:
                pytest.fail(f"Operation {op.id} requires unknown dataset: {dataset_key}")


def test_all_tool_names_are_unique() -> None:
    """Every non-None tool_name must be unique across operations."""
    tool_names = [op.tool_name for op in iter_operations() if op.tool_name]
    if len(tool_names) != len(set(tool_names)):
        duplicates = [n for n in tool_names if tool_names.count(n) > 1]
        pytest.fail(f"Duplicate MCP tool names detected: {duplicates}")


def test_tool_names_follow_convention() -> None:
    """Tool names should follow snake_case convention."""
    for op in iter_operations():
        if op.tool_name and not (op.tool_name.islower() or "_" in op.tool_name):
            pytest.fail(f"Tool name {op.tool_name} for {op.id} should be snake_case")


def test_http_paths_are_unique() -> None:
    """Every non-None http_path must be unique (ignoring path params)."""

    def normalize_path(path: str) -> str:
        return re.sub(r"\{[^}]+\}", "{}", path)

    paths_with_methods: list[tuple[str, str]] = []
    for op in iter_operations():
        if op.http_path and op.http_method:
            normalized = normalize_path(op.http_path)
            paths_with_methods.append((op.http_method, normalized))

    if len(paths_with_methods) != len(set(paths_with_methods)):
        pytest.fail("Duplicate HTTP method+path combinations detected")


def test_http_operations_have_both_method_and_path() -> None:
    """Operations with http_method must have http_path and vice versa."""
    for op in iter_operations():
        if op.http_method is not None and op.http_path is None:
            pytest.fail(f"Operation {op.id} has http_method but no http_path")
        if op.http_path is not None and op.http_method is None:
            pytest.fail(f"Operation {op.id} has http_path but no http_method")
