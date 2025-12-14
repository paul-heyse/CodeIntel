"""Tests for JSON export service.

Tests the export_all_constraints_json, export_dataset_catalog_json,
export_dependency_graph_json, and export_to_file functions.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.build.hamilton.contracts.schemas.constraints import ConstraintKind
from codeintel.build.hamilton.contracts.schemas.export import (
    export_all_constraints_json,
    export_dataset_catalog_json,
    export_dependency_graph_json,
    export_to_file,
    get_constraint_summary,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def _require(*, condition: bool, message: str) -> None:
    """Assert a condition using pytest.fail for S101 compliance."""
    if not condition:
        pytest.fail(message)


def _expect_equal(actual: object, expected: object, label: str) -> None:
    """Check equality with clear failure message."""
    if actual != expected:
        pytest.fail(f"{label}: expected {expected!r}, got {actual!r}")


def _expect_in(key: str, container: Mapping[str, object], label: str) -> None:
    """Check key is in mapping with clear failure message."""
    if key not in container:
        pytest.fail(f"{label}: key '{key}' not in {list(container.keys())}")


def test_export_all_constraints_json_returns_dict() -> None:
    """Verify export_all_constraints_json returns a dict."""
    result = export_all_constraints_json()
    _require(condition=isinstance(result, dict), message="should return dict")


def test_export_all_constraints_json_has_meta() -> None:
    """Verify result has meta section."""
    result = export_all_constraints_json()
    _expect_in("meta", result, "result")
    meta = result["meta"]
    _require(condition=isinstance(meta, dict), message="meta should be dict")
    _expect_in("export_type", meta, "meta")
    _expect_equal(meta["export_type"], "constraints", "export_type")


def test_export_all_constraints_json_has_datasets() -> None:
    """Verify result has datasets section."""
    result = export_all_constraints_json()
    _expect_in("datasets", result, "result")
    _require(condition=isinstance(result["datasets"], dict), message="datasets should be dict")


def test_export_all_constraints_json_structure() -> None:
    """Verify dataset entries have expected structure."""
    result = export_all_constraints_json()
    datasets = result["datasets"]
    if not isinstance(datasets, dict):
        pytest.skip("No datasets in export")

    for table_key, data in datasets.items():
        if not isinstance(data, dict):
            continue
        _require(condition=isinstance(table_key, str), message="table_key should be str")
        _expect_in("columns", data, f"{table_key}")
        _expect_in("table_level", data, f"{table_key}")
        _expect_in("constraint_count", data, f"{table_key}")


def test_export_dataset_catalog_json_returns_dict() -> None:
    """Verify export_dataset_catalog_json returns a dict."""
    result = export_dataset_catalog_json()
    _require(condition=isinstance(result, dict), message="should return dict")


def test_export_dataset_catalog_json_has_meta() -> None:
    """Verify result has meta section."""
    result = export_dataset_catalog_json()
    _expect_in("meta", result, "result")
    meta = result["meta"]
    _require(condition=isinstance(meta, dict), message="meta should be dict")
    _expect_equal(meta["export_type"], "catalog", "export_type")


def test_export_dataset_catalog_json_has_datasets() -> None:
    """Verify result has datasets section."""
    result = export_dataset_catalog_json()
    _expect_in("datasets", result, "result")
    _require(condition=isinstance(result["datasets"], dict), message="datasets should be dict")


def test_export_dataset_catalog_json_structure() -> None:
    """Verify dataset entries have expected structure."""
    result = export_dataset_catalog_json()
    datasets = result["datasets"]
    if not isinstance(datasets, dict):
        pytest.skip("No datasets in export")

    for table_key, data in datasets.items():
        if not isinstance(data, dict):
            continue
        _require(condition=isinstance(table_key, str), message="table_key should be str")
        _expect_in("name", data, f"{table_key}")
        _expect_in("columns", data, f"{table_key}")
        _expect_in("column_count", data, f"{table_key}")


def test_export_dependency_graph_json_returns_dict() -> None:
    """Verify export_dependency_graph_json returns a dict."""
    result = export_dependency_graph_json()
    _require(condition=isinstance(result, dict), message="should return dict")


def test_export_dependency_graph_json_has_meta() -> None:
    """Verify result has meta section."""
    result = export_dependency_graph_json()
    _expect_in("meta", result, "result")
    meta = result["meta"]
    _require(condition=isinstance(meta, dict), message="meta should be dict")
    _expect_equal(meta["export_type"], "dependency_graph", "export_type")


def test_export_dependency_graph_json_has_nodes() -> None:
    """Verify result has nodes section."""
    result = export_dependency_graph_json()
    _expect_in("nodes", result, "result")
    _require(condition=isinstance(result["nodes"], dict), message="nodes should be dict")


def test_export_dependency_graph_json_has_edges() -> None:
    """Verify result has edges section."""
    result = export_dependency_graph_json()
    _expect_in("edges", result, "result")
    _require(condition=isinstance(result["edges"], list), message="edges should be list")


def test_export_dependency_graph_json_has_topological_order() -> None:
    """Verify result has topological_order."""
    result = export_dependency_graph_json()
    _expect_in("topological_order", result, "result")
    _require(
        condition=isinstance(result["topological_order"], list),
        message="topological_order should be list",
    )


def test_export_dependency_graph_json_node_structure() -> None:
    """Verify node entries have expected structure."""
    result = export_dependency_graph_json()
    nodes = result["nodes"]
    if not isinstance(nodes, dict):
        pytest.skip("No nodes in export")

    for table_key, data in nodes.items():
        if not isinstance(data, dict):
            continue
        _require(condition=isinstance(table_key, str), message="table_key should be str")
        _expect_in("upstream", data, f"{table_key}")
        _expect_in("downstream", data, f"{table_key}")
        _expect_in("is_root", data, f"{table_key}")
        _expect_in("is_leaf", data, f"{table_key}")


def test_export_to_file_constraints() -> None:
    """Verify export_to_file writes constraints."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        bytes_written = export_to_file("constraints", path)
        _require(condition=bytes_written > 0, message="should write bytes")
        _require(condition=path.exists(), message="file should exist")

        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        _expect_in("meta", data, "data")
        _expect_equal(data["meta"]["export_type"], "constraints", "export_type")
    finally:
        if path.exists():
            path.unlink()


def test_export_to_file_catalog() -> None:
    """Verify export_to_file writes catalog."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        bytes_written = export_to_file("catalog", path)
        _require(condition=bytes_written > 0, message="should write bytes")
        _require(condition=path.exists(), message="file should exist")

        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        _expect_in("meta", data, "data")
        _expect_equal(data["meta"]["export_type"], "catalog", "export_type")
    finally:
        if path.exists():
            path.unlink()


def test_export_to_file_graph() -> None:
    """Verify export_to_file writes graph."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        bytes_written = export_to_file("graph", path)
        _require(condition=bytes_written > 0, message="should write bytes")
        _require(condition=path.exists(), message="file should exist")

        content = path.read_text(encoding="utf-8")
        data = json.loads(content)
        _expect_in("meta", data, "data")
        _expect_equal(data["meta"]["export_type"], "dependency_graph", "export_type")
    finally:
        if path.exists():
            path.unlink()


def test_export_to_file_invalid_type() -> None:
    """Verify export_to_file raises for invalid type."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = Path(f.name)

    try:
        try:
            export_to_file("invalid_type", path)
            pytest.fail("Should have raised ValueError")
        except ValueError:
            pass
    finally:
        if path.exists():
            path.unlink()


def test_export_to_file_creates_directories() -> None:
    """Verify export_to_file creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "nested" / "dir" / "output.json"
        bytes_written = export_to_file("catalog", path)
        _require(condition=bytes_written > 0, message="should write bytes")
        _require(condition=path.exists(), message="file should exist")


def test_get_constraint_summary_returns_dict() -> None:
    """Verify get_constraint_summary returns a dict."""
    result = get_constraint_summary()
    _require(condition=isinstance(result, dict), message="should return dict")


def test_get_constraint_summary_has_total_datasets() -> None:
    """Verify result has total_datasets."""
    result = get_constraint_summary()
    _expect_in("total_datasets", result, "result")
    _require(
        condition=isinstance(result["total_datasets"], int),
        message="total_datasets should be int",
    )


def test_get_constraint_summary_has_by_kind() -> None:
    """Verify result has by_kind breakdown."""
    result = get_constraint_summary()
    _expect_in("by_kind", result, "result")
    by_kind = result["by_kind"]
    _require(condition=isinstance(by_kind, dict), message="by_kind should be dict")


def test_get_constraint_summary_by_kind_has_all_kinds() -> None:
    """Verify by_kind has all constraint kinds."""
    result = get_constraint_summary()
    by_kind_obj = result["by_kind"]
    if not isinstance(by_kind_obj, dict):
        pytest.skip("by_kind not a dict")

    by_kind: Mapping[str, object] = by_kind_obj
    for kind in ConstraintKind:
        _expect_in(kind.value, by_kind, "by_kind")
