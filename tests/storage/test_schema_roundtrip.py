"""Schema generation and round-trip validation tests."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import duckdb
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from codeintel.config.schemas.tables import TABLE_SCHEMAS
from codeintel.storage.datasets import load_dataset_registry
from codeintel.storage.metadata_bootstrap import bootstrap_metadata_datasets
from codeintel.storage.rows import (
    BehavioralCoverageRowModel,
    CallGraphEdgeRow,
    SymbolUseRow,
    TestCoverageEdgeRow,
    behavioral_coverage_row_to_tuple,
    call_graph_edge_to_tuple,
    serialize_test_coverage_edge,
    symbol_use_to_tuple,
)
from codeintel.storage.schema_generation import (
    generate_export_schemas,
    json_schema_from_typeddict,
    validate_row_with_schema,
)
from codeintel.storage.schemas import apply_all_schemas


def _json_safe(value: object) -> object:
    """
    Coerce arbitrary values into JSON-serializable shapes for validation.

    Returns
    -------
    object
        JSON-friendly value (basic types, lists/dicts, ISO datetimes, or strings).
    """
    if value is NotImplemented:
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


@settings(max_examples=15)
@given(st.from_type(CallGraphEdgeRow))
def test_call_graph_edge_round_trip(row: CallGraphEdgeRow) -> None:
    """Generated schemas from TypedDict should validate generated call graph edges."""
    schema = json_schema_from_typeddict(CallGraphEdgeRow)
    validate_row_with_schema({key: _json_safe(value) for key, value in row.items()}, schema)
    values = call_graph_edge_to_tuple(row)
    expected_len = len(TABLE_SCHEMAS["graph.call_graph_edges"].columns)
    if len(values) != expected_len:
        pytest.fail(f"Expected {expected_len} values, got {len(values)}")


@settings(max_examples=15)
@given(st.from_type(SymbolUseRow))
def test_symbol_use_round_trip(row: SymbolUseRow) -> None:
    """Generated schemas should align with symbol use TypedDict and serializer."""
    schema = json_schema_from_typeddict(SymbolUseRow)
    validate_row_with_schema({key: _json_safe(value) for key, value in row.items()}, schema)
    values = symbol_use_to_tuple(row)
    expected_len = len(TABLE_SCHEMAS["graph.symbol_use_edges"].columns)
    if len(values) != expected_len:
        pytest.fail(f"Expected {expected_len} values, got {len(values)}")


@settings(max_examples=15)
@given(st.from_type(TestCoverageEdgeRow))
def test_test_coverage_round_trip(row: TestCoverageEdgeRow) -> None:
    """Generated schemas should align with test coverage edge TypedDict and serializer."""
    schema = json_schema_from_typeddict(TestCoverageEdgeRow)
    validate_row_with_schema({key: _json_safe(value) for key, value in row.items()}, schema)
    values = serialize_test_coverage_edge(row)
    expected_len = len(TABLE_SCHEMAS["analytics.test_coverage_edges"].columns)
    if len(values) != expected_len:
        pytest.fail(f"Expected {expected_len} values, got {len(values)}")


@settings(max_examples=15)
@given(st.from_type(BehavioralCoverageRowModel))
def test_behavioral_coverage_round_trip(row: BehavioralCoverageRowModel) -> None:
    """Generated schemas should align with behavioral coverage TypedDict and serializer."""
    schema = json_schema_from_typeddict(BehavioralCoverageRowModel)
    validate_row_with_schema({key: _json_safe(value) for key, value in row.items()}, schema)
    values = behavioral_coverage_row_to_tuple(row)
    expected_len = len(TABLE_SCHEMAS["analytics.behavioral_coverage"].columns)
    if len(values) != expected_len:
        pytest.fail(f"Expected {expected_len} values, got {len(values)}")


def test_generate_export_schemas_writes_files(tmp_path: Path) -> None:
    """Codegen should write schemas for datasets with row bindings."""
    con = duckdb.connect(database=":memory:")
    apply_all_schemas(con)
    bootstrap_metadata_datasets(con)
    registry = load_dataset_registry(con)
    written = generate_export_schemas(
        registry,
        output_dir=tmp_path,
        include_datasets={"call_graph_edges"},
    )
    schema_path = tmp_path / "call_graph_edges.json"
    if not schema_path.exists():
        pytest.fail("Expected generated schema for call_graph_edges")
    if schema_path not in written:
        pytest.fail("Generated schemas list missing call_graph_edges.json")
    doc = json.loads(schema_path.read_text(encoding="utf-8"))
    if "properties" not in doc or "repo" not in doc["properties"]:
        pytest.fail("Generated schema missing expected repo property")
