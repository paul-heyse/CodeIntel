"""Schema generation and round-trip validation tests."""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime
from typing import TYPE_CHECKING, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from codeintel.build.schemas import iter_contracts_by_table_key
from codeintel.core.data_models.rows import (
    SymbolUseRow,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsBehavioralCoverageRow as BehavioralCoverageRowModel,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsTestCoverageEdgesRow as TestCoverageEdgeRow,
)
from codeintel.core.schemas.generated_rows.graph import (
    GraphCallGraphEdgesRow as CallGraphEdgeRow,
)
from codeintel.storage.datasets import load_dataset_registry
from codeintel.storage.metadata import bootstrap_metadata_datasets
from codeintel.storage.schema.json_schema import (
    generate_export_schemas,
    json_schema_from_typeddict,
    validate_row_with_schema,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway


MAX_HYPOTHESIS_EXAMPLES = 15


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


@settings(max_examples=MAX_HYPOTHESIS_EXAMPLES)
@given(st.from_type(CallGraphEdgeRow))
def test_call_graph_edge_round_trip(row: CallGraphEdgeRow) -> None:
    """Generate schemas from TypedDict should validate generated call graph edges."""
    schema = json_schema_from_typeddict(CallGraphEdgeRow)
    validate_row_with_schema({key: _json_safe(value) for key, value in row.items()}, schema)
    contracts = dict(iter_contracts_by_table_key())
    contract = contracts["graph.call_graph_edges"]
    expected_len = len(contract.schema.columns) if contract.schema else 0
    columns = [col.name for col in contract.schema.columns] if contract.schema else []
    values = tuple(cast("dict[str, object]", row)[col] for col in columns)
    if len(values) != expected_len:
        pytest.fail(f"Expected {expected_len} values, got {len(values)}")


@settings(max_examples=MAX_HYPOTHESIS_EXAMPLES)
@given(st.from_type(SymbolUseRow))
def test_symbol_use_round_trip(row: SymbolUseRow) -> None:
    """Generate schemas should align with symbol use dataclass and serializer."""
    schema = json_schema_from_typeddict(SymbolUseRow)
    row_dict = dataclasses.asdict(row)
    validate_row_with_schema({key: _json_safe(value) for key, value in row_dict.items()}, schema)
    values = row.to_tuple()
    contracts = dict(iter_contracts_by_table_key())
    contract = contracts["graph.symbol_use_edges"]
    expected_len = len(contract.schema.columns) if contract.schema else 0
    if len(values) != expected_len:
        pytest.fail(f"Expected {expected_len} values, got {len(values)}")


@settings(max_examples=MAX_HYPOTHESIS_EXAMPLES)
@given(st.from_type(TestCoverageEdgeRow))
def test_test_coverage_round_trip(row: TestCoverageEdgeRow) -> None:
    """Generate schemas should align with test coverage edge TypedDict and serializer."""
    schema = json_schema_from_typeddict(TestCoverageEdgeRow)
    validate_row_with_schema({key: _json_safe(value) for key, value in row.items()}, schema)
    contracts = dict(iter_contracts_by_table_key())
    contract = contracts["analytics.test_coverage_edges"]
    expected_len = len(contract.schema.columns) if contract.schema else 0
    columns = [col.name for col in contract.schema.columns] if contract.schema else []
    values = tuple(cast("dict[str, object]", row)[col] for col in columns)
    if len(values) != expected_len:
        pytest.fail(f"Expected {expected_len} values, got {len(values)}")


@settings(max_examples=MAX_HYPOTHESIS_EXAMPLES)
@given(st.from_type(BehavioralCoverageRowModel))
def test_behavioral_coverage_round_trip(row: BehavioralCoverageRowModel) -> None:
    """Generate schemas should align with behavioral coverage TypedDict and serializer."""
    schema = json_schema_from_typeddict(BehavioralCoverageRowModel)
    validate_row_with_schema({key: _json_safe(value) for key, value in row.items()}, schema)
    contracts = dict(iter_contracts_by_table_key())
    contract = contracts["analytics.behavioral_coverage"]
    expected_len = len(contract.schema.columns) if contract.schema else 0
    columns = [col.name for col in contract.schema.columns] if contract.schema else []
    values = tuple(cast("dict[str, object]", row)[col] for col in columns)
    if len(values) != expected_len:
        pytest.fail(f"Expected {expected_len} values, got {len(values)}")


def test_generate_export_schemas_writes_files(
    schema_gateway: StorageGateway, tmp_path: Path
) -> None:
    """Codegen should write schemas for datasets with row bindings."""
    bootstrap_metadata_datasets(schema_gateway.con)
    registry = load_dataset_registry(schema_gateway.con)
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
