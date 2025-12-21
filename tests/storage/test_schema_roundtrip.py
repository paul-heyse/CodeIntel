"""Schema generation and round-trip validation tests."""

from __future__ import annotations

import dataclasses
import json
from datetime import UTC, datetime
from functools import lru_cache
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

    from hypothesis.strategies import SearchStrategy

    from codeintel.core.schemas.contract_primitives import DatasetContract
    from codeintel.storage.gateway import StorageGateway


MAX_HYPOTHESIS_EXAMPLES = 15
TEXT_SAMPLES = ["alpha", "bravo", "charlie", "delta", "echo"]
CALL_GRAPH_EDGE_SAMPLES: list[CallGraphEdgeRow] = [
    {
        "repo": "alpha",
        "commit": "bravo",
        "caller_goid_h128": 1,
        "callee_goid_h128": 2,
        "callsite_path": "src/app.py",
        "callsite_line": 10,
        "callsite_col": 4,
        "language": "python",
        "kind": "function",
        "resolved_via": "static",
        "confidence": 0.9,
        "evidence_json": {"evidence": "value"},
    },
    {
        "repo": "charlie",
        "commit": "delta",
        "caller_goid_h128": 3,
        "callee_goid_h128": None,
        "callsite_path": "src/lib.ts",
        "callsite_line": 42,
        "callsite_col": 0,
        "language": "javascript",
        "kind": "method",
        "resolved_via": None,
        "confidence": None,
        "evidence_json": None,
    },
]
TEST_COVERAGE_EDGE_SAMPLES: list[TestCoverageEdgeRow] = [
    {
        "test_id": "test_demo",
        "test_goid_h128": 101,
        "function_goid_h128": 202,
        "urn": "urn:ci:test:demo",
        "repo": "alpha",
        "commit": "bravo",
        "rel_path": "tests/test_demo.py",
        "qualname": "TestDemo.test_demo",
        "covered_lines": 12,
        "executable_lines": 20,
        "coverage_ratio": 0.6,
        "last_status": "passed",
        "created_at": datetime(2023, 1, 1, tzinfo=UTC),
    },
    {
        "test_id": None,
        "test_goid_h128": None,
        "function_goid_h128": None,
        "urn": None,
        "repo": None,
        "commit": None,
        "rel_path": None,
        "qualname": None,
        "covered_lines": None,
        "executable_lines": None,
        "coverage_ratio": None,
        "last_status": None,
        "created_at": None,
    },
]


def _naive_datetime(year: int, month: int, day: int) -> datetime:
    return datetime(year, month, day, tzinfo=UTC).replace(tzinfo=None)


def _short_text() -> SearchStrategy[str]:
    return st.sampled_from(TEXT_SAMPLES)


def _optional_text() -> SearchStrategy[str | None]:
    return st.sampled_from([None, *TEXT_SAMPLES])


def _small_int() -> SearchStrategy[int]:
    return st.sampled_from([0, 1, 2, 3, 5, 8, 13])


def _optional_int() -> SearchStrategy[int | None]:
    return st.sampled_from([None, 0, 1, 2, 3, 5, 8, 13])


def _small_float() -> SearchStrategy[float]:
    return st.sampled_from([0.0, 0.5, 1.0])


def _optional_float() -> SearchStrategy[float | None]:
    return st.sampled_from([None, 0.0, 0.5, 1.0])


def _call_graph_edge_strategy() -> SearchStrategy[CallGraphEdgeRow]:
    return st.sampled_from(CALL_GRAPH_EDGE_SAMPLES)


def _test_coverage_edge_strategy() -> SearchStrategy[TestCoverageEdgeRow]:
    return st.sampled_from(TEST_COVERAGE_EDGE_SAMPLES)


def _symbol_use_row_strategy() -> SearchStrategy[SymbolUseRow]:
    return st.builds(
        SymbolUseRow,
        symbol=_short_text(),
        def_path=_short_text(),
        use_path=_short_text(),
        same_file=st.booleans(),
        same_module=st.booleans(),
        def_goid_h128=_optional_int(),
        use_goid_h128=_optional_int(),
    )


def _behavioral_coverage_strategy() -> SearchStrategy[BehavioralCoverageRowModel]:
    mapping: dict[str, SearchStrategy[object]] = {
        "repo": _short_text(),
        "commit": _short_text(),
        "test_id": _short_text(),
        "test_goid_h128": _optional_int(),
        "rel_path": _short_text(),
        "qualname": _optional_text(),
        "behavior_tags": st.just({"tag": "value"}),
        "tag_source": _short_text(),
        "heuristic_version": _optional_text(),
        "llm_model": _optional_text(),
        "llm_run_id": _optional_text(),
        "created_at": st.datetimes(
            min_value=_naive_datetime(2000, 1, 1),
            max_value=_naive_datetime(2030, 1, 1),
            timezones=st.just(UTC),
        ),
    }
    return cast("SearchStrategy[BehavioralCoverageRowModel]", st.fixed_dictionaries(mapping))


@lru_cache(maxsize=1)
def _contracts_by_table_key() -> dict[str, DatasetContract]:
    return dict(iter_contracts_by_table_key())


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
@given(_call_graph_edge_strategy())
def test_call_graph_edge_round_trip(row: CallGraphEdgeRow) -> None:
    """Generate schemas from TypedDict should validate generated call graph edges."""
    schema = json_schema_from_typeddict(CallGraphEdgeRow)
    validate_row_with_schema({key: _json_safe(value) for key, value in row.items()}, schema)
    contract = _contracts_by_table_key()["graph.call_graph_edges"]
    expected_len = len(contract.schema.columns) if contract.schema else 0
    columns = [col.name for col in contract.schema.columns] if contract.schema else []
    values = tuple(cast("dict[str, object]", row)[col] for col in columns)
    if len(values) != expected_len:
        pytest.fail(f"Expected {expected_len} values, got {len(values)}")


@settings(max_examples=MAX_HYPOTHESIS_EXAMPLES)
@given(_symbol_use_row_strategy())
def test_symbol_use_round_trip(row: SymbolUseRow) -> None:
    """Generate schemas should align with symbol use dataclass and serializer."""
    schema = json_schema_from_typeddict(SymbolUseRow)
    row_dict = dataclasses.asdict(row)
    validate_row_with_schema({key: _json_safe(value) for key, value in row_dict.items()}, schema)
    values = row.to_tuple()
    contract = _contracts_by_table_key()["graph.symbol_use_edges"]
    expected_len = len(contract.schema.columns) if contract.schema else 0
    if len(values) != expected_len:
        pytest.fail(f"Expected {expected_len} values, got {len(values)}")


@settings(max_examples=MAX_HYPOTHESIS_EXAMPLES)
@given(_test_coverage_edge_strategy())
def test_test_coverage_round_trip(row: TestCoverageEdgeRow) -> None:
    """Generate schemas should align with test coverage edge TypedDict and serializer."""
    schema = json_schema_from_typeddict(TestCoverageEdgeRow)
    validate_row_with_schema({key: _json_safe(value) for key, value in row.items()}, schema)
    contract = _contracts_by_table_key()["analytics.test_coverage_edges"]
    expected_len = len(contract.schema.columns) if contract.schema else 0
    columns = [col.name for col in contract.schema.columns] if contract.schema else []
    values = tuple(cast("dict[str, object]", row)[col] for col in columns)
    if len(values) != expected_len:
        pytest.fail(f"Expected {expected_len} values, got {len(values)}")


@settings(max_examples=MAX_HYPOTHESIS_EXAMPLES)
@given(_behavioral_coverage_strategy())
def test_behavioral_coverage_round_trip(row: BehavioralCoverageRowModel) -> None:
    """Generate schemas should align with behavioral coverage TypedDict and serializer."""
    schema = json_schema_from_typeddict(BehavioralCoverageRowModel)
    validate_row_with_schema({key: _json_safe(value) for key, value in row.items()}, schema)
    contract = _contracts_by_table_key()["analytics.behavioral_coverage"]
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
