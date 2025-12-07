"""Domain builder unit tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import pytest

from codeintel.serving import domain_models as dm
from codeintel.serving.backend.domain_builders import (
    DatasetSchemaInput,
    build_callgraph_neighbors,
    build_dataset_schema,
    build_file_hints,
    build_file_profile,
    build_file_summary,
    build_function_architecture,
    build_function_profile,
    build_function_summary,
    build_graph_neighborhood,
    build_high_risk_functions,
    build_import_boundary,
    build_module_architecture,
    build_module_profile,
    build_module_subsystems,
    build_subsystem_coverage,
    build_subsystem_modules,
    build_subsystem_profile,
    build_subsystem_search,
    build_subsystem_summary,
    build_tests_for_function,
)
from tests._helpers.assertions import assert_mapping_value

BuilderReturn = (
    dm.FunctionSummaryResult
    | dm.FunctionProfileResult
    | dm.FileProfileResult
    | dm.ModuleProfileResult
    | dm.FunctionArchitectureResult
    | dm.ModuleArchitectureResult
)
Builder = Callable[..., BuilderReturn]


def _expect(*, condition: bool, message: str) -> None:
    """Raise an assertion error when condition is false.

    Raises
    ------
    AssertionError
        When the condition evaluates to False.
    """
    if not condition:
        raise AssertionError(message)


@pytest.mark.parametrize(
    ("builder", "field_name"),
    [
        (build_function_summary, "summary"),
        (build_function_profile, "profile"),
        (build_file_profile, "profile"),
        (build_module_profile, "profile"),
        (build_function_architecture, "architecture"),
        (build_module_architecture, "architecture"),
    ],
)
def test_object_builders_return_not_found_on_none(builder: Builder, field_name: str) -> None:
    """Ensure builders mark results as not found when rows are missing."""
    meta = dm.ResponseMeta(messages=[dm.Message(code="none", severity="info", detail=None)])
    result = builder(row=None, meta=meta)
    _expect(condition=result.found is False, message="Result should be marked as not found")
    _expect(
        condition=getattr(result, field_name) is None,
        message=f"{field_name} should be cleared",
    )
    _expect(condition=result.meta is meta, message="Metadata should be reused")


def test_build_file_summary_injects_rel_path() -> None:
    """Ensure file summaries include the requested rel_path."""
    meta = dm.ResponseMeta()
    result = build_file_summary({"module": "demo"}, rel_path="src/main.py", meta=meta)
    _expect(condition=result.found is True, message="File summary should be marked found")
    if result.file is None:
        pytest.fail("File summary payload should not be None")
    file_summary = result.file
    _expect(
        condition=file_summary["rel_path"] == "src/main.py",
        message="rel_path should be injected",
    )
    _expect(condition=result.meta is meta, message="Metadata should be reused")


def test_build_high_risk_functions_uses_meta_truncation() -> None:
    """Ensure high-risk builders respect meta truncation."""
    meta = dm.ResponseMeta(truncated=True)
    rows = [{"goid_h128": 1, "rel_path": "a.py"}]
    result = build_high_risk_functions(rows, meta=meta)
    _expect(condition=result.functions == rows, message="Function rows should be preserved")
    _expect(
        condition=result.truncated is True,
        message="Truncation flag should be copied from meta",
    )
    _expect(condition=result.meta is meta, message="Metadata should be reused")


def test_build_callgraph_neighbors_passes_metadata() -> None:
    """Ensure callgraph neighbors include outgoing, incoming, and messages."""
    meta = dm.ResponseMeta(messages=[dm.Message(code="limit", severity="warning", detail="warn")])
    outgoing = [{"caller_goid_h128": 1, "callee_goid_h128": 2}]
    incoming = [{"caller_goid_h128": 3, "callee_goid_h128": 1}]
    result = build_callgraph_neighbors(outgoing, incoming, meta=meta)
    _expect(condition=result.outgoing == outgoing, message="Outgoing edges should be preserved")
    _expect(condition=result.incoming == incoming, message="Incoming edges should be preserved")
    _expect(
        condition=result.meta.messages[0].code == "limit",
        message="Messages should be carried through",
    )


def test_build_tests_for_function_preserves_meta() -> None:
    """Ensure tests-for-function builder preserves messages."""
    meta = dm.ResponseMeta(truncated=False)
    rows = [{"urn": "urn:demo:test"}]
    result = build_tests_for_function(rows, meta=meta)
    _expect(condition=result.tests == rows, message="Test rows should be preserved")
    _expect(condition=result.meta is meta, message="Metadata should be reused")


def test_build_graph_neighborhood_retains_nodes_and_edges() -> None:
    """Ensure graph neighborhood builder keeps nodes, edges, and truncation."""
    meta = dm.ResponseMeta(truncated=True)
    nodes = [{"function_goid_h128": 1}]
    edges = [{"caller_goid_h128": 1, "callee_goid_h128": 2}]
    result = build_graph_neighborhood(nodes, edges, meta=meta)
    _expect(condition=result.nodes == nodes, message="Nodes should be preserved")
    _expect(condition=result.edges == edges, message="Edges should be preserved")
    _expect(condition=result.meta.truncated is True, message="Truncation should be preserved")


def test_build_import_boundary_constructs_node_ids() -> None:
    """Ensure import boundary builder adds id entries to nodes."""
    meta = dm.ResponseMeta()
    nodes = ["alpha", "beta"]
    edges = [{"source": "alpha", "target": "beta"}]
    result = build_import_boundary(nodes, edges, meta=meta)
    _expect(condition={"id": "alpha"} in result.nodes, message="Nodes should include ids")
    _expect(condition=result.edges == edges, message="Edges should be preserved")
    _expect(condition=result.meta is meta, message="Metadata should be reused")


def test_build_file_hints_sets_found_false_when_empty() -> None:
    """Ensure file hints mark not found when empty."""
    meta = dm.ResponseMeta()
    empty_result = build_file_hints([], rel_path="missing.py", meta=meta)
    _expect(
        condition=empty_result.found is False,
        message="Empty hints should return found=False",
    )
    _expect(condition=empty_result.hints == [], message="Empty hints should be preserved")
    rows = [{"hint": "use dataclasses"}]
    populated = build_file_hints(rows, rel_path="file.py", meta=meta)
    _expect(
        condition=populated.found is True,
        message="Hints should be marked found when present",
    )
    _expect(condition=populated.hints == rows, message="Hints should be preserved")


def test_build_subsystem_modules_handles_missing_subsystem() -> None:
    """Ensure subsystem modules handle missing subsystems gracefully."""
    meta = dm.ResponseMeta()
    missing = build_subsystem_modules(None, [], meta=meta)
    _expect(condition=missing.found is False, message="Missing subsystem should mark found=False")
    subsystem = {"id": "subsystem-1"}
    rows = [{"module": "pkg.module"}]
    present = build_subsystem_modules(subsystem, rows, meta=meta)
    if present.subsystem is None:
        pytest.fail("Subsystem should be included when present")
    subsystem_payload = present.subsystem
    _expect(
        condition=present.found is True,
        message="Subsystem should be marked found when provided",
    )
    _expect(
        condition=subsystem_payload["id"] == "subsystem-1",
        message="Subsystem id should be copied",
    )
    _expect(condition=present.modules == rows, message="Module rows should be preserved")


def test_build_module_and_subsystem_lists() -> None:
    """Ensure subsystem helper builders preserve provided rows."""
    meta = dm.ResponseMeta()
    subsystems = build_subsystem_summary([{"id": "s1"}], meta=meta)
    _expect(
        condition=subsystems.subsystems[0]["id"] == "s1",
        message="Subsystem id should be preserved",
    )
    memberships = build_module_subsystems([{"subsystem_id": "s1"}], meta=meta)
    _expect(
        condition=memberships.found is True,
        message="Memberships should be marked found",
    )
    _expect(
        condition=memberships.memberships[0]["subsystem_id"] == "s1",
        message="Membership rows should be preserved",
    )
    search = build_subsystem_search([{"name": "Search"}], meta=meta)
    _expect(
        condition=search.subsystems[0]["name"] == "Search",
        message="Search rows should be preserved",
    )
    profiles = build_subsystem_profile([{"subsystem_id": "s1"}], meta=meta)
    first_profile = assert_mapping_value({"row": profiles.profiles[0]}, "row", Mapping)
    _expect(
        condition=first_profile.get("subsystem_id") == "s1",
        message="Profile rows should be preserved",
    )
    coverage = build_subsystem_coverage([{"subsystem_id": "s1"}], meta=meta)
    first_coverage = assert_mapping_value({"row": coverage.coverage[0]}, "row", Mapping)
    _expect(
        condition=first_coverage.get("subsystem_id") == "s1",
        message="Coverage rows should be preserved",
    )


def test_build_dataset_schema_preserves_fields() -> None:
    """Ensure dataset schema builder preserves all fields."""
    meta = dm.ResponseMeta(applied_limit=3)
    schema_input = DatasetSchemaInput(
        dataset_name="demo",
        table_key="docs.demo",
        duckdb_schema=[{"name": "id", "type": "INTEGER", "nullable": False}],
        json_schema={"type": "object"},
        sample_rows=[{"id": 1}],
        capabilities={"docs_view": True},
        owner="owner",
        freshness_sla="daily",
        retention_policy="30d",
        schema_version="v1",
        stable_id="stable",
        validation_profile="strict",
        meta=meta,
    )
    result = build_dataset_schema(schema_input)
    _expect(condition=result.dataset_name == "demo", message="Dataset name should be preserved")
    _expect(
        condition=result.duckdb_schema[0]["name"] == "id",
        message="DuckDB schema should be preserved",
    )
    _expect(condition=result.sample_rows[0]["id"] == 1, message="Sample rows should be preserved")
    _expect(condition=result.meta is meta, message="Metadata should be reused")
