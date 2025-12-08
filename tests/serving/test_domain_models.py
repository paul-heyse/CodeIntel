"""Tests for serving layer domain models.

This module tests the domain models defined in domain_models.py, verifying
their construction, serialization via model_dump(), and compatibility with
the serving protocols.
"""

from __future__ import annotations

import pytest

from codeintel.serving.domain_models import (
    CallGraphNeighbors,
    DatasetDescriptorDomain,
    DatasetRows,
    DatasetSchema,
    FileHintsResult,
    FileProfileResult,
    FileSummary,
    FileSummaryResult,
    FunctionArchitectureResult,
    FunctionProfileResult,
    FunctionSummary,
    FunctionSummaryResult,
    GraphNeighborhood,
    GraphPlan,
    HighRiskFunction,
    HighRiskFunctions,
    HighRiskFunctionsResult,
    ImportBoundary,
    Message,
    ModuleArchitectureResult,
    ModuleProfileResult,
    ModuleSubsystemResult,
    ResponseMeta,
    SubsystemCoverageResult,
    SubsystemModulesResult,
    SubsystemProfileResult,
    SubsystemSearchResult,
    SubsystemSummaryResult,
    TestsForFunctionResult,
)
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_true,
)

# Constants for tests
LIMIT_VALUE = 10
OFFSET_VALUE = 0
GOID_VALUE = 12345
RISK_SCORE = 0.85
REQUESTED_LIMIT = 100
APPLIED_LIMIT_50 = 50
COUNT_TWO = 2
COUNT_THREE = 3
OFFSET_FIVE = 5
RISK_SCORE_HIGH = 0.9


# =============================================================================
# Message Tests
# =============================================================================


def test_message_with_all_fields() -> None:
    """Verify Message construction with all fields."""
    msg = Message(
        code="WARN_TRUNCATED",
        severity="warning",
        detail="Results were truncated to limit",
        context={"limit": LIMIT_VALUE, "total": 100},
    )

    expect_equal(msg.code, "WARN_TRUNCATED")
    expect_equal(msg.severity, "warning")
    expect_equal(msg.detail, "Results were truncated to limit")
    expect_equal(msg.context, {"limit": LIMIT_VALUE, "total": 100})


def test_message_with_minimal_fields() -> None:
    """Verify Message construction with minimal fields."""
    msg = Message(code="OK", severity="info")

    expect_equal(msg.code, "OK")
    expect_equal(msg.severity, "info")
    expect_true(msg.detail is None)
    expect_equal(msg.context, {})


def test_message_severity_types() -> None:
    """Verify all valid severity levels work."""
    for severity in ("info", "warning", "error"):
        msg = Message(code="TEST", severity=severity)
        expect_equal(msg.severity, severity)


# =============================================================================
# ResponseMeta Tests
# =============================================================================


def test_response_meta_defaults() -> None:
    """Verify ResponseMeta default values."""
    meta = ResponseMeta()

    expect_true(meta.requested_limit is None)
    expect_true(meta.applied_limit is None)
    expect_true(meta.requested_offset is None)
    expect_true(meta.applied_offset is None)
    expect_false(meta.truncated)
    expect_equal(meta.messages, [])


def test_response_meta_with_all_fields() -> None:
    """Verify ResponseMeta with all fields populated."""
    meta = ResponseMeta(
        requested_limit=REQUESTED_LIMIT,
        applied_limit=APPLIED_LIMIT_50,
        requested_offset=0,
        applied_offset=0,
        truncated=True,
        messages=[Message(code="TRUNCATED", severity="info")],
    )

    expect_equal(meta.requested_limit, REQUESTED_LIMIT)
    expect_equal(meta.applied_limit, APPLIED_LIMIT_50)
    expect_true(meta.truncated)
    expect_length(meta.messages, 1)


def test_response_meta_model_dump_empty() -> None:
    """Verify model_dump with default values."""
    meta = ResponseMeta()
    dumped = meta.model_dump()

    expect_true(dumped["requested_limit"] is None)
    expect_true(dumped["applied_limit"] is None)
    expect_true(dumped["requested_offset"] is None)
    expect_true(dumped["applied_offset"] is None)
    expect_false(dumped["truncated"])
    expect_equal(dumped["messages"], [])


def test_response_meta_model_dump_with_messages() -> None:
    """Verify model_dump includes message details."""
    meta = ResponseMeta(
        applied_limit=LIMIT_VALUE,
        truncated=True,
        messages=[
            Message(
                code="WARN",
                severity="warning",
                detail="Test detail",
                context={"key": "value"},
            ),
            Message(code="INFO", severity="info"),
        ],
    )

    dumped = meta.model_dump()

    expect_equal(dumped["applied_limit"], LIMIT_VALUE)
    expect_true(dumped["truncated"])
    expect_length(dumped["messages"], COUNT_TWO)

    # First message fully populated
    msg1 = dumped["messages"][0]
    expect_equal(msg1["code"], "WARN")
    expect_equal(msg1["severity"], "warning")
    expect_equal(msg1["detail"], "Test detail")
    expect_equal(msg1["context"], {"key": "value"})

    # Second message minimal
    msg2 = dumped["messages"][1]
    expect_equal(msg2["code"], "INFO")
    expect_equal(msg2["severity"], "info")
    expect_true(msg2["detail"] is None)
    expect_equal(msg2["context"], {})


# =============================================================================
# DatasetRows Tests
# =============================================================================


def test_dataset_rows_construction() -> None:
    """Verify DatasetRows construction."""
    meta = ResponseMeta(applied_limit=LIMIT_VALUE, truncated=False)
    rows = DatasetRows(
        dataset_name="analytics.functions",
        limit=LIMIT_VALUE,
        offset=OFFSET_VALUE,
        rows=[{"id": 1, "name": "func1"}, {"id": 2, "name": "func2"}],
        meta=meta,
    )

    expect_equal(rows.dataset_name, "analytics.functions")
    expect_equal(rows.limit, LIMIT_VALUE)
    expect_equal(rows.offset, OFFSET_VALUE)
    expect_length(rows.rows, COUNT_TWO)
    expect_false(rows.meta.truncated)


def test_dataset_rows_model_dump() -> None:
    """Verify DatasetRows model_dump includes nested meta."""
    meta = ResponseMeta(
        applied_limit=LIMIT_VALUE, truncated=True, messages=[Message(code="T", severity="info")]
    )
    rows = DatasetRows(
        dataset_name="test.dataset",
        limit=LIMIT_VALUE,
        offset=OFFSET_FIVE,
        rows=[{"col": "val"}],
        meta=meta,
    )

    dumped = rows.model_dump()

    # Top-level fields use expected keys
    expect_equal(dumped["dataset"], "test.dataset")
    expect_equal(dumped["limit"], LIMIT_VALUE)
    expect_equal(dumped["offset"], OFFSET_FIVE)
    expect_equal(dumped["rows"], [{"col": "val"}])

    # Nested meta is fully dumped
    expect_in("meta", dumped)
    expect_equal(dumped["meta"]["applied_limit"], LIMIT_VALUE)
    expect_true(dumped["meta"]["truncated"])
    expect_length(dumped["meta"]["messages"], 1)


def test_dataset_rows_empty() -> None:
    """Verify DatasetRows with empty rows list."""
    meta = ResponseMeta()
    rows = DatasetRows(
        dataset_name="empty.dataset",
        limit=LIMIT_VALUE,
        offset=OFFSET_VALUE,
        rows=[],
        meta=meta,
    )

    dumped = rows.model_dump()

    expect_equal(dumped["rows"], [])
    expect_false(dumped["meta"]["truncated"])


# =============================================================================
# FunctionSummary Tests
# =============================================================================


def test_function_summary_construction() -> None:
    """Verify FunctionSummary construction."""
    meta = ResponseMeta(applied_limit=1)
    summary = FunctionSummary(
        urn="repo::pkg.mod::func",
        goid_h128=GOID_VALUE,
        rel_path="pkg/mod.py",
        qualname="pkg.mod.func",
        short_summary="Short description",
        long_summary="Longer detailed description",
        is_test=False,
        meta=meta,
    )

    expect_equal(summary.urn, "repo::pkg.mod::func")
    expect_equal(summary.goid_h128, GOID_VALUE)
    expect_equal(summary.rel_path, "pkg/mod.py")
    expect_equal(summary.qualname, "pkg.mod.func")
    expect_equal(summary.short_summary, "Short description")
    expect_equal(summary.long_summary, "Longer detailed description")
    expect_false(summary.is_test)


def test_function_summary_nullable_summaries() -> None:
    """Verify FunctionSummary with None summaries."""
    meta = ResponseMeta()
    summary = FunctionSummary(
        urn="repo::pkg.mod::func2",
        goid_h128=GOID_VALUE,
        rel_path="pkg/mod.py",
        qualname="pkg.mod.func2",
        short_summary=None,
        long_summary=None,
        is_test=True,
        meta=meta,
    )

    expect_is_none(summary.short_summary)
    expect_is_none(summary.long_summary)
    expect_true(summary.is_test)


# =============================================================================
# HighRiskFunction Tests
# =============================================================================


def test_high_risk_function_construction() -> None:
    """Verify HighRiskFunction construction."""
    func = HighRiskFunction(
        goid_h128=GOID_VALUE,
        qualname="pkg.mod.risky_func",
        rel_path="pkg/mod.py",
        risk_score=RISK_SCORE,
        is_tested=False,
    )

    expect_equal(func.goid_h128, GOID_VALUE)
    expect_equal(func.qualname, "pkg.mod.risky_func")
    expect_equal(func.rel_path, "pkg/mod.py")
    expect_equal(func.risk_score, RISK_SCORE)
    expect_false(func.is_tested)


def test_high_risk_functions_collection() -> None:
    """Verify HighRiskFunctions collection construction."""
    funcs = [
        HighRiskFunction(
            goid_h128=1, qualname="f1", rel_path="p1.py", risk_score=RISK_SCORE_HIGH, is_tested=True
        ),
        HighRiskFunction(
            goid_h128=2, qualname="f2", rel_path="p2.py", risk_score=0.8, is_tested=False
        ),
    ]
    meta = ResponseMeta(applied_limit=COUNT_TWO, truncated=False)
    result = HighRiskFunctions(functions=funcs, meta=meta)

    expect_length(result.functions, COUNT_TWO)
    expect_equal(result.functions[0].risk_score, RISK_SCORE_HIGH)
    expect_equal(result.meta.applied_limit, COUNT_TWO)


# =============================================================================
# FileSummary Tests
# =============================================================================


def test_file_summary_construction() -> None:
    """Verify FileSummary construction."""
    meta = ResponseMeta()
    func_meta = ResponseMeta()
    func_summary = FunctionSummary(
        urn="repo::pkg.mod::func",
        goid_h128=GOID_VALUE,
        rel_path="pkg/mod.py",
        qualname="pkg.mod.func",
        short_summary="Test",
        long_summary=None,
        is_test=False,
        meta=func_meta,
    )
    file_summary = FileSummary(
        rel_path="pkg/mod.py",
        module="pkg.mod",
        functions=[func_summary],
        meta=meta,
    )

    expect_equal(file_summary.rel_path, "pkg/mod.py")
    expect_equal(file_summary.module, "pkg.mod")
    expect_length(file_summary.functions, 1)


def test_file_summary_no_module() -> None:
    """Verify FileSummary with None module."""
    meta = ResponseMeta()
    file_summary = FileSummary(
        rel_path="script.py",
        module=None,
        functions=[],
        meta=meta,
    )

    expect_is_none(file_summary.module)
    expect_equal(file_summary.functions, [])


# =============================================================================
# DatasetDescriptorDomain Tests
# =============================================================================


def test_dataset_descriptor_full() -> None:
    """Verify DatasetDescriptorDomain with all fields."""
    desc = DatasetDescriptorDomain(
        name="analytics.functions",
        table="analytics.functions",
        description="Function metadata",
        family="analytics",
        owner="analytics-team",
        schema_version="1.0.0",
        stable_id="functions-v1",
        is_docs_view=False,
        is_read_only=True,
    )

    expect_equal(desc.name, "analytics.functions")
    expect_equal(desc.table, "analytics.functions")
    expect_equal(desc.description, "Function metadata")
    expect_equal(desc.family, "analytics")
    expect_equal(desc.owner, "analytics-team")
    expect_true(desc.is_read_only)


def test_dataset_descriptor_minimal() -> None:
    """Verify DatasetDescriptorDomain with minimal fields."""
    desc = DatasetDescriptorDomain(
        name="test.dataset",
        table="test.dataset",
        description="Test",
    )

    expect_is_none(desc.family)
    expect_is_none(desc.owner)
    expect_is_none(desc.schema_version)
    expect_false(desc.is_docs_view)
    expect_false(desc.is_read_only)


# =============================================================================
# DatasetSchema Tests
# =============================================================================


def test_dataset_schema_construction() -> None:
    """Verify DatasetSchema construction."""
    schema = DatasetSchema(
        dataset_name="analytics.functions",
        table_key="analytics.functions",
        duckdb_schema=[
            {"name": "id", "type": "INTEGER"},
            {"name": "name", "type": "VARCHAR"},
        ],
        json_schema={"type": "object"},
        sample_rows=[{"id": 1, "name": "test"}],
        capabilities={"read": True, "write": False},
        owner="team",
        freshness_sla="1h",
        retention_policy="30d",
        schema_version="1.0",
        stable_id="funcs-v1",
        validation_profile="strict",
        meta=ResponseMeta(),
    )

    expect_equal(schema.dataset_name, "analytics.functions")
    expect_length(schema.duckdb_schema, COUNT_TWO)
    expect_true(schema.capabilities["read"])
    expect_equal(schema.validation_profile, "strict")


def test_dataset_schema_minimal() -> None:
    """Verify DatasetSchema with minimal/None fields."""
    schema = DatasetSchema(
        dataset_name="test",
        table_key="test",
        duckdb_schema=[],
        json_schema=None,
        sample_rows=[],
        capabilities={},
        owner=None,
        freshness_sla=None,
        retention_policy=None,
        schema_version=None,
        stable_id=None,
        validation_profile=None,
    )

    expect_is_none(schema.json_schema)
    expect_is_none(schema.meta)


# =============================================================================
# GraphPlan Tests
# =============================================================================


def test_graph_plan_construction() -> None:
    """Verify GraphPlan construction."""
    plan = GraphPlan(
        plan_id="plan-123",
        ordered_plugins=("plugin_a", "plugin_b", "plugin_c"),
        skipped_plugins=[{"name": "plugin_d", "reason": "disabled"}],
        dep_graph={"plugin_b": ("plugin_a",), "plugin_c": ("plugin_b",)},
        plugin_metadata={"plugin_a": {"version": "1.0"}},
    )

    expect_equal(plan.plan_id, "plan-123")
    expect_length(plan.ordered_plugins, COUNT_THREE)
    expect_equal(plan.ordered_plugins[0], "plugin_a")
    expect_length(plan.skipped_plugins, 1)
    expect_in("plugin_b", plan.dep_graph)


def test_graph_plan_empty() -> None:
    """Verify GraphPlan with defaults."""
    plan = GraphPlan(
        plan_id="empty-plan",
        ordered_plugins=(),
    )

    expect_equal(plan.ordered_plugins, ())
    expect_equal(plan.skipped_plugins, [])
    expect_equal(plan.dep_graph, {})
    expect_equal(plan.plugin_metadata, {})


# =============================================================================
# Result Dataclasses Tests
# =============================================================================


def test_function_summary_result() -> None:
    """Verify FunctionSummaryResult construction."""
    meta = ResponseMeta(applied_limit=1)
    result = FunctionSummaryResult(
        found=True,
        summary={"goid_h128": GOID_VALUE, "qualname": "test.func"},
        meta=meta,
    )

    expect_true(result.found)
    if result.summary is None:
        pytest.fail("Expected summary to be present")
    expect_equal(result.summary["qualname"], "test.func")


def test_function_summary_result_not_found() -> None:
    """Verify FunctionSummaryResult when not found."""
    meta = ResponseMeta()
    result = FunctionSummaryResult(found=False, summary=None, meta=meta)

    expect_false(result.found)
    expect_is_none(result.summary)


def test_high_risk_functions_result() -> None:
    """Verify HighRiskFunctionsResult construction."""
    meta = ResponseMeta()
    result = HighRiskFunctionsResult(
        functions=[{"goid": 1}, {"goid": 2}],
        truncated=True,
        meta=meta,
    )

    expect_length(result.functions, COUNT_TWO)
    expect_true(result.truncated)


def test_call_graph_neighbors() -> None:
    """Verify CallGraphNeighbors construction."""
    meta = ResponseMeta()
    neighbors = CallGraphNeighbors(
        outgoing=[{"target": "func_a"}],
        incoming=[{"source": "func_b"}, {"source": "func_c"}],
        meta=meta,
    )

    expect_length(neighbors.outgoing, 1)
    expect_length(neighbors.incoming, COUNT_TWO)


def test_tests_for_function_result() -> None:
    """Verify TestsForFunctionResult construction."""
    meta = ResponseMeta()
    result = TestsForFunctionResult(
        tests=[{"test_name": "test_func1"}, {"test_name": "test_func2"}],
        meta=meta,
    )

    expect_length(result.tests, COUNT_TWO)


def test_graph_neighborhood() -> None:
    """Verify GraphNeighborhood construction."""
    meta = ResponseMeta()
    neighborhood = GraphNeighborhood(
        nodes=[{"id": "n1"}, {"id": "n2"}],
        edges=[{"source": "n1", "target": "n2"}],
        meta=meta,
    )

    expect_length(neighborhood.nodes, COUNT_TWO)
    expect_length(neighborhood.edges, 1)


def test_import_boundary() -> None:
    """Verify ImportBoundary construction."""
    meta = ResponseMeta()
    boundary = ImportBoundary(
        nodes=[{"module": "pkg.a"}, {"module": "pkg.b"}],
        edges=[{"from": "pkg.a", "to": "pkg.b"}],
        meta=meta,
    )

    expect_length(boundary.nodes, COUNT_TWO)
    expect_length(boundary.edges, 1)


def test_file_summary_result() -> None:
    """Verify FileSummaryResult construction."""
    meta = ResponseMeta()
    result = FileSummaryResult(
        found=True,
        file={"rel_path": "pkg/mod.py", "module": "pkg.mod"},
        meta=meta,
    )

    expect_true(result.found)
    expect_is_not_none(result.file)


def test_function_profile_result() -> None:
    """Verify FunctionProfileResult construction."""
    meta = ResponseMeta()
    result = FunctionProfileResult(
        found=True,
        profile={"complexity": 5, "lines": 20},
        meta=meta,
    )

    expect_true(result.found)
    expect_is_not_none(result.profile)


def test_file_profile_result() -> None:
    """Verify FileProfileResult construction."""
    meta = ResponseMeta()
    result = FileProfileResult(
        found=False,
        profile=None,
        meta=meta,
    )

    expect_false(result.found)


def test_module_profile_result() -> None:
    """Verify ModuleProfileResult construction."""
    meta = ResponseMeta()
    result = ModuleProfileResult(
        found=True,
        profile={"module": "pkg.mod"},
        meta=meta,
    )

    expect_true(result.found)


def test_function_architecture_result() -> None:
    """Verify FunctionArchitectureResult construction."""
    meta = ResponseMeta()
    result = FunctionArchitectureResult(
        found=True,
        architecture={"layers": ["domain", "service"]},
        meta=meta,
    )

    expect_true(result.found)
    expect_is_not_none(result.architecture)


def test_module_architecture_result() -> None:
    """Verify ModuleArchitectureResult construction."""
    meta = ResponseMeta()
    result = ModuleArchitectureResult(
        found=True,
        architecture={"subsystem": "core"},
        meta=meta,
    )

    expect_true(result.found)


def test_subsystem_summary_result() -> None:
    """Verify SubsystemSummaryResult construction."""
    meta = ResponseMeta()
    result = SubsystemSummaryResult(
        subsystems=[{"id": "core"}, {"id": "api"}],
        meta=meta,
    )

    expect_length(result.subsystems, COUNT_TWO)


def test_module_subsystem_result() -> None:
    """Verify ModuleSubsystemResult construction."""
    meta = ResponseMeta()
    result = ModuleSubsystemResult(
        found=True,
        memberships=[{"subsystem_id": "core", "role": "member"}],
        meta=meta,
    )

    expect_true(result.found)
    expect_length(result.memberships, 1)


def test_file_hints_result() -> None:
    """Verify FileHintsResult construction."""
    meta = ResponseMeta()
    result = FileHintsResult(
        found=True,
        hints=[{"type": "unused_import", "line": 5}],
        meta=meta,
    )

    expect_true(result.found)
    expect_length(result.hints, 1)


def test_subsystem_modules_result() -> None:
    """Verify SubsystemModulesResult construction."""
    meta = ResponseMeta()
    result = SubsystemModulesResult(
        found=True,
        subsystem={"id": "core", "name": "Core"},
        modules=[{"module": "pkg.a"}, {"module": "pkg.b"}],
        meta=meta,
    )

    expect_true(result.found)
    expect_is_not_none(result.subsystem)
    expect_length(result.modules, COUNT_TWO)


def test_subsystem_search_result() -> None:
    """Verify SubsystemSearchResult construction."""
    meta = ResponseMeta()
    result = SubsystemSearchResult(
        subsystems=[{"id": "core", "score": 1.0}],
        meta=meta,
    )

    expect_length(result.subsystems, 1)


def test_subsystem_profile_result() -> None:
    """Verify SubsystemProfileResult construction."""
    meta = ResponseMeta()
    result = SubsystemProfileResult(
        profiles=[{"metric": "coverage", "value": 0.85}],
        meta=meta,
    )

    expect_length(result.profiles, 1)


def test_subsystem_coverage_result() -> None:
    """Verify SubsystemCoverageResult construction."""
    meta = ResponseMeta()
    result = SubsystemCoverageResult(
        coverage=[{"subsystem": "core", "covered": 85, "total": 100}],
        meta=meta,
    )

    expect_length(result.coverage, 1)
