"""Tests for serving layer domain models.

This module tests the domain models defined in domain_models.py, verifying
their construction, serialization via model_dump(), and compatibility with
the serving protocols.
"""

from __future__ import annotations

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

    assert msg.code == "WARN_TRUNCATED"
    assert msg.severity == "warning"
    assert msg.detail == "Results were truncated to limit"
    assert msg.context == {"limit": LIMIT_VALUE, "total": 100}


def test_message_with_minimal_fields() -> None:
    """Verify Message construction with minimal fields."""
    msg = Message(code="OK", severity="info")

    assert msg.code == "OK"
    assert msg.severity == "info"
    assert msg.detail is None
    assert msg.context == {}


def test_message_severity_types() -> None:
    """Verify all valid severity levels work."""
    for severity in ("info", "warning", "error"):
        msg = Message(code="TEST", severity=severity)
        assert msg.severity == severity


# =============================================================================
# ResponseMeta Tests
# =============================================================================


def test_response_meta_defaults() -> None:
    """Verify ResponseMeta default values."""
    meta = ResponseMeta()

    assert meta.requested_limit is None
    assert meta.applied_limit is None
    assert meta.requested_offset is None
    assert meta.applied_offset is None
    assert meta.truncated is False
    assert meta.messages == []


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

    assert meta.requested_limit == REQUESTED_LIMIT
    assert meta.applied_limit == APPLIED_LIMIT_50
    assert meta.truncated is True
    assert len(meta.messages) == 1


def test_response_meta_model_dump_empty() -> None:
    """Verify model_dump with default values."""
    meta = ResponseMeta()
    dumped = meta.model_dump()

    assert dumped["requested_limit"] is None
    assert dumped["applied_limit"] is None
    assert dumped["requested_offset"] is None
    assert dumped["applied_offset"] is None
    assert dumped["truncated"] is False
    assert dumped["messages"] == []


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

    assert dumped["applied_limit"] == LIMIT_VALUE
    assert dumped["truncated"] is True
    assert len(dumped["messages"]) == COUNT_TWO

    # First message fully populated
    msg1 = dumped["messages"][0]
    assert msg1["code"] == "WARN"
    assert msg1["severity"] == "warning"
    assert msg1["detail"] == "Test detail"
    assert msg1["context"] == {"key": "value"}

    # Second message minimal
    msg2 = dumped["messages"][1]
    assert msg2["code"] == "INFO"
    assert msg2["severity"] == "info"
    assert msg2["detail"] is None
    assert msg2["context"] == {}


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

    assert rows.dataset_name == "analytics.functions"
    assert rows.limit == LIMIT_VALUE
    assert rows.offset == OFFSET_VALUE
    assert len(rows.rows) == COUNT_TWO
    assert rows.meta.truncated is False


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
    assert dumped["dataset"] == "test.dataset"
    assert dumped["limit"] == LIMIT_VALUE
    assert dumped["offset"] == OFFSET_FIVE
    assert dumped["rows"] == [{"col": "val"}]

    # Nested meta is fully dumped
    assert "meta" in dumped
    assert dumped["meta"]["applied_limit"] == LIMIT_VALUE
    assert dumped["meta"]["truncated"] is True
    assert len(dumped["meta"]["messages"]) == 1


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

    assert dumped["rows"] == []
    assert dumped["meta"]["truncated"] is False


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

    assert summary.urn == "repo::pkg.mod::func"
    assert summary.goid_h128 == GOID_VALUE
    assert summary.rel_path == "pkg/mod.py"
    assert summary.qualname == "pkg.mod.func"
    assert summary.short_summary == "Short description"
    assert summary.long_summary == "Longer detailed description"
    assert summary.is_test is False


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

    assert summary.short_summary is None
    assert summary.long_summary is None
    assert summary.is_test is True


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

    assert func.goid_h128 == GOID_VALUE
    assert func.qualname == "pkg.mod.risky_func"
    assert func.rel_path == "pkg/mod.py"
    assert func.risk_score == RISK_SCORE
    assert func.is_tested is False


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

    assert len(result.functions) == COUNT_TWO
    assert result.functions[0].risk_score == RISK_SCORE_HIGH
    assert result.meta.applied_limit == COUNT_TWO


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

    assert file_summary.rel_path == "pkg/mod.py"
    assert file_summary.module == "pkg.mod"
    assert len(file_summary.functions) == 1


def test_file_summary_no_module() -> None:
    """Verify FileSummary with None module."""
    meta = ResponseMeta()
    file_summary = FileSummary(
        rel_path="script.py",
        module=None,
        functions=[],
        meta=meta,
    )

    assert file_summary.module is None
    assert file_summary.functions == []


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

    assert desc.name == "analytics.functions"
    assert desc.table == "analytics.functions"
    assert desc.description == "Function metadata"
    assert desc.family == "analytics"
    assert desc.owner == "analytics-team"
    assert desc.is_read_only is True


def test_dataset_descriptor_minimal() -> None:
    """Verify DatasetDescriptorDomain with minimal fields."""
    desc = DatasetDescriptorDomain(
        name="test.dataset",
        table="test.dataset",
        description="Test",
    )

    assert desc.family is None
    assert desc.owner is None
    assert desc.schema_version is None
    assert desc.is_docs_view is False
    assert desc.is_read_only is False


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

    assert schema.dataset_name == "analytics.functions"
    assert len(schema.duckdb_schema) == COUNT_TWO
    assert schema.capabilities["read"] is True
    assert schema.validation_profile == "strict"


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

    assert schema.json_schema is None
    assert schema.meta is None


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

    assert plan.plan_id == "plan-123"
    assert len(plan.ordered_plugins) == COUNT_THREE
    assert plan.ordered_plugins[0] == "plugin_a"
    assert len(plan.skipped_plugins) == 1
    assert "plugin_b" in plan.dep_graph


def test_graph_plan_empty() -> None:
    """Verify GraphPlan with defaults."""
    plan = GraphPlan(
        plan_id="empty-plan",
        ordered_plugins=(),
    )

    assert plan.ordered_plugins == ()
    assert plan.skipped_plugins == []
    assert plan.dep_graph == {}
    assert plan.plugin_metadata == {}


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

    assert result.found is True
    assert result.summary is not None
    assert result.summary["qualname"] == "test.func"


def test_function_summary_result_not_found() -> None:
    """Verify FunctionSummaryResult when not found."""
    meta = ResponseMeta()
    result = FunctionSummaryResult(found=False, summary=None, meta=meta)

    assert result.found is False
    assert result.summary is None


def test_high_risk_functions_result() -> None:
    """Verify HighRiskFunctionsResult construction."""
    meta = ResponseMeta()
    result = HighRiskFunctionsResult(
        functions=[{"goid": 1}, {"goid": 2}],
        truncated=True,
        meta=meta,
    )

    assert len(result.functions) == COUNT_TWO
    assert result.truncated is True


def test_call_graph_neighbors() -> None:
    """Verify CallGraphNeighbors construction."""
    meta = ResponseMeta()
    neighbors = CallGraphNeighbors(
        outgoing=[{"target": "func_a"}],
        incoming=[{"source": "func_b"}, {"source": "func_c"}],
        meta=meta,
    )

    assert len(neighbors.outgoing) == 1
    assert len(neighbors.incoming) == COUNT_TWO


def test_tests_for_function_result() -> None:
    """Verify TestsForFunctionResult construction."""
    meta = ResponseMeta()
    result = TestsForFunctionResult(
        tests=[{"test_name": "test_func1"}, {"test_name": "test_func2"}],
        meta=meta,
    )

    assert len(result.tests) == COUNT_TWO


def test_graph_neighborhood() -> None:
    """Verify GraphNeighborhood construction."""
    meta = ResponseMeta()
    neighborhood = GraphNeighborhood(
        nodes=[{"id": "n1"}, {"id": "n2"}],
        edges=[{"source": "n1", "target": "n2"}],
        meta=meta,
    )

    assert len(neighborhood.nodes) == COUNT_TWO
    assert len(neighborhood.edges) == 1


def test_import_boundary() -> None:
    """Verify ImportBoundary construction."""
    meta = ResponseMeta()
    boundary = ImportBoundary(
        nodes=[{"module": "pkg.a"}, {"module": "pkg.b"}],
        edges=[{"from": "pkg.a", "to": "pkg.b"}],
        meta=meta,
    )

    assert len(boundary.nodes) == COUNT_TWO
    assert len(boundary.edges) == 1


def test_file_summary_result() -> None:
    """Verify FileSummaryResult construction."""
    meta = ResponseMeta()
    result = FileSummaryResult(
        found=True,
        file={"rel_path": "pkg/mod.py", "module": "pkg.mod"},
        meta=meta,
    )

    assert result.found is True
    assert result.file is not None


def test_function_profile_result() -> None:
    """Verify FunctionProfileResult construction."""
    meta = ResponseMeta()
    result = FunctionProfileResult(
        found=True,
        profile={"complexity": 5, "lines": 20},
        meta=meta,
    )

    assert result.found is True
    assert result.profile is not None


def test_file_profile_result() -> None:
    """Verify FileProfileResult construction."""
    meta = ResponseMeta()
    result = FileProfileResult(
        found=False,
        profile=None,
        meta=meta,
    )

    assert result.found is False


def test_module_profile_result() -> None:
    """Verify ModuleProfileResult construction."""
    meta = ResponseMeta()
    result = ModuleProfileResult(
        found=True,
        profile={"module": "pkg.mod"},
        meta=meta,
    )

    assert result.found is True


def test_function_architecture_result() -> None:
    """Verify FunctionArchitectureResult construction."""
    meta = ResponseMeta()
    result = FunctionArchitectureResult(
        found=True,
        architecture={"layers": ["domain", "service"]},
        meta=meta,
    )

    assert result.found is True
    assert result.architecture is not None


def test_module_architecture_result() -> None:
    """Verify ModuleArchitectureResult construction."""
    meta = ResponseMeta()
    result = ModuleArchitectureResult(
        found=True,
        architecture={"subsystem": "core"},
        meta=meta,
    )

    assert result.found is True


def test_subsystem_summary_result() -> None:
    """Verify SubsystemSummaryResult construction."""
    meta = ResponseMeta()
    result = SubsystemSummaryResult(
        subsystems=[{"id": "core"}, {"id": "api"}],
        meta=meta,
    )

    assert len(result.subsystems) == COUNT_TWO


def test_module_subsystem_result() -> None:
    """Verify ModuleSubsystemResult construction."""
    meta = ResponseMeta()
    result = ModuleSubsystemResult(
        found=True,
        memberships=[{"subsystem_id": "core", "role": "member"}],
        meta=meta,
    )

    assert result.found is True
    assert len(result.memberships) == 1


def test_file_hints_result() -> None:
    """Verify FileHintsResult construction."""
    meta = ResponseMeta()
    result = FileHintsResult(
        found=True,
        hints=[{"type": "unused_import", "line": 5}],
        meta=meta,
    )

    assert result.found is True
    assert len(result.hints) == 1


def test_subsystem_modules_result() -> None:
    """Verify SubsystemModulesResult construction."""
    meta = ResponseMeta()
    result = SubsystemModulesResult(
        found=True,
        subsystem={"id": "core", "name": "Core"},
        modules=[{"module": "pkg.a"}, {"module": "pkg.b"}],
        meta=meta,
    )

    assert result.found is True
    assert result.subsystem is not None
    assert len(result.modules) == COUNT_TWO


def test_subsystem_search_result() -> None:
    """Verify SubsystemSearchResult construction."""
    meta = ResponseMeta()
    result = SubsystemSearchResult(
        subsystems=[{"id": "core", "score": 1.0}],
        meta=meta,
    )

    assert len(result.subsystems) == 1


def test_subsystem_profile_result() -> None:
    """Verify SubsystemProfileResult construction."""
    meta = ResponseMeta()
    result = SubsystemProfileResult(
        profiles=[{"metric": "coverage", "value": 0.85}],
        meta=meta,
    )

    assert len(result.profiles) == 1


def test_subsystem_coverage_result() -> None:
    """Verify SubsystemCoverageResult construction."""
    meta = ResponseMeta()
    result = SubsystemCoverageResult(
        coverage=[{"subsystem": "core", "covered": 85, "total": 100}],
        meta=meta,
    )

    assert len(result.coverage) == 1
