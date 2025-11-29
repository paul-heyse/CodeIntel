"""Row assembly and writers for test and behavioral coverage profiles."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import cast

from codeintel.analytics.profiles.utils import optional_int
from codeintel.analytics.profiles.writer_guard import (
    SerializeRow,
    WriterContext,
    write_rows_with_registry_guard,
)
from codeintel.analytics.tests_profiles.coverage_inputs import (
    FunctionCoverageEntry,
    SubsystemCoverageEntry,
    TestGraphMetrics,
)
from codeintel.analytics.tests_profiles.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.analytics.tests_profiles.types import (
    FunctionCoverageEntryProtocol,
    ImportanceInputs,
    SubsystemCoverageEntryProtocol,
    TestAstInfo,
    TestGraphMetricsProtocol,
    TestProfileContext,
    TestRecord,
)
from codeintel.config import BehavioralCoverageStepConfig, TestProfileStepConfig
from codeintel.config.schemas.registry_adapter import load_registry_columns
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.rows import (
    BEHAVIORAL_COVERAGE_COLUMNS,
    TEST_PROFILE_COLUMNS,
    BehavioralCoverageRowModel,
    TestProfileRowModel,
    behavioral_coverage_row_to_tuple,
    serialize_test_profile_row,
)
from codeintel.storage.sql_helpers import ensure_schema, prepared_statements_dynamic


def build_test_profile_context(
    *,
    cfg: TestProfileStepConfig,
    functions_covered: Mapping[str, FunctionCoverageEntryProtocol],
    subsystems_covered: Mapping[str, SubsystemCoverageEntryProtocol],
    tg_metrics: Mapping[str, TestGraphMetricsProtocol],
    ast_info: Mapping[str, TestAstInfo],
) -> TestProfileContext:
    """
    Construct the shared context required for test_profile row assembly.

    Returns
    -------
    TestProfileContext
        Snapshot-scoped context used when building test profile rows.
    """
    max_function_count = max((entry.count for entry in functions_covered.values()), default=0)
    max_weighted_degree = max(
        (metrics.weighted_degree or 0.0 for metrics in tg_metrics.values()), default=0.0
    )
    max_subsystem_risk = max(
        (entry.max_risk_score or 0.0 for entry in subsystems_covered.values()),
        default=0.0,
    )
    return TestProfileContext(
        cfg=cfg,
        now=datetime.now(tz=UTC),
        max_function_count=max_function_count,
        max_weighted_degree=max_weighted_degree,
        max_subsystem_risk=max_subsystem_risk,
        functions_covered=functions_covered,
        subsystems_covered=subsystems_covered,
        tg_metrics=tg_metrics,
        ast_info=ast_info,
    )


def build_test_profile_rows(
    tests: Iterable[TestRecord],
    ctx: TestProfileContext,
) -> list[TestProfileRowModel]:
    """
    Build test_profile row models using the current helpers.

    Returns
    -------
    list[TestProfileRowModel]
        Row models ready for insertion.
    """
    return [_build_test_profile_model(test, ctx) for test in tests]


def _build_test_profile_model(test: TestRecord, ctx: TestProfileContext) -> TestProfileRowModel:
    markers = _normalize_markers(test.markers)
    ast_details = ctx.ast_info.get(test.test_id, TestAstInfo())
    cov_entry = ctx.functions_covered.get(
        test.test_id,
        FunctionCoverageEntry(functions=[], count=0, primary=[]),
    )
    subs_entry = ctx.subsystems_covered.get(
        test.test_id,
        SubsystemCoverageEntry(
            subsystems=[],
            count=0,
            primary_subsystem_id=None,
            max_risk_score=0.0,
        ),
    )
    tg_entry = ctx.tg_metrics.get(
        test.test_id,
        TestGraphMetrics(
            degree=None,
            weighted_degree=None,
            proj_degree=None,
            proj_weight=None,
            proj_clustering=None,
            proj_betweenness=None,
        ),
    )
    uses_parametrize = _uses_parametrize(test, markers)
    uses_fixtures = ast_details.uses_fixtures or _markers_use_fixtures(markers)
    flakiness = compute_flakiness_score(
        status=test.status,
        markers=markers,
        duration_ms=test.duration_ms,
        io_flags=ast_details.io_flags,
        slow_test_threshold_ms=ctx.cfg.slow_test_threshold_ms,
    )
    importance_inputs = ImportanceInputs(
        functions_covered_count=cov_entry.count,
        weighted_degree=tg_entry.weighted_degree,
        max_function_count=ctx.max_function_count,
        max_weighted_degree=ctx.max_weighted_degree,
        subsystem_risk=subs_entry.max_risk_score,
        max_subsystem_risk=ctx.max_subsystem_risk,
    )
    importance = compute_importance_score(importance_inputs)
    now = ctx.now
    return TestProfileRowModel(
        repo=ctx.cfg.repo,
        commit=ctx.cfg.commit,
        test_id=test.test_id,
        test_goid_h128=test.test_goid_h128,
        urn=test.urn,
        rel_path=test.rel_path,
        module=test.module,
        qualname=test.qualname,
        language=test.language or "python",
        kind=test.kind,
        status=test.status,
        duration_ms=test.duration_ms,
        markers=markers,
        flaky=test.flaky,
        last_run_at=now,
        functions_covered=list(cov_entry.functions),
        functions_covered_count=cov_entry.count,
        primary_function_goids=list(cov_entry.primary),
        subsystems_covered=list(subs_entry.subsystems),
        subsystems_covered_count=subs_entry.count,
        primary_subsystem_id=subs_entry.primary_subsystem_id,
        assert_count=ast_details.assert_count,
        raise_count=ast_details.raise_count,
        uses_parametrize=uses_parametrize,
        uses_fixtures=uses_fixtures,
        io_bound=ast_details.io_flags.io_bound,
        uses_network=ast_details.io_flags.uses_network,
        uses_db=ast_details.io_flags.uses_db,
        uses_filesystem=ast_details.io_flags.uses_filesystem,
        uses_subprocess=ast_details.io_flags.uses_subprocess,
        flakiness_score=flakiness,
        importance_score=importance,
        notes=None,
        tg_degree=tg_entry.degree,
        tg_weighted_degree=tg_entry.weighted_degree,
        tg_proj_degree=tg_entry.proj_degree,
        tg_proj_weight=tg_entry.proj_weight,
        tg_proj_clustering=tg_entry.proj_clustering,
        tg_proj_betweenness=tg_entry.proj_betweenness,
        created_at=now,
    )


def write_test_profile_rows(
    gateway: StorageGateway,
    cfg: TestProfileStepConfig,
    rows: Iterable[TestProfileRowModel],
) -> int:
    """
    Insert rows into analytics.test_profile with registry alignment checks.

    Returns
    -------
    int
        Number of inserted rows.
    """
    rows_list = list(rows)
    return write_rows_with_registry_guard(
        gateway.con,
        rows=rows_list,
        context=WriterContext(
            table_key="analytics.test_profile",
            columns=TEST_PROFILE_COLUMNS,
            serialize_row=cast("SerializeRow", serialize_test_profile_row),
            repo=cfg.repo,
            commit=cfg.commit,
            delete_sql="DELETE FROM analytics.test_profile WHERE repo = ? AND commit = ?",
            ensure_schema_fn=ensure_schema,
            load_registry_columns_fn=load_registry_columns,
            prepared_statements_fn=prepared_statements_dynamic,
        ),
    )


def build_behavioral_coverage_rows(
    rows: Iterable[tuple[object, ...]],
) -> list[BehavioralCoverageRowModel]:
    """
    Build BehavioralCoverageRowModel entries from tuples returned by the behavior helper.

    Returns
    -------
    list[BehavioralCoverageRowModel]
        Row models for behavioral coverage.
    """
    models: list[BehavioralCoverageRowModel] = []
    for row in rows:
        (
            repo,
            commit,
            test_id,
            test_goid_h128,
            rel_path,
            qualname,
            behavior_tags,
            tag_source,
            heuristic_version,
            llm_model,
            llm_run_id,
            created_at,
        ) = row
        created_at_value = created_at if isinstance(created_at, datetime) else datetime.now(tz=UTC)
        models.append(
            BehavioralCoverageRowModel(
                repo=str(repo),
                commit=str(commit),
                test_id=str(test_id),
                test_goid_h128=optional_int(test_goid_h128),
                rel_path=str(rel_path),
                qualname=str(qualname) if qualname is not None else None,
                behavior_tags=behavior_tags,
                tag_source=str(tag_source),
                heuristic_version=str(heuristic_version) if heuristic_version is not None else None,
                llm_model=str(llm_model) if llm_model is not None else None,
                llm_run_id=str(llm_run_id) if llm_run_id is not None else None,
                created_at=created_at_value,
            )
        )
    return models


def write_behavioral_coverage_rows(
    gateway: StorageGateway,
    cfg: BehavioralCoverageStepConfig,
    rows: Iterable[BehavioralCoverageRowModel],
) -> int:
    """
    Insert rows into analytics.behavioral_coverage with registry alignment checks.

    Returns
    -------
    int
        Number of inserted rows.
    """
    rows_list = list(rows)
    return write_rows_with_registry_guard(
        gateway.con,
        rows=rows_list,
        context=WriterContext(
            table_key="analytics.behavioral_coverage",
            columns=BEHAVIORAL_COVERAGE_COLUMNS,
            serialize_row=cast("SerializeRow", behavioral_coverage_row_to_tuple),
            repo=cfg.repo,
            commit=cfg.commit,
            delete_sql=("DELETE FROM analytics.behavioral_coverage WHERE repo = ? AND commit = ?"),
            ensure_schema_fn=ensure_schema,
            load_registry_columns_fn=load_registry_columns,
            prepared_statements_fn=prepared_statements_dynamic,
        ),
    )


def _normalize_markers(markers: list[str] | None) -> list[str]:
    if markers is None:
        return []
    return [str(marker) for marker in markers]


def _uses_parametrize(test: TestRecord, markers: Iterable[str]) -> bool:
    markers_lower = [marker.lower() for marker in markers]
    if test.kind == "parametrized_case":
        return True
    if any("parametrize" in marker for marker in markers_lower):
        return True
    qual = test.qualname or ""
    return "[" in qual and "]" in qual


def _markers_use_fixtures(markers: Iterable[str]) -> bool:
    return any("usefixtures" in marker.lower() for marker in markers)
