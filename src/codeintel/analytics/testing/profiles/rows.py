"""Row assembly and writers for test and behavioral coverage profiles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.analytics.profiles.writer_guard import (
    PolicyWriterConfig,
    write_rows_via_policy_backend,
)
from codeintel.analytics.testing.behavioral.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.analytics.testing.coverage.inputs import (
    FunctionCoverageEntry,
    SubsystemCoverageEntry,
    TestGraphMetrics,
)
from codeintel.analytics.testing.profiles.types import (
    ImportanceInputs,
    TestAstInfo,
    TestProfileContext,
    TestProfileOptions,
)
from codeintel.analytics.utilities.type_coercion import optional_int
from codeintel.core.schemas.generated_types import BehavioralCoverageRowModel, ProfileRowModel

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.analytics.testing.profiles.types import (
        FunctionCoverageEntryProtocol,
        SubsystemCoverageEntryProtocol,
        TestGraphMetricsProtocol,
        TestRecord,
    )
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class TestProfileInputs:
    """Bundled inputs for test profile context construction.

    Groups coverage and metric data needed to build test profiles.

    Parameters
    ----------
    functions_covered
        Function coverage entries keyed by test_id.
    subsystems_covered
        Subsystem coverage entries keyed by test_id.
    tg_metrics
        Test graph metrics keyed by test_id.
    ast_info
        AST-derived info keyed by test_id.
    """

    functions_covered: Mapping[str, FunctionCoverageEntryProtocol]
    subsystems_covered: Mapping[str, SubsystemCoverageEntryProtocol]
    tg_metrics: Mapping[str, TestGraphMetricsProtocol]
    ast_info: Mapping[str, TestAstInfo]


def build_test_profile_context(
    *,
    snapshot: SnapshotRef,
    inputs: TestProfileInputs,
    options: TestProfileOptions | None = None,
) -> TestProfileContext:
    """Construct the shared context required for test_profile row assembly.

    Parameters
    ----------
    snapshot
        Snapshot reference.
    inputs
        Bundled coverage and metric inputs.
    options
        Optional test profile options.

    Returns
    -------
    TestProfileContext
        Snapshot-scoped context used when building test profile rows.
    """
    opts = options or TestProfileOptions()
    max_function_count = max(
        (entry.count for entry in inputs.functions_covered.values()), default=0
    )
    max_weighted_degree = max(
        (metrics.weighted_degree or 0.0 for metrics in inputs.tg_metrics.values()), default=0.0
    )
    max_subsystem_risk = max(
        (entry.max_risk_score or 0.0 for entry in inputs.subsystems_covered.values()),
        default=0.0,
    )
    return TestProfileContext(
        snapshot=snapshot,
        options=opts,
        now=datetime.now(tz=UTC),
        max_function_count=max_function_count,
        max_weighted_degree=max_weighted_degree,
        max_subsystem_risk=max_subsystem_risk,
        functions_covered=inputs.functions_covered,
        subsystems_covered=inputs.subsystems_covered,
        tg_metrics=inputs.tg_metrics,
        ast_info=inputs.ast_info,
    )


def build_test_profile_rows(
    tests: Iterable[TestRecord],
    ctx: TestProfileContext,
) -> list[ProfileRowModel]:
    """Build test_profile row models using the current helpers.

    Returns
    -------
    list[ProfileRowModel]
        Row models ready for insertion.
    """
    return [_build_test_profile_model(test, ctx) for test in tests]


def _build_test_profile_model(test: TestRecord, ctx: TestProfileContext) -> ProfileRowModel:
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
        slow_test_threshold_ms=ctx.options.slow_test_threshold_ms,
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
    return ProfileRowModel(
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
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
    snapshot: SnapshotRef,
    rows: Iterable[ProfileRowModel],
) -> int:
    """Insert rows into analytics.test_profile via policy backend.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.
    rows
        Row models to insert.

    Returns
    -------
    int
        Number of inserted rows.
    """
    rows_list = list(rows)
    if not rows_list:
        return 0
    config = PolicyWriterConfig(
        table_key="analytics.test_profile",
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    return write_rows_via_policy_backend(gateway, rows=rows_list, config=config)


def build_behavioral_coverage_rows(
    rows: Iterable[tuple[object, ...]],
) -> list[BehavioralCoverageRowModel]:
    """Build BehavioralCoverageRowModel entries from tuples returned by the behavior helper.

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
    snapshot: SnapshotRef,
    rows: Iterable[BehavioralCoverageRowModel],
) -> int:
    """Insert rows into analytics.behavioral_coverage via policy backend.

    Parameters
    ----------
    gateway
        Storage gateway.
    snapshot
        Snapshot reference.
    rows
        Row models to insert.

    Returns
    -------
    int
        Number of inserted rows.
    """
    rows_list = list(rows)
    if not rows_list:
        return 0
    config = PolicyWriterConfig(
        table_key="analytics.behavioral_coverage",
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    return write_rows_via_policy_backend(gateway, rows=rows_list, config=config)


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
