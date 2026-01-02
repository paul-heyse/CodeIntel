"""Row assembly helpers for test profiles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.analytics.testing.behavioral.importance import (
    compute_flakiness_score,
    compute_importance_score,
)
from codeintel.build.analytics.testing.profiles.types import (
    FunctionCoverageEntry,
    ImportanceInputs,
    SubsystemCoverageEntry,
    TestAstInfo,
    TestGraphMetrics,
    TestProfileContext,
    TestProfileOptions,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsTestProfileRow as ProfileRowModel,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from codeintel.build.analytics.testing.profiles.types import (
        FunctionCoverageEntryProtocol,
        SubsystemCoverageEntryProtocol,
        TestGraphMetricsProtocol,
        TestRecord,
    )
    from codeintel.config.primitives import SnapshotRef


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
