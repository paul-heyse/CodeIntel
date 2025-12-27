"""Native Hamilton implementation for subsystem cache targets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.analytics.subsystems.cache import (
    build_subsystem_coverage_cache_rows,
    build_subsystem_profile_cache_rows,
)
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.patterns.savers import (
    SaverContext,
    TableSaveSpec,
    save_rows,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
)
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.core.schemas.row_serialization import row_to_tuple
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsSubsystemCoverageCacheRow as SubsystemCoverageCacheRow,
    )
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsSubsystemProfileCacheRow as SubsystemProfileCacheRow,
    )

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord)

SUBSYSTEM_CACHES_TARGET_NAME = "subsystem_caches"

SUBSYSTEM_PROFILE_CACHE_TABLE_KEY = "analytics.subsystem_profile_cache"
SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY = "analytics.subsystem_coverage_cache"
SUBSYSTEM_CACHE_TABLE_KEYS = (
    SUBSYSTEM_PROFILE_CACHE_TABLE_KEY,
    SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY,
)
SUBSYSTEM_CACHES_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=SUBSYSTEM_CACHES_TARGET_NAME,
)


@tag_helper(domain="analytics")
def gateway(env: BuildEnv) -> StorageGateway:
    """Expose the storage gateway for subsystem cache nodes.

    Returns
    -------
    StorageGateway
        Storage gateway for the current build environment.
    """
    return env.gateway






@dataclass(frozen=True)
class SubsystemCachesComputeResult:
    """Result from subsystem cache computation."""

    profile_rows: list[SubsystemProfileCacheRow] | None
    coverage_rows: list[SubsystemCoverageCacheRow] | None
    error: str | None = None


@dataclass(frozen=True)
class SubsystemCacheInputs:
    """Bundled inputs for subsystem cache computation."""

    gateway: StorageGateway
    subsystems: TargetRunRecord
    subsystem_graph_metrics: TargetRunRecord
    test_profile: TargetRunRecord


@tag_helper(domain="analytics")
def subsystem_caches__inputs(
    gateway: StorageGateway,
    t__subsystems: TargetRunRecord,
    t__subsystem_graph_metrics: TargetRunRecord,
    t__test_profile: TargetRunRecord,
) -> SubsystemCacheInputs:
    """Bundle subsystem cache inputs for reuse.

    Returns
    -------
    SubsystemCacheInputs
        Bundled inputs for subsystem cache computation.
    """
    return SubsystemCacheInputs(
        gateway=gateway,
        subsystems=t__subsystems,
        subsystem_graph_metrics=t__subsystem_graph_metrics,
        test_profile=t__test_profile,
    )


@tag_compute(domain="analytics", target=SUBSYSTEM_CACHES_TARGET_NAME)
def t__subsystem_caches__compute(
    env: BuildEnv,
    subsystem_caches__inputs: SubsystemCacheInputs,
) -> SubsystemCachesComputeResult | None:
    """Compute subsystem cache rows from base subsystem tables.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    subsystem_caches__inputs
        Bundled inputs including gateway and upstream target results.
        Skip flag derived from manifest-based input hash evaluation.

    Returns
    -------
    SubsystemCachesComputeResult | None
        Computed cache rows or None when skipped.
    """
    gateway = subsystem_caches__inputs.gateway

    if subsystem_caches__inputs.subsystems.status != "succeeded":
        return SubsystemCachesComputeResult(
            profile_rows=None,
            coverage_rows=None,
            error=(
                f"Upstream subsystems target failed: {subsystem_caches__inputs.subsystems.error}"
            ),
        )

    if subsystem_caches__inputs.subsystem_graph_metrics.status != "succeeded":
        return SubsystemCachesComputeResult(
            profile_rows=None,
            coverage_rows=None,
            error=(
                "Upstream subsystem_graph_metrics target failed: "
                f"{subsystem_caches__inputs.subsystem_graph_metrics.error}"
            ),
        )

    if subsystem_caches__inputs.test_profile.status != "succeeded":
        return SubsystemCachesComputeResult(
            profile_rows=None,
            coverage_rows=None,
            error=(
                "Upstream test_profile target failed: "
                f"{subsystem_caches__inputs.test_profile.error}"
            ),
        )

    try:
        profile_rows = build_subsystem_profile_cache_rows(gateway, env.snapshot)
        coverage_rows = build_subsystem_coverage_cache_rows(gateway, env.snapshot)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        log.exception("subsystem_caches: build cache rows failed")
        return SubsystemCachesComputeResult(
            profile_rows=None,
            coverage_rows=None,
            error=str(exc),
        )

    return SubsystemCachesComputeResult(
        profile_rows=profile_rows,
        coverage_rows=coverage_rows,
    )


@save_rows(
    context=SUBSYSTEM_CACHES_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SUBSYSTEM_PROFILE_CACHE_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=SUBSYSTEM_CACHES_TARGET_NAME,
    target_="subsystem_profile_cache__rows",
)
def subsystem_profile_cache__rows(
    t__subsystem_caches__compute: SubsystemCachesComputeResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.subsystem_profile_cache table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if (
        t__subsystem_caches__compute is None
        or t__subsystem_caches__compute.error
        or t__subsystem_caches__compute.profile_rows is None
    ):
        return None
    return tuple(
        row_to_tuple(SUBSYSTEM_PROFILE_CACHE_TABLE_KEY, row)
        for row in t__subsystem_caches__compute.profile_rows
    )


@save_rows(
    context=SUBSYSTEM_CACHES_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY),
)
@tag_compute(
    domain="analytics",
    target=SUBSYSTEM_CACHES_TARGET_NAME,
    target_="subsystem_coverage_cache__rows",
)
def subsystem_coverage_cache__rows(
    t__subsystem_caches__compute: SubsystemCachesComputeResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.subsystem_coverage_cache table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if (
        t__subsystem_caches__compute is None
        or t__subsystem_caches__compute.error
        or t__subsystem_caches__compute.coverage_rows is None
    ):
        return None
    return tuple(
        row_to_tuple(SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY, row)
        for row in t__subsystem_caches__compute.coverage_rows
    )


@codeintel_target(domain="analytics", target=SUBSYSTEM_CACHES_TARGET_NAME)
def t__subsystem_caches(
    env: BuildEnv,
    catalog: DagCatalog,
    t__subsystem_caches__compute: SubsystemCachesComputeResult | None,
    subsystem_caches__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Materialize cached subsystem profile and coverage tables.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    if t__subsystem_caches__compute is not None and t__subsystem_caches__compute.error:
        options_hash = options_hash_for_target(env, SUBSYSTEM_CACHES_TARGET_NAME)
        return TargetRunRecord(
            target=SUBSYSTEM_CACHES_TARGET_NAME,
            impl_kind="native",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__subsystem_caches__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=SUBSYSTEM_CACHES_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=subsystem_caches__table_materializations,
    )


subsystem_caches__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=SUBSYSTEM_CACHES_TARGET_NAME,
    table_keys=SUBSYSTEM_CACHE_TABLE_KEYS,
)


__all__ = [
    "SubsystemCachesComputeResult",
    "subsystem_caches__table_materializations",
    "subsystem_coverage_cache__rows",
    "subsystem_profile_cache__rows",
    "t__subsystem_caches",
    "t__subsystem_caches__compute",
]
