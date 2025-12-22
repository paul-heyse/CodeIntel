"""Native Hamilton implementation for subsystem cache targets."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from hamilton.function_modifiers import source, value

from codeintel.analytics.subsystems.cache import (
    build_subsystem_coverage_cache_rows,
    build_subsystem_profile_cache_rows,
)
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.target_override_tables import SUBSYSTEM_CACHE_OVERRIDE_TABLES
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
    register_output_targets,
)
from codeintel.build.hamilton.run_records import (
    TargetRunRecord,
    options_hash_for_target,
    should_skip_native_target,
)
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize
from codeintel.build.hashing import InputHashOptions, compute_input_hash
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.targets import TargetGraph
from codeintel.core.schemas.row_serialization import row_to_tuple

if TYPE_CHECKING:
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsSubsystemCoverageCacheRow as SubsystemCoverageCacheRow,
    )
    from codeintel.core.schemas.generated_rows.analytics import (
        AnalyticsSubsystemProfileCacheRow as SubsystemProfileCacheRow,
    )

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

SUBSYSTEM_CACHES_TARGET_NAME = "subsystem_caches"

SUBSYSTEM_PROFILE_CACHE_TABLE_KEY = "analytics.subsystem_profile_cache"
SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY = "analytics.subsystem_coverage_cache"
SUBSYSTEM_CACHE_TABLE_KEYS = (
    SUBSYSTEM_PROFILE_CACHE_TABLE_KEY,
    SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY,
)

register_output_targets(
    make_output_target(
        name=SUBSYSTEM_CACHES_TARGET_NAME,
        module="analytics",
        description="Cached subsystem profile and coverage tables.",
        options=TargetSpecOptions(
            table_keys=SUBSYSTEM_CACHE_TABLE_KEYS,
            override_tables=SUBSYSTEM_CACHE_OVERRIDE_TABLES,
        ),
    ),
)


@dataclass(frozen=True)
class SubsystemCachesComputeResult:
    """Result from subsystem cache computation."""

    profile_rows: list[SubsystemProfileCacheRow] | None
    coverage_rows: list[SubsystemCoverageCacheRow] | None
    error: str | None = None


@tag_compute(domain="analytics", target=SUBSYSTEM_CACHES_TARGET_NAME)
def t__subsystem_caches__compute(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystems: TargetRunRecord,
    t__subsystem_graph_metrics: TargetRunRecord,
    t__test_profile: TargetRunRecord,
) -> SubsystemCachesComputeResult | None:
    """Compute subsystem cache rows from base subsystem tables.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup and skip checks.
    t__subsystems
        Upstream subsystems target result.
    t__subsystem_graph_metrics
        Upstream subsystem graph metrics result.
    t__test_profile
        Upstream test profile result.

    Returns
    -------
    SubsystemCachesComputeResult | None
        Computed cache rows or None when skipped.
    """
    target = graph.get(SUBSYSTEM_CACHES_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, SUBSYSTEM_CACHES_TARGET_NAME)
        hash_options = InputHashOptions(options_hash=options_hash, manifests=env.manifest_index)
        input_hash = compute_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=hash_options,
        )
        if should_skip_native_target(env, target, input_hash):
            return None

    if t__subsystems.status != "succeeded":
        return SubsystemCachesComputeResult(
            profile_rows=None,
            coverage_rows=None,
            error=f"Upstream subsystems target failed: {t__subsystems.error}",
        )

    if t__subsystem_graph_metrics.status != "succeeded":
        return SubsystemCachesComputeResult(
            profile_rows=None,
            coverage_rows=None,
            error=(
                "Upstream subsystem_graph_metrics target failed: "
                f"{t__subsystem_graph_metrics.error}"
            ),
        )

    if t__test_profile.status != "succeeded":
        return SubsystemCachesComputeResult(
            profile_rows=None,
            coverage_rows=None,
            error=f"Upstream test_profile target failed: {t__test_profile.error}",
        )

    try:
        profile_rows = build_subsystem_profile_cache_rows(env.gateway, env.snapshot)
        coverage_rows = build_subsystem_coverage_cache_rows(env.gateway, env.snapshot)
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


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(SUBSYSTEM_PROFILE_CACHE_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SUBSYSTEM_CACHES_TARGET_NAME),
    table_key=value(SUBSYSTEM_PROFILE_CACHE_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(SUBSYSTEM_PROFILE_CACHE_TABLE_KEY)),
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


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SUBSYSTEM_CACHES_TARGET_NAME),
    table_key=value(SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY)),
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


@tag_materialize(domain="analytics", target=SUBSYSTEM_CACHES_TARGET_NAME)
def t__subsystem_caches(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystem_caches__compute: SubsystemCachesComputeResult | None,
    m__analytics__subsystem_profile_cache: MaterializationMetadata,
    m__analytics__subsystem_coverage_cache: MaterializationMetadata,
) -> TargetRunRecord:
    """Materialize subsystem cache tables from computed rows.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    if t__subsystem_caches__compute is not None and t__subsystem_caches__compute.error:
        options_hash = options_hash_for_target(env, SUBSYSTEM_CACHES_TARGET_NAME)
        return TargetRunRecord(
            target=SUBSYSTEM_CACHES_TARGET_NAME,
            plugin_name=f"native:{SUBSYSTEM_CACHES_TARGET_NAME}",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__subsystem_caches__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=SUBSYSTEM_CACHES_TARGET_NAME,
        materializations={
            SUBSYSTEM_PROFILE_CACHE_TABLE_KEY: m__analytics__subsystem_profile_cache,
            SUBSYSTEM_COVERAGE_CACHE_TABLE_KEY: m__analytics__subsystem_coverage_cache,
        },
    )


__all__ = [
    "SubsystemCachesComputeResult",
    "subsystem_coverage_cache__rows",
    "subsystem_profile_cache__rows",
    "t__subsystem_caches",
    "t__subsystem_caches__compute",
]
