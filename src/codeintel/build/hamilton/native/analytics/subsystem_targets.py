"""Native Hamilton implementation for the `subsystems` analytics target.

This target is implemented as a native execution boundary that reuses the
canonical subsystem inference pipeline in `codeintel.analytics.subsystems`.

The subsystem pipeline materializes both:
- ``analytics.subsystems``
- ``analytics.subsystem_modules``
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from hamilton.function_modifiers import source, value

from codeintel.analytics.subsystems.materialize import SubsystemRows, build_subsystem_rows
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
)
from codeintel.build.hamilton.native.target_override_tables import SUBSYSTEMS_OVERRIDE_TABLES
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

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, SubsystemRows)

SUBSYSTEMS_TARGET_NAME = "subsystems"

SUBSYSTEMS_TABLE_KEY = "analytics.subsystems"
SUBSYSTEM_MODULES_TABLE_KEY = "analytics.subsystem_modules"
SUBSYSTEMS_TABLE_KEYS = (SUBSYSTEMS_TABLE_KEY, SUBSYSTEM_MODULES_TABLE_KEY)

register_output_targets(
    make_output_target(
        name=SUBSYSTEMS_TARGET_NAME,
        module="analytics",
        description="Architectural subsystem inference.",
        options=TargetSpecOptions(
            table_keys=SUBSYSTEMS_TABLE_KEYS,
            override_tables=SUBSYSTEMS_OVERRIDE_TABLES,
        ),
    ),
)


@dataclass(frozen=True)
class SubsystemsComputeResult:
    """Result from subsystem inference computation."""

    rows: SubsystemRows | None
    error: str | None = None


@tag_compute(domain="analytics", target=SUBSYSTEMS_TARGET_NAME)
def t__subsystems__compute(
    env: BuildEnv,
    graph: TargetGraph,
    t__import_graph: TargetRunRecord,
    t__semantic_roles: TargetRunRecord,
) -> SubsystemsComputeResult | None:
    """Compute subsystems by executing the subsystem inference pipeline.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup and skip detection.
    t__import_graph
        Upstream import graph record.
    t__semantic_roles
        Upstream semantic roles record.

    Returns
    -------
    SubsystemsComputeResult | None
        Computed subsystem rows or None when skipped.
    """
    target = graph.get(SUBSYSTEMS_TARGET_NAME)
    if target is not None:
        options_hash = options_hash_for_target(env, SUBSYSTEMS_TARGET_NAME)
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

    if t__import_graph.status != "succeeded":
        return SubsystemsComputeResult(
            rows=None,
            error=(
                f"Upstream import_graph target failed: {t__import_graph.error or 'unknown error'}"
            ),
        )

    if t__semantic_roles.status != "succeeded":
        return SubsystemsComputeResult(
            rows=None,
            error=(
                f"Upstream semantic_roles target failed: {t__semantic_roles.error or 'unknown error'}"
            ),
        )

    try:
        rows = build_subsystem_rows(env.gateway, env.snapshot)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        log.exception("subsystems: build_subsystem_rows failed")
        return SubsystemsComputeResult(rows=None, error=str(exc))

    return SubsystemsComputeResult(rows=rows)


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(SUBSYSTEMS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SUBSYSTEMS_TARGET_NAME),
    table_key=value(SUBSYSTEMS_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(SUBSYSTEMS_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=SUBSYSTEMS_TARGET_NAME,
    target_="subsystems__rows",
)
def subsystems__rows(
    t__subsystems__compute: SubsystemsComputeResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.subsystems table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if (
        t__subsystems__compute is None
        or t__subsystems__compute.error
        or t__subsystems__compute.rows is None
    ):
        return None
    return tuple(t__subsystems__compute.rows.subsystem_rows)


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(SUBSYSTEM_MODULES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SUBSYSTEMS_TARGET_NAME),
    table_key=value(SUBSYSTEM_MODULES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(SUBSYSTEM_MODULES_TABLE_KEY)),
)
@tag_compute(
    domain="analytics",
    target=SUBSYSTEMS_TARGET_NAME,
    target_="subsystem_modules__rows",
)
def subsystem_modules__rows(
    t__subsystems__compute: SubsystemsComputeResult | None,
) -> tuple[tuple[object, ...], ...] | None:
    """Extract rows for analytics.subsystem_modules table.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Row tuples ready for materialization, or ``None`` when unavailable.
    """
    if (
        t__subsystems__compute is None
        or t__subsystems__compute.error
        or t__subsystems__compute.rows is None
    ):
        return None
    return tuple(t__subsystems__compute.rows.membership_rows)


@tag_materialize(domain="analytics", target=SUBSYSTEMS_TARGET_NAME)
def t__subsystems(
    env: BuildEnv,
    graph: TargetGraph,
    t__subsystems__compute: SubsystemsComputeResult | None,
    m__analytics__subsystems: MaterializationMetadata,
    m__analytics__subsystem_modules: MaterializationMetadata,
) -> TargetRunRecord:
    """Materialize a TargetRunRecord for subsystems from a compute result.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    graph
        Target graph for metadata lookup and skip detection.
    t__subsystems__compute
        Computed subsystem rows and optional error.
    m__analytics__subsystems
        Materialization metadata for analytics.subsystems.
    m__analytics__subsystem_modules
        Materialization metadata for analytics.subsystem_modules.

    Returns
    -------
    TargetRunRecord
        Final execution record for the target.
    """
    if t__subsystems__compute is not None and t__subsystems__compute.error:
        options_hash = options_hash_for_target(env, SUBSYSTEMS_TARGET_NAME)
        return TargetRunRecord(
            target=SUBSYSTEMS_TARGET_NAME,
            plugin_name=f"native:{SUBSYSTEMS_TARGET_NAME}",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__subsystems__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=SUBSYSTEMS_TARGET_NAME,
        materializations={
            SUBSYSTEMS_TABLE_KEY: m__analytics__subsystems,
            SUBSYSTEM_MODULES_TABLE_KEY: m__analytics__subsystem_modules,
        },
    )


__all__ = [
    "SubsystemsComputeResult",
    "subsystem_modules__rows",
    "subsystems__rows",
    "t__subsystems",
    "t__subsystems__compute",
]
