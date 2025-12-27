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

from codeintel.analytics.subsystems.materialize import SubsystemRows, build_subsystem_rows
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
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
from codeintel.build.hashing import InputHashOptions
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, SubsystemRows)

SUBSYSTEMS_TARGET_NAME = "subsystems"

SUBSYSTEMS_TABLE_KEY = "analytics.subsystems"
SUBSYSTEM_MODULES_TABLE_KEY = "analytics.subsystem_modules"
SUBSYSTEMS_TABLE_KEYS = (
    SUBSYSTEMS_TABLE_KEY,
    SUBSYSTEM_MODULES_TABLE_KEY,
)
SUBSYSTEMS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=SUBSYSTEMS_TARGET_NAME,
    hash_options_node="subsystems__hash_options",
)


@tag_helper(domain="analytics")
def gateway(env: BuildEnv) -> StorageGateway:
    """Expose the storage gateway for subsystem nodes.

    Returns
    -------
    StorageGateway
        Storage gateway for the current build environment.
    """
    return env.gateway


@tag_helper(domain="analytics", target=SUBSYSTEMS_TARGET_NAME)
def subsystems__hash_options(env: BuildEnv) -> InputHashOptions:
    """Build hash inputs for subsystems execution.

    Returns
    -------
    InputHashOptions
        Hash inputs for manifest-based skip evaluation.
    """
    return InputHashOptions(
        options_hash=options_hash_for_target(env, SUBSYSTEMS_TARGET_NAME),
        manifests=env.manifest_index,
    )


@tag_helper(domain="analytics", target=SUBSYSTEMS_TARGET_NAME)
def subsystems__skip(
    env: BuildEnv,
    catalog: DagCatalog,
    subsystems__hash_options: InputHashOptions,
) -> bool:
    """Return True when subsystems should be skipped.

    Returns
    -------
    bool
        True when the target should be skipped.
    """
    executor = NativeTargetExecutor.for_target(
        env,
        catalog,
        SUBSYSTEMS_TARGET_NAME,
        hash_options=subsystems__hash_options,
    )
    return executor.should_skip()


@dataclass(frozen=True)
class SubsystemsComputeResult:
    """Result from subsystem inference computation."""

    rows: SubsystemRows | None
    error: str | None = None


@tag_compute(domain="analytics", target=SUBSYSTEMS_TARGET_NAME)
def t__subsystems__compute(
    env: BuildEnv,
    gateway: StorageGateway,
    t__import_graph: TargetRunRecord,
    t__semantic_roles: TargetRunRecord,
    *,
    subsystems__skip: bool,
) -> SubsystemsComputeResult | None:
    """Compute subsystems by executing the subsystem inference pipeline.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    gateway
        Storage gateway for analytics queries.
    t__import_graph
        Upstream import graph record.
    t__semantic_roles
        Upstream semantic roles record.
    subsystems__skip
        Skip flag derived from manifest-based input hash evaluation.

    Returns
    -------
    SubsystemsComputeResult | None
        Computed subsystem rows or None when skipped.
    """
    if subsystems__skip:
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
                "Upstream semantic_roles target failed: "
                f"{t__semantic_roles.error or 'unknown error'}"
            ),
        )

    try:
        rows = build_subsystem_rows(gateway, env.snapshot)
    except (KeyError, RuntimeError, TypeError, ValueError) as exc:
        log.exception("subsystems: build_subsystem_rows failed")
        return SubsystemsComputeResult(rows=None, error=str(exc))

    return SubsystemsComputeResult(rows=rows)


@save_rows(
    context=SUBSYSTEMS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SUBSYSTEMS_TABLE_KEY),
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


@save_rows(
    context=SUBSYSTEMS_SAVE_CONTEXT,
    spec=TableSaveSpec(table_key=SUBSYSTEM_MODULES_TABLE_KEY),
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


@codeintel_target(domain="analytics", target=SUBSYSTEMS_TARGET_NAME)
def t__subsystems(
    env: BuildEnv,
    catalog: DagCatalog,
    t__subsystems__compute: SubsystemsComputeResult | None,
    subsystems__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Infer architectural subsystems.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot info.
    catalog
        DAG catalog for metadata lookup and skip detection.
    t__subsystems__compute
        Computed subsystem rows and optional error.
    subsystems__table_materializations
        Materialization results for subsystem tables.

    Returns
    -------
    TargetRunRecord
        Final execution record for the target.
    """
    if t__subsystems__compute is not None and t__subsystems__compute.error:
        options_hash = options_hash_for_target(env, SUBSYSTEMS_TARGET_NAME)
        return TargetRunRecord(
            target=SUBSYSTEMS_TARGET_NAME,
            impl_kind="native",
            status="failed",
            input_hash="",
            options_hash=options_hash,
            duration_ms=0.0,
            row_counts={},
            error=t__subsystems__compute.error,
            datasets=(),
            artifacts=(),
        )

    return record_from_materializations(
        context=MaterializationRecordContext(
            env=env,
            catalog=catalog,
            target_name=SUBSYSTEMS_TARGET_NAME,
        ),
        artifact_materializations=None,
        table_materializations=subsystems__table_materializations,
    )


subsystems__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=SUBSYSTEMS_TARGET_NAME,
    table_keys=SUBSYSTEMS_TABLE_KEYS,
)


__all__ = [
    "SubsystemsComputeResult",
    "subsystem_modules__rows",
    "subsystems__hash_options",
    "subsystems__rows",
    "subsystems__skip",
    "subsystems__table_materializations",
    "t__subsystems",
    "t__subsystems__compute",
]
