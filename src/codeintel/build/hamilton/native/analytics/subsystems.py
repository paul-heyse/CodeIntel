"""Subsystem analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import pyarrow as pa

from codeintel.build.analytics.subsystems.materialize import (
    SubsystemBuildInputs,
    SubsystemRows,
    build_subsystem_modules_table,
    build_subsystem_rows,
    build_subsystems_table,
)
from codeintel.build.contracts.ref import contract_ref_for_table
from codeintel.build.graphs.external_plan import run_rustworkx_external_plan
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.finalize_helpers import (
    finalize_analytics_reader,
)
from codeintel.build.hamilton.native.patterns import (
    MultiTableTargetContext,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec_from_contexts,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SUBSYSTEMS_TARGET_NAME = "subsystems"
SUBSYSTEMS_TABLE_KEY = "analytics.subsystems"
SUBSYSTEM_MODULES_TABLE_KEY = "analytics.subsystem_modules"
SUBSYSTEMS_CONTRACT = contract_ref_for_table(
    table_key=SUBSYSTEMS_TABLE_KEY,
    target_name=SUBSYSTEMS_TARGET_NAME,
    input_name="subsystems__base",
    required_cols=(),
    clip_column=None,
)
SUBSYSTEM_MODULES_CONTRACT = contract_ref_for_table(
    table_key=SUBSYSTEM_MODULES_TABLE_KEY,
    target_name=SUBSYSTEMS_TARGET_NAME,
    input_name="subsystem_modules__base",
    required_cols=(),
    clip_column=None,
)


@dataclass(frozen=True)
class SubsystemCoreFrames:
    """Core subsystem inputs sourced from graph tables."""

    modules: InferableTabularInput
    import_graph_edges: InferableTabularInput
    symbol_use_edges: InferableTabularInput


@dataclass(frozen=True)
class SubsystemAnalyticsFrames:
    """Analytics inputs sourced from derived tables."""

    config_values: InferableTabularInput


def subsystem_core_frames(
    q__core__modules: InferableTabularInput,
    q__graph__import_graph_edges: InferableTabularInput,
    q__graph__symbol_use_edges: InferableTabularInput,
) -> SubsystemCoreFrames:
    """Bundle core graph inputs for subsystem inference.

    Returns
    -------
    SubsystemCoreFrames
        Core graph inputs for subsystem inference.
    """
    return SubsystemCoreFrames(
        modules=q__core__modules,
        import_graph_edges=q__graph__import_graph_edges,
        symbol_use_edges=q__graph__symbol_use_edges,
    )


def subsystem_analytics_frames(
    q__analytics__config_values: InferableTabularInput,
) -> SubsystemAnalyticsFrames:
    """Bundle analytics inputs for subsystem inference.

    Returns
    -------
    SubsystemAnalyticsFrames
        Analytics inputs for subsystem inference.
    """
    return SubsystemAnalyticsFrames(
        config_values=q__analytics__config_values,
    )


def subsystem_rows(
    env: BuildEnv,
    subsystem_core_frames: SubsystemCoreFrames,
    subsystem_analytics_frames: SubsystemAnalyticsFrames,
) -> SubsystemRows:
    """Compute subsystem inference rows for subsystems and memberships.

    Returns
    -------
    SubsystemRows
        Subsystem summary and membership rows.
    """
    return build_subsystem_rows(
        env.snapshot,
        _build_subsystem_inputs(
            env,
            subsystem_core_frames=subsystem_core_frames,
            subsystem_analytics_frames=subsystem_analytics_frames,
        ),
    )


def _build_subsystem_inputs(
    env: BuildEnv,
    *,
    subsystem_core_frames: SubsystemCoreFrames,
    subsystem_analytics_frames: SubsystemAnalyticsFrames,
) -> SubsystemBuildInputs:
    scope = SnapshotScope.from_snapshot(env.snapshot)
    return SubsystemBuildInputs(
        modules_frame=tabular_to_scoped_table(
            subsystem_core_frames.modules,
            columns=None,
            scope=scope,
            require_scope_columns=True,
        ),
        import_graph_edges_frame=tabular_to_scoped_table(
            subsystem_core_frames.import_graph_edges,
            columns=None,
            scope=scope,
            require_scope_columns=True,
        ),
        symbol_use_edges_frame=tabular_to_scoped_table(
            subsystem_core_frames.symbol_use_edges,
            columns=None,
            scope=scope,
            require_scope_columns=True,
        ),
        config_values_frame=tabular_to_scoped_table(
            subsystem_analytics_frames.config_values,
            columns=None,
            scope=scope,
            require_scope_columns=True,
        ),
    )


def subsystems__base(
    env: BuildEnv,
    subsystem_core_frames: SubsystemCoreFrames,
    subsystem_analytics_frames: SubsystemAnalyticsFrames,
) -> pa.Table:
    """Build subsystem summary rows.

    Returns
    -------
    pa.Table
        Reader containing subsystem rows.
    """
    inputs = _build_subsystem_inputs(
        env,
        subsystem_core_frames=subsystem_core_frames,
        subsystem_analytics_frames=subsystem_analytics_frames,
    )
    reader = run_rustworkx_external_plan(
        builder=build_subsystems_table,
        kwargs={
            "snapshot": env.snapshot,
            "inputs": inputs,
        },
    )
    return finalize_analytics_reader(SUBSYSTEMS_TABLE_KEY, reader)


def subsystem_modules__base(
    env: BuildEnv,
    subsystem_core_frames: SubsystemCoreFrames,
    subsystem_analytics_frames: SubsystemAnalyticsFrames,
) -> pa.Table:
    """Build subsystem membership rows.

    Returns
    -------
    pa.Table
        Reader containing subsystem membership rows.
    """
    inputs = _build_subsystem_inputs(
        env,
        subsystem_core_frames=subsystem_core_frames,
        subsystem_analytics_frames=subsystem_analytics_frames,
    )
    reader = run_rustworkx_external_plan(
        builder=build_subsystem_modules_table,
        kwargs={
            "snapshot": env.snapshot,
            "inputs": inputs,
        },
    )
    return finalize_analytics_reader(SUBSYSTEM_MODULES_TABLE_KEY, reader)


_MODULE = sys.modules[__name__]
_SUBSYSTEMS_TABLE_CONTEXTS = (
    TableTargetTableContext.from_contract_ref(
        contract_ref=SUBSYSTEMS_CONTRACT,
        node_name="subsystems__table",
        input_type=pa.Table,
    ),
    TableTargetTableContext.from_contract_ref(
        contract_ref=SUBSYSTEM_MODULES_CONTRACT,
        node_name="subsystem_modules__table",
        input_type=pa.Table,
    ),
)
_SUBSYSTEMS_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="analytics",
        target_name=SUBSYSTEMS_TARGET_NAME,
        tables=(),
        table_materializations_node="subsystems__table_materializations",
        anchor_node_name="t__subsystems",
    ),
    table_contexts=_SUBSYSTEMS_TABLE_CONTEXTS,
)
attach_table_target_template(_MODULE, spec=_SUBSYSTEMS_TABLE_TARGET_SPEC)
subsystems__table = _MODULE.subsystems__table
subsystem_modules__table = _MODULE.subsystem_modules__table
subsystems__table_materializations = _MODULE.subsystems__table_materializations
t__subsystems = _MODULE.t__subsystems


__all__ = [
    "subsystem_modules__base",
    "subsystem_modules__table",
    "subsystem_rows",
    "subsystems__base",
    "subsystems__table",
    "subsystems__table_materializations",
    "t__subsystems",
]
