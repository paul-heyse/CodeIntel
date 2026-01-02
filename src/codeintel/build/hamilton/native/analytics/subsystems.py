"""Subsystem analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import polars as pl

from codeintel.build.analytics.subsystems.materialize import (
    SubsystemBuildInputs,
    SubsystemRows,
    build_subsystem_rows,
)
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.tabular.frames import rows_to_frame
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SUBSYSTEMS_TARGET_NAME = "subsystems"
SUBSYSTEMS_TABLE_KEY = "analytics.subsystems"
SUBSYSTEM_MODULES_TABLE_KEY = "analytics.subsystem_modules"
SUBSYSTEMS_CONTRACT = TableContractSpec(
    table_key=SUBSYSTEMS_TABLE_KEY,
    domain="analytics",
    target=SUBSYSTEMS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="subsystems__base",
)
SUBSYSTEM_MODULES_CONTRACT = TableContractSpec(
    table_key=SUBSYSTEM_MODULES_TABLE_KEY,
    domain="analytics",
    target=SUBSYSTEMS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="subsystem_modules__base",
)

SUBSYSTEMS_COLUMNS = (
    "repo",
    "commit",
    "subsystem_id",
    "name",
    "description",
    "module_count",
    "modules_json",
    "entrypoints_json",
    "internal_edge_count",
    "external_edge_count",
    "fan_in",
    "fan_out",
    "function_count",
    "avg_risk_score",
    "max_risk_score",
    "high_risk_function_count",
    "risk_level",
    "created_at",
)

SUBSYSTEM_MODULES_COLUMNS = (
    "repo",
    "commit",
    "subsystem_id",
    "module",
    "role",
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
        SubsystemBuildInputs(
            modules_frame=tabular_to_lazyframe(subsystem_core_frames.modules).collect(),
            import_graph_edges_frame=tabular_to_lazyframe(
                subsystem_core_frames.import_graph_edges
            ).collect(),
            symbol_use_edges_frame=tabular_to_lazyframe(
                subsystem_core_frames.symbol_use_edges
            ).collect(),
            config_values_frame=tabular_to_lazyframe(
                subsystem_analytics_frames.config_values
            ).collect(),
        ),
    )


def subsystems__base(subsystem_rows: SubsystemRows) -> pl.LazyFrame:
    """Build subsystem summary rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing subsystem rows.
    """
    return rows_to_frame(
        SUBSYSTEMS_TABLE_KEY,
        subsystem_rows.subsystem_rows,
        columns=SUBSYSTEMS_COLUMNS,
    )


def subsystem_modules__base(subsystem_rows: SubsystemRows) -> pl.LazyFrame:
    """Build subsystem membership rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing subsystem membership rows.
    """
    return rows_to_frame(
        SUBSYSTEM_MODULES_TABLE_KEY,
        subsystem_rows.membership_rows,
        columns=SUBSYSTEM_MODULES_COLUMNS,
    )


_MODULE = sys.modules[__name__]
_SUBSYSTEMS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=SUBSYSTEMS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=SUBSYSTEMS_TABLE_KEY,
            base_node="subsystems__base",
            contract=SUBSYSTEMS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=SUBSYSTEMS_TABLE_KEY),
            node_name="subsystems__table",
        ),
        TableTargetTableSpec(
            table_key=SUBSYSTEM_MODULES_TABLE_KEY,
            base_node="subsystem_modules__base",
            contract=SUBSYSTEM_MODULES_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=SUBSYSTEM_MODULES_TABLE_KEY),
            node_name="subsystem_modules__table",
        ),
    ),
    table_materializations_node="subsystems__table_materializations",
    anchor_node_name="t__subsystems",
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
