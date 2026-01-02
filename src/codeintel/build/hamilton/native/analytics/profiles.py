"""Profile analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import polars as pl

from codeintel.build.analytics.hotspots import compute_hotspot_rows
from codeintel.build.analytics.profiles.files import (
    build_file_profile_rows,
    compute_file_profile_inputs,
)
from codeintel.build.analytics.profiles.functions import (
    FunctionProfileViews,
    build_function_profile_rows,
    compute_function_profile_inputs,
    join_function_contracts,
    join_function_docs,
    join_function_effects,
    join_function_risk,
    join_function_roles,
    load_function_base_info,
)
from codeintel.build.analytics.profiles.graph_features import summarize_graph_for_function_profile
from codeintel.build.analytics.profiles.types import (
    FileProfileFrames,
    FileProfileInputs,
    FunctionProfileFrames,
    FunctionProfileInputs,
)
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import rows_to_frame
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

FUNCTION_PROFILE_TARGET_NAME = "function_profile"
FUNCTION_PROFILE_TABLE_KEY = "analytics.function_profile"
FUNCTION_PROFILE_CONTRACT = TableContractSpec(
    table_key=FUNCTION_PROFILE_TABLE_KEY,
    domain="analytics",
    target=FUNCTION_PROFILE_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="function_profile__base",
)

FILE_PROFILE_TARGET_NAME = "file_profile"
FILE_PROFILE_TABLE_KEY = "analytics.file_profile"
FILE_PROFILE_CONTRACT = TableContractSpec(
    table_key=FILE_PROFILE_TABLE_KEY,
    domain="analytics",
    target=FILE_PROFILE_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="file_profile__base",
)

HOTSPOTS_TARGET_NAME = "hotspots"
HOTSPOTS_TABLE_KEY = "analytics.hotspots"
HOTSPOTS_CONTRACT = TableContractSpec(
    table_key=HOTSPOTS_TABLE_KEY,
    domain="analytics",
    target=HOTSPOTS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="hotspots__base",
)


@dataclass(frozen=True)
class FunctionProfileCoreFrames:
    function_metrics: pl.DataFrame
    function_types: pl.DataFrame
    typedness: pl.DataFrame
    diagnostics: pl.DataFrame
    modules: pl.DataFrame


@dataclass(frozen=True)
class FunctionProfileRiskFrames:
    goid_risk_factors: pl.DataFrame
    hotspots: pl.DataFrame


@dataclass(frozen=True)
class FunctionProfileGraphFrames:
    graph_metrics_functions: pl.DataFrame
    call_graph_edges: pl.DataFrame
    call_graph_nodes: pl.DataFrame


@dataclass(frozen=True)
class FunctionProfileEffectFrames:
    function_effects: pl.DataFrame
    function_contracts: pl.DataFrame
    semantic_roles_functions: pl.DataFrame
    docstrings: pl.DataFrame


def function_profile_core_frames(
    q__analytics__function_metrics: InferableTabularInput,
    q__analytics__function_types: InferableTabularInput,
    q__analytics__typedness: InferableTabularInput,
    q__analytics__static_diagnostics: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> FunctionProfileCoreFrames:
    return FunctionProfileCoreFrames(
        function_metrics=tabular_to_lazyframe(q__analytics__function_metrics).collect(),
        function_types=tabular_to_lazyframe(q__analytics__function_types).collect(),
        typedness=tabular_to_lazyframe(q__analytics__typedness).collect(),
        diagnostics=tabular_to_lazyframe(q__analytics__static_diagnostics).collect(),
        modules=tabular_to_lazyframe(q__core__modules).collect(),
    )


def function_profile_risk_frames(
    q__analytics__goid_risk_factors: InferableTabularInput,
    q__analytics__hotspots: InferableTabularInput,
) -> FunctionProfileRiskFrames:
    return FunctionProfileRiskFrames(
        goid_risk_factors=tabular_to_lazyframe(q__analytics__goid_risk_factors).collect(),
        hotspots=tabular_to_lazyframe(q__analytics__hotspots).collect(),
    )


def function_profile_graph_frames(
    q__analytics__graph_metrics_functions: InferableTabularInput,
    q__graph__call_graph_edges: InferableTabularInput,
    q__graph__call_graph_nodes: InferableTabularInput,
) -> FunctionProfileGraphFrames:
    return FunctionProfileGraphFrames(
        graph_metrics_functions=tabular_to_lazyframe(
            q__analytics__graph_metrics_functions
        ).collect(),
        call_graph_edges=tabular_to_lazyframe(q__graph__call_graph_edges).collect(),
        call_graph_nodes=tabular_to_lazyframe(q__graph__call_graph_nodes).collect(),
    )


def function_profile_effect_frames(
    q__analytics__function_effects: InferableTabularInput,
    q__analytics__function_contracts: InferableTabularInput,
    q__analytics__semantic_roles_functions: InferableTabularInput,
    q__core__docstrings: InferableTabularInput,
) -> FunctionProfileEffectFrames:
    return FunctionProfileEffectFrames(
        function_effects=tabular_to_lazyframe(q__analytics__function_effects).collect(),
        function_contracts=tabular_to_lazyframe(q__analytics__function_contracts).collect(),
        semantic_roles_functions=tabular_to_lazyframe(
            q__analytics__semantic_roles_functions
        ).collect(),
        docstrings=tabular_to_lazyframe(q__core__docstrings).collect(),
    )


def function_profile_frames(
    function_profile_core_frames: FunctionProfileCoreFrames,
    function_profile_risk_frames: FunctionProfileRiskFrames,
    function_profile_graph_frames: FunctionProfileGraphFrames,
    function_profile_effect_frames: FunctionProfileEffectFrames,
) -> FunctionProfileFrames:
    return FunctionProfileFrames(
        function_metrics=function_profile_core_frames.function_metrics,
        function_types=function_profile_core_frames.function_types,
        modules=function_profile_core_frames.modules,
        typedness=function_profile_core_frames.typedness,
        diagnostics=function_profile_core_frames.diagnostics,
        goid_risk_factors=function_profile_risk_frames.goid_risk_factors,
        graph_metrics_functions=function_profile_graph_frames.graph_metrics_functions,
        function_effects=function_profile_effect_frames.function_effects,
        function_contracts=function_profile_effect_frames.function_contracts,
        semantic_roles_functions=function_profile_effect_frames.semantic_roles_functions,
        docstrings=function_profile_effect_frames.docstrings,
        hotspots=function_profile_risk_frames.hotspots,
        call_graph_edges=function_profile_graph_frames.call_graph_edges,
        call_graph_nodes=function_profile_graph_frames.call_graph_nodes,
    )


@dataclass(frozen=True)
class FileProfileCoreFrames:
    function_profile: pl.DataFrame
    ast_metrics: pl.DataFrame
    hotspots: pl.DataFrame
    typedness: pl.DataFrame
    modules: pl.DataFrame


@dataclass(frozen=True)
class FileProfileDiagnosticsFrames:
    static_diagnostics: pl.DataFrame


def file_profile_core_frames(
    q__analytics__function_profile: InferableTabularInput,
    q__core__ast_metrics: InferableTabularInput,
    q__analytics__hotspots: InferableTabularInput,
    q__analytics__typedness: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> FileProfileCoreFrames:
    return FileProfileCoreFrames(
        function_profile=tabular_to_lazyframe(q__analytics__function_profile).collect(),
        ast_metrics=tabular_to_lazyframe(q__core__ast_metrics).collect(),
        hotspots=tabular_to_lazyframe(q__analytics__hotspots).collect(),
        typedness=tabular_to_lazyframe(q__analytics__typedness).collect(),
        modules=tabular_to_lazyframe(q__core__modules).collect(),
    )


def file_profile_diagnostics_frames(
    q__analytics__static_diagnostics: InferableTabularInput,
) -> FileProfileDiagnosticsFrames:
    return FileProfileDiagnosticsFrames(
        static_diagnostics=tabular_to_lazyframe(q__analytics__static_diagnostics).collect(),
    )


def file_profile_frames(
    file_profile_core_frames: FileProfileCoreFrames,
    file_profile_diagnostics_frames: FileProfileDiagnosticsFrames,
) -> FileProfileFrames:
    return FileProfileFrames(
        function_profile=file_profile_core_frames.function_profile,
        ast_metrics=file_profile_core_frames.ast_metrics,
        hotspots=file_profile_core_frames.hotspots,
        typedness=file_profile_core_frames.typedness,
        static_diagnostics=file_profile_diagnostics_frames.static_diagnostics,
        modules=file_profile_core_frames.modules,
    )


def function_profile_inputs(
    env: BuildEnv,
    function_profile_frames: FunctionProfileFrames,
) -> FunctionProfileInputs:
    return compute_function_profile_inputs(env.snapshot, function_profile_frames)


def function_profile__base(function_profile_inputs: FunctionProfileInputs) -> pl.LazyFrame:
    """Build function profile rows using tabular inputs.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing function profile rows.
    """
    inputs = function_profile_inputs
    views = FunctionProfileViews(
        base_by_func=load_function_base_info(inputs),
        risk_by_func=join_function_risk(inputs),
        graph_by_func=summarize_graph_for_function_profile(inputs),
        effects_by_func=join_function_effects(inputs),
        contracts_by_func=join_function_contracts(inputs),
        roles_by_func=join_function_roles(inputs),
        docs_by_func=join_function_docs(inputs),
    )
    rows = list(build_function_profile_rows(inputs, views=views))
    return rows_to_frame(FUNCTION_PROFILE_TABLE_KEY, rows)


def file_profile_inputs(
    env: BuildEnv,
    file_profile_frames: FileProfileFrames,
) -> FileProfileInputs:
    return compute_file_profile_inputs(env.snapshot, file_profile_frames)


def file_profile__base(file_profile_inputs: FileProfileInputs) -> pl.LazyFrame:
    """Build file profile rows using tabular inputs.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing file profile rows.
    """
    inputs = file_profile_inputs
    rows = list(build_file_profile_rows(inputs))
    return rows_to_frame(FILE_PROFILE_TABLE_KEY, rows)


def hotspots__base(q__core__ast_metrics: InferableTabularInput) -> pl.LazyFrame:
    """Build hotspot rows from core AST metrics.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing hotspot rows.
    """
    metrics_frame = tabular_to_lazyframe(q__core__ast_metrics).collect()
    ast_metrics: list[tuple[str, float]] = []
    for row in metrics_frame.iter_rows(named=True):
        rel_path = row.get("rel_path")
        if not isinstance(rel_path, str):
            continue
        complexity_raw = row.get("complexity")
        complexity = float(complexity_raw or 0.0)
        ast_metrics.append((rel_path, complexity))
    rows = compute_hotspot_rows(ast_metrics)
    return rows_to_frame(HOTSPOTS_TABLE_KEY, rows)


_MODULE = sys.modules[__name__]
_FUNCTION_PROFILE_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=FUNCTION_PROFILE_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=FUNCTION_PROFILE_TABLE_KEY,
            base_node="function_profile__base",
            contract=FUNCTION_PROFILE_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=FUNCTION_PROFILE_TABLE_KEY),
            node_name="function_profile__table",
        ),
    ),
    table_materializations_node="function_profile__table_materializations",
    anchor_node_name="t__function_profile",
)
attach_table_target_template(_MODULE, spec=_FUNCTION_PROFILE_TABLE_TARGET_SPEC)
function_profile__table = _MODULE.function_profile__table
function_profile__table_materializations = _MODULE.function_profile__table_materializations
t__function_profile = _MODULE.t__function_profile

_FILE_PROFILE_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=FILE_PROFILE_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=FILE_PROFILE_TABLE_KEY,
            base_node="file_profile__base",
            contract=FILE_PROFILE_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=FILE_PROFILE_TABLE_KEY),
            node_name="file_profile__table",
        ),
    ),
    table_materializations_node="file_profile__table_materializations",
    anchor_node_name="t__file_profile",
)
attach_table_target_template(_MODULE, spec=_FILE_PROFILE_TABLE_TARGET_SPEC)
file_profile__table = _MODULE.file_profile__table
file_profile__table_materializations = _MODULE.file_profile__table_materializations
t__file_profile = _MODULE.t__file_profile

_HOTSPOTS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=HOTSPOTS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=HOTSPOTS_TABLE_KEY,
            base_node="hotspots__base",
            contract=HOTSPOTS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=HOTSPOTS_TABLE_KEY),
            node_name="hotspots__table",
        ),
    ),
    table_materializations_node="hotspots__table_materializations",
    anchor_node_name="t__hotspots",
)
attach_table_target_template(_MODULE, spec=_HOTSPOTS_TABLE_TARGET_SPEC)
hotspots__table = _MODULE.hotspots__table
hotspots__table_materializations = _MODULE.hotspots__table_materializations
t__hotspots = _MODULE.t__hotspots


__all__ = [
    "file_profile__base",
    "file_profile__table",
    "function_profile__base",
    "function_profile__table",
    "hotspots__base",
    "hotspots__table",
    "t__file_profile",
    "t__function_profile",
    "t__hotspots",
]
