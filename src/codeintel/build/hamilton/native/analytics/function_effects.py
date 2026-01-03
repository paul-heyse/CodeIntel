"""Function effects table built with inferable tabular nodes."""

from __future__ import annotations

import sys

import polars as pl

from codeintel.build.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    build_function_effects_rows,
)
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.utilities.catalogs import catalog_provider_from_frames
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.frames import rows_to_frame
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_EFFECTS_TARGET_NAME = "function_effects"
FUNCTION_EFFECTS_TABLE_KEY = "analytics.function_effects"
FUNCTION_EFFECTS_CONTRACT = TableContractSpec(
    table_key=FUNCTION_EFFECTS_TABLE_KEY,
    domain="analytics",
    target=FUNCTION_EFFECTS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="function_effects__base",
)


def function_effects__base(
    env: BuildEnv,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
    q__graph__call_graph_edges: InferableTabularInput,
    q__graph__call_graph_nodes: InferableTabularInput,
) -> pl.LazyFrame:
    """Build function effects rows using tabular inputs.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with function effects columns.
    """
    goids_frame = tabular_to_lazyframe(q__core__goids).collect()
    modules_frame = tabular_to_lazyframe(q__core__modules).collect()
    catalog = catalog_provider_from_frames(goids_frame=goids_frame, modules_frame=modules_frame)
    request = FunctionAstLoadRequest(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.snapshot.repo_root,
        catalog_provider=catalog,
    )
    ast_map, missing = load_function_asts(request)
    inputs = FunctionEffectsInputs(
        catalog_provider=catalog,
        ast_map=ast_map,
        missing_goids=missing,
        call_graph_edges=tabular_to_lazyframe(q__graph__call_graph_edges).collect(),
        call_graph_nodes=tabular_to_lazyframe(q__graph__call_graph_nodes).collect(),
    )
    rows = build_function_effects_rows(env.snapshot, inputs=inputs)
    return rows_to_frame(FUNCTION_EFFECTS_TABLE_KEY, rows)


_MODULE = sys.modules[__name__]
_FUNCTION_EFFECTS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=FUNCTION_EFFECTS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=FUNCTION_EFFECTS_TABLE_KEY,
            base_node="function_effects__base",
            contract=FUNCTION_EFFECTS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=FUNCTION_EFFECTS_TABLE_KEY),
            node_name="function_effects__table",
        ),
    ),
    table_materializations_node="function_effects__table_materializations",
    anchor_node_name="t__function_effects",
)
attach_table_target_template(_MODULE, spec=_FUNCTION_EFFECTS_TABLE_TARGET_SPEC)
function_effects__table = _MODULE.function_effects__table
function_effects__table_materializations = _MODULE.function_effects__table_materializations
t__function_effects = _MODULE.t__function_effects


__all__ = [
    "function_effects__base",
    "function_effects__table",
    "t__function_effects",
]
