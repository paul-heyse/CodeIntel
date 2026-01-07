"""Function effects table built with inferable tabular nodes."""

from __future__ import annotations

import sys

import pyarrow as pa

from codeintel.build.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    build_function_effects_rows,
)
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.utilities.catalogs import catalog_provider_from_frames
from codeintel.build.contracts.registry import contract_for_table
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import table_for_rows

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_EFFECTS_TARGET_NAME = "function_effects"
FUNCTION_EFFECTS_TABLE_KEY = "analytics.function_effects"
FUNCTION_EFFECTS_CONTRACT = contract_for_table(
    table_key=FUNCTION_EFFECTS_TABLE_KEY,
    target_name=FUNCTION_EFFECTS_TARGET_NAME,
    input_name="function_effects__base",
    required_cols=(),
    clip_column=None,
)


def function_effects__base(
    env: BuildEnv,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
    q__graph__call_graph_edges: InferableTabularInput,
    q__graph__call_graph_nodes: InferableTabularInput,
) -> pa.Table:
    """Build function effects rows using tabular inputs.

    Returns
    -------
    pa.Table
        Reader with function effects rows.
    """
    goids_frame = tabular_to_arrow_table(q__core__goids).select(
        ["goid_h128", "rel_path", "qualname", "start_line", "end_line", "urn", "kind"]
    )
    modules_frame = tabular_to_arrow_table(q__core__modules).select(["path", "module"])
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
        call_graph_edges=tabular_to_arrow_table(q__graph__call_graph_edges).select(
            ["repo", "commit", "caller_goid_h128", "callee_goid_h128"]
        ),
        call_graph_nodes=tabular_to_arrow_table(q__graph__call_graph_nodes).select(
            ["goid_h128", "kind"]
        ),
    )
    rows = build_function_effects_rows(env.snapshot, inputs=inputs)
    reader, _ = table_for_rows(FUNCTION_EFFECTS_TABLE_KEY, rows)
    return reader


_MODULE = sys.modules[__name__]
_FUNCTION_EFFECTS_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract(
        contract=FUNCTION_EFFECTS_CONTRACT,
        input_type=pa.Table,
    )
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
