"""Function contracts table built with inferable tabular nodes."""

from __future__ import annotations

import sys

import pyarrow as pa

from codeintel.build.analytics.functions.function_contracts import (
    FunctionContractInputs,
    build_function_contracts_rows,
)
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.utilities.catalogs import catalog_provider_from_frames
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_CONTRACTS_TARGET_NAME = "function_contracts"
FUNCTION_CONTRACTS_TABLE_KEY = "analytics.function_contracts"
FUNCTION_CONTRACTS_CONTRACT = TableContractSpec(
    table_key=FUNCTION_CONTRACTS_TABLE_KEY,
    domain="analytics",
    target=FUNCTION_CONTRACTS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="function_contracts__base",
)


def function_contracts__base(
    env: BuildEnv,
    q__core__docstrings: InferableTabularInput,
    q__analytics__function_types: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> pa.Table:
    """Build function contracts rows using tabular inputs.

    Returns
    -------
    pa.Table
        Reader containing function contract rows.
    """
    goids_frame = tabular_to_arrow_table(q__core__goids)
    modules_frame = tabular_to_arrow_table(q__core__modules)
    docstrings_frame = tabular_to_arrow_table(q__core__docstrings)
    function_types_frame = tabular_to_arrow_table(q__analytics__function_types)
    catalog = catalog_provider_from_frames(goids_frame=goids_frame, modules_frame=modules_frame)
    request = FunctionAstLoadRequest(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.snapshot.repo_root,
        catalog_provider=catalog,
    )
    ast_map, _missing = load_function_asts(request)
    rows = build_function_contracts_rows(
        env.snapshot,
        FunctionContractInputs(
            function_ast_map=ast_map,
            catalog=catalog,
            docstrings_frame=docstrings_frame,
            function_types_frame=function_types_frame,
        ),
    )
    if not rows:
        return empty_table_for_table(FUNCTION_CONTRACTS_TABLE_KEY)
    reader, _ = table_for_rows(FUNCTION_CONTRACTS_TABLE_KEY, rows)
    return reader


_MODULE = sys.modules[__name__]
_FUNCTION_CONTRACTS_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext(
        domain="analytics",
        target_name=FUNCTION_CONTRACTS_TARGET_NAME,
        table_key=FUNCTION_CONTRACTS_TABLE_KEY,
        base_node="function_contracts__base",
        contract=FUNCTION_CONTRACTS_CONTRACT,
        input_type=pa.Table,
    )
)
attach_table_target_template(_MODULE, spec=_FUNCTION_CONTRACTS_TABLE_TARGET_SPEC)
function_contracts__table = _MODULE.function_contracts__table
function_contracts__table_materializations = _MODULE.function_contracts__table_materializations
t__function_contracts = _MODULE.t__function_contracts


__all__ = [
    "function_contracts__base",
    "function_contracts__table",
    "t__function_contracts",
]
