"""Function contracts table built with inferable tabular nodes."""

from __future__ import annotations

import sys

import pyarrow as pa

from codeintel.build.analytics.functions.function_contracts import (
    FunctionContractInputs,
    build_function_contracts_rows,
)
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.parsing.worklists import build_function_ast_worklist
from codeintel.build.analytics.utilities.catalogs import (
    CatalogProviderRequest,
    CatalogScope,
    catalog_provider_from_frames,
)
from codeintel.build.contracts.ref import contract_ref_for_table
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.finalize_helpers import finalize_analytics_rows
from codeintel.build.hamilton.native.patterns import (
    TableTargetContext,
    attach_table_target_template,
    build_single_table_target_spec,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_CONTRACTS_TARGET_NAME = "function_contracts"
FUNCTION_CONTRACTS_TABLE_KEY = "analytics.function_contracts"
FUNCTION_CONTRACTS_CONTRACT = contract_ref_for_table(
    table_key=FUNCTION_CONTRACTS_TABLE_KEY,
    target_name=FUNCTION_CONTRACTS_TARGET_NAME,
    input_name="function_contracts__base",
    required_cols=(),
    clip_column=None,
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
    scope = SnapshotScope.from_snapshot(env.snapshot)
    goids_frame = tabular_to_scoped_table(
        q__core__goids,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    modules_frame = tabular_to_scoped_table(
        q__core__modules,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    docstrings_frame = tabular_to_scoped_table(
        q__core__docstrings,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    function_types_frame = tabular_to_scoped_table(
        q__analytics__function_types,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    catalog = catalog_provider_from_frames(
        CatalogProviderRequest(
            goids_frame=goids_frame,
            modules_frame=modules_frame,
            scope=CatalogScope(
                repo=env.repo,
                commit=env.commit,
                ctx=env.execution_context,
            ),
        )
    )
    worklist = build_function_ast_worklist(
        goids_frame,
        repo=env.repo,
        commit=env.commit,
        ctx=env.execution_context,
    )
    request = FunctionAstLoadRequest(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.snapshot.repo_root,
        catalog_provider=catalog,
        worklist=worklist,
    )
    ast_map, _missing = load_function_asts(request)
    rows = build_function_contracts_rows(
        env.snapshot,
        FunctionContractInputs(
            function_ast_map=ast_map,
            catalog=catalog,
            docstrings_frame=docstrings_frame,
            function_types_frame=function_types_frame,
            ctx=env.execution_context,
        ),
    )
    if not rows:
        return empty_table_for_table(FUNCTION_CONTRACTS_TABLE_KEY)
    return finalize_analytics_rows(FUNCTION_CONTRACTS_TABLE_KEY, rows)


_MODULE = sys.modules[__name__]
_FUNCTION_CONTRACTS_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract_ref(
        contract_ref=FUNCTION_CONTRACTS_CONTRACT,
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
