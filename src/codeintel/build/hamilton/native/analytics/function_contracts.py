"""Function contracts table built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.build.analytics.functions.function_contracts import build_function_contracts_rows
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.utilities.catalogs import catalog_provider_from_frames
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import rows_to_frame
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    SaverContext,
    save_dataset,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_CONTRACTS_TARGET_NAME = "function_contracts"
FUNCTION_CONTRACTS_TABLE_KEY = "analytics.function_contracts"
FUNCTION_CONTRACTS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=FUNCTION_CONTRACTS_TARGET_NAME,
)
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
) -> pl.LazyFrame:
    """Build function contracts rows using tabular inputs.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing function contract rows.
    """
    goids_frame = tabular_to_lazyframe(q__core__goids).collect()
    modules_frame = tabular_to_lazyframe(q__core__modules).collect()
    docstrings_frame = tabular_to_lazyframe(q__core__docstrings).collect()
    function_types_frame = tabular_to_lazyframe(q__analytics__function_types).collect()
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
        function_ast_map=ast_map,
        catalog=catalog,
        docstrings_frame=docstrings_frame,
        function_types_frame=function_types_frame,
    )
    return rows_to_frame(FUNCTION_CONTRACTS_TABLE_KEY, rows)


@save_dataset(
    context=FUNCTION_CONTRACTS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=FUNCTION_CONTRACTS_TABLE_KEY),
)
@table_contract(FUNCTION_CONTRACTS_CONTRACT)
def function_contracts__table(function_contracts__base: pl.LazyFrame) -> pl.LazyFrame:
    """Return the cleaned/enriched function contracts frame.

    Returns
    -------
    pl.LazyFrame
        Cleaned/enriched function contracts frame.
    """
    return function_contracts__base


@codeintel_target(domain="analytics", target=FUNCTION_CONTRACTS_TARGET_NAME)
def t__function_contracts(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__function_contracts: MaterializationResult,
) -> TargetRunRecord:
    """Finalize function_contracts target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the function_contracts target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=FUNCTION_CONTRACTS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            FUNCTION_CONTRACTS_TABLE_KEY: m__analytics__function_contracts,
        },
    )


__all__ = [
    "function_contracts__base",
    "function_contracts__table",
    "t__function_contracts",
]
