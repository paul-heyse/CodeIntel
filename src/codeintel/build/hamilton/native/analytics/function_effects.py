"""Function effects table built with inferable tabular nodes."""

from __future__ import annotations

import polars as pl

from codeintel.analytics.functions.function_effects import (
    FunctionEffectsInputs,
    build_function_effects_rows,
)
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
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
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.catalog.service import CatalogService

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_EFFECTS_TARGET_NAME = "function_effects"
FUNCTION_EFFECTS_TABLE_KEY = "analytics.function_effects"
FUNCTION_EFFECTS_SAVE_CONTEXT = SaverContext(
    domain="analytics",
    target=FUNCTION_EFFECTS_TARGET_NAME,
)
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
    _q__core__goids: InferableTabularInput,
    _q__graph__call_graph_edges: InferableTabularInput,
) -> pl.LazyFrame:
    """Build function effects rows using gateway-backed helpers.

    Returns
    -------
    pl.LazyFrame
        Lazy frame with function effects columns.
    """
    catalog = CatalogService.from_db(env.gateway, repo=env.repo, commit=env.commit)
    request = FunctionAstLoadRequest(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.snapshot.repo_root,
        catalog_provider=catalog,
    )
    ast_map, missing = load_function_asts(env.gateway, request)
    inputs = FunctionEffectsInputs(
        catalog_provider=catalog,
        ast_map=ast_map,
        missing_goids=missing,
    )
    rows = build_function_effects_rows(env.gateway, env.snapshot, inputs=inputs)
    return rows_to_frame(FUNCTION_EFFECTS_TABLE_KEY, rows)


@save_dataset(
    context=FUNCTION_EFFECTS_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=FUNCTION_EFFECTS_TABLE_KEY),
)
@table_contract(FUNCTION_EFFECTS_CONTRACT)
def function_effects__table(function_effects__base: pl.LazyFrame) -> pl.LazyFrame:
    """Return the cleaned/enriched function effects frame.

    Returns
    -------
    pl.LazyFrame
        Cleaned/enriched function effects frame.
    """
    return function_effects__base


@codeintel_target(domain="analytics", target=FUNCTION_EFFECTS_TARGET_NAME)
def t__function_effects(
    env: BuildEnv,
    catalog: DagCatalog,
    m__analytics__function_effects: MaterializationResult,
) -> TargetRunRecord:
    """Finalize function_effects target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the function_effects target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=FUNCTION_EFFECTS_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations={
            FUNCTION_EFFECTS_TABLE_KEY: m__analytics__function_effects,
        },
    )


__all__ = [
    "function_effects__base",
    "function_effects__table",
    "t__function_effects",
]
