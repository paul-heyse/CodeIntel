"""Semantic role analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import json

import polars as pl

from codeintel.build.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.semantic_roles.core import (
    SemanticRolesResult,
    build_semantic_roles_rows,
)
from codeintel.build.analytics.utilities.catalogs import catalog_provider_from_frames
from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import (
    empty_frame_for_table,
    rows_to_frame,
)
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    DatasetSaveSpec,
    SaverContext,
    make_table_materializations_collector,
    save_dataset,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_dataset
from codeintel.build.hamilton.transforms.table_contract import TableContractSpec, table_contract
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.data_models.ids import normalize_decimal_id

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SEMANTIC_ROLES_TARGET_NAME = "semantic_roles"
SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY = "analytics.semantic_roles_functions"
SEMANTIC_ROLES_MODULES_TABLE_KEY = "analytics.semantic_roles_modules"
SEMANTIC_ROLES_TABLE_KEYS = (
    SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY,
    SEMANTIC_ROLES_MODULES_TABLE_KEY,
)
SEMANTIC_ROLES_SAVE_CONTEXT = SaverContext(domain="analytics", target=SEMANTIC_ROLES_TARGET_NAME)
SEMANTIC_ROLES_FUNCTIONS_CONTRACT = TableContractSpec(
    table_key=SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY,
    domain="analytics",
    target=SEMANTIC_ROLES_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="semantic_roles_functions__base",
)
SEMANTIC_ROLES_MODULES_CONTRACT = TableContractSpec(
    table_key=SEMANTIC_ROLES_MODULES_TABLE_KEY,
    domain="analytics",
    target=SEMANTIC_ROLES_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="semantic_roles_modules__base",
)


def _parse_json_list(value: object) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return []


def _module_map(modules_frame: pl.DataFrame) -> dict[str, str]:
    module_map: dict[str, str] = {}
    for row in modules_frame.iter_rows(named=True):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_map[path] = module
    return module_map


def _features_by_goid(features_frame: pl.DataFrame) -> dict[int, FunctionAstFeatures]:
    features_map: dict[int, FunctionAstFeatures] = {}
    for row in features_frame.iter_rows(named=True):
        goid_raw = row.get("function_goid_h128")
        goid_value = normalize_decimal_id(goid_raw)
        if goid_value is None:
            continue
        rel_path = row.get("rel_path")
        qualname = row.get("qualname")
        if not isinstance(rel_path, str) or not isinstance(qualname, str):
            continue
        features_map[int(goid_value)] = FunctionAstFeatures(
            goid=int(goid_value),
            rel_path=rel_path,
            qualname=qualname,
            is_async=bool(row.get("is_async")),
            decorators=tuple(_parse_json_list(row.get("decorators"))),
            imports={},
            libraries_used=frozenset(_parse_json_list(row.get("libraries_used"))),
            io_flags=IoFlags(
                uses_network=bool(row.get("uses_network")),
                uses_db=bool(row.get("uses_db")),
                uses_filesystem=bool(row.get("uses_filesystem")),
                uses_subprocess=bool(row.get("uses_subprocess")),
            ),
            uses_concurrency_lib=bool(row.get("uses_concurrency_lib")),
            uses_threading=bool(row.get("uses_threading")),
            uses_asyncio_lib=bool(row.get("uses_asyncio_lib")),
            http_client_libs=frozenset(_parse_json_list(row.get("http_client_libs"))),
            http_server_libs=frozenset(_parse_json_list(row.get("http_server_libs"))),
            db_libs=frozenset(_parse_json_list(row.get("db_libs"))),
            message_libs=frozenset(_parse_json_list(row.get("message_libs"))),
            config_read_count=int(row.get("config_read_count") or 0),
            feature_flag_count=int(row.get("feature_flag_count") or 0),
            extra={},
        )
    return features_map


def semantic_roles_result(
    env: BuildEnv,
    q__core__modules: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__analytics__function_ast_features: InferableTabularInput,
    q__analytics__function_metrics: InferableTabularInput,
    q__analytics__function_effects: InferableTabularInput,
    q__analytics__function_contracts: InferableTabularInput,
    q__analytics__graph_metrics_functions: InferableTabularInput,
) -> SemanticRolesResult:
    """Compute semantic role rows for functions and modules.

    Returns
    -------
    SemanticRolesResult
        Container with semantic role rows for functions and modules.
    """
    modules_frame = tabular_to_lazyframe(q__core__modules).collect()
    goids_frame = tabular_to_lazyframe(q__core__goids).collect()
    features_frame = tabular_to_lazyframe(q__analytics__function_ast_features).collect()
    function_metrics_frame = tabular_to_lazyframe(q__analytics__function_metrics).collect()
    function_effects_frame = tabular_to_lazyframe(q__analytics__function_effects).collect()
    function_contracts_frame = tabular_to_lazyframe(q__analytics__function_contracts).collect()
    graph_metrics_frame = tabular_to_lazyframe(q__analytics__graph_metrics_functions).collect()
    module_map = _module_map(modules_frame)
    if not module_map:
        return SemanticRolesResult(function_rows=[], module_rows=[])
    catalog = catalog_provider_from_frames(goids_frame=goids_frame, modules_frame=modules_frame)
    request = FunctionAstLoadRequest(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.snapshot.repo_root,
        catalog_provider=catalog,
    )
    ast_map, _missing = load_function_asts(request)
    return build_semantic_roles_rows(
        env.snapshot,
        module_by_path=module_map,
        ast_map=ast_map,
        features_map=_features_by_goid(features_frame),
        function_metrics_frame=function_metrics_frame,
        function_effects_frame=function_effects_frame,
        function_contracts_frame=function_contracts_frame,
        graph_metrics_frame=graph_metrics_frame,
        modules_frame=modules_frame,
    )


def semantic_roles_functions__base(semantic_roles_result: SemanticRolesResult) -> pl.LazyFrame:
    """Build semantic role rows for functions.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing semantic role rows for functions.
    """
    if not semantic_roles_result.function_rows:
        return empty_frame_for_table(SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY)
    return rows_to_frame(SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY, semantic_roles_result.function_rows)


@save_dataset(
    context=SEMANTIC_ROLES_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=SEMANTIC_ROLES_TARGET_NAME,
    table_key=SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY,
)
@table_contract(SEMANTIC_ROLES_FUNCTIONS_CONTRACT)
def semantic_roles_functions__table(
    semantic_roles_functions__base: pl.LazyFrame,
) -> pl.LazyFrame:
    """Persist semantic role rows for functions.

    Returns
    -------
    pl.LazyFrame
        Persisted semantic roles functions frame.
    """
    return semantic_roles_functions__base


def semantic_roles_modules__base(semantic_roles_result: SemanticRolesResult) -> pl.LazyFrame:
    """Build semantic role rows for modules.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing semantic role rows for modules.
    """
    if not semantic_roles_result.module_rows:
        return empty_frame_for_table(SEMANTIC_ROLES_MODULES_TABLE_KEY)
    return rows_to_frame(SEMANTIC_ROLES_MODULES_TABLE_KEY, semantic_roles_result.module_rows)


@save_dataset(
    context=SEMANTIC_ROLES_SAVE_CONTEXT,
    spec=DatasetSaveSpec(table_key=SEMANTIC_ROLES_MODULES_TABLE_KEY),
)
@tag_dataset(
    domain="analytics",
    target=SEMANTIC_ROLES_TARGET_NAME,
    table_key=SEMANTIC_ROLES_MODULES_TABLE_KEY,
)
@table_contract(SEMANTIC_ROLES_MODULES_CONTRACT)
def semantic_roles_modules__table(semantic_roles_modules__base: pl.LazyFrame) -> pl.LazyFrame:
    """Persist semantic role rows for modules.

    Returns
    -------
    pl.LazyFrame
        Persisted semantic roles modules frame.
    """
    return semantic_roles_modules__base


semantic_roles__table_materializations = make_table_materializations_collector(
    domain="analytics",
    target=SEMANTIC_ROLES_TARGET_NAME,
    table_keys=SEMANTIC_ROLES_TABLE_KEYS,
    node_name="semantic_roles__table_materializations",
)


@codeintel_target(domain="analytics", target=SEMANTIC_ROLES_TARGET_NAME)
def t__semantic_roles(
    env: BuildEnv,
    catalog: DagCatalog,
    semantic_roles__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize semantic roles target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the semantic roles target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=SEMANTIC_ROLES_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=semantic_roles__table_materializations,
    )


__all__ = [
    "semantic_roles__table_materializations",
    "semantic_roles_functions__base",
    "semantic_roles_functions__table",
    "semantic_roles_modules__base",
    "semantic_roles_modules__table",
    "t__semantic_roles",
]
