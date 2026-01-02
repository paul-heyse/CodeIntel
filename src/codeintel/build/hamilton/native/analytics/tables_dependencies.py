"""Dependency analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import json
import sys

import polars as pl

from codeintel.build.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.build.analytics.dependencies.compute import (
    EXTERNAL_DEPENDENCIES_COLS,
    EXTERNAL_DEPENDENCY_CALLS_COLS,
    compute_dependency_calls_pure,
    compute_external_dependencies_pure,
)
from codeintel.build.analytics.dependencies.core import ExternalDependencyInputs
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.utilities.catalogs import catalog_provider_from_frames
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.analytics.table_utils import (
    empty_frame_for_table,
    rows_to_frame,
)
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
from codeintel.core.data_models.ids import normalize_decimal_id

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

EXTERNAL_DEPS_TARGET_NAME = "external_deps"
EXTERNAL_DEPENDENCIES_TABLE_KEY = "analytics.external_dependencies"
EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY = "analytics.external_dependency_calls"
EXTERNAL_DEPENDENCIES_CONTRACT = TableContractSpec(
    table_key=EXTERNAL_DEPENDENCIES_TABLE_KEY,
    domain="analytics",
    target=EXTERNAL_DEPS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="external_dependencies__base",
)
EXTERNAL_DEPENDENCY_CALLS_CONTRACT = TableContractSpec(
    table_key=EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY,
    domain="analytics",
    target=EXTERNAL_DEPS_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="external_dependency_calls__base",
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


def external_dependency_calls__base(
    env: BuildEnv,
    q__core__modules: InferableTabularInput,
    q__analytics__function_ast_features: InferableTabularInput,
    q__core__goids: InferableTabularInput,
) -> pl.LazyFrame:
    """Build external dependency call rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing external dependency call rows.
    """
    modules_frame = tabular_to_lazyframe(q__core__modules).collect()
    goids_frame = tabular_to_lazyframe(q__core__goids).collect()
    features_frame = tabular_to_lazyframe(q__analytics__function_ast_features).collect()
    module_map = _module_map(modules_frame)
    if not module_map:
        return empty_frame_for_table(EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY)
    catalog = catalog_provider_from_frames(goids_frame=goids_frame, modules_frame=modules_frame)
    request = FunctionAstLoadRequest(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.snapshot.repo_root,
        catalog_provider=catalog,
    )
    ast_map, missing = load_function_asts(request)
    inputs = ExternalDependencyInputs(
        catalog_provider=catalog,
        module_map=module_map,
        ast_by_goid=ast_map,
        features_map=_features_by_goid(features_frame),
        missing_goids=missing,
    )
    result = compute_dependency_calls_pure(env.snapshot, inputs)
    return rows_to_frame(
        EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY,
        result.rows,
        columns=EXTERNAL_DEPENDENCY_CALLS_COLS,
    )


def external_dependencies__base(
    env: BuildEnv,
    q__analytics__external_dependency_calls: InferableTabularInput,
    q__analytics__config_values: InferableTabularInput,
) -> pl.LazyFrame:
    """Build external dependencies summary rows.

    Returns
    -------
    pl.LazyFrame
        Lazy frame containing external dependency summary rows.
    """
    dependency_calls_frame = tabular_to_lazyframe(q__analytics__external_dependency_calls).collect()
    config_values_frame = tabular_to_lazyframe(q__analytics__config_values).collect()
    result = compute_external_dependencies_pure(
        env.snapshot,
        dependency_calls_frame=dependency_calls_frame,
        config_values_frame=config_values_frame,
    )
    return rows_to_frame(
        EXTERNAL_DEPENDENCIES_TABLE_KEY,
        result.rows,
        columns=EXTERNAL_DEPENDENCIES_COLS,
    )


_MODULE = sys.modules[__name__]
_EXTERNAL_DEPS_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=EXTERNAL_DEPS_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY,
            base_node="external_dependency_calls__base",
            contract=EXTERNAL_DEPENDENCY_CALLS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=EXTERNAL_DEPENDENCY_CALLS_TABLE_KEY),
            node_name="external_dependency_calls__table",
        ),
        TableTargetTableSpec(
            table_key=EXTERNAL_DEPENDENCIES_TABLE_KEY,
            base_node="external_dependencies__base",
            contract=EXTERNAL_DEPENDENCIES_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=EXTERNAL_DEPENDENCIES_TABLE_KEY),
            node_name="external_dependencies__table",
        ),
    ),
    table_materializations_node="external_deps__table_materializations",
    anchor_node_name="t__external_deps",
)
attach_table_target_template(_MODULE, spec=_EXTERNAL_DEPS_TABLE_TARGET_SPEC)
external_dependency_calls__table = _MODULE.external_dependency_calls__table
external_dependencies__table = _MODULE.external_dependencies__table
external_deps__table_materializations = _MODULE.external_deps__table_materializations
t__external_deps = _MODULE.t__external_deps


__all__ = [
    "external_dependencies__base",
    "external_dependencies__table",
    "external_dependency_calls__base",
    "external_dependency_calls__table",
    "external_deps__table_materializations",
    "t__external_deps",
]
