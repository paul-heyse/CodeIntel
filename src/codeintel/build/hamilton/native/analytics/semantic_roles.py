"""Semantic role analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import polars as pl
from hamilton.function_modifiers import cache

from codeintel.build.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.semantic_roles.core import (
    SemanticRoleInputs,
    SemanticRolesResult,
    build_semantic_roles_rows,
)
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
from codeintel.build.tabular.conversion import tabular_to_frame
from codeintel.build.tabular.frames import (
    empty_frame_for_table,
    rows_to_frame,
)
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.serialization.json import decode_json_list
from codeintel.core.serialization.payload import decode_payload

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SEMANTIC_ROLES_TARGET_NAME = "semantic_roles"
SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY = "analytics.semantic_roles_functions"
SEMANTIC_ROLES_MODULES_TABLE_KEY = "analytics.semantic_roles_modules"
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


@dataclass(frozen=True)
class SemanticRoleModuleFrames:
    modules_frame: pl.DataFrame
    goids_frame: pl.DataFrame
    features_frame: pl.DataFrame


@dataclass(frozen=True)
class SemanticRoleEffectFrames:
    function_effects_frame: pl.DataFrame
    function_contracts_frame: pl.DataFrame


@dataclass(frozen=True)
class SemanticRoleGraphFrames:
    graph_metrics_frame: pl.DataFrame


def semantic_role_module_frames(
    q__core__modules: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__analytics__function_ast_features: InferableTabularInput,
) -> SemanticRoleModuleFrames:
    return SemanticRoleModuleFrames(
        modules_frame=tabular_to_frame(q__core__modules),
        goids_frame=tabular_to_frame(q__core__goids),
        features_frame=tabular_to_frame(q__analytics__function_ast_features),
    )


def semantic_role_effect_frames(
    q__analytics__function_effects: InferableTabularInput,
    q__analytics__function_contracts: InferableTabularInput,
) -> SemanticRoleEffectFrames:
    return SemanticRoleEffectFrames(
        function_effects_frame=tabular_to_frame(q__analytics__function_effects),
        function_contracts_frame=tabular_to_frame(q__analytics__function_contracts),
    )


def semantic_role_graph_frames(
    q__analytics__graph_metrics_functions: InferableTabularInput,
) -> SemanticRoleGraphFrames:
    return SemanticRoleGraphFrames(
        graph_metrics_frame=tabular_to_frame(q__analytics__graph_metrics_functions),
    )


def _parse_json_list(value: object) -> list[str]:
    decoded = decode_payload(value)
    if isinstance(decoded, list):
        return [str(item) for item in decoded]
    if isinstance(decoded, str):
        return [str(item) for item in decode_json_list(decoded)]
    return []


def _module_map(modules_frame: pl.DataFrame) -> dict[str, str]:
    module_map: dict[str, str] = {}
    if modules_frame.is_empty():
        return module_map
    if not {"path", "module"}.issubset(set(modules_frame.columns)):
        return module_map
    data = modules_frame.select(["path", "module"]).to_dict(as_series=False)
    module_map.update(
        {
            path: module
            for path, module in zip(data["path"], data["module"], strict=True)
            if isinstance(path, str) and isinstance(module, str)
        }
    )
    return module_map


def _features_by_goid(features_frame: pl.DataFrame) -> dict[int, FunctionAstFeatures]:
    features_map: dict[int, FunctionAstFeatures] = {}
    if features_frame.is_empty():
        return features_map
    required = {"function_goid_h128", "rel_path", "qualname"}
    if not required.issubset(set(features_frame.columns)):
        return features_map
    columns = [
        "function_goid_h128",
        "rel_path",
        "qualname",
        "is_async",
        "decorators",
        "libraries_used",
        "uses_network",
        "uses_db",
        "uses_filesystem",
        "uses_subprocess",
        "uses_concurrency_lib",
        "uses_threading",
        "uses_asyncio_lib",
        "http_client_libs",
        "http_server_libs",
        "db_libs",
        "message_libs",
        "config_read_count",
        "feature_flag_count",
    ]
    data = features_frame.select(columns).to_dict(as_series=False)
    for (
        goid_raw,
        rel_path,
        qualname,
        is_async,
        decorators,
        libraries_used,
        uses_network,
        uses_db,
        uses_filesystem,
        uses_subprocess,
        uses_concurrency_lib,
        uses_threading,
        uses_asyncio_lib,
        http_client_libs,
        http_server_libs,
        db_libs,
        message_libs,
        config_read_count,
        feature_flag_count,
    ) in zip(
        data["function_goid_h128"],
        data["rel_path"],
        data["qualname"],
        data["is_async"],
        data["decorators"],
        data["libraries_used"],
        data["uses_network"],
        data["uses_db"],
        data["uses_filesystem"],
        data["uses_subprocess"],
        data["uses_concurrency_lib"],
        data["uses_threading"],
        data["uses_asyncio_lib"],
        data["http_client_libs"],
        data["http_server_libs"],
        data["db_libs"],
        data["message_libs"],
        data["config_read_count"],
        data["feature_flag_count"],
        strict=True,
    ):
        goid_value = normalize_decimal_id(goid_raw)
        if goid_value is None:
            continue
        if not isinstance(rel_path, str) or not isinstance(qualname, str):
            continue
        features_map[int(goid_value)] = FunctionAstFeatures(
            goid=int(goid_value),
            rel_path=rel_path,
            qualname=qualname,
            is_async=bool(is_async),
            decorators=tuple(_parse_json_list(decorators)),
            imports={},
            libraries_used=frozenset(_parse_json_list(libraries_used)),
            io_flags=IoFlags(
                uses_network=bool(uses_network),
                uses_db=bool(uses_db),
                uses_filesystem=bool(uses_filesystem),
                uses_subprocess=bool(uses_subprocess),
            ),
            uses_concurrency_lib=bool(uses_concurrency_lib),
            uses_threading=bool(uses_threading),
            uses_asyncio_lib=bool(uses_asyncio_lib),
            http_client_libs=frozenset(_parse_json_list(http_client_libs)),
            http_server_libs=frozenset(_parse_json_list(http_server_libs)),
            db_libs=frozenset(_parse_json_list(db_libs)),
            message_libs=frozenset(_parse_json_list(message_libs)),
            config_read_count=int(config_read_count or 0),
            feature_flag_count=int(feature_flag_count or 0),
            extra={},
        )
    return features_map


@cache()
def semantic_roles_result(
    env: BuildEnv,
    semantic_role_module_frames: SemanticRoleModuleFrames,
    semantic_role_effect_frames: SemanticRoleEffectFrames,
    semantic_role_graph_frames: SemanticRoleGraphFrames,
) -> SemanticRolesResult:
    """Compute semantic role rows for functions and modules.

    Returns
    -------
    SemanticRolesResult
        Container with semantic role rows for functions and modules.
    """
    module_map = _module_map(semantic_role_module_frames.modules_frame)
    if not module_map:
        return SemanticRolesResult(function_rows=[], module_rows=[])
    catalog = catalog_provider_from_frames(
        goids_frame=semantic_role_module_frames.goids_frame,
        modules_frame=semantic_role_module_frames.modules_frame,
    )
    request = FunctionAstLoadRequest(
        repo=env.repo,
        commit=env.commit,
        repo_root=env.snapshot.repo_root,
        catalog_provider=catalog,
    )
    ast_map, _missing = load_function_asts(request)
    return build_semantic_roles_rows(
        env.snapshot,
        SemanticRoleInputs(
            module_by_path=module_map,
            ast_map=ast_map,
            features_map=_features_by_goid(semantic_role_module_frames.features_frame),
            goids_frame=semantic_role_module_frames.goids_frame,
            function_effects_frame=semantic_role_effect_frames.function_effects_frame,
            function_contracts_frame=semantic_role_effect_frames.function_contracts_frame,
            graph_metrics_frame=semantic_role_graph_frames.graph_metrics_frame,
            modules_frame=semantic_role_module_frames.modules_frame,
        ),
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


_MODULE = sys.modules[__name__]
_SEMANTIC_ROLES_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=SEMANTIC_ROLES_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY,
            base_node="semantic_roles_functions__base",
            contract=SEMANTIC_ROLES_FUNCTIONS_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY),
            node_name="semantic_roles_functions__table",
        ),
        TableTargetTableSpec(
            table_key=SEMANTIC_ROLES_MODULES_TABLE_KEY,
            base_node="semantic_roles_modules__base",
            contract=SEMANTIC_ROLES_MODULES_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=SEMANTIC_ROLES_MODULES_TABLE_KEY),
            node_name="semantic_roles_modules__table",
        ),
    ),
    table_materializations_node="semantic_roles__table_materializations",
    anchor_node_name="t__semantic_roles",
)
attach_table_target_template(_MODULE, spec=_SEMANTIC_ROLES_TABLE_TARGET_SPEC)
semantic_roles_functions__table = _MODULE.semantic_roles_functions__table
semantic_roles_modules__table = _MODULE.semantic_roles_modules__table
semantic_roles__table_materializations = _MODULE.semantic_roles__table_materializations
t__semantic_roles = _MODULE.t__semantic_roles


__all__ = [
    "semantic_roles__table_materializations",
    "semantic_roles_functions__base",
    "semantic_roles_functions__table",
    "semantic_roles_modules__base",
    "semantic_roles_modules__table",
    "t__semantic_roles",
]
