"""Semantic role analytics tables built with inferable tabular nodes."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass

import pyarrow as pa
from hamilton.function_modifiers import cache

from codeintel.build.analytics.ast_features.model import FunctionAstFeatures, IoFlags
from codeintel.build.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.analytics.parsing.worklists import build_function_ast_worklist
from codeintel.build.analytics.semantic_roles.core import (
    SemanticRoleInputs,
    SemanticRolesResult,
    build_semantic_roles_rows,
)
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
    MultiTableTargetContext,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec_from_contexts,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.core.query_results import coerce_optional_int

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

SEMANTIC_ROLES_TARGET_NAME = "semantic_roles"
SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY = "analytics.semantic_roles_functions"
SEMANTIC_ROLES_MODULES_TABLE_KEY = "analytics.semantic_roles_modules"
SEMANTIC_ROLES_FUNCTIONS_CONTRACT = contract_ref_for_table(
    table_key=SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY,
    target_name=SEMANTIC_ROLES_TARGET_NAME,
    input_name="semantic_roles_functions__base",
    required_cols=(),
    clip_column=None,
)
SEMANTIC_ROLES_MODULES_CONTRACT = contract_ref_for_table(
    table_key=SEMANTIC_ROLES_MODULES_TABLE_KEY,
    target_name=SEMANTIC_ROLES_TARGET_NAME,
    input_name="semantic_roles_modules__base",
    required_cols=(),
    clip_column=None,
)


@dataclass(frozen=True)
class SemanticRoleModuleFrames:
    modules_frame: pa.Table
    goids_frame: pa.Table
    features_frame: pa.Table


@dataclass(frozen=True)
class SemanticRoleEffectFrames:
    function_effects_frame: pa.Table
    function_contracts_frame: pa.Table


@dataclass(frozen=True)
class SemanticRoleGraphFrames:
    graph_metrics_frame: pa.Table


def semantic_role_module_frames(
    env: BuildEnv,
    q__core__modules: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__analytics__function_ast_features: InferableTabularInput,
) -> SemanticRoleModuleFrames:
    scope = SnapshotScope.from_snapshot(env.snapshot)
    return SemanticRoleModuleFrames(
        modules_frame=tabular_to_scoped_table(
            q__core__modules,
            columns=None,
            scope=scope,
            require_scope_columns=True,
        ),
        goids_frame=tabular_to_scoped_table(
            q__core__goids,
            columns=None,
            scope=scope,
            require_scope_columns=True,
        ),
        features_frame=tabular_to_scoped_table(
            q__analytics__function_ast_features,
            columns=None,
            scope=scope,
            require_scope_columns=True,
        ),
    )


def semantic_role_effect_frames(
    env: BuildEnv,
    q__analytics__function_effects: InferableTabularInput,
    q__analytics__function_contracts: InferableTabularInput,
) -> SemanticRoleEffectFrames:
    scope = SnapshotScope.from_snapshot(env.snapshot)
    return SemanticRoleEffectFrames(
        function_effects_frame=tabular_to_scoped_table(
            q__analytics__function_effects,
            columns=None,
            scope=scope,
            require_scope_columns=True,
        ),
        function_contracts_frame=tabular_to_scoped_table(
            q__analytics__function_contracts,
            columns=None,
            scope=scope,
            require_scope_columns=True,
        ),
    )


def semantic_role_graph_frames(
    env: BuildEnv,
    q__analytics__graph_metrics_functions: InferableTabularInput,
) -> SemanticRoleGraphFrames:
    scope = SnapshotScope.from_snapshot(env.snapshot)
    return SemanticRoleGraphFrames(
        graph_metrics_frame=tabular_to_scoped_table(
            q__analytics__graph_metrics_functions,
            columns=None,
            scope=scope,
            require_scope_columns=True,
        ),
    )


def _extras_list(extras: Mapping[str, object] | None, key: str) -> list[str]:
    if extras is None:
        return []
    value = extras.get(key)
    if isinstance(value, list):
        return [str(item) for item in value]
    return []


def _module_map(modules_frame: pa.Table) -> dict[str, str]:
    module_map: dict[str, str] = {}
    if modules_frame.num_rows == 0:
        return module_map
    if not {"path", "module"}.issubset(set(modules_frame.column_names)):
        return module_map
    for row in iter_rows(modules_frame, ["path", "module"]):
        path = row.get("path")
        module = row.get("module")
        if isinstance(path, str) and isinstance(module, str):
            module_map[path] = module
    return module_map


def _features_by_goid(features_frame: pa.Table) -> dict[int, FunctionAstFeatures]:
    features_map: dict[int, FunctionAstFeatures] = {}
    if features_frame.num_rows == 0:
        return features_map
    required = {"function_goid_h128", "rel_path", "qualname"}
    if not required.issubset(set(features_frame.column_names)):
        return features_map
    columns = [
        "function_goid_h128",
        "rel_path",
        "qualname",
        "is_async",
        "uses_network",
        "uses_db",
        "uses_filesystem",
        "uses_subprocess",
        "uses_concurrency_lib",
        "uses_threading",
        "uses_asyncio_lib",
        "config_read_count",
        "feature_flag_count",
        "extras",
    ]
    for row in iter_rows(features_frame, columns):
        feature = _feature_from_row(row)
        if feature is None:
            continue
        features_map[feature.goid] = feature
    return features_map


def _feature_from_row(row: dict[str, object]) -> FunctionAstFeatures | None:
    goid_value = normalize_decimal_id(row.get("function_goid_h128"))
    rel_path = row.get("rel_path")
    qualname = row.get("qualname")
    if goid_value is None or not isinstance(rel_path, str) or not isinstance(qualname, str):
        return None
    extras = row.get("extras")
    extras_map = extras if isinstance(extras, Mapping) else None
    return FunctionAstFeatures(
        goid=int(goid_value),
        rel_path=rel_path,
        qualname=qualname,
        is_async=bool(row.get("is_async")),
        decorators=tuple(_extras_list(extras_map, "decorators")),
        imports={},
        libraries_used=frozenset(_extras_list(extras_map, "libraries_used")),
        io_flags=IoFlags(
            uses_network=bool(row.get("uses_network")),
            uses_db=bool(row.get("uses_db")),
            uses_filesystem=bool(row.get("uses_filesystem")),
            uses_subprocess=bool(row.get("uses_subprocess")),
        ),
        uses_concurrency_lib=bool(row.get("uses_concurrency_lib")),
        uses_threading=bool(row.get("uses_threading")),
        uses_asyncio_lib=bool(row.get("uses_asyncio_lib")),
        http_client_libs=frozenset(_extras_list(extras_map, "http_client_libs")),
        http_server_libs=frozenset(_extras_list(extras_map, "http_server_libs")),
        db_libs=frozenset(_extras_list(extras_map, "db_libs")),
        message_libs=frozenset(_extras_list(extras_map, "message_libs")),
        config_read_count=coerce_optional_int(
            row.get("config_read_count"),
            ctx="semantic_roles.config_read_count",
        )
        or 0,
        feature_flag_count=coerce_optional_int(
            row.get("feature_flag_count"),
            ctx="semantic_roles.feature_flag_count",
        )
        or 0,
        extra={},
    )


@cache(behavior="default")
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
        CatalogProviderRequest(
            goids_frame=semantic_role_module_frames.goids_frame,
            modules_frame=semantic_role_module_frames.modules_frame,
            scope=CatalogScope(
                repo=env.repo,
                commit=env.commit,
                ctx=env.execution_context,
            ),
        )
    )
    worklist = build_function_ast_worklist(
        semantic_role_module_frames.goids_frame,
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
            ctx=env.execution_context,
        ),
    )


def semantic_roles_functions__base(
    semantic_roles_result: SemanticRolesResult,
) -> pa.Table:
    """Build semantic role rows for functions.

    Returns
    -------
    pa.Table
        Reader containing semantic role rows for functions.
    """
    if not semantic_roles_result.function_rows:
        return empty_table_for_table(SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY)
    return finalize_analytics_rows(
        SEMANTIC_ROLES_FUNCTIONS_TABLE_KEY,
        semantic_roles_result.function_rows,
    )


def semantic_roles_modules__base(
    semantic_roles_result: SemanticRolesResult,
) -> pa.Table:
    """Build semantic role rows for modules.

    Returns
    -------
    pa.Table
        Reader containing semantic role rows for modules.
    """
    if not semantic_roles_result.module_rows:
        return empty_table_for_table(SEMANTIC_ROLES_MODULES_TABLE_KEY)
    return finalize_analytics_rows(
        SEMANTIC_ROLES_MODULES_TABLE_KEY,
        semantic_roles_result.module_rows,
    )


_MODULE = sys.modules[__name__]
_SEMANTIC_ROLES_TABLE_CONTEXTS = (
    TableTargetTableContext.from_contract_ref(
        contract_ref=SEMANTIC_ROLES_FUNCTIONS_CONTRACT,
        node_name="semantic_roles_functions__table",
        input_type=pa.Table,
    ),
    TableTargetTableContext.from_contract_ref(
        contract_ref=SEMANTIC_ROLES_MODULES_CONTRACT,
        node_name="semantic_roles_modules__table",
        input_type=pa.Table,
    ),
)
_SEMANTIC_ROLES_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="analytics",
        target_name=SEMANTIC_ROLES_TARGET_NAME,
        tables=(),
        table_materializations_node="semantic_roles__table_materializations",
        anchor_node_name="t__semantic_roles",
    ),
    table_contexts=_SEMANTIC_ROLES_TABLE_CONTEXTS,
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
