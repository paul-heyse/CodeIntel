"""Function AST feature tables built with inferable tabular nodes."""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow as pa

from codeintel.build.analytics.ast_features.extract import compute_function_features
from codeintel.build.analytics.compute.row_builders import row_tuple_for_table
from codeintel.build.analytics.parsing.ast_cache import FunctionAst
from codeintel.build.analytics.parsing.worklists import build_function_ast_worklist
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
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_AST_FEATURES_TARGET_NAME = "function_ast_features"
FUNCTION_AST_FEATURES_TABLE_KEY = "analytics.function_ast_features"
FUNCTION_AST_FEATURES_CONTRACT = contract_ref_for_table(
    table_key=FUNCTION_AST_FEATURES_TABLE_KEY,
    target_name=FUNCTION_AST_FEATURES_TARGET_NAME,
    input_name="function_ast_features__base",
    required_cols=(),
    clip_column=None,
)


@dataclass(frozen=True, slots=True)
class _FunctionNodeInfo:
    qualname: str
    start_line: int
    end_line: int
    node: ast.FunctionDef | ast.AsyncFunctionDef


def _collect_function_nodes(tree: ast.AST, module_name: str) -> list[_FunctionNodeInfo]:
    results: list[_FunctionNodeInfo] = []
    scope: list[str] = []

    def _visit(node: ast.AST) -> None:
        if isinstance(node, ast.ClassDef):
            scope.append(node.name)
            for child in node.body:
                _visit(child)
            scope.pop()
            return
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = ".".join([module_name, *scope, node.name]) if module_name else node.name
            start_line = getattr(node, "lineno", None)
            end_line = getattr(node, "end_lineno", None)
            if start_line is not None and end_line is not None:
                results.append(
                    _FunctionNodeInfo(
                        qualname=qualname,
                        start_line=start_line,
                        end_line=end_line,
                        node=node,
                    )
                )
            scope.append(node.name)
            for child in node.body:
                _visit(child)
            scope.pop()
            return
        if isinstance(node, ast.Module):
            for child in node.body:
                _visit(child)

    _visit(tree)
    return results


@dataclass(frozen=True, slots=True)
class _FeatureRowRequest:
    repo: str
    commit: str
    goid: object
    rel_path: str | None
    qualname: str | None
    created_at: object | None
    start_line: object | None = None
    end_line: object | None = None
    nodes_by_path: dict[str, dict[tuple[str, int], _FunctionNodeInfo]] | None = None
    lines_by_path: dict[str, list[str]] | None = None
    trees_by_path: dict[str, ast.AST] | None = None
    repo_root: Path | None = None


def _default_feature_row(request: _FeatureRowRequest) -> dict[str, object]:
    return {
        "repo": request.repo,
        "commit": request.commit,
        "function_goid_h128": request.goid,
        "rel_path": request.rel_path,
        "qualname": request.qualname,
        "is_async": False,
        "uses_network": False,
        "uses_db": False,
        "uses_filesystem": False,
        "uses_subprocess": False,
        "uses_concurrency_lib": False,
        "uses_threading": False,
        "uses_asyncio_lib": False,
        "config_read_count": 0,
        "feature_flag_count": 0,
        "extras": {
            "http_client_libs": [],
            "http_server_libs": [],
            "db_libs": [],
            "message_libs": [],
            "decorators": [],
            "libraries_used": [],
        },
        "created_at": request.created_at,
    }


def _module_by_path(modules_frame: pa.Table) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    columns = ["path", "module"]
    include_language = "language" in modules_frame.column_names
    if include_language:
        columns.append("language")
    for row in iter_tuples(modules_frame.to_reader(), columns=tuple(columns)):
        rel_path = row[0]
        module_name = row[1]
        language = row[2] if include_language else None
        if language not in {None, "python"}:
            continue
        if isinstance(rel_path, str) and isinstance(module_name, str):
            module_by_path[rel_path] = module_name
    return module_by_path


def _load_module_nodes(
    env: BuildEnv,
    module_by_path: dict[str, str],
) -> tuple[
    dict[str, dict[tuple[str, int], _FunctionNodeInfo]],
    dict[str, list[str]],
    dict[str, ast.AST],
]:
    nodes_by_path: dict[str, dict[tuple[str, int], _FunctionNodeInfo]] = {}
    lines_by_path: dict[str, list[str]] = {}
    trees_by_path: dict[str, ast.AST] = {}
    repo_root = Path(env.snapshot.repo_root)
    for rel_path, module_name in module_by_path.items():
        module_path = repo_root / rel_path
        parsed = parse_python_module(module_path)
        if parsed is None:
            continue
        lines, tree = parsed
        lines_by_path[rel_path] = lines
        trees_by_path[rel_path] = tree
        node_map: dict[tuple[str, int], _FunctionNodeInfo] = {}
        for info in _collect_function_nodes(tree, module_name):
            node_map[info.qualname, info.start_line] = info
        nodes_by_path[rel_path] = node_map
    return nodes_by_path, lines_by_path, trees_by_path


def _feature_row_from_worklist(request: _FeatureRowRequest) -> dict[str, object]:
    rel_path = request.rel_path
    qualname = request.qualname
    if not isinstance(rel_path, str) or not isinstance(qualname, str):
        return _default_feature_row(
            replace(
                request,
                rel_path=rel_path if isinstance(rel_path, str) else None,
                qualname=qualname if isinstance(qualname, str) else None,
            )
        )
    if not isinstance(request.start_line, int):
        return _default_feature_row(
            replace(request, rel_path=rel_path, qualname=qualname)
        )
    nodes_by_path = request.nodes_by_path or {}
    lines_by_path = request.lines_by_path or {}
    trees_by_path = request.trees_by_path or {}
    info = nodes_by_path.get(rel_path, {}).get((qualname, request.start_line))
    if info is None:
        return _default_feature_row(
            replace(request, rel_path=rel_path, qualname=qualname)
        )
    goid_id = normalize_decimal_id(request.goid)
    if goid_id is None:
        return _default_feature_row(
            replace(request, rel_path=rel_path, qualname=qualname)
        )
    lines = lines_by_path.get(rel_path, [])
    fn = FunctionAst(
        goid=goid_id,
        rel_path=rel_path,
        qualname=qualname,
        start_line=request.start_line,
        end_line=info.end_line,
        node=info.node,
        lines=lines,
    )
    try:
        features = compute_function_features(
            fn,
            repo_root=request.repo_root or Path(),
            module_tree=trees_by_path.get(rel_path),
        )
    except (SyntaxError, ValueError, TypeError):
        return _default_feature_row(
            replace(request, rel_path=rel_path, qualname=qualname)
        )
    return {
        "repo": request.repo,
        "commit": request.commit,
        "function_goid_h128": request.goid,
        "rel_path": features.rel_path,
        "qualname": features.qualname,
        "is_async": features.is_async,
        "uses_network": features.io_flags.uses_network,
        "uses_db": features.io_flags.uses_db,
        "uses_filesystem": features.io_flags.uses_filesystem,
        "uses_subprocess": features.io_flags.uses_subprocess,
        "uses_concurrency_lib": features.uses_concurrency_lib,
        "uses_threading": features.uses_threading,
        "uses_asyncio_lib": features.uses_asyncio_lib,
        "config_read_count": features.config_read_count,
        "feature_flag_count": features.feature_flag_count,
        "extras": {
            "http_client_libs": sorted(features.http_client_libs),
            "http_server_libs": sorted(features.http_server_libs),
            "db_libs": sorted(features.db_libs),
            "message_libs": sorted(features.message_libs),
            "decorators": list(features.decorators),
            "libraries_used": sorted(features.libraries_used),
        },
        "created_at": request.created_at,
    }


def function_ast_features__base(
    env: BuildEnv,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> pa.Table:
    """Build function AST features using parsed source files.

    Returns
    -------
    pa.Table
        Reader with function AST feature rows.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    goids = tabular_to_scoped_table(
        q__core__goids,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    if goids.num_rows == 0:
        return empty_table_for_table(FUNCTION_AST_FEATURES_TABLE_KEY)

    modules = tabular_to_scoped_table(
        q__core__modules,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    module_by_path = _module_by_path(modules)
    nodes_by_path, lines_by_path, trees_by_path = _load_module_nodes(env, module_by_path)

    worklist = build_function_ast_worklist(
        goids,
        repo=env.repo,
        commit=env.commit,
        ctx=env.execution_context,
    )
    if worklist.num_rows == 0:
        return empty_table_for_table(FUNCTION_AST_FEATURES_TABLE_KEY)
    rows: list[tuple[object, ...]] = []
    repo_root = Path(env.snapshot.repo_root)
    for row in iter_tuples(
        worklist.to_reader(),
        columns=(
            "goid_h128",
            "rel_path",
            "qualname",
            "start_line",
            "end_line",
            "created_at",
        ),
    ):
        rel_path = row[1] if isinstance(row[1], str) else None
        qualname = row[2] if isinstance(row[2], str) else None
        row_dict = _feature_row_from_worklist(
            _FeatureRowRequest(
                repo=env.repo,
                commit=env.commit,
                goid=row[0],
                rel_path=rel_path,
                qualname=qualname,
                start_line=row[3],
                end_line=row[4],
                created_at=row[5],
                nodes_by_path=nodes_by_path,
                lines_by_path=lines_by_path,
                trees_by_path=trees_by_path,
                repo_root=repo_root,
            )
        )
        rows.append(row_tuple_for_table(FUNCTION_AST_FEATURES_TABLE_KEY, row_dict))

    if not rows:
        return empty_table_for_table(FUNCTION_AST_FEATURES_TABLE_KEY)
    return finalize_analytics_rows(FUNCTION_AST_FEATURES_TABLE_KEY, rows)


_MODULE = sys.modules[__name__]
_FUNCTION_AST_FEATURES_TABLE_TARGET_SPEC = build_single_table_target_spec(
    context=TableTargetContext.from_contract_ref(
        contract_ref=FUNCTION_AST_FEATURES_CONTRACT,
        input_type=pa.Table,
    )
)
attach_table_target_template(_MODULE, spec=_FUNCTION_AST_FEATURES_TABLE_TARGET_SPEC)
function_ast_features__table = _MODULE.function_ast_features__table
function_ast_features__table_materializations = (
    _MODULE.function_ast_features__table_materializations
)
t__function_ast_features = _MODULE.t__function_ast_features


__all__ = [
    "function_ast_features__base",
    "function_ast_features__table",
    "t__function_ast_features",
]
