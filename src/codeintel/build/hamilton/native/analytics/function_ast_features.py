"""Function AST feature tables built with inferable tabular nodes."""

from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from codeintel.build.analytics.ast_features.extract import compute_function_features
from codeintel.build.analytics.parsing.ast_cache import FunctionAst
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.tabular.frames import empty_frame_for_table
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
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

FUNCTION_AST_FEATURES_TARGET_NAME = "function_ast_features"
FUNCTION_AST_FEATURES_TABLE_KEY = "analytics.function_ast_features"
FUNCTION_AST_FEATURES_CONTRACT = TableContractSpec(
    table_key=FUNCTION_AST_FEATURES_TABLE_KEY,
    domain="analytics",
    target=FUNCTION_AST_FEATURES_TARGET_NAME,
    ops_module=None,
    columns_to_pass=(),
    required_cols=(),
    clip_column=None,
    input_name="function_ast_features__base",
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


def _default_feature_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "repo": row.get("repo"),
        "commit": row.get("commit"),
        "function_goid_h128": row.get("goid_h128"),
        "rel_path": row.get("rel_path"),
        "qualname": row.get("qualname"),
        "is_async": False,
        "uses_network": False,
        "uses_db": False,
        "uses_filesystem": False,
        "uses_subprocess": False,
        "uses_concurrency_lib": False,
        "uses_threading": False,
        "uses_asyncio_lib": False,
        "http_client_libs": "[]",
        "http_server_libs": "[]",
        "db_libs": "[]",
        "message_libs": "[]",
        "config_read_count": 0,
        "feature_flag_count": 0,
        "decorators": "[]",
        "libraries_used": "[]",
        "created_at": row.get("created_at"),
    }


def _module_by_path(modules_frame: pl.DataFrame) -> dict[str, str]:
    module_by_path: dict[str, str] = {}
    for row in modules_frame.iter_rows(named=True):
        rel_path = row.get("path")
        module_name = row.get("module")
        language = row.get("language")
        if language not in {None, "python"}:
            continue
        if isinstance(rel_path, str) and isinstance(module_name, str):
            module_by_path[rel_path] = module_name
    return module_by_path


def _load_module_nodes(
    env: BuildEnv,
    module_by_path: dict[str, str],
) -> tuple[dict[str, dict[tuple[str, int], _FunctionNodeInfo]], dict[str, list[str]]]:
    nodes_by_path: dict[str, dict[tuple[str, int], _FunctionNodeInfo]] = {}
    lines_by_path: dict[str, list[str]] = {}
    repo_root = Path(env.snapshot.repo_root)
    for rel_path, module_name in module_by_path.items():
        module_path = repo_root / rel_path
        parsed = parse_python_module(module_path)
        if parsed is None:
            continue
        lines, tree = parsed
        lines_by_path[rel_path] = lines
        node_map: dict[tuple[str, int], _FunctionNodeInfo] = {}
        for info in _collect_function_nodes(tree, module_name):
            node_map[info.qualname, info.start_line] = info
        nodes_by_path[rel_path] = node_map
    return nodes_by_path, lines_by_path


def _feature_row_from_goid(
    row: dict[str, object],
    nodes_by_path: dict[str, dict[tuple[str, int], _FunctionNodeInfo]],
    lines_by_path: dict[str, list[str]],
    *,
    repo_root: Path,
) -> dict[str, object]:
    rel_path = row.get("rel_path")
    qualname = row.get("qualname")
    start_line = row.get("start_line")
    if not isinstance(rel_path, str) or not isinstance(qualname, str):
        return _default_feature_row(row)
    if not isinstance(start_line, int):
        return _default_feature_row(row)
    info = nodes_by_path.get(rel_path, {}).get((qualname, start_line))
    if info is None:
        return _default_feature_row(row)
    goid = row.get("goid_h128")
    if not isinstance(goid, int):
        return _default_feature_row(row)
    lines = lines_by_path.get(rel_path, [])
    fn = FunctionAst(
        goid=goid,
        rel_path=rel_path,
        qualname=qualname,
        start_line=start_line,
        end_line=info.end_line,
        node=info.node,
        lines=lines,
    )
    try:
        features = compute_function_features(fn, repo_root=repo_root)
    except (SyntaxError, ValueError, TypeError):
        return _default_feature_row(row)
    return {
        "repo": row.get("repo"),
        "commit": row.get("commit"),
        "function_goid_h128": row.get("goid_h128"),
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
        "http_client_libs": json.dumps(sorted(features.http_client_libs)),
        "http_server_libs": json.dumps(sorted(features.http_server_libs)),
        "db_libs": json.dumps(sorted(features.db_libs)),
        "message_libs": json.dumps(sorted(features.message_libs)),
        "config_read_count": features.config_read_count,
        "feature_flag_count": features.feature_flag_count,
        "decorators": json.dumps(list(features.decorators)),
        "libraries_used": json.dumps(sorted(features.libraries_used)),
        "created_at": row.get("created_at"),
    }


def function_ast_features__base(
    env: BuildEnv,
    q__core__goids: InferableTabularInput,
    q__core__modules: InferableTabularInput,
) -> pl.LazyFrame:
    """Build function AST features using parsed source files.

    Returns
    -------
    polars.LazyFrame
        Lazy frame with function AST feature columns.
    """
    goids = tabular_to_lazyframe(q__core__goids).collect()
    if goids.is_empty():
        return empty_frame_for_table(FUNCTION_AST_FEATURES_TABLE_KEY)

    modules = tabular_to_lazyframe(q__core__modules).collect()
    module_by_path = _module_by_path(modules)
    nodes_by_path, lines_by_path = _load_module_nodes(env, module_by_path)

    rows: list[dict[str, object]] = []
    repo_root = Path(env.snapshot.repo_root)
    for row in goids.iter_rows(named=True):
        if row.get("kind") not in {"function", "method"}:
            continue
        rows.append(
            _feature_row_from_goid(
                row,
                nodes_by_path,
                lines_by_path,
                repo_root=repo_root,
            )
        )

    if not rows:
        return empty_frame_for_table(FUNCTION_AST_FEATURES_TABLE_KEY)
    frame = pl.DataFrame(rows)
    return frame.lazy().select(
        [
            "repo",
            "commit",
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
            "http_client_libs",
            "http_server_libs",
            "db_libs",
            "message_libs",
            "config_read_count",
            "feature_flag_count",
            "decorators",
            "libraries_used",
            "created_at",
        ]
    )


_MODULE = sys.modules[__name__]
_FUNCTION_AST_FEATURES_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="analytics",
    target_name=FUNCTION_AST_FEATURES_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=FUNCTION_AST_FEATURES_TABLE_KEY,
            base_node="function_ast_features__base",
            contract=FUNCTION_AST_FEATURES_CONTRACT,
            save_spec=DatasetSaveSpec(table_key=FUNCTION_AST_FEATURES_TABLE_KEY),
            node_name="function_ast_features__table",
        ),
    ),
    table_materializations_node="function_ast_features__table_materializations",
    anchor_node_name="t__function_ast_features",
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
