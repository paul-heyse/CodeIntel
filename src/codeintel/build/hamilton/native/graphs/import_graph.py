"""Import graph relation sources for graph targets."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import pyarrow as pa

from codeintel.build.graphs.assembly import tabular_to_table
from codeintel.build.graphs.compute.imports import (
    ImportAnalysisResult,
    ImportEdge,
    analyze_imports,
    build_import_edge_rows,
    build_import_module_rows,
    collect_import_edges,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns.loaders import load_snapshot_tabular
from codeintel.build.tabular.conversion import table_to_reader
from codeintel.build.tabular.expr_vocab import E, Expression
from codeintel.build.tabular.finalize_ops import finalize_reader, finalize_spec_for_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.arrowdsl import ExecutionPlan
from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.execution_context import resolve_execution_context
from codeintel.core.columnar.iter import iter_tuples
from codeintel.core.columnar.kernels import SortKey
from codeintel.core.columnar.plan_builder import TablePlanOptions, build_table_plan
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module

IMPORT_MODULES_TABLE_KEY = "graph.import_modules"
IMPORT_GRAPH_EDGES_TABLE_KEY = "graph.import_graph_edges"
IMPORT_MODULES_SORT_KEYS: tuple[SortKey, ...] = (
    ("repo", "ascending"),
    ("commit", "ascending"),
    ("module", "ascending"),
)
IMPORT_GRAPH_EDGES_SORT_KEYS: tuple[SortKey, ...] = (
    ("repo", "ascending"),
    ("commit", "ascending"),
    ("src_module", "ascending"),
    ("dst_module", "ascending"),
)


def _resolve_import_from(
    *,
    current_module: str,
    module_part: str | None,
    level: int,
) -> str | None:
    if level <= 0:
        return module_part or None
    parts = current_module.split(".")
    base_parts = parts[:-level] if level <= len(parts) else []
    if module_part:
        if base_parts:
            return ".".join([*base_parts, module_part])
        return module_part
    if base_parts:
        return ".".join(base_parts)
    return None


def _collect_imports(current_module: str, tree: ast.AST) -> list[tuple[str, tuple[str, ...]]]:
    imports: list[tuple[str, tuple[str, ...]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module_part = node.module
            base = _resolve_import_from(
                current_module=current_module,
                module_part=module_part,
                level=node.level or 0,
            )
            if base is None:
                continue
            if module_part is None:
                imports.extend((f"{base}.{alias.name}", ()) for alias in node.names)
            else:
                imports.append((base, tuple(alias.name for alias in node.names)))
    return imports


def import_graph_analysis(
    env: BuildEnv,
    q__core__modules: InferableTabularInput,
) -> ImportAnalysisResult:
    """Parse module imports and build the import graph analysis result.

    Parameters
    ----------
    env
        Build environment for repository metadata.
    q__core__modules
        Relation for ``core.modules``.

    Returns
    -------
    ImportAnalysisResult
        Import graph analysis derived from module sources.
    """
    modules_table = _python_modules_table(tabular_to_table(q__core__modules))
    modules: set[str] = set()
    edges: list[ImportEdge] = []
    repo_root = env.snapshot.repo_root

    for module_name, rel_path, language in iter_tuples(
        table_to_reader(modules_table),
        columns=("module", "path", "language"),
    ):
        if not isinstance(module_name, str) or not module_name:
            continue
        modules.add(module_name)
        if language not in {None, "python"}:
            continue
        if not isinstance(rel_path, str) or not rel_path:
            continue
        module_path = Path(repo_root) / rel_path
        parsed = parse_python_module(module_path)
        if parsed is None:
            continue
        _, tree = parsed
        imports = _collect_imports(module_name, tree)
        edges.extend(collect_import_edges(module_name, imports))

    return analyze_imports(edges, modules)


def import_modules_compute(
    env: BuildEnv,
    import_graph_analysis: ImportAnalysisResult,
) -> InferableTabularInput:
    """Build import module rows from computed import graph analysis.

    Returns
    -------
    InferableTabularInput
        Arrow reader for computed import modules.
    """
    rows = build_import_module_rows(env.repo, env.commit, import_graph_analysis)
    if not rows:
        return empty_table_for_table(IMPORT_MODULES_TABLE_KEY)
    table, _ = table_for_rows(
        IMPORT_MODULES_TABLE_KEY,
        (dataclasses.asdict(row) for row in rows),
    )
    reader = table_to_reader(table, batch_size=None)
    result = finalize_reader(
        reader,
        spec=finalize_spec_for_table(
            IMPORT_MODULES_TABLE_KEY,
            mode="strict",
            order_by=IMPORT_MODULES_SORT_KEYS,
        ),
    )
    return result.good


def import_graph_edges_compute(
    env: BuildEnv,
    import_graph_analysis: ImportAnalysisResult,
) -> InferableTabularInput:
    """Build import graph edges from computed import graph analysis.

    Returns
    -------
    InferableTabularInput
        Tabular input for computed import graph edges.
    """
    rows = (
        dataclasses.asdict(row)
        for row in build_import_edge_rows(env.repo, env.commit, import_graph_analysis)
    )
    table, _ = table_for_rows(IMPORT_GRAPH_EDGES_TABLE_KEY, rows)
    reader = table_to_reader(table, batch_size=None)
    result = finalize_reader(
        reader,
        spec=finalize_spec_for_table(
            IMPORT_GRAPH_EDGES_TABLE_KEY,
            mode="strict",
            order_by=IMPORT_GRAPH_EDGES_SORT_KEYS,
        ),
    )
    return result.good


def import_modules_existing(env: BuildEnv) -> InferableTabularInput:
    """Load import modules from the dataset snapshot.

    Returns
    -------
    InferableTabularInput
        Tabular input for existing import modules.
    """
    return load_snapshot_tabular(
        env=env,
        table_key=IMPORT_MODULES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def import_graph_edges_existing(env: BuildEnv) -> InferableTabularInput:
    """Load import graph edges from the dataset snapshot.

    Returns
    -------
    InferableTabularInput
        Tabular input for existing import graph edges.
    """
    return load_snapshot_tabular(
        env=env,
        table_key=IMPORT_GRAPH_EDGES_TABLE_KEY,
        snapshot_id=env.commit,
    )


def import_modules_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for import modules.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for import modules.
    """
    _ = env
    return empty_table_for_table(IMPORT_MODULES_TABLE_KEY)


def import_graph_edges_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for import graph edges.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for import graph edges.
    """
    _ = env
    return empty_table_for_table(IMPORT_GRAPH_EDGES_TABLE_KEY)


def _python_modules_table(modules_table: pa.Table) -> pa.Table:
    required = {"path", "module"}
    if modules_table.num_rows == 0 or not required.issubset(set(modules_table.column_names)):
        return modules_table
    projection = {
        "path": E.cast(E.field("path"), "string"),
        "module": E.cast(E.field("module"), "string"),
    }
    if "language" in modules_table.column_names:
        projection["language"] = E.cast(E.field("language"), "string")
    exprs: list[Expression] = [_non_empty_expr("path"), _non_empty_expr("module")]
    if "language" in projection:
        exprs.append(_python_language_expr())
    plan = build_table_plan(
        table=modules_table,
        options=TablePlanOptions(
            projection=projection,
            filter_expr=E.and_(*exprs),
        ),
    )
    execution_ctx = resolve_execution_context(None)
    reader = ExecutionPlan.from_plan(plan).to_reader(ctx=execution_ctx)
    return reader_to_table(reader)


def _python_language_expr() -> Expression:
    return E.or_(E.is_null("language"), E.field("language") == E.scalar("python"))


def _non_empty_expr(name: str) -> Expression:
    return E.and_(E.is_valid(name), E.field(name) != E.scalar(""))


__all__ = [
    "IMPORT_GRAPH_EDGES_TABLE_KEY",
    "IMPORT_MODULES_TABLE_KEY",
    "import_graph_analysis",
    "import_graph_edges_compute",
    "import_graph_edges_empty",
    "import_graph_edges_existing",
    "import_modules_compute",
    "import_modules_empty",
    "import_modules_existing",
]
