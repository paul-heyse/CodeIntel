"""Import graph relation sources for graph targets."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path

import polars as pl

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
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.frames import empty_frame_for_table
from codeintel.build.tabular.types import InferableTabularInput, TabularFrame
from codeintel.core.columnar.rows import empty_reader_for_table, record_batch_reader_for_rows
from codeintel.ingestion.infrastructure.ast_utils import parse_python_module

IMPORT_MODULES_TABLE_KEY = "graph.import_modules"
IMPORT_GRAPH_EDGES_TABLE_KEY = "graph.import_graph_edges"


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
    modules_frame = tabular_to_lazyframe(q__core__modules).collect()
    modules: set[str] = set()
    edges: list[ImportEdge] = []
    repo_root = env.snapshot.repo_root

    for row in modules_frame.iter_rows(named=True):
        module_name = row.get("module")
        rel_path = row.get("path")
        language = row.get("language")
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
) -> TabularFrame:
    """Build import module rows from computed import graph analysis.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for computed import modules.
    """
    rows = build_import_module_rows(env.repo, env.commit, import_graph_analysis)
    if not rows:
        return empty_frame_for_table(IMPORT_MODULES_TABLE_KEY)
    frame = pl.DataFrame([dataclasses.asdict(row) for row in rows])
    return frame.lazy().select(
        [
            "repo",
            "commit",
            "module",
            "scc_id",
            "component_size",
            "layer",
            "cycle_group",
        ]
    )


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
    reader, _ = record_batch_reader_for_rows(IMPORT_GRAPH_EDGES_TABLE_KEY, rows)
    return reader


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
    return empty_reader_for_table(IMPORT_MODULES_TABLE_KEY)


def import_graph_edges_empty(env: BuildEnv) -> InferableTabularInput:
    """Return an empty frame for import graph edges.

    Returns
    -------
    InferableTabularInput
        Empty tabular input for import graph edges.
    """
    _ = env
    return empty_reader_for_table(IMPORT_GRAPH_EDGES_TABLE_KEY)


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
