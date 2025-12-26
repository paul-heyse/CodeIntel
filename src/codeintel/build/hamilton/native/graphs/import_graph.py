"""Native Hamilton implementation for import_graph target.

This module implements import graph construction as a native Hamilton pipeline with:
- t__import_graph__extract: Parse source files and extract imports
- t__import_graph: Materialize with validators and return TargetRunRecord

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import ast
import logging
from typing import TYPE_CHECKING

import ibis.expr.types as ir
from hamilton.function_modifiers.dependencies import source, value

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.helpers import filter_mapping, get_source_root
from codeintel.build.hamilton.materialization_helpers import executor_materialize
from codeintel.build.hamilton.materialize_options import materialize_options
from codeintel.build.hamilton.materializers import DuckDBRowsSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.options.graphs import ImportGraphOptions
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.build.schemas import deferred_columns_for_table_key
from codeintel.build.targets import TargetGraph
from codeintel.core.ibis_typing import filter_by
from codeintel.core.paths import normalize_path
from codeintel.graphs.compute import imports as imports_compute
from codeintel.storage.gateway import DuckDBError, StorageGateway

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)

IMPORT_GRAPH_TARGET_NAME = "import_graph"
IMPORT_MODULES_TABLE_KEY = "graph.import_modules"
IMPORT_GRAPH_EDGES_TABLE_KEY = "graph.import_graph_edges"
IMPORT_GRAPH_TABLE_KEYS = (
    IMPORT_MODULES_TABLE_KEY,
    IMPORT_GRAPH_EDGES_TABLE_KEY,
)


@tag_helper(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def gateway(env: BuildEnv) -> StorageGateway:
    """Expose the storage gateway for import graph nodes.

    Returns
    -------
    StorageGateway
        Storage gateway for the current build environment.
    """
    return env.gateway


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(IMPORT_MODULES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(IMPORT_GRAPH_TARGET_NAME),
    table_key=value(IMPORT_MODULES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(IMPORT_MODULES_TABLE_KEY)),
)
@tag_compute(
    domain="graphs", target=IMPORT_GRAPH_TARGET_NAME, target_="import_graph__modules_marker"
)
def import_graph__modules_marker() -> tuple[tuple[object, ...], ...] | None:
    """Declare import graph modules output for inventory checks.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Always ``None`` so the saver node is used only for metadata.
    """
    return None


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(IMPORT_GRAPH_EDGES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(IMPORT_GRAPH_TARGET_NAME),
    table_key=value(IMPORT_GRAPH_EDGES_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(IMPORT_GRAPH_EDGES_TABLE_KEY)),
)
@tag_compute(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME, target_="import_graph__edges_marker")
def import_graph__edges_marker() -> tuple[tuple[object, ...], ...] | None:
    """Declare import graph edges output for inventory checks.

    Returns
    -------
    tuple[tuple[object, ...], ...] | None
        Always ``None`` so the saver node is used only for metadata.
    """
    return None


@tag_helper(domain="graphs")
def _load_modules(
    q__core__modules: ir.Table,
    repo: str,
    commit: str,
) -> dict[str, str]:
    """Load module information from core.modules.

    Parameters
    ----------
    q__core__modules
        Ibis table expression for core.modules.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    dict[str, str]
        Mapping of relative path to module name.
    """
    try:
        expr = filter_by(
            q__core__modules,
            q__core__modules.repo == repo,
            q__core__modules.commit == commit,
        ).select(q__core__modules.path, q__core__modules.module)
        df = expr.execute()
        return {
            normalize_path(str(path)): str(module)
            for path, module in df.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


@tag_helper(domain="graphs")
def _extract_imports_from_file(file_path: Path) -> list[tuple[str, tuple[str, ...]]]:
    """Extract imports from a Python file.

    Parameters
    ----------
    file_path
        Absolute path to the file.

    Returns
    -------
    list[tuple[str, tuple[str, ...]]]
        List of (module_name, imported_names) tuples.
    """
    if not file_path.exists():
        return []

    try:
        source = file_path.read_text(encoding="utf8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []

    imports: list[tuple[str, tuple[str, ...]]] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend((alias.name, ()) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                names = tuple(alias.name for alias in node.names)
                imports.append((module, names))

    return imports


def _collect_import_edges(
    *,
    source_root: Path,
    module_by_path: dict[str, str],
) -> list[imports_compute.ImportEdge]:
    edges: list[imports_compute.ImportEdge] = []
    for rel_path, module_name in module_by_path.items():
        edges.extend(
            imports_compute.collect_import_edges(
                module_name,
                _extract_imports_from_file(source_root / rel_path),
            )
        )
    return edges


def _materialize_import_graph(
    env: BuildEnv,
    *,
    repo: str,
    commit: str,
    analysis: imports_compute.ImportAnalysisResult,
) -> tuple[int, int]:
    options = materialize_options(env, owner_target=IMPORT_GRAPH_TARGET_NAME)
    module_count = int(
        env.warehouse.materialize_rows(
            IMPORT_MODULES_TABLE_KEY,
            [
                row.to_tuple()
                for row in imports_compute.build_import_module_rows(repo, commit, analysis)
            ],
            columns=None,
            options=options,
        ).rows_written
        or 0
    )
    edge_count = int(
        env.warehouse.materialize_rows(
            IMPORT_GRAPH_EDGES_TABLE_KEY,
            [
                row.to_tuple()
                for row in imports_compute.build_import_edge_rows(repo, commit, analysis)
            ],
            columns=None,
            options=options,
        ).rows_written
        or 0
    )
    return module_count, edge_count


@tag_tool(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def t__import_graph__extract(
    env: BuildEnv,
    gateway: StorageGateway,
    q__core__modules: ir.Table,
    t__modules: TargetRunRecord,
) -> ExecutionResult:
    """Execute import graph extraction on repository modules.

    This is the compute node for the import_graph target. It parses Python
    source files to extract import statements and builds a module-level
    import graph with SCC and layer analysis.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    gateway
        Storage gateway for graph data access.
    q__core__modules
        Ibis table expression for core.modules.
    t__modules
        Upstream modules target result (for dependency).

    Returns
    -------
    ExecutionResult
        Execution status and table row counts.

    Notes
    -----
    Produces:
    - graph.import_modules: Module metadata with SCC and layer info
    - graph.import_graph_edges: Import relationships
    """
    if t__modules.status != "succeeded":
        return ExecutionResult.failed(
            f"Upstream modules target failed: {t__modules.error}"
        )

    try:
        snapshot = env.snapshot
        opts = load_target_options(
            env,
            target_name=IMPORT_GRAPH_TARGET_NAME,
            options_type=ImportGraphOptions,
        )

        source_root = snapshot.repo_root or get_source_root(
            gateway,
            snapshot.repo,
            snapshot.commit,
        )
        module_by_path = filter_mapping(
            _load_modules(q__core__modules, snapshot.repo, snapshot.commit),
            scope_paths=opts.scope_paths,
        )

        if not module_by_path:
            log.info("import_graph: No modules found, skipping")
            return ExecutionResult.ok(
                table_counts={
                    IMPORT_MODULES_TABLE_KEY: 0,
                    IMPORT_GRAPH_EDGES_TABLE_KEY: 0,
                }
            )

        edges = _collect_import_edges(source_root=source_root, module_by_path=module_by_path)
        analysis = imports_compute.analyze_imports(edges, set(module_by_path.values()))
        log.info(
            "import_graph: %d edges, %d SCCs",
            len(edges),
            len(set(analysis.scc_map.values())),
        )
        module_count, edge_count = _materialize_import_graph(
            env,
            repo=snapshot.repo,
            commit=snapshot.commit,
            analysis=analysis,
        )

        log.info("import_graph: Persisted %d modules, %d edges", module_count, edge_count)

        return ExecutionResult.ok(
            table_counts={
                IMPORT_MODULES_TABLE_KEY: module_count,
                IMPORT_GRAPH_EDGES_TABLE_KEY: edge_count,
            }
        )

    except Exception as exc:
        log.exception("Import graph extraction failed")
        return ExecutionResult.failed(str(exc))


@codeintel_target(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
def t__import_graph(
    env: BuildEnv,
    graph: TargetGraph,
    t__import_graph__extract: ExecutionResult,
) -> TargetRunRecord:
    """Construct a module import graph.

    This is the entry point for the import_graph target. It orchestrates
    import graph extraction and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__import_graph__extract
        Execution result produced by the extract node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    return executor_materialize(env, graph, IMPORT_GRAPH_TARGET_NAME, t__import_graph__extract)


__all__ = [
    "t__import_graph",
    "t__import_graph__extract",
]
