"""Native Hamilton implementation for import_graph target.

This module implements import graph construction as a native Hamilton pipeline with:
- t__import_graph__extract: Parse source files and extract imports
- t__import_graph: Materialize with validators and return TargetRunRecord

Phase 3: Graphs domain migration with Hamilton-native validation.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.helpers import filter_mapping, get_source_root, persist_rows
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.options.graphs import ImportGraphOptions
from codeintel.build.targets import TargetGraph
from codeintel.core.paths import normalize_path
from codeintel.graphs.compute import imports as imports_compute
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


@dataclass(frozen=True)
class ImportGraphExtractResult:
    """Result from import graph extraction.

    Attributes
    ----------
    success
        Whether extraction completed successfully.
    module_count
        Number of modules processed.
    edge_count
        Number of import edges extracted.
    table_counts
        Row counts per produced table.
    error
        Fatal error message if extraction failed.
    """

    success: bool
    module_count: int = 0
    edge_count: int = 0
    table_counts: dict[str, int] = field(default_factory=dict)
    error: str | None = None


@tag(node_type="helper")
def _load_modules(
    gateway: StorageGateway,
    repo: str,
    commit: str,
) -> dict[str, str]:
    """Load module information from core.modules.

    Parameters
    ----------
    gateway
        Storage gateway.
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
        modules = gateway.ibis.table("core.modules")
        expr = modules.filter(
            cast("Any", modules.repo == repo) & cast("Any", modules.commit == commit)
        ).select(modules.path, modules.module)
        df = expr.execute()
        return {
            normalize_path(str(path)): str(module)
            for path, module in df.itertuples(index=False, name=None)
        }
    except DuckDBError:
        return {}


@tag(node_type="helper")
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


@tag(domain="graphs", target="import_graph", node_type="compute")
def t__import_graph__extract(
    env: BuildEnv,
    t__modules: TargetRunRecord,
) -> ImportGraphExtractResult:
    """Execute import graph extraction on repository modules.

    This is the compute node for the import_graph target. It parses Python
    source files to extract import statements and builds a module-level
    import graph with SCC and layer analysis.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    t__modules
        Upstream modules target result (for dependency).

    Returns
    -------
    ImportGraphExtractResult
        Result containing module and edge counts.

    Notes
    -----
    Produces:
    - graph.import_modules: Module metadata with SCC and layer info
    - graph.import_graph_edges: Import relationships
    """
    if t__modules.status != "succeeded":
        return ImportGraphExtractResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    try:
        gateway = env.gateway
        repo = env.snapshot.repo
        commit = env.snapshot.commit
        opts = ImportGraphOptions()

        source_root = env.snapshot.repo_root or get_source_root(gateway, repo, commit)
        module_by_path = filter_mapping(
            _load_modules(gateway, repo, commit),
            scope_paths=opts.scope_paths,
        )

        if not module_by_path:
            log.info("import_graph: No modules found, skipping")
            return ImportGraphExtractResult(
                success=True,
                module_count=0,
                edge_count=0,
                table_counts={
                    "graph.import_modules": 0,
                    "graph.import_graph_edges": 0,
                },
            )

        edges: list[imports_compute.ImportEdge] = []
        for rel_path, module_name in module_by_path.items():
            edges.extend(
                imports_compute.collect_import_edges(
                    module_name, _extract_imports_from_file(source_root / rel_path)
                )
            )

        modules = set(module_by_path.values())
        result = imports_compute.analyze_imports(edges, modules)
        log.info(
            "import_graph: %d edges, %d SCCs", len(edges), len(set(result.scc_map.values()))
        )

        mc = persist_rows(
            gateway,
            "graph.import_modules",
            imports_compute.build_import_module_rows(repo, commit, result),
            repo=repo,
            commit=commit,
        )
        ec = persist_rows(
            gateway,
            "graph.import_graph_edges",
            imports_compute.build_import_edge_rows(repo, commit, result),
            repo=repo,
            commit=commit,
        )

        log.info("import_graph: Persisted %d modules, %d edges", mc, ec)

        return ImportGraphExtractResult(
            success=True,
            module_count=mc,
            edge_count=ec,
            table_counts={
                "graph.import_modules": mc,
                "graph.import_graph_edges": ec,
            },
        )

    except Exception as exc:
        log.exception("Import graph extraction failed: %s", exc)
        return ImportGraphExtractResult(
            success=False,
            error=str(exc),
        )


@tag(domain="graphs", target="import_graph", node_type="materialize")
def t__import_graph(
    env: BuildEnv,
    graph: TargetGraph,
    t__import_graph__extract: ImportGraphExtractResult,
) -> TargetRunRecord:
    """Materialize import graph target with validation.

    This is the entry point for the import_graph target. It orchestrates
    import graph extraction and returns a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment with gateway and snapshot.
    graph
        Target graph for metadata lookup.
    t__import_graph__extract
        Extraction result from upstream compute node.

    Returns
    -------
    TargetRunRecord
        Record with status, datasets, and execution metadata.
    """
    executor = NativeTargetExecutor.for_target(env, graph, "import_graph")

    if executor.should_skip():
        return executor.skip()

    if not t__import_graph__extract.success:
        return executor.fail(
            RuntimeError(t__import_graph__extract.error or "Import graph extraction failed")
        )

    def compute() -> dict[str, int]:
        return dict(t__import_graph__extract.table_counts)

    return executor.execute(compute)


__all__ = [
    "ImportGraphExtractResult",
    "t__import_graph",
    "t__import_graph__extract",
]
