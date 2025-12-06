"""Import graph builder plugin.

This module provides the import graph builder as a build target plugin.

Architecture
------------
The import graph plugin performs the following steps:

1. Load module information from core.modules
2. Extract imports from each Python file
3. Analyze imports to compute SCCs and layers
4. Persist module and edge data to graph.import_*
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.config import ImportGraphStepConfig
from codeintel.core.data_models import ImportEdgeRow, ImportModuleRow
from codeintel.graphs.compute import imports as imports_compute
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.ingestion.infrastructure.paths import normalize_rel_path

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def _get_source_root(gateway: StorageGateway, repo: str, commit: str) -> Path | None:
    """Retrieve source root from core.snapshots.

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
    Path | None
        Absolute path to source root, or None if not found.
    """
    con = gateway.con
    try:
        row = con.execute(
            "SELECT source_root FROM core.snapshots WHERE repo = ? AND commit = ?",
            [repo, commit],
        ).fetchone()
        if row and row[0]:
            return Path(row[0])
    except Exception as e:  # noqa: BLE001
        log.debug("import_graph: Could not get source root: %s", e)
    return None


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
    con = gateway.con
    try:
        rows = con.execute(
            """
            SELECT path, module
            FROM core.modules
            WHERE repo = ? AND commit = ?
            """,
            [repo, commit],
        ).fetchall()
        return {normalize_rel_path(str(row[0])): str(row[1]) for row in rows}
    except Exception:  # noqa: BLE001
        return {}


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


def _persist_import_modules(
    gateway: StorageGateway,
    rows: list[ImportModuleRow],
    repo: str,
    commit: str,
) -> int:
    """Persist import module rows.

    Parameters
    ----------
    gateway
        Storage gateway.
    rows
        Module rows to persist.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of rows persisted.
    """
    if not rows:
        return 0

    storage = IngestStorageService.from_gateway(gateway)
    storage.run_batch(
        "graph.import_modules",
        [row.to_tuple() for row in rows],
        delete_params=[repo, commit],
        scope="import_modules",
    )
    return len(rows)


def _persist_import_edges(
    gateway: StorageGateway,
    rows: list[ImportEdgeRow],
    repo: str,
    commit: str,
) -> int:
    """Persist import edge rows.

    Parameters
    ----------
    gateway
        Storage gateway.
    rows
        Edge rows to persist.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of rows persisted.
    """
    if not rows:
        return 0

    storage = IngestStorageService.from_gateway(gateway)
    storage.run_batch(
        "graph.import_graph_edges",
        [row.to_tuple() for row in rows],
        delete_params=[repo, commit],
        scope="import_graph_edges",
    )
    return len(rows)


class ImportGraphPlugin(TargetPlugin):
    """Build module-level import graph.

    This plugin performs full import graph construction:
    1. Loads module information from core.modules
    2. Parses source files to extract imports
    3. Analyzes imports to compute SCCs and layers
    4. Persists to graph.import_modules and graph.import_graph_edges

    Outputs
    -------
    - graph.import_modules: Module metadata with SCC and layer info
    - graph.import_graph_edges: Import relationships
    """

    plugin_name: ClassVar[str] = "import_graph"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build module-level import graph."

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute import graph construction.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result with row counts.
        """
        _ = self  # Protocol method requires instance
        config = ImportGraphStepConfig(snapshot=ctx.snapshot)
        gateway, repo, commit = ctx.gateway, config.repo, config.commit

        try:
            # Use snapshot repo_root directly, fall back to db or cwd
            source_root = (
                ctx.snapshot.repo_root or _get_source_root(gateway, repo, commit) or Path.cwd()
            )
            module_by_path = _load_modules(gateway, repo, commit)

            if not module_by_path:
                log.info("import_graph: No modules found, skipping")
                return TargetResult.succeeded(
                    row_counts={"graph.import_modules": 0, "graph.import_graph_edges": 0}
                )

            # Collect edges and analyze
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

            # Persist
            mc = _persist_import_modules(
                gateway,
                imports_compute.build_import_module_rows(repo, commit, result),
                repo,
                commit,
            )
            ec = _persist_import_edges(
                gateway, imports_compute.build_import_edge_rows(repo, commit, result), repo, commit
            )

            log.info("import_graph: Persisted %d modules, %d edges", mc, ec)
            return TargetResult.succeeded(
                row_counts={"graph.import_modules": mc, "graph.import_graph_edges": ec}
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Import graph build failed: {e}")


__all__ = ["ImportGraphPlugin"]
