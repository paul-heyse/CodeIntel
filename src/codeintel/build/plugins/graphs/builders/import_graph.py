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
from typing import TYPE_CHECKING, Any, ClassVar, cast

from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.build.plugins._helpers import filter_mapping, get_source_root, persist_rows
from codeintel.build.plugins.graphs.builders.import_graph_options import ImportGraphOptions
from codeintel.core.paths import normalize_path
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.graphs.compute import imports as imports_compute
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


IMPORT_GRAPH_METADATA = CorePluginMetadata(
    name="graphs.import_graph",
    version="2.0.0",
    description="Build module import graph.",
    domain=PluginDomain.GRAPH,
    kind="builder",
    stage="edges",
    provides=("graph.import_graph",),
    requires=("core.modules",),
    produces_tables=(
        "graph.import_modules",
        "graph.import_graph_edges",
    ),
    consumes_tables=("core.modules",),
    scope_aware=True,
    options_model=ImportGraphOptions,
    extra={"graph_kinds": ("import_graph",)},
)


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


class ImportGraphPlugin(MetadataPlugin):
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

    _core_metadata: ClassVar[CorePluginMetadata] = IMPORT_GRAPH_METADATA

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
        _ = self
        opts = self.resolve_options(ImportGraphOptions)
        gateway, repo, commit = ctx.gateway, ctx.snapshot.repo, ctx.snapshot.commit

        try:
            source_root = ctx.snapshot.repo_root or get_source_root(gateway, repo, commit)
            module_by_path = filter_mapping(
                _load_modules(gateway, repo, commit),
                scope_paths=opts.scope_paths,
            )

            if not module_by_path:
                log.info("import_graph: No modules found, skipping")
                return TargetResult.succeeded(
                    row_counts={"graph.import_modules": 0, "graph.import_graph_edges": 0}
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
            return TargetResult.succeeded(
                row_counts={"graph.import_modules": mc, "graph.import_graph_edges": ec}
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Import graph build failed: {e}")


__all__ = [
    "IMPORT_GRAPH_METADATA",
    "ImportGraphPlugin",
]
