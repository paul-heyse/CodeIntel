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
from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugins._metadata import to_plugin_metadata
from codeintel.build.plugins.graphs.builders.import_graph_options import ImportGraphOptions
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.graphs.compute import imports as imports_compute
from codeintel.ingestion.infrastructure.paths import normalize_rel_path
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.data_models import ImportEdgeRow, ImportModuleRow
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
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
    try:
        snapshots = gateway.ibis.table("core.snapshots")
        expr = (
            snapshots.filter(
                cast("Any", snapshots.repo == repo) & cast("Any", snapshots.commit == commit)
            )
            .select(snapshots.source_root)
            .limit(1)
        )
        df = expr.execute()
        if not getattr(df, "empty", True):
            value = df.iloc[0][0]
            if value:
                return Path(str(value))
    except DuckDBError as exc:
        log.debug("import_graph: Could not get source root: %s", exc)
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
    try:
        modules = gateway.ibis.table("core.modules")
        expr = modules.filter(
            cast("Any", modules.repo == repo) & cast("Any", modules.commit == commit)
        ).select(modules.path, modules.module)
        df = expr.execute()
        return {
            normalize_rel_path(str(path)): str(module)
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

    gateway.policy.ensure_table("graph.import_modules")
    gateway.policy.delete_for_snapshot("graph.import_modules", repo=repo, commit=commit)
    gateway.policy.bulk_insert("graph.import_modules", [row.to_tuple() for row in rows])
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

    gateway.policy.ensure_table("graph.import_graph_edges")
    gateway.policy.delete_for_snapshot("graph.import_graph_edges", repo=repo, commit=commit)
    gateway.policy.bulk_insert("graph.import_graph_edges", [row.to_tuple() for row in rows])
    return len(rows)


def _filter_paths_by_scope(
    module_by_path: Mapping[str, str],
    scope_paths: list[str] | None,
) -> dict[str, str]:
    """Filter module map by configured scope prefixes.

    Returns
    -------
    dict[str, str]
        Filtered mapping keyed by relative path.
    """
    if not scope_paths:
        return dict(module_by_path)
    prefixes = tuple(scope_paths)
    return {path: module for path, module in module_by_path.items() if path.startswith(prefixes)}


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
    _core_metadata: ClassVar[CorePluginMetadata] = IMPORT_GRAPH_METADATA

    def __init__(self, *, options_resolver: PluginOptionsResolver | None = None) -> None:
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return full core metadata."""
        return self._core_metadata

    def resolve_options(
        self,
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> ImportGraphOptions:
        """Resolve typed options from configuration.

        Returns
        -------
        ImportGraphOptions
            Resolved options instance.
        """
        if self._options_resolver is None:
            if dynamic_overrides:
                return ImportGraphOptions(**dynamic_overrides)
            return ImportGraphOptions()

        return self._options_resolver.get_options(
            self._core_metadata,
            ImportGraphOptions,
            dynamic_overrides=dynamic_overrides,
        )

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
        opts = self.resolve_options()
        gateway, repo, commit = ctx.gateway, ctx.snapshot.repo, ctx.snapshot.commit

        try:
            source_root = (
                ctx.snapshot.repo_root or _get_source_root(gateway, repo, commit) or Path.cwd()
            )
            module_by_path = _filter_paths_by_scope(
                _load_modules(gateway, repo, commit),
                opts.scope_paths,
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


__all__ = [
    "IMPORT_GRAPH_METADATA",
    "ImportGraphPlugin",
]
