"""Control flow graph and data flow graph builder plugin.

This module provides the CFG/DFG builder as a build target plugin.

Architecture
------------
The CFG/DFG plugin performs the following steps:

1. Load function spans from `core.goids` to identify functions
2. For each Python file with functions:
   - Parse the file to AST
   - For each function, build CFG using CFGBuilder
   - Build DFG from CFG using DFGBuilder
   - Convert results to row format
3. Persist CFG blocks and edges to graph.cfg_*
4. Persist DFG edges to graph.dfg_edges
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, cast

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugins.graphs.builders.cfg_dfg_options import CfgDfgOptions
from codeintel.config import CFGBuilderStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.graphs.catalog import load_function_index
from codeintel.graphs.compute import cfg as cfg_compute
from codeintel.graphs.compute import dfg as dfg_compute
from codeintel.storage.gateway import DuckDBError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.data_models import CFGBlockRow, CFGEdgeRow, DFGEdgeRow
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.core.plugins.types.protocol import PluginKind, PluginStage
    from codeintel.graphs.catalog import FunctionSpanIndex
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


CFG_DFG_METADATA = CorePluginMetadata(
    name="graphs.cfg_dfg",
    version="3.0.0",
    description="Build control flow and data flow graphs.",
    domain=PluginDomain.GRAPH,
    kind="builder",
    stage="edges",
    provides=("graph.cfg", "graph.dfg"),
    requires=("core.goids",),
    produces_tables=("graph.cfg_blocks", "graph.cfg_edges", "graph.dfg_edges"),
    consumes_tables=("core.goids",),
    supports_incremental=False,
    scope_aware=True,
    options_model=CfgDfgOptions,
    extra={"graph_kinds": ("cfg", "dfg")},
)


def _to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to PluginMetadata for protocol compliance.

    Returns
    -------
    PluginMetadata
        Protocol-compatible metadata instance.
    """
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=cast("PluginKind", core.kind),
        stage=cast("PluginStage", core.stage or "edges"),
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )


def _is_test_path(path: str) -> bool:
    """Return True when the path looks like a test file.

    Returns
    -------
    bool
        True when the path is considered a test path.
    """
    lowered = path.lower()
    return (
        "tests/" in lowered
        or lowered.endswith("_test.py")
        or "/test_" in lowered
        or lowered.startswith("test_")
    )


def _filter_paths(paths: list[str], options: CfgDfgOptions) -> list[str]:
    """Filter function paths by scope and test inclusion.

    Returns
    -------
    list[str]
        Filtered list of relative paths.
    """
    filtered = list(paths)

    if options.scope_paths:
        prefixes = tuple(options.scope_paths)
        filtered = [path for path in filtered if path.startswith(prefixes)]

    if not options.include_test_files:
        filtered = [path for path in filtered if not _is_test_path(path)]

    return filtered


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
        log.debug("cfg_dfg: Could not get source root: %s", exc)
    return None


def _parse_file_functions(
    file_path: Path,
) -> list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, int, int]]:
    """Parse a Python file and extract function nodes with line ranges.

    Parameters
    ----------
    file_path
        Absolute path to the Python file.

    Returns
    -------
    list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, int, int]]
        List of (function_node, start_line, end_line) tuples.
    """
    if not file_path.exists():
        return []

    try:
        source = file_path.read_text(encoding="utf8")
        tree = ast.parse(source)
    except (OSError, UnicodeDecodeError, SyntaxError):
        return []

    functions: list[tuple[ast.FunctionDef | ast.AsyncFunctionDef, int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", 0)
            end = getattr(node, "end_lineno", start)
            functions.append((node, start, end if end is not None else start))
    return functions


def _build_cfg_dfg_for_function(
    goid: int,
    func_node: ast.FunctionDef | ast.AsyncFunctionDef,
    file_path: str,
    start_line: int,
    end_line: int,
) -> tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]:
    """Build CFG and DFG for a single function.

    Parameters
    ----------
    goid
        Function GOID.
    func_node
        Function AST node.
    file_path
        Relative file path.
    start_line
        Function start line.
    end_line
        Function end line.

    Returns
    -------
    tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]
        CFG blocks, CFG edges, and DFG edges.
    """
    cfg_result = cfg_compute.build_cfg(goid, func_node, file_path)

    cfg_blocks, cfg_edges = cfg_compute.cfg_to_rows(cfg_result, file_path, start_line, end_line)

    dfg_result = dfg_compute.build_dfg(goid, cfg_result.blocks, cfg_result.edges)

    dfg_edges = dfg_compute.dfg_to_rows(dfg_result)

    return list(cfg_blocks), list(cfg_edges), list(dfg_edges)


def _persist_cfg_blocks(
    gateway: StorageGateway,
    blocks: list[CFGBlockRow],
    repo: str,
    commit: str,
) -> int:
    """Persist CFG blocks.

    Parameters
    ----------
    gateway
        Storage gateway.
    blocks
        Block rows to persist.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of blocks persisted.
    """
    if not blocks:
        return 0

    gateway.policy.ensure_table("graph.cfg_blocks")
    gateway.policy.delete_for_snapshot("graph.cfg_blocks", repo=repo, commit=commit)
    gateway.policy.bulk_insert(
        "graph.cfg_blocks",
        [block.to_tuple() for block in blocks],
    )
    return len(blocks)


def _persist_cfg_edges(
    gateway: StorageGateway,
    edges: list[CFGEdgeRow],
    repo: str,
    commit: str,
) -> int:
    """Persist CFG edges.

    Parameters
    ----------
    gateway
        Storage gateway.
    edges
        Edge rows to persist.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of edges persisted.
    """
    if not edges:
        return 0

    gateway.policy.ensure_table("graph.cfg_edges")
    gateway.policy.delete_for_snapshot("graph.cfg_edges", repo=repo, commit=commit)
    gateway.policy.bulk_insert(
        "graph.cfg_edges",
        [edge.to_tuple() for edge in edges],
    )
    return len(edges)


def _persist_dfg_edges(
    gateway: StorageGateway,
    edges: list[DFGEdgeRow],
    repo: str,
    commit: str,
) -> int:
    """Persist DFG edges.

    Parameters
    ----------
    gateway
        Storage gateway.
    edges
        Edge rows to persist.
    repo
        Repository identifier.
    commit
        Commit SHA.

    Returns
    -------
    int
        Number of edges persisted.
    """
    if not edges:
        return 0

    gateway.policy.ensure_table("graph.dfg_edges")
    gateway.policy.delete_for_snapshot("graph.dfg_edges", repo=repo, commit=commit)
    gateway.policy.bulk_insert(
        "graph.dfg_edges",
        [edge.to_tuple() for edge in edges],
    )
    return len(edges)


def _process_all_files(
    paths: list[str],
    function_index: FunctionSpanIndex,
    source_root: Path,
) -> tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]:
    """Process all files and collect CFG/DFG data.

    Parameters
    ----------
    paths
        Paths with functions.
    function_index
        Function span index.
    source_root
        Source root directory.

    Returns
    -------
    tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]
        All collected blocks and edges.
    """
    blocks: list[CFGBlockRow] = []
    cfg_edges: list[CFGEdgeRow] = []
    dfg_edges: list[DFGEdgeRow] = []

    for rel_path in paths:
        functions = _parse_file_functions(source_root / rel_path)
        if not functions:
            continue

        spans = function_index.spans_for_path(rel_path)
        span_map = {(s.start_line, s.end_line): s.goid for s in spans}

        for func_node, start, end in functions:
            goid = span_map.get((start, end)) or function_index.lookup(rel_path, start, end)
            if goid is None:
                continue

            try:
                b, ce, de = _build_cfg_dfg_for_function(goid, func_node, rel_path, start, end)
                blocks.extend(b)
                cfg_edges.extend(ce)
                dfg_edges.extend(de)
            except (ValueError, RuntimeError) as exc:
                log.debug("cfg_dfg: Failed to process function %d: %s", goid, exc)

    return blocks, cfg_edges, dfg_edges


class CfgDfgPlugin(TargetPlugin):
    """Build control flow and data flow graphs.

    This plugin performs full CFG/DFG construction:
    1. Loads function metadata from core.goids
    2. Parses source files and builds CFG for each function
    3. Builds DFG from CFG using reaching definitions
    4. Persists blocks and edges to graph.*

    Outputs
    -------
    - graph.cfg_blocks: CFG basic blocks
    - graph.cfg_edges: CFG edges
    - graph.dfg_edges: DFG data-flow edges
    """

    plugin_name: ClassVar[str] = "cfg_dfg"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Build control flow and data flow graphs."
    _core_metadata: ClassVar[CorePluginMetadata] = CFG_DFG_METADATA

    def __init__(self, *, options_resolver: PluginOptionsResolver | None = None) -> None:
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return _to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return full core metadata."""
        return self._core_metadata

    def resolve_options(
        self,
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> CfgDfgOptions:
        """Resolve typed options from configuration.

        Returns
        -------
        CfgDfgOptions
            Resolved options instance.
        """
        if self._options_resolver is None:
            if dynamic_overrides:
                return CfgDfgOptions(**dynamic_overrides)
            return CfgDfgOptions()

        return self._options_resolver.get_options(
            self._core_metadata,
            CfgDfgOptions,
            dynamic_overrides=dynamic_overrides,
        )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute CFG/DFG construction.

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
        config = CFGBuilderStepConfig(snapshot=ctx.snapshot)
        opts = self.resolve_options()
        gateway, repo, commit = ctx.gateway, config.repo, config.commit

        try:
            function_index = load_function_index(gateway, repo=repo, commit=commit)
            paths = _filter_paths(function_index.paths(), opts)

            if not paths:
                log.info("cfg_dfg: No functions found, skipping")
                return TargetResult.succeeded(
                    row_counts={"graph.cfg_blocks": 0, "graph.cfg_edges": 0, "graph.dfg_edges": 0}
                )

            source_root = (
                ctx.snapshot.repo_root or _get_source_root(gateway, repo, commit) or Path.cwd()
            )
            blocks, cfg_edges, dfg_edges = _process_all_files(paths, function_index, source_root)

            log.info(
                "cfg_dfg: Collected %d blocks, %d cfg_edges, %d dfg_edges from %d files",
                len(blocks),
                len(cfg_edges),
                len(dfg_edges),
                len(paths),
            )

            bc = _persist_cfg_blocks(gateway, blocks, repo, commit)
            ce = _persist_cfg_edges(gateway, cfg_edges, repo, commit)
            de = _persist_dfg_edges(gateway, dfg_edges, repo, commit)

            log.info("cfg_dfg: Persisted %d blocks, %d cfg_edges, %d dfg_edges", bc, ce, de)
            return TargetResult.succeeded(
                row_counts={"graph.cfg_blocks": bc, "graph.cfg_edges": ce, "graph.dfg_edges": de}
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"CFG/DFG build failed: {e}")


__all__ = ["CfgDfgPlugin"]
