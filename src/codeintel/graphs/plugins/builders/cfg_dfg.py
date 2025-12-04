"""CFG and DFG builder plugin using factory pattern.

This module provides the control-flow graph (CFG) and data-flow graph (DFG)
builder as a graph plugin. All orchestration logic is contained here.

Uses resource injection pattern via ctx.require() to access storage.

Architecture notes:
- Pure computation functions are in graphs.compute.cfg and graphs.compute.dfg
- This plugin orchestrates file I/O and database persistence
- The compute layer is stateless and testable in isolation
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import networkx as nx

from codeintel.config import CFGBuilderStepConfig
from codeintel.config.datasets import CFGBlockRow as DatasetCFGBlockRow
from codeintel.config.datasets import CFGEdgeRow as DatasetCFGEdgeRow
from codeintel.config.datasets import DFGEdgeRow as DatasetDFGEdgeRow
from codeintel.config.datasets import (
    cfg_block_to_tuple,
    cfg_edge_to_tuple,
    dfg_edge_to_tuple,
)
from codeintel.graphs.catalog import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.graphs.compute.cfg import (
    BasicBlock as Block,
)
from codeintel.graphs.compute.cfg import (
    CFGEdge as Edge,
)
from codeintel.graphs.compute.cfg import (
    build_cfg,
    cfg_to_rows,
)
from codeintel.graphs.compute.dfg import (
    DFGBuilder as ComputeDFGBuilder,
)
from codeintel.graphs.compute.dfg import (
    build_dfg,
    dfg_to_rows,
)
from codeintel.graphs.core import (
    ComputationResult,
    GraphPluginExecutionContext,
    GraphPluginProtocol,
    make_builder_plugin,
)
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.resources import StorageResource
from codeintel.ingestion.adapters import IngestStorageService
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


class DFGBuilder:
    """Build DFG from CFG blocks - delegates to compute layer.

    This class wraps the compute layer DFGBuilder to convert results
    to dataset row types for persistence.
    """

    def __init__(self, goid: int, blocks: list[Block], edges: list[Edge]) -> None:
        """Initialize DFG builder with CFG data.

        Parameters
        ----------
        goid
            Function GOID.
        blocks
            CFG basic blocks.
        edges
            CFG edges.
        """
        self.goid = goid
        self.blocks = blocks
        self.edges = edges
        cfg_edges = [Edge(src=e.src, dst=e.dst, kind=e.kind) for e in edges]
        self._builder = ComputeDFGBuilder(goid, blocks, cfg_edges)
        self.dfg_edges: list[DatasetDFGEdgeRow] = []

    def build(self) -> list[DatasetDFGEdgeRow]:
        """Construct DFG edges using reaching definitions.

        Delegates to the compute layer and converts results to dataset types.

        Returns
        -------
        list[DatasetDFGEdgeRow]
            Data flow edges linking definitions to uses.
        """
        result = self._builder.build()
        compute_rows = dfg_to_rows(result)
        self.dfg_edges = [
            DatasetDFGEdgeRow(
                function_goid_h128=row.function_goid_h128,
                src_block_id=row.src_block_id,
                dst_block_id=row.dst_block_id,
                src_var=row.src_var,
                dst_var=row.dst_var,
                edge_kind=row.edge_kind,
                via_phi=row.via_phi,
                use_kind=row.use_kind,
            )
            for row in compute_rows
        ]
        return self.dfg_edges

    def as_nx_digraph(self) -> nx.DiGraph:
        """Convert emitted DFG edges into a NetworkX DiGraph.

        Returns
        -------
        nx.DiGraph
            Directed data-flow graph keyed by block indices.
        """
        graph = nx.DiGraph()
        for block in self.blocks:
            graph.add_node(block.idx, kind=block.kind, label=block.label)
        for edge in self.dfg_edges:
            src_idx = _parse_block_idx(edge["src_block_id"])
            dst_idx = _parse_block_idx(edge["dst_block_id"])
            if src_idx is None or dst_idx is None:
                continue
            graph.add_edge(
                src_idx,
                dst_idx,
                src_var=edge["src_var"],
                dst_var=edge["dst_var"],
                edge_kind=edge["edge_kind"],
            )
        return graph


def _parse_block_idx(block_id: str) -> int | None:
    """Extract the integer block index suffix from a block identifier.

    Parameters
    ----------
    block_id
        Block identifier shaped like "<goid>:block<idx>".

    Returns
    -------
    int | None
        Parsed block index or None when parsing fails.
    """
    if "block" not in block_id:
        return None
    try:
        return int(block_id.rsplit("block", 1)[-1])
    except ValueError:
        return None


@dataclass(frozen=True)
class FunctionBuildSpec:
    """Specification of a single function to build CFG/DFG rows for."""

    goid: int
    repo_root: Path
    rel_path: str
    lines: tuple[int, int]
    qualname: str


def _load_source(spec: FunctionBuildSpec, file_cache: dict[str, str]) -> str:
    """Load source code for a function, caching file contents.

    Parameters
    ----------
    spec
        Function specification with path information.
    file_cache
        Cache of file contents to avoid repeated reads.

    Returns
    -------
    str
        Source code contents, or empty string on read failure.
    """
    if spec.rel_path not in file_cache:
        try:
            file_cache[spec.rel_path] = (spec.repo_root / spec.rel_path).read_text(encoding="utf8")
        except OSError:
            file_cache[spec.rel_path] = ""
    return file_cache[spec.rel_path]


def _parse_function_ast(source: str) -> ast.Module | None:
    """Parse source code into an AST module.

    Parameters
    ----------
    source
        Python source code.

    Returns
    -------
    ast.Module | None
        Parsed AST module or None on syntax error.
    """
    if not source:
        return None
    try:
        return ast.parse(source)
    except SyntaxError:
        return None


def _select_target_node(
    tree: ast.Module, start_line: int, end_line: int, qualname: str
) -> ast.FunctionDef | ast.AsyncFunctionDef | None:
    """Find the target function node in the AST.

    Parameters
    ----------
    tree
        Parsed AST module.
    start_line
        Expected start line of the function.
    end_line
        Expected end line of the function.
    qualname
        Qualified name of the function.

    Returns
    -------
    ast.FunctionDef | ast.AsyncFunctionDef | None
        The matching function node or None if not found.
    """
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.lineno == start_line:
                return node
            if qualname.endswith(node.name) and start_line <= node.lineno <= end_line:
                return node
    return None


def _build_cfg_for_function(
    spec: FunctionBuildSpec, file_cache: dict[str, str]
) -> tuple[list[DatasetCFGBlockRow], list[DatasetCFGEdgeRow], list[DatasetDFGEdgeRow]]:
    """Build CFG and DFG for a single function using the compute layer.

    Parameters
    ----------
    spec
        Function specification with GOID, path, and line range.
    file_cache
        Cache of file contents to avoid repeated reads.

    Returns
    -------
    tuple[list[DatasetCFGBlockRow], list[DatasetCFGEdgeRow], list[DatasetDFGEdgeRow]]
        Block rows, CFG edge rows, and DFG edge rows.
    """
    start_line, end_line = spec.lines
    source = _load_source(spec, file_cache)
    tree = _parse_function_ast(source)
    if tree is None:
        return [], [], []

    target_node = _select_target_node(tree, start_line, end_line, spec.qualname)
    if target_node is None:
        return [], [], []

    # Delegate CFG construction to compute layer
    cfg_result = build_cfg(spec.goid, target_node, spec.rel_path)

    # Convert compute layer results to dataset types
    compute_block_rows, compute_edge_rows = cfg_to_rows(
        cfg_result, spec.rel_path, start_line, end_line
    )

    block_rows = [
        DatasetCFGBlockRow(
            function_goid_h128=row.function_goid_h128,
            block_idx=row.block_idx,
            block_id=row.block_id,
            label=row.label,
            file_path=row.file_path,
            start_line=row.start_line,
            end_line=row.end_line,
            kind=row.kind,
            stmts_json=row.stmts_json,
            in_degree=row.in_degree,
            out_degree=row.out_degree,
        )
        for row in compute_block_rows
    ]

    edge_rows = [
        DatasetCFGEdgeRow(
            function_goid_h128=row.function_goid_h128,
            src_block_id=row.src_block_id,
            dst_block_id=row.dst_block_id,
            edge_kind=row.edge_kind,
        )
        for row in compute_edge_rows
    ]

    # Delegate DFG construction to compute layer
    dfg_result = build_dfg(spec.goid, list(cfg_result.blocks), list(cfg_result.edges))
    compute_dfg_rows = dfg_to_rows(dfg_result)

    dfg_rows = [
        DatasetDFGEdgeRow(
            function_goid_h128=row.function_goid_h128,
            src_block_id=row.src_block_id,
            dst_block_id=row.dst_block_id,
            src_var=row.src_var,
            dst_var=row.dst_var,
            edge_kind=row.edge_kind,
            via_phi=row.via_phi,
            use_kind=row.use_kind,
        )
        for row in compute_dfg_rows
    ]

    return block_rows, edge_rows, dfg_rows


def _flush(
    gateway: StorageGateway,
    blocks: list[DatasetCFGBlockRow],
    cfg_edges: list[DatasetCFGEdgeRow],
    dfg_edges: list[DatasetDFGEdgeRow],
) -> None:
    """Persist CFG and DFG data to storage.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    blocks
        CFG block rows to persist.
    cfg_edges
        CFG edge rows to persist.
    dfg_edges
        DFG edge rows to persist.
    """
    storage_service = IngestStorageService.from_gateway(gateway)
    if blocks:
        storage_service.run_batch(
            "graph.cfg_blocks",
            [cfg_block_to_tuple(r) for r in blocks],
            delete_params=[],  # Append only
            scope="cfg_blocks",
        )
    if cfg_edges:
        storage_service.run_batch(
            "graph.cfg_edges",
            [cfg_edge_to_tuple(r) for r in cfg_edges],
            scope="cfg_edges",
        )
    if dfg_edges:
        storage_service.run_batch(
            "graph.dfg_edges",
            [dfg_edge_to_tuple(r) for r in dfg_edges],
            scope="dfg_edges",
        )


def build_cfg_and_dfg(
    gateway: StorageGateway,
    cfg: CFGBuilderStepConfig,
    *,
    cfg_builder: Callable[
        [FunctionBuildSpec, dict[str, str]],
        tuple[list[DatasetCFGBlockRow], list[DatasetCFGEdgeRow], list[DatasetDFGEdgeRow]],
    ]
    | None = None,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> None:
    """Emit CFG and DFG edges for each function GOID.

    This function orchestrates:
    1. Loading function metadata from the catalog
    2. Delegating pure CFG/DFG construction to the compute layer
    3. Persisting results to storage

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    cfg
        CFG builder step configuration.
    cfg_builder
        Optional custom builder function for testing.
    catalog_provider
        Optional catalog provider for function metadata.
    """
    con = gateway.con
    # Clear existing data for this repo/commit to allow idempotent re-runs
    log.info("Clearing existing CFG/DFG data for %s@%s", cfg.repo, cfg.commit)
    con.execute(
        """
        DELETE FROM graph.cfg_blocks
        WHERE function_goid_h128 IN (
            SELECT goid_h128 FROM core.goids WHERE repo = ? AND commit = ?
        )
        """,
        [cfg.repo, cfg.commit],
    )
    con.execute(
        """
        DELETE FROM graph.cfg_edges
        WHERE function_goid_h128 IN (
            SELECT goid_h128 FROM core.goids WHERE repo = ? AND commit = ?
        )
        """,
        [cfg.repo, cfg.commit],
    )
    con.execute(
        """
        DELETE FROM graph.dfg_edges
        WHERE function_goid_h128 IN (
            SELECT goid_h128 FROM core.goids WHERE repo = ? AND commit = ?
        )
        """,
        [cfg.repo, cfg.commit],
    )

    provider = catalog_provider or FunctionCatalogService.from_db(
        gateway, repo=cfg.repo, commit=cfg.commit
    )
    function_spans = provider.catalog().function_spans
    if not function_spans:
        log.warning("No function GOIDs found; skipping CFG/DFG build.")
        return

    log.info("Building CFG/DFG for %d functions...", len(function_spans))

    all_blocks: list[DatasetCFGBlockRow] = []
    all_cfg_edges: list[DatasetCFGEdgeRow] = []
    all_dfg_edges: list[DatasetDFGEdgeRow] = []

    file_cache: dict[str, str] = {}

    for span in function_spans:
        start = span.start_line
        end = span.end_line

        spec = FunctionBuildSpec(
            goid=span.goid,
            repo_root=cfg.repo_root,
            rel_path=span.rel_path,
            lines=(start, end),
            qualname=span.qualname,
        )
        builder = cfg_builder or cfg.cfg_builder or _build_cfg_for_function
        blocks, edges, dfg = builder(spec, file_cache)

        all_blocks.extend(blocks)
        all_cfg_edges.extend(edges)
        all_dfg_edges.extend(dfg)

    _flush(gateway, all_blocks, all_cfg_edges, all_dfg_edges)

    log.info("CFG/DFG build complete.")


def _build_cfg_and_dfg(ctx: GraphPluginExecutionContext) -> ComputationResult:
    """Build control-flow and data-flow graphs for functions.

    Uses resource injection to access storage.

    Returns
    -------
    ComputationResult
        Success result after building CFG and DFG artifacts.
    """
    storage = ctx.require(StorageResource)
    gateway = storage.gateway

    cfg = CFGBuilderStepConfig(snapshot=ctx.snapshot)
    build_cfg_and_dfg(gateway, cfg)
    return ComputationResult.ok()


cfg_dfg_builder_plugin = make_builder_plugin(
    name="cfg_dfg_builder",
    computation=_build_cfg_and_dfg,
    stage="edges",
    produces_graph_kinds=(GraphKind.CFG_GRAPH,),
    depends_on=("goid_builder",),
    provides=("cfg_graph", "dfg_graph"),
    produces_tables=("graph.cfg_blocks", "graph.cfg_edges", "graph.dfg_edges"),
)


def get_cfg_dfg_builder_plugin() -> GraphPluginProtocol:
    """Return the CFG/DFG builder plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The configured CFG/DFG builder plugin.
    """
    return cfg_dfg_builder_plugin


__all__ = [
    "Block",
    "DFGBuilder",
    "Edge",
    "FunctionBuildSpec",
    "build_cfg_and_dfg",
    "cfg_dfg_builder_plugin",
    "get_cfg_dfg_builder_plugin",
]
