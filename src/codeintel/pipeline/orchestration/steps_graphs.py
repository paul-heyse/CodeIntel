"""Graph-related pipeline steps."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from codeintel.graphs.callgraph_builder import build_call_graph
from codeintel.graphs.cfg_builder import build_cfg_and_dfg
from codeintel.graphs.goid_builder import build_goids
from codeintel.graphs.import_graph import build_import_graph
from codeintel.graphs.symbol_uses import build_symbol_use_edges
from codeintel.graphs.validation import run_graph_validations
from codeintel.pipeline.orchestration.core import (
    PipelineContext,
    PipelineStep,
    StepPhase,
    _function_catalog,
    _log_step,
    ensure_graph_runtime,
)

log = logging.getLogger(__name__)


@dataclass
class GoidsStep:
    """Build core.goids and core.goid_crosswalk from AST."""

    name: str = "goids"
    description: str = "Build core.goids and core.goid_crosswalk from AST nodes."
    phase: StepPhase = StepPhase.GRAPHS
    deps: Sequence[str] = ("ast_extract",)

    def run(self, ctx: PipelineContext) -> None:
        """Build GOID registry and crosswalk tables."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = ctx.config_builder().goid_builder(language="python")
        build_goids(gateway, cfg)


@dataclass
class CallGraphStep:
    """Build graph.call_graph_nodes and graph.call_graph_edges."""

    name: str = "callgraph"
    description: str = "Build static call graph nodes and edges from CST/AST analysis."
    phase: StepPhase = StepPhase.GRAPHS
    deps: Sequence[str] = ("goids", "repo_scan")

    def run(self, ctx: PipelineContext) -> None:
        """Construct static call graph nodes and edges."""
        _log_step(self.name)
        gateway = ctx.gateway
        catalog = _function_catalog(ctx)
        cfg = ctx.config_builder().call_graph(
            cst_collector=ctx.cst_collector,
            ast_collector=ctx.ast_collector,
        )
        build_call_graph(gateway, cfg, catalog_provider=catalog)


@dataclass
class CFGStep:
    """Build graph.cfg_blocks, graph.cfg_edges, and graph.dfg_edges."""

    name: str = "cfg"
    description: str = "Build control-flow and data-flow graph structures."
    phase: StepPhase = StepPhase.GRAPHS
    deps: Sequence[str] = ("function_metrics",)

    def run(self, ctx: PipelineContext) -> None:
        """Create minimal CFG/DFG scaffolding."""
        _log_step(self.name)
        gateway = ctx.gateway
        catalog = _function_catalog(ctx)
        cfg = ctx.config_builder().cfg_builder(cfg_builder=ctx.cfg_builder)
        build_cfg_and_dfg(gateway, cfg, catalog_provider=catalog)


@dataclass
class ImportGraphStep:
    """Build graph.import_graph_edges from LibCST imports."""

    name: str = "import_graph"
    description: str = "Build module import graph edges from LibCST analysis."
    phase: StepPhase = StepPhase.GRAPHS
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext) -> None:
        """Construct module import graph edges."""
        _log_step(self.name)
        gateway = ctx.gateway
        cfg = ctx.config_builder().import_graph()
        build_import_graph(gateway, cfg)


@dataclass
class SymbolUsesStep:
    """Build graph.symbol_use_edges from index.scip.json."""

    name: str = "symbol_uses"
    description: str = "Build symbol definition-to-use edges from SCIP index."
    phase: StepPhase = StepPhase.GRAPHS
    deps: Sequence[str] = ("repo_scan", "scip_ingest")

    def run(self, ctx: PipelineContext) -> None:
        """Derive symbol definition→use edges from SCIP JSON."""
        _log_step(self.name)
        gateway = ctx.gateway
        catalog = _function_catalog(ctx)
        scip_json = ctx.build_dir / "scip" / "index.scip.json"
        if not scip_json.is_file():
            log.info("Skipping symbol_uses: SCIP JSON missing at %s", scip_json)
            return
        cfg = ctx.config_builder().symbol_uses(scip_json_path=scip_json)
        build_symbol_use_edges(gateway, cfg, catalog_provider=catalog)


@dataclass
class GraphValidationStep:
    """Run integrity validations over graph datasets."""

    name: str = "graph_validation"
    description: str = "Validate graph integrity for GOIDs, spans, and orphan nodes."
    phase: StepPhase = StepPhase.GRAPHS
    deps: Sequence[str] = ("callgraph", "cfg")

    def run(self, ctx: PipelineContext) -> None:
        """Emit warnings for missing GOIDs, span mismatches, and orphans."""
        _log_step(self.name)
        gateway = ctx.gateway
        catalog = _function_catalog(ctx)
        runtime = ensure_graph_runtime(ctx)
        run_graph_validations(
            gateway,
            repo=ctx.repo,
            commit=ctx.commit,
            catalog_provider=catalog,
            logger=log,
            runtime=runtime,
        )


GRAPH_STEPS: dict[str, PipelineStep] = {
    "goids": GoidsStep(),
    "callgraph": CallGraphStep(),
    "cfg": CFGStep(),
    "import_graph": ImportGraphStep(),
    "symbol_uses": SymbolUsesStep(),
    "graph_validation": GraphValidationStep(),
}


__all__ = [
    "GRAPH_STEPS",
    "CFGStep",
    "CallGraphStep",
    "GoidsStep",
    "GraphValidationStep",
    "ImportGraphStep",
    "SymbolUsesStep",
]
