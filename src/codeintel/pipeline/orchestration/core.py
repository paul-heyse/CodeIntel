"""Shared orchestration primitives for pipeline steps.

NOTE: Imports inside methods are intentional to avoid circular dependencies.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

from codeintel.analytics.graph_runtime import (
    GraphKind,
    GraphRuntime,
    GraphRuntimeOptions,
    build_graph_runtime,
)
from codeintel.config import (
    ConfigBuilder,
    GraphBackendConfig,
    GraphRunScope,
    ScanProfiles,
    SnapshotRef,
    ToolsConfig,
)
from codeintel.config.parser_types import FunctionParserKind
from codeintel.config.primitives import BuildPaths
from codeintel.graphs.catalog import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.graphs.engine import GraphEngine
from codeintel.ingestion.change_tracker import ChangeTracker
from codeintel.ingestion.core.execution_context import IngestExecutionContext
from codeintel.ingestion.infrastructure_utilities.source_scanner import ScanProfile
from codeintel.ingestion.infrastructure_utilities.tool_runner import ToolRunner
from codeintel.ingestion.plugins.protocol import IngestRuntimeScratch
from codeintel.ingestion.steps.scip_ingest import ScipIngestResult
from codeintel.ingestion.tool_service import ToolService
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from coverage import Coverage

    from codeintel.config import TestCoverageStepConfig
    from codeintel.config.datasets import (
        CallGraphEdgeRow,
        CFGBlockRow,
        CFGEdgeRow,
        DFGEdgeRow,
    )
    from codeintel.runtime import RunContext

log = logging.getLogger(__name__)


class StepPhase(Enum):
    """Classification of pipeline step phases."""

    INGESTION = "ingestion"
    GRAPHS = "graphs"
    ANALYTICS = "analytics"
    EXPORT = "export"


@dataclass(frozen=True)
class StepMetadata:
    """
    Machine-readable metadata for a pipeline step.

    Parameters
    ----------
    name
        Unique step identifier.
    description
        Human-readable description of what the step does.
    phase
        Pipeline phase this step belongs to.
    deps
        Names of steps this step depends on.
    """

    name: str
    description: str
    phase: StepPhase
    deps: tuple[str, ...]


def _log_step(name: str) -> None:
    """Log step execution at debug level."""
    log.debug("Running pipeline step: %s", name)


@dataclass
class PipelineContext:
    """
    Shared context passed to every pipeline step.

    This matches the repo layout described in your architecture:
      repo_root/
        src/
        Document Output/
        build/
    """

    snapshot: SnapshotRef
    paths: BuildPaths
    gateway: StorageGateway
    code_profile_cfg: ScanProfile
    config_profile_cfg: ScanProfile
    graph_backend_cfg: GraphBackendConfig
    tool_runner: ToolRunner | None = None
    tool_service: ToolService | None = None
    coverage_loader: Callable[[TestCoverageStepConfig], Coverage | None] | None = None
    scip_runner: Callable[..., ScipIngestResult] | None = None
    cst_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
    ast_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
    cfg_builder: (
        Callable[..., tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]] | None
    ) = None
    artifact_writer: Callable[[Path, Path, Path], None] | None = None
    function_catalog: FunctionCatalogProvider | None = None
    extra: dict[str, object] = field(default_factory=dict)
    graph_scope: GraphRunScope | None = None
    function_fail_on_missing_spans: bool = False
    function_parser: FunctionParserKind | None = None
    change_tracker: ChangeTracker | None = None
    tools: ToolsConfig | None = None
    graph_runtime: GraphRuntime | None = None
    export_datasets: tuple[str, ...] | None = None
    export_validation_profile: Literal["strict", "lenient"] | None = None
    force_full_export: bool = False
    run_id: str = ""
    run_context: RunContext | None = None

    @property
    def document_output_dir(self) -> Path:
        """Document Output directory resolved under repo root."""
        return self.paths.document_output_dir

    @property
    def repo_root(self) -> Path:
        """Repository root for the current snapshot."""
        return self.snapshot.repo_root

    @property
    def repo(self) -> str:
        """Repository slug for the current snapshot."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier for the current snapshot."""
        return self.snapshot.commit

    @property
    def db_path(self) -> Path:
        """DuckDB backing database path."""
        return self.gateway.config.db_path

    @property
    def build_dir(self) -> Path:
        """Build directory resolved from execution config."""
        return self.paths.build_dir

    @property
    def tools_config(self) -> ToolsConfig:
        """Resolve the active tools configuration."""
        return self.tools or ToolsConfig.default()

    @property
    def code_profile(self) -> ScanProfile:
        """Code scan profile for this run."""
        return self.code_profile_cfg

    @property
    def config_profile(self) -> ScanProfile:
        """Config scan profile for this run."""
        return self.config_profile_cfg

    @property
    def graph_backend(self) -> GraphBackendConfig:
        """Graph backend configuration."""
        return self.graph_backend_cfg

    @property
    def graph_engine(self) -> GraphEngine | None:
        """Convenience accessor for the shared graph engine."""
        return self.graph_runtime.engine if self.graph_runtime is not None else None

    @property
    def effective_run_id(self) -> str:
        """Get run ID preferring unified RunContext if present.

        Returns
        -------
        str
            Run ID from run_context if set, otherwise falls back to run_id.
        """
        if self.run_context is not None:
            return self.run_context.run_id
        return self.run_id

    def config_builder(self) -> ConfigBuilder:
        """
        Create a ConfigBuilder from this pipeline context.

        Returns
        -------
        ConfigBuilder
            Builder configured with snapshot and paths from this context.
        """
        return ConfigBuilder.from_primitives(
            snapshot=self.snapshot,
            paths=self.paths,
            profiles=ScanProfiles(code=self.code_profile, config=self.config_profile),
            graph_backend=self.graph_backend,
        )


class PipelineStep(Protocol):
    """
    Contract for pipeline steps.

    Each step must define:
    - name: Unique identifier for the step.
    - description: Human-readable description of the step's purpose.
    - phase: The pipeline phase this step belongs to.
    - deps: Sequence of step names this step depends on.
    - run(): Method to execute the step with a PipelineContext.
    """

    name: str
    description: str
    phase: StepPhase
    deps: Sequence[str]

    def run(self, ctx: PipelineContext) -> None:
        """Execute the step using shared context."""


def _resolve_code_profile(ctx: PipelineContext) -> ScanProfile:
    """
    Return the configured code scan profile.

    Returns
    -------
    ScanProfile
        Code scan profile for the current run.
    """
    return ctx.code_profile


def _resolve_config_profile(ctx: PipelineContext) -> ScanProfile:
    """
    Return the configured config scan profile.

    Returns
    -------
    ScanProfile
        Config scan profile for the current run.
    """
    return ctx.config_profile


def _plugin_ctx(
    ctx: PipelineContext,
    *,
    scratch: IngestRuntimeScratch | None = None,
    plugin_name: str | None = None,
) -> IngestExecutionContext:
    """Build an IngestExecutionContext from a pipeline context.

    Parameters
    ----------
    ctx
        Pipeline context to convert.
    scratch
        Optional shared scratch space; creates new one if not provided.
    plugin_name
        Optional name of the plugin being executed.

    Returns
    -------
    IngestExecutionContext
        Context suitable for new plugin architecture.
    """
    # Import resource providers (inside function to avoid circular imports)
    from codeintel.ingestion.resources.modules import ModuleProvider
    from codeintel.ingestion.resources.registry import ResourceRegistry
    from codeintel.ingestion.resources.tools import ToolsProvider

    # Build resource registry with required providers
    resources = ResourceRegistry()

    # Register ModuleProvider for module access
    module_provider = ModuleProvider(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        tracker=ctx.change_tracker,
        profile=ctx.code_profile,
    )
    resources.register(ModuleProvider, module_provider)

    # Register ToolsProvider for external tool access
    tools_provider = ToolsProvider(
        tools_config=ctx.tools_config,
        cache_dir=ctx.paths.tool_cache,
    )
    resources.register(ToolsProvider, tools_provider)

    return IngestExecutionContext(
        gateway=ctx.gateway,
        snapshot=ctx.snapshot,
        paths=ctx.paths,
        tools=ctx.tools_config,
        code_profile=ctx.code_profile,
        config_profile=ctx.config_profile,
        resources=resources,
        scratch=scratch or IngestRuntimeScratch(),
        plugin_name=plugin_name,
        run_id=ctx.run_id or "",
        run_context=ctx.run_context,
    )


def _function_catalog(ctx: PipelineContext) -> FunctionCatalogProvider:
    """
    Return a cached FunctionCatalogService, constructing it on first access.

    Returns
    -------
    FunctionCatalogProvider
        Catalog service for the current repo and commit.
    """
    if ctx.function_catalog is None:
        ctx.function_catalog = FunctionCatalogService.from_db(
            ctx.gateway, repo=ctx.repo, commit=ctx.commit
        )
    return ctx.function_catalog


def _graph_engine(ctx: PipelineContext) -> GraphEngine:
    """
    Construct or reuse a shared graph engine for the pipeline run.

    Returns
    -------
    GraphEngine
        Engine bound to the pipeline snapshot.
    """
    return _graph_runtime(ctx).engine


def _graph_runtime(ctx: PipelineContext) -> GraphRuntime:
    """
    Build a runtime bundle for graph consumers using the shared engine.

    Returns
    -------
    GraphRuntime
        Runtime configured to reuse the shared engine and backend.
    """
    if ctx.graph_runtime is not None:
        return ctx.graph_runtime
    options = GraphRuntimeOptions(
        snapshot=ctx.snapshot,
        backend=ctx.graph_backend,
        graphs=GraphKind.ALL,
        eager=False,
        validate=False,
    )
    runtime = build_graph_runtime(
        ctx.gateway,
        options,
    )
    ctx.graph_runtime = runtime
    return runtime


def ensure_graph_engine(ctx: PipelineContext) -> GraphEngine:
    """
    Public helper to retrieve the shared graph engine for the pipeline snapshot.

    Returns
    -------
    GraphEngine
        Shared engine keyed to the pipeline's repo and commit.
    """
    return _graph_engine(ctx)


def ensure_graph_runtime(ctx: PipelineContext) -> GraphRuntime:
    """
    Public helper to construct a shared graph runtime for the pipeline snapshot.

    Returns
    -------
    GraphRuntime
        Runtime bound to the shared engine and backend preferences.
    """
    return _graph_runtime(ctx)


__all__ = [
    "PipelineContext",
    "PipelineStep",
    "StepMetadata",
    "StepPhase",
    "_function_catalog",
    "_graph_engine",
    "_graph_runtime",
    "_log_step",
    "_plugin_ctx",
    "_resolve_code_profile",
    "_resolve_config_profile",
    "ensure_graph_engine",
    "ensure_graph_runtime",
]
