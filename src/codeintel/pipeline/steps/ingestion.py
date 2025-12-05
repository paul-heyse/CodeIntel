"""Pipeline ingestion step implementations using the new plugin architecture.

This module provides pipeline step wrappers that delegate to the ingestion plugin
registry, enabling a unified plugin-driven ingestion approach.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from codeintel.core.plugins.execution.context import PluginScratch
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.ingestion.plugins.registry import get_ingest_registry
from codeintel.ingestion.tracker import ChangeTracker
from codeintel.pipeline.execution.context import (
    PipelineContext,
    _log_step,
    _plugin_ctx,
)
from codeintel.pipeline.steps.base import PipelineStep, StepPhase, step_to_plugin_metadata

log = logging.getLogger(__name__)


def _get_shared_scratch() -> PluginScratch:
    """Get or create the shared scratch space for the current pipeline run.

    Returns
    -------
    PluginScratch
        Shared scratch space for storing intermediate data during ingestion.
    """
    return _shared_scratch_cache()


def reset_shared_scratch() -> None:
    """Reset the shared scratch space between pipeline runs."""
    scratch = _shared_scratch_cache()
    scratch.cleanup()
    _shared_scratch_cache.cache_clear()


@lru_cache(maxsize=1)
def _shared_scratch_cache() -> PluginScratch:
    return PluginScratch()


def _execute_plugin(ctx: PipelineContext, plugin_name: str) -> None:
    """Execute an ingestion plugin by name.

    Parameters
    ----------
    ctx
        Pipeline context.
    plugin_name
        Name of the plugin to execute.
    """
    registry = get_ingest_registry()
    plugin = registry.get(plugin_name)
    scratch = _get_shared_scratch()

    # Build plugin context with shared scratch
    plugin_ctx = _plugin_ctx(ctx, scratch=scratch, plugin_name=plugin_name)

    # If change_tracker is available in pipeline context, populate scratch
    if ctx.change_tracker is not None and not scratch.has("change_tracker"):
        scratch.declare("change_tracker", ctx.change_tracker)

    result = plugin.execute(plugin_ctx)

    # Extract change_tracker from scratch if repo_scan populated it
    if plugin_name == "repo_scan" and result.success:
        tracker = scratch.consume("change_tracker")
        if tracker is not None and isinstance(tracker, ChangeTracker):
            ctx.change_tracker = tracker

    if not result.success:
        log.warning(
            "Plugin %s failed: %s (kind=%s)",
            plugin_name,
            result.error,
            result.error_kind,
        )
    elif result.skipped:
        log.info("Plugin %s skipped: %s", plugin_name, result.skip_reason)


@dataclass
class SchemaBootstrapStep:
    """Apply schemas and create views before ingestion."""

    name: str = "schema_bootstrap"
    description: str = "Apply database schemas and create views before ingestion."
    phase: StepPhase = StepPhase.INGESTION
    deps: Sequence[str] = ()

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility."""
        return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)

    def run(self, ctx: PipelineContext) -> None:
        """Reset shared scratch and prepare for new run."""
        _log_step(self.name)
        reset_shared_scratch()
        _ = ctx


@dataclass
class RepoScanStep:
    """Ingest repository modules and repo_map."""

    name: str = "repo_scan"
    description: str = "Scan repository and ingest modules into core.modules."
    phase: StepPhase = StepPhase.INGESTION
    deps: Sequence[str] = ("schema_bootstrap",)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility."""
        return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)

    def run(self, ctx: PipelineContext) -> None:
        """Execute repository scan ingestion via plugin."""
        _log_step(self.name)
        _execute_plugin(ctx, "repo_scan")


@dataclass
class SCIPIngestStep:
    """Run scip-python and register SCIP artifacts/view."""

    name: str = "scip_ingest"
    description: str = "Run scip-python and register SCIP artifacts and symbols."
    phase: StepPhase = StepPhase.INGESTION
    deps: Sequence[str] = ("repo_scan",)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility."""
        return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)

    def run(self, ctx: PipelineContext) -> None:
        """Register SCIP artifacts and populate SCIP symbols in crosswalk."""
        _log_step(self.name)
        _execute_plugin(ctx, "scip_ingest")


@dataclass
class CSTStep:
    """Parse CST and persist rows."""

    name: str = "cst_extract"
    description: str = "Parse CST using LibCST and persist rows into core.cst_nodes."
    phase: StepPhase = StepPhase.INGESTION
    deps: Sequence[str] = ("repo_scan",)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility."""
        return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)

    def run(self, ctx: PipelineContext) -> None:
        """Extract CST rows into core.cst_nodes."""
        _log_step(self.name)
        _execute_plugin(ctx, "cst_extract")


@dataclass
class AstStep:
    """Parse stdlib AST and persist rows/metrics."""

    name: str = "ast_extract"
    description: str = "Parse Python AST and persist rows and metrics into core tables."
    phase: StepPhase = StepPhase.INGESTION
    deps: Sequence[str] = ("repo_scan",)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility."""
        return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)

    def run(self, ctx: PipelineContext) -> None:
        """Extract AST rows and metrics into core tables."""
        _log_step(self.name)
        _execute_plugin(ctx, "ast_extract")


@dataclass
class CoverageIngestStep:
    """Load coverage.py data into analytics.coverage_lines."""

    name: str = "coverage_ingest"
    description: str = "Load coverage.py data into analytics.coverage_lines."
    phase: StepPhase = StepPhase.INGESTION
    deps: Sequence[str] = ("repo_scan",)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility."""
        return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)

    def run(self, ctx: PipelineContext) -> None:
        """Ingest line-level coverage signals."""
        _log_step(self.name)
        _execute_plugin(ctx, "coverage_ingest")


@dataclass
class TestsIngestStep:
    """Load pytest JSON report into analytics.test_catalog."""

    name: str = "tests_ingest"
    description: str = "Load pytest JSON report into analytics.test_catalog."
    phase: StepPhase = StepPhase.INGESTION
    deps: Sequence[str] = ("repo_scan",)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility."""
        return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)

    def run(self, ctx: PipelineContext) -> None:
        """Ingest pytest test catalog."""
        _log_step(self.name)
        _execute_plugin(ctx, "tests_ingest")


@dataclass
class TypingIngestStep:
    """Collect typedness/static diagnostics."""

    name: str = "typing_ingest"
    description: str = "Collect typedness and static diagnostics from AST and pyright."
    phase: StepPhase = StepPhase.INGESTION
    deps: Sequence[str] = ("repo_scan",)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility."""
        return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)

    def run(self, ctx: PipelineContext) -> None:
        """Ingest typing signals from ast + pyright."""
        _log_step(self.name)
        _execute_plugin(ctx, "typing_ingest")


@dataclass
class DocstringsIngestStep:
    """Extract and persist structured docstrings."""

    name: str = "docstrings_ingest"
    description: str = "Extract and persist structured docstrings from Python modules."
    phase: StepPhase = StepPhase.INGESTION
    deps: Sequence[str] = ("repo_scan",)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility."""
        return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)

    def run(self, ctx: PipelineContext) -> None:
        """Ingest docstrings for all Python modules."""
        _log_step(self.name)
        _execute_plugin(ctx, "docstrings_ingest")


@dataclass
class ConfigIngestStep:
    """Flatten config files into analytics.config_values."""

    name: str = "config_ingest"
    description: str = "Flatten configuration files into analytics.config_values."
    phase: StepPhase = StepPhase.INGESTION
    deps: Sequence[str] = ("repo_scan",)

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata for registry compatibility."""
        return step_to_plugin_metadata(self.name, self.description, self.phase, self.deps)

    def run(self, ctx: PipelineContext) -> None:
        """Ingest configuration files from repo root."""
        _log_step(self.name)
        _execute_plugin(ctx, "config_ingest")


INGESTION_STEPS: dict[str, PipelineStep] = {
    "schema_bootstrap": SchemaBootstrapStep(),
    "repo_scan": RepoScanStep(),
    "scip_ingest": SCIPIngestStep(),
    "cst_extract": CSTStep(),
    "ast_extract": AstStep(),
    "coverage_ingest": CoverageIngestStep(),
    "tests_ingest": TestsIngestStep(),
    "typing_ingest": TypingIngestStep(),
    "docstrings_ingest": DocstringsIngestStep(),
    "config_ingest": ConfigIngestStep(),
}


__all__ = [
    "INGESTION_STEPS",
    "AstStep",
    "CSTStep",
    "ConfigIngestStep",
    "CoverageIngestStep",
    "DocstringsIngestStep",
    "RepoScanStep",
    "SCIPIngestStep",
    "SchemaBootstrapStep",
    "TestsIngestStep",
    "TypingIngestStep",
    "reset_shared_scratch",
]
