"""CLI adapter for unified pipeline execution.

This module provides a bridge between CLI arguments and the spec-based
pipeline execution system via :class:`CliPipelineArgs`.

The adapter handles:
- Converting CLI options to :class:`PipelinePlanOptions`
- Resolving configuration with environment overrides
- Building :class:`SnapshotRef` and :class:`BuildPaths` from user inputs
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from codeintel.config import GraphRunScope, SnapshotRef
from codeintel.config.primitives import BuildPaths, GraphBackendConfig
from codeintel.core.execution import TriggerKind
from codeintel.pipeline.planning.planner import PipelinePlanOptions

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.config.parser_types import FunctionParserKind
    from codeintel.ingestion.infrastructure.scanning import ScanProfile
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class CliPipelineArgs:
    """CLI-layer arguments that translate to PipelinePlanOptions.

    This dataclass captures all CLI arguments needed to execute a pipeline
    and provides conversion methods to the spec-based execution system.

    Parameters
    ----------
    repo_root
        Path to the repository root directory.
    repo
        Repository identifier (e.g., "org/repo").
    commit
        Commit SHA to process.
    db_path
        Path to the DuckDB database file.
    build_dir
        Build directory for intermediate artifacts.
    tools
        Optional tools configuration.
    code_profile
        Optional code scan profile.
    config_profile
        Optional config scan profile.
    graph_backend
        Graph backend configuration.
    graph_scope
        Graph run scope filtering.
    function_fail_on_missing_spans
        Fail when function spans are missing.
    function_parser
        Parser selector for function analytics.
    history_commits
        Commits to include in history timeseries.
    history_db_dir
        Directory containing per-commit DuckDB snapshots.
    export_datasets
        Datasets to export during docs export step.
    export_validation_profile
        Override validation profile: strict or lenient.
    force_full_export
        Force re-export even when incremental markers match.
    log_db_path
        Optional path to the log database.
    trigger
        How the pipeline run was triggered.
    """

    repo_root: Path
    repo: str
    commit: str
    db_path: Path
    build_dir: Path
    tools: ToolsConfig | None = None
    code_profile: ScanProfile | None = None
    config_profile: ScanProfile | None = None
    graph_backend: GraphBackendConfig | None = None
    graph_scope: GraphRunScope | None = None
    function_fail_on_missing_spans: bool = False
    function_parser: FunctionParserKind | None = None
    history_commits: tuple[str, ...] | None = None
    history_db_dir: Path | None = None
    export_datasets: tuple[str, ...] | None = None
    export_validation_profile: Literal["strict", "lenient"] | None = None
    force_full_export: bool = False
    log_db_path: Path | None = None
    trigger: TriggerKind = "cli"

    def snapshot_ref(self) -> SnapshotRef:
        """Build a SnapshotRef from CLI arguments.

        Returns
        -------
        SnapshotRef
            Normalized snapshot descriptor.
        """
        return SnapshotRef(
            repo_root=self.repo_root,
            repo=self.repo,
            commit=self.commit,
        )

    def build_paths(self) -> BuildPaths:
        """Build paths configuration from CLI arguments.

        Returns
        -------
        BuildPaths
            Normalized build paths anchored to repo_root/build.
        """
        return BuildPaths.from_layout(
            repo_root=self.repo_root,
            build_dir=self.build_dir,
            db_path=self.db_path,
            document_output_dir=self.repo_root / "Document Output",
            log_db_path=self.log_db_path,
        )

    def to_plan_options(self, gateway: StorageGateway, tools: ToolsConfig) -> PipelinePlanOptions:
        """Convert CLI args to execution plan options.

        Parameters
        ----------
        gateway
            Storage gateway for database operations.
        tools
            Resolved tools configuration.

        Returns
        -------
        PipelinePlanOptions
            Options suitable for :func:`build_pipeline_plan`.
        """
        return PipelinePlanOptions(
            snapshot=self.snapshot_ref(),
            paths=self.build_paths(),
            gateway=gateway,
            tools=tools,
            trigger=self.trigger,
        )


__all__ = [
    "CliPipelineArgs",
]
