"""Core snapshot and execution configuration primitives."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codeintel.config.models import GraphBackendConfig, ToolsConfig
from codeintel.ingestion.source_scanner import ScanProfile


@dataclass(frozen=True)
class SnapshotConfig:
    """Immutable description of the code snapshot under analysis."""

    repo_root: Path
    repo_slug: str
    commit: str
    branch: str | None = None

    @classmethod
    def from_args(
        cls,
        repo_root: Path,
        repo_slug: str,
        commit: str,
        branch: str | None = None,
    ) -> SnapshotConfig:
        """Construct a snapshot configuration from primitive arguments.

        Parameters
        ----------
        repo_root
            Root directory for the repository.
        repo_slug
            Repository identifier (e.g., owner/name).
        commit
            Commit hash or identifier for the snapshot.
        branch
            Optional branch name associated with the commit.

        Returns
        -------
        SnapshotConfig
            Normalized snapshot configuration.
        """
        return cls(
            repo_root=repo_root,
            repo_slug=repo_slug,
            commit=commit,
            branch=branch,
        )


@dataclass(frozen=True)
class ScanProfilesConfig:
    """Bundle of code and config scan profiles for a pipeline run."""

    code: ScanProfile
    config: ScanProfile


@dataclass(frozen=True)
class ExecutionOptions:
    """Optional execution tuning parameters."""

    history_db_dir: Path | None = None
    history_commits: tuple[str, ...] = ()
    function_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize tuple fields for immutability."""
        object.__setattr__(self, "history_commits", tuple(self.history_commits))
        object.__setattr__(self, "function_overrides", tuple(self.function_overrides))


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime execution configuration for a pipeline run."""

    build_dir: Path
    tools: ToolsConfig
    code_profile: ScanProfile
    config_profile: ScanProfile
    graph_backend: GraphBackendConfig
    history_db_dir: Path | None = None
    history_commits: tuple[str, ...] = ()
    function_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Resolve paths and normalize tuple fields after initialization."""
        object.__setattr__(self, "build_dir", self.build_dir.resolve())
        object.__setattr__(self, "history_commits", tuple(self.history_commits))
        object.__setattr__(self, "function_overrides", tuple(self.function_overrides))

    @classmethod
    def for_default_pipeline(
        cls,
        *,
        build_dir: Path,
        tools: ToolsConfig,
        profiles: ScanProfilesConfig,
        graph_backend: GraphBackendConfig,
        options: ExecutionOptions | None = None,
    ) -> ExecutionConfig:
        """Build an execution config with normalized optional fields.

        Parameters
        ----------
        build_dir
            Root build directory for all generated artifacts.
        tools
            Toolchain configuration for external binaries.
        profiles
            Code and config scan profiles for this run.
        graph_backend
            Graph backend preferences.
        options
            Optional history and override settings.

        Returns
        -------
        ExecutionConfig
            Normalized execution configuration for the pipeline.
        """
        opts = options or ExecutionOptions()
        return cls(
            build_dir=build_dir,
            tools=tools,
            code_profile=profiles.code,
            config_profile=profiles.config,
            graph_backend=graph_backend,
            history_db_dir=opts.history_db_dir,
            history_commits=opts.history_commits,
            function_overrides=opts.function_overrides,
        )


@dataclass(frozen=True)
class PathsConfig:
    """Derived build paths for a given snapshot and execution config."""

    snapshot: SnapshotConfig
    execution: ExecutionConfig

    @property
    def build_dir(self) -> Path:
        """Root build directory for the run."""
        return self.execution.build_dir

    @property
    def document_output_dir(self) -> Path:
        """Document output directory relative to the repository root."""
        return (self.snapshot.repo_root / "Document Output").resolve()

    @property
    def coverage_json(self) -> Path:
        """Path for coverage JSON output."""
        return (self.build_dir / "coverage" / "coverage.json").resolve()

    @property
    def tool_cache(self) -> Path:
        """Cache directory for external tool artifacts."""
        return (self.build_dir / ".tool_cache").resolve()

    @property
    def pytest_report(self) -> Path:
        """Path for pytest JSON report output."""
        return (self.build_dir / "pytest" / "report.json").resolve()

    @property
    def scip_temp_dir(self) -> Path:
        """Temporary directory for SCIP artifacts."""
        return (self.build_dir / "scip").resolve()

    @property
    def log_db_path(self) -> Path:
        """Path to the pipeline logging DuckDB database."""
        return (self.build_dir / "db" / "codeintel_logs.duckdb").resolve()
