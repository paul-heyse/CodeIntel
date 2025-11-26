"""Core snapshot and execution configuration primitives.

.. deprecated::
    This module is deprecated. Use `codeintel.config.primitives` for the canonical
    primitive types:

    Old:
        from codeintel.core.config import SnapshotConfig, ExecutionOptions

    New:
        from codeintel.config.primitives import SnapshotConfig, ExecutionOptions

    The `ExecutionConfig` and `PathsConfig` classes are retained here because they
    depend on `ToolsConfig` (Pydantic) which would create circular imports if moved.
    New code should prefer `codeintel.config.ConfigBuilder` for configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

# Re-export primitive types from the canonical location
from codeintel.config.primitives import (
    ExecutionOptions,
    GraphBackendConfig,
    ScanProfilesConfig,
    SnapshotConfig,
)

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.source_scanner import ScanProfile

__all__ = [
    "ExecutionConfig",
    "ExecutionOptions",
    "PathsConfig",
    "ScanProfilesConfig",
    "SnapshotConfig",
]


@dataclass(frozen=True)
class ExecutionConfig:
    """Runtime execution configuration for a pipeline run.

    .. note::
        This class references `ToolsConfig` (Pydantic) and `ScanProfile` which
        prevents it from being moved to `codeintel.config.primitives`. For new
        code, consider using `ConfigBuilder` from `codeintel.config.builder`.

    Attributes
    ----------
    build_dir : Path
        Root build directory for all generated artifacts.
    tools : ToolsConfig
        Toolchain configuration for external binaries.
    code_profile : ScanProfile
        Profile for scanning Python source files.
    config_profile : ScanProfile
        Profile for scanning configuration files.
    graph_backend : GraphBackendConfig
        Graph backend preferences (CPU vs GPU).
    history_db_dir : Path | None
        Directory containing historical DuckDB snapshots.
    history_commits : tuple[str, ...]
        Commit identifiers to include in historical analysis.
    function_overrides : tuple[str, ...]
        Function identifiers with explicit override settings.
    """

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
    """Derived build paths for a given snapshot and execution config.

    .. note::
        This class composes `SnapshotConfig` and `ExecutionConfig`. For new code,
        consider using `BuildPaths` from `codeintel.config.primitives` or the
        `DerivedPaths` class for the same computed-property pattern.

    Attributes
    ----------
    snapshot : SnapshotConfig
        The snapshot configuration this paths config derives from.
    execution : ExecutionConfig
        The execution configuration providing build_dir and tools.
    """

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
