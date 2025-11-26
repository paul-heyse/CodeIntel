"""Core configuration primitives for composition across pipeline steps.

This module defines the foundational configuration types used throughout the
CodeIntel pipeline. These frozen dataclasses serve as composable building blocks
for step-specific configurations, eliminating repetitive field definitions.

Design Principles
-----------------
1. All primitives are frozen dataclasses for immutability and hashability.
2. Path resolution happens at construction time via factory methods.
3. These types are internal; Pydantic models at CLI/API boundaries convert to these.
4. Composition over inheritance: step configs embed these primitives.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

if TYPE_CHECKING:
    from codeintel.ingestion.source_scanner import ScanProfile


@dataclass(frozen=True)
class SnapshotRef:
    """Immutable reference to a repository snapshot under analysis.

    This is the canonical representation of repo identity used throughout the
    pipeline. All step configs compose this rather than duplicating repo/commit
    fields.

    Attributes
    ----------
    repo : str
        Repository slug (e.g., "my-org/my-repo").
    commit : str
        Commit SHA or identifier for this analysis run.
    repo_root : Path
        Absolute path to the repository root directory.
    branch : str | None
        Optional branch name associated with the commit.
    """

    repo: str
    commit: str
    repo_root: Path
    branch: str | None = None

    def __post_init__(self) -> None:
        """Resolve repo_root to absolute path."""
        if not self.repo_root.is_absolute():
            object.__setattr__(self, "repo_root", self.repo_root.resolve())

    @classmethod
    def from_args(
        cls,
        repo: str,
        commit: str,
        repo_root: Path,
        branch: str | None = None,
    ) -> Self:
        """Construct a snapshot reference from primitive arguments.

        Parameters
        ----------
        repo
            Repository slug identifier.
        commit
            Commit SHA or identifier.
        repo_root
            Path to repository root (will be resolved to absolute).
        branch
            Optional branch name.

        Returns
        -------
        Self
            Normalized snapshot reference.
        """
        return cls(
            repo=repo,
            commit=commit,
            repo_root=repo_root.resolve(),
            branch=branch,
        )


@dataclass(frozen=True)
class BuildPaths:
    """Derived paths for a pipeline run.

    Centralizes all build-related paths to avoid repetitive path construction
    across step configs. All paths are resolved to absolute at construction.

    Attributes
    ----------
    build_dir : Path
        Root build directory for all generated artifacts.
    db_path : Path
        Path to the primary DuckDB database file.
    document_output_dir : Path
        Directory for final exported datasets (JSONL, Parquet).
    scip_dir : Path
        Directory for SCIP index artifacts.
    coverage_json : Path
        Path for coverage JSON output.
    pytest_report : Path
        Path for pytest JSON report output.
    tool_cache : Path
        Cache directory for external tool artifacts.
    log_db_path : Path
        Path to the pipeline logging DuckDB database.
    """

    build_dir: Path
    db_path: Path
    document_output_dir: Path
    scip_dir: Path
    coverage_json: Path
    pytest_report: Path
    tool_cache: Path
    log_db_path: Path

    def __post_init__(self) -> None:
        """Resolve all paths to absolute."""
        for field_name in (
            "build_dir",
            "db_path",
            "document_output_dir",
            "scip_dir",
            "coverage_json",
            "pytest_report",
            "tool_cache",
            "log_db_path",
        ):
            path = getattr(self, field_name)
            if not path.is_absolute():
                object.__setattr__(self, field_name, path.resolve())

    @classmethod
    def from_repo_root(
        cls,
        repo_root: Path,
        build_dir: Path | None = None,
    ) -> Self:
        """Derive all paths from repo root with sensible defaults.

        Parameters
        ----------
        repo_root
            Root directory of the repository.
        build_dir
            Optional override for build directory; defaults to repo_root/build.

        Returns
        -------
        Self
            BuildPaths with all paths resolved.
        """
        resolved_root = repo_root.resolve()
        resolved_build = (build_dir or resolved_root / "build").resolve()
        return cls(
            build_dir=resolved_build,
            db_path=resolved_build / "db" / "codeintel.duckdb",
            document_output_dir=resolved_root / "Document Output",
            scip_dir=resolved_build / "scip",
            coverage_json=resolved_build / "coverage" / "coverage.json",
            pytest_report=resolved_build / "test-results" / "pytest-report.json",
            tool_cache=resolved_build / ".tool_cache",
            log_db_path=resolved_build / "db" / "codeintel_logs.duckdb",
        )

    @classmethod
    def from_explicit(
        cls,
        *,
        build_dir: Path,
        db_path: Path | None = None,
        document_output_dir: Path | None = None,
        scip_dir: Path | None = None,
        coverage_json: Path | None = None,
        pytest_report: Path | None = None,
        tool_cache: Path | None = None,
        log_db_path: Path | None = None,
    ) -> Self:
        """Construct BuildPaths with explicit overrides for specific paths.

        Parameters
        ----------
        build_dir
            Root build directory (required).
        db_path
            Optional override for database path.
        document_output_dir
            Optional override for document output directory.
        scip_dir
            Optional override for SCIP artifacts directory.
        coverage_json
            Optional override for coverage JSON path.
        pytest_report
            Optional override for pytest report path.
        tool_cache
            Optional override for tool cache directory.
        log_db_path
            Optional override for log database path.

        Returns
        -------
        Self
            BuildPaths with specified overrides applied.
        """
        resolved_build = build_dir.resolve()
        return cls(
            build_dir=resolved_build,
            db_path=(db_path or resolved_build / "db" / "codeintel.duckdb").resolve(),
            document_output_dir=(
                document_output_dir or resolved_build.parent / "Document Output"
            ).resolve(),
            scip_dir=(scip_dir or resolved_build / "scip").resolve(),
            coverage_json=(
                coverage_json or resolved_build / "coverage" / "coverage.json"
            ).resolve(),
            pytest_report=(
                pytest_report or resolved_build / "test-results" / "pytest-report.json"
            ).resolve(),
            tool_cache=(tool_cache or resolved_build / ".tool_cache").resolve(),
            log_db_path=(log_db_path or resolved_build / "db" / "codeintel_logs.duckdb").resolve(),
        )


@dataclass(frozen=True)
class ToolBinaries:
    """Frozen configuration for external tool binary paths.

    This is the internal representation of tool paths. Pydantic ToolsConfig
    at CLI boundaries converts to this type via `.to_binaries()`.

    Attributes
    ----------
    scip_python_bin : str
        Path or name of scip-python binary.
    scip_bin : str
        Path or name of scip binary.
    pyright_bin : str
        Path or name of pyright binary.
    pyrefly_bin : str
        Path or name of pyrefly binary.
    ruff_bin : str
        Path or name of ruff binary.
    coverage_bin : str
        Path or name of coverage.py CLI.
    pytest_bin : str
        Path or name of pytest binary.
    git_bin : str
        Path or name of git binary.
    default_timeout_s : float
        Default timeout in seconds for tool invocations.
    """

    scip_python_bin: str = "scip-python"
    scip_bin: str = "scip"
    pyright_bin: str = "pyright"
    pyrefly_bin: str = "pyrefly"
    ruff_bin: str = "ruff"
    coverage_bin: str = "coverage"
    pytest_bin: str = "pytest"
    git_bin: str = "git"
    default_timeout_s: float = 300.0

    def resolve_path(self, tool: str) -> str:
        """Return the configured path for a tool name.

        Parameters
        ----------
        tool
            Tool identifier (e.g., "scip-python", "pyright").

        Returns
        -------
        str
            Configured path or the tool name as fallback.
        """
        mapping = {
            "scip-python": self.scip_python_bin,
            "scip": self.scip_bin,
            "pyright": self.pyright_bin,
            "pyrefly": self.pyrefly_bin,
            "coverage": self.coverage_bin,
            "pytest": self.pytest_bin,
            "ruff": self.ruff_bin,
            "git": self.git_bin,
        }
        return mapping.get(tool, tool)


@dataclass(frozen=True)
class StepConfig:
    """Base configuration for all pipeline step configurations.

    Step-specific configs inherit from this and add only their unique fields.
    This eliminates the need for repetitive repo/commit/repo_root fields and
    `from_paths()` factory methods across 25+ config classes.

    Attributes
    ----------
    snapshot : SnapshotRef
        Repository snapshot reference.
    paths : BuildPaths
        Derived build paths for the pipeline run.
    """

    snapshot: SnapshotRef
    paths: BuildPaths

    @property
    def repo(self) -> str:
        """Repository slug from the snapshot reference."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier from the snapshot reference."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path from the snapshot reference."""
        return self.snapshot.repo_root

    @property
    def build_dir(self) -> Path:
        """Build directory from the paths configuration."""
        return self.paths.build_dir


@dataclass(frozen=True)
class ScanProfiles:
    """Bundle of code and config scan profiles for a pipeline run.

    Attributes
    ----------
    code : ScanProfile
        Profile for scanning Python source files.
    config : ScanProfile
        Profile for scanning configuration files.
    """

    code: ScanProfile
    config: ScanProfile


@dataclass(frozen=True)
class ExecutionOptions:
    """Optional execution tuning parameters for pipeline runs.

    Attributes
    ----------
    history_db_dir : Path | None
        Directory containing historical DuckDB snapshots.
    history_commits : tuple[str, ...]
        Commit identifiers to include in historical analysis.
    function_overrides : tuple[str, ...]
        Function identifiers with explicit override settings.
    """

    history_db_dir: Path | None = None
    history_commits: tuple[str, ...] = ()
    function_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Normalize tuple fields for immutability."""
        object.__setattr__(self, "history_commits", tuple(self.history_commits))
        object.__setattr__(self, "function_overrides", tuple(self.function_overrides))


@dataclass(frozen=True)
class GraphBackendConfig:
    """Configuration for selecting the NetworkX execution backend.

    Attributes
    ----------
    use_gpu : bool
        Prefer GPU-capable backend such as nx-cugraph when available.
    backend : str
        Backend identifier: "auto", "cpu", or "nx-cugraph".
    strict : bool
        Raise when the requested backend cannot be enabled.
    """

    use_gpu: bool = False
    backend: Literal["auto", "cpu", "nx-cugraph"] = "auto"
    strict: bool = False


# ---------------------------------------------------------------------------
# Aliases for backward compatibility with core.config
# ---------------------------------------------------------------------------


# SnapshotConfig is an alias for SnapshotRef with repo_slug property
@dataclass(frozen=True)
class SnapshotConfig:
    """Immutable description of the code snapshot under analysis.

    This is provided for backward compatibility with `codeintel.core.config`.
    New code should use `SnapshotRef` directly.

    Attributes
    ----------
    repo_root : Path
        Absolute path to the repository root directory.
    repo_slug : str
        Repository slug (e.g., "my-org/my-repo"). Alias for `repo`.
    commit : str
        Commit SHA or identifier for this analysis run.
    branch : str | None
        Optional branch name associated with the commit.
    """

    repo_root: Path
    repo_slug: str
    commit: str
    branch: str | None = None

    def __post_init__(self) -> None:
        """Resolve repo_root to absolute path."""
        if not self.repo_root.is_absolute():
            object.__setattr__(self, "repo_root", self.repo_root.resolve())

    @property
    def repo(self) -> str:
        """Repository slug (alias for backward compatibility)."""
        return self.repo_slug

    @classmethod
    def from_args(
        cls,
        repo_root: Path,
        repo_slug: str,
        commit: str,
        branch: str | None = None,
    ) -> Self:
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
        Self
            Normalized snapshot configuration.
        """
        return cls(
            repo_root=repo_root.resolve(),
            repo_slug=repo_slug,
            commit=commit,
            branch=branch,
        )

    def to_snapshot_ref(self) -> SnapshotRef:
        """Convert to the canonical SnapshotRef type."""
        return SnapshotRef(
            repo=self.repo_slug,
            commit=self.commit,
            repo_root=self.repo_root,
            branch=self.branch,
        )


# ScanProfilesConfig is an alias for ScanProfiles
ScanProfilesConfig = ScanProfiles


@dataclass(frozen=True)
class DerivedPaths:
    """Derived build paths for a given snapshot and execution context.

    This class composes `SnapshotConfig` and runtime settings to provide
    computed path properties. It is the canonical replacement for the old
    `PathsConfig` from `codeintel.core.config`.

    Attributes
    ----------
    snapshot : SnapshotConfig
        The snapshot configuration this paths config derives from.
    build_dir : Path
        Root build directory for all generated artifacts.
    """

    snapshot: SnapshotConfig
    build_dir: Path

    def __post_init__(self) -> None:
        """Resolve build_dir to absolute path."""
        if not self.build_dir.is_absolute():
            object.__setattr__(self, "build_dir", self.build_dir.resolve())

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

    def to_build_paths(self) -> BuildPaths:
        """Convert to the canonical BuildPaths type."""
        return BuildPaths(
            build_dir=self.build_dir,
            db_path=self.build_dir / "db" / "codeintel.duckdb",
            document_output_dir=self.document_output_dir,
            scip_dir=self.scip_temp_dir,
            coverage_json=self.coverage_json,
            pytest_report=self.pytest_report,
            tool_cache=self.tool_cache,
            log_db_path=self.log_db_path,
        )


__all__ = [
    "BuildPaths",
    "DerivedPaths",
    "ExecutionOptions",
    "GraphBackendConfig",
    "ScanProfiles",
    "ScanProfilesConfig",
    "SnapshotConfig",
    "SnapshotRef",
    "StepConfig",
    "ToolBinaries",
]
