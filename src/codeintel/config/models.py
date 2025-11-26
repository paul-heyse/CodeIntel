"""Configuration models used by the CodeIntel CLI and pipeline steps.

This module contains:
- **CLI Boundary Models** (Pydantic): `RepoConfig`, `CliPathsInput`, `ToolsConfig`,
  `CodeIntelConfig` - use these for CLI argument parsing and validation.
- **Legacy Step Configs** (frozen dataclasses): These are DEPRECATED and will be
  removed in a future version.

Migration Guide
---------------
For step configurations, prefer the new composition-based system:

Preferred:
    from codeintel.config import ConfigBuilder
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path("."))
    cfg = builder.graph_metrics()

See `codeintel.config.builder` for the new ConfigBuilder API.
See `codeintel.config.compat` for converters between old and new configs.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from codeintel.config.primitives import BuildPaths, GraphBackendConfig

if TYPE_CHECKING:
    from codeintel.ingestion.tool_runner import ToolName


DEFAULT_TOOL_TIMEOUT_S = 300.0


class RepoConfig(BaseModel):
    """
    Repository identity used across the pipeline.

    These values are embedded into GOIDs and exported into the Document Output
    datasets (goids.*, coverage_lines.*, etc.).
    """

    repo: str = Field(..., description="Repository slug, e.g. 'my-org/my-repo'")
    commit: str = Field(..., description="Commit SHA for this analysis run")


class CliPathsInput(BaseModel):
    """Filesystem layout for a single CodeIntel run (CLI boundary model).

    This Pydantic model is used for CLI argument parsing and validation. It
    normalizes paths and expands user home directories. For internal use,
    convert to `BuildPaths` using the `to_build_paths()` method.

    Attributes
    ----------
    repo_root : Path
        Path to repository root.
    build_dir : Path
        Build directory (holds db/, logs/, etc.).
    db_path : Path
        DuckDB database path.
    document_output_dir : Path | None
        Directory for final datasets (defaults to repo_root / 'Document Output').

    Example
    -------
        # CLI input parsing
        paths = CliPathsInput(repo_root=Path("."))

        # Convert to internal type
        build_paths = paths.to_build_paths()
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_root: Path = Field(..., description="Path to repository root")
    build_dir: Path = Field(
        default=Path("build"),
        description="Build directory (holds db/, logs/, etc.)",
    )
    db_path: Path = Field(
        default=Path("build/db/codeintel_prefect.duckdb"),
        description="DuckDB database path",
    )
    document_output_dir: Path | None = Field(
        default=None,
        description="Directory for final datasets (defaults to repo_root / 'Document Output')",
    )

    @field_validator("repo_root", "build_dir", "db_path", "document_output_dir", mode="before")
    @classmethod
    def _expand_user(cls, v: Path | str | None) -> Path | None:
        """Expand user home markers for any path-like CLI inputs.

        Parameters
        ----------
        v : Path | str | None
            Raw value provided via CLI or configuration.

        Returns
        -------
        Path | None
            Expanded pathlib object or None for unset values.
        """
        if v is None:
            return None
        if isinstance(v, Path):
            return v.expanduser()
        return Path(str(v)).expanduser()

    @model_validator(mode="after")
    def _resolve_paths(self) -> CliPathsInput:
        """Resolve relative paths against repo_root/build_dir and set defaults.

        Returns
        -------
        CliPathsInput
            New instance with absolute paths and defaults applied.
        """
        repo_root = self.repo_root.resolve()

        build_dir = self.build_dir
        if not build_dir.is_absolute():
            build_dir = (repo_root / build_dir).resolve()

        db_path = self.db_path
        if not db_path.is_absolute():
            db_path = (build_dir / db_path).resolve()

        doc_dir = self.document_output_dir
        if doc_dir is None:
            doc_dir = (repo_root / "Document Output").resolve()
        elif not doc_dir.is_absolute():
            doc_dir = (repo_root / doc_dir).resolve()

        self.repo_root = repo_root
        self.build_dir = build_dir
        self.db_path = db_path
        self.document_output_dir = doc_dir
        return self

    @property
    def db_dir(self) -> Path:
        """Directory containing the DuckDB database file.

        Returns
        -------
        Path
            Parent directory of `db_path`.
        """
        return self.db_path.parent

    @property
    def logs_dir(self) -> Path:
        """Directory for pipeline log files under the build directory.

        Returns
        -------
        Path
            Absolute path to the logs directory.
        """
        return (self.build_dir / "logs").resolve()

    @property
    def scip_dir(self) -> Path:
        """Directory where SCIP index artifacts are stored.

        Returns
        -------
        Path
            Absolute path to the SCIP artifacts directory.
        """
        return (self.build_dir / "scip").resolve()

    def to_build_paths(self) -> BuildPaths:
        """Convert to the canonical BuildPaths type for internal use.

        Returns
        -------
        BuildPaths
            Internal paths configuration.
        """
        doc_dir = self.document_output_dir or (self.repo_root / "Document Output")
        return BuildPaths(
            build_dir=self.build_dir,
            db_path=self.db_path,
            document_output_dir=doc_dir,
            scip_dir=self.scip_dir,
            coverage_json=self.build_dir / "coverage" / "coverage.json",
            pytest_report=self.build_dir / "test-results" / "pytest-report.json",
            tool_cache=self.build_dir / ".tool_cache",
            log_db_path=self.build_dir / "db" / "codeintel_logs.duckdb",
        )


class ToolsConfig(BaseModel):
    """
    External tool configuration used by ingestion and analytics pipelines.

    The fields capture executable paths and report locations for SCIP index
    generation, static typing diagnostics, and coverage/test ingestion.
    """

    scip_python_bin: str = Field("scip-python", description="Path to scip-python binary")
    scip_bin: str = Field("scip", description="Path to scip binary")
    pyright_bin: str = Field("pyright", description="Path to pyright binary")
    pyrefly_bin: str = Field("pyrefly", description="Path to pyrefly binary")
    ruff_bin: str = Field("ruff", description="Path to ruff binary")
    coverage_bin: str = Field("coverage", description="Path to coverage.py CLI")
    pytest_bin: str = Field("pytest", description="Path to pytest binary")
    git_bin: str = Field("git", description="Path to git binary")
    default_timeout_s: float = Field(
        DEFAULT_TOOL_TIMEOUT_S,
        description="Default timeout (seconds) for external tool invocations",
    )

    coverage_file: Path | None = Field(
        default=None,
        description="Path to .coverage database (defaults to repo_root/.coverage at call site)",
    )
    pytest_report_path: Path | None = Field(
        default=None,
        description="Path to pytest JSON report (default: repo_root/build/pytest-report.json)",
    )

    @field_validator("coverage_file", "pytest_report_path", mode="before")
    @classmethod
    def _normalize_optional_path(cls, v: Path | str | None) -> Path | None:
        if v is None:
            return None
        if isinstance(v, Path):
            return v.expanduser()
        return Path(str(v)).expanduser()

    @classmethod
    def default(cls) -> ToolsConfig:
        """
        Return a fully-populated tool configuration with baked-in defaults.

        Returns
        -------
        ToolsConfig
            Configuration populated with built-in binary names.
        """
        return cls.model_validate({})

    @classmethod
    def with_overrides(cls, **overrides: str | float | Path | None) -> ToolsConfig:
        """
        Construct a ToolsConfig using defaults merged with provided overrides.

        Parameters
        ----------
        **overrides
            Keyword overrides for binary paths or optional report locations.

        Returns
        -------
        ToolsConfig
            Fully-populated configuration with overrides applied.
        """
        return cls.default().model_copy(update=overrides)

    def resolve_path(self, tool: ToolName | str) -> str:
        """
        Return the configured executable path for a tool or fall back to its name.

        Parameters
        ----------
        tool
            Tool identifier (enum or string).

        Returns
        -------
        str
            Executable path or name to invoke.
        """
        name = str(tool)
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
        return str(mapping.get(name, name))

    def build_env(
        self,
        tool: ToolName | str,
        *,
        base_env: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """
        Construct an environment mapping for a tool invocation.

        Parameters
        ----------
        tool
            Tool identifier (unused but reserved for future tool-specific envs).
        base_env
            Baseline environment to merge into the returned mapping.

        Returns
        -------
        dict[str, str]
            Environment variables to supply to the subprocess call.
        """
        _ = tool
        env: dict[str, str] = dict(base_env or {})
        env.setdefault("CODEINTEL_TOOL_TIMEOUT", str(int(self.default_timeout_s)))
        return env


class CodeIntelConfig(BaseModel):
    """
    Top-level configuration used by the CLI.

    Typical construction:

        cfg = CodeIntelConfig(
            repo=RepoConfig(repo="myrepo", commit="deadbeef"),
            paths=CliPathsInput(repo_root=".", ...),
        )
    """

    repo: RepoConfig
    paths: CliPathsInput
    tools: ToolsConfig = Field(default_factory=ToolsConfig.default)

    default_targets: list[str] = Field(
        default_factory=lambda: ["export_docs"],
        description="Default pipeline target(s) when none are specified",
    )
    graph_backend: GraphBackendConfig = Field(default_factory=GraphBackendConfig)

    @property
    def document_output_dir(self) -> Path:
        """
        Document Output directory resolved in paths config.

        Returns
        -------
        Path
            Populated document_output_dir from paths configuration.

        Raises
        ------
        RuntimeError
            If document_output_dir was not populated.
        """
        doc_dir = self.paths.document_output_dir
        if doc_dir is None:
            message = "document_output_dir was not resolved; ensure CliPathsInput validation ran."
            raise RuntimeError(message)
        return doc_dir

    @property
    def build_paths(self) -> BuildPaths:
        """
        Convert CLI path inputs to the internal BuildPaths representation.

        Returns
        -------
        BuildPaths
            Normalized build paths derived from the CLI inputs.
        """
        return self.paths.to_build_paths()

    @classmethod
    def from_cli_args(
        cls,
        *,
        repo_cfg: RepoConfig,
        paths_cfg: CliPathsInput,
        tools_cfg: ToolsConfig | None = None,
        default_targets: list[str] | None = None,
        graph_backend: GraphBackendConfig | None = None,
    ) -> CodeIntelConfig:
        """
        Build a CodeIntelConfig from pre-parsed CLI models.

        Parameters
        ----------
        repo_cfg : RepoConfig
            Repository identity (slug and commit).
        paths_cfg : CliPathsInput
            Normalized filesystem layout.
        tools_cfg : ToolsConfig | None
            Optional external tool configuration; defaults to ToolsConfig().
        default_targets : list[str] | None
            Optional override for default pipeline targets.
        graph_backend : GraphBackendConfig | None
            Optional NetworkX backend selection.

        Returns
        -------
        CodeIntelConfig
            Fully constructed configuration model.
        """
        return cls(
            repo=repo_cfg,
            paths=paths_cfg,
            tools=tools_cfg or ToolsConfig.default(),
            default_targets=default_targets or ["export_docs"],
            graph_backend=graph_backend or GraphBackendConfig(),
        )
