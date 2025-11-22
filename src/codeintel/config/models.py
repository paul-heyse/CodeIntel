"""
Configuration models used by the CodeIntel CLI and pipeline steps.

These Pydantic models normalize repository identity, filesystem layout, and tool
paths so downstream ingestion and analytics stages can rely on consistent
settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RepoConfig(BaseModel):
    """
    Repository identity used across the pipeline.

    These values are embedded into GOIDs and exported into the
    Document Output datasets (goids.*, coverage_lines.*, etc.). :contentReference[oaicite:1]{index=1}
    """

    repo: str = Field(..., description="Repository slug, e.g. 'my-org/my-repo'")
    commit: str = Field(..., description="Commit SHA for this analysis run")


class PathsConfig(BaseModel):
    """
    Filesystem layout for a single CodeIntel run.

    We assume the repo layout described in the architecture:

      repo_root/
        src/
        Document Output/
        build/
          db/codeintel.duckdb
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    repo_root: Path = Field(..., description="Path to repository root")
    build_dir: Path = Field(
        default=Path("build"),
        description="Build directory (holds db/, logs/, etc.)",
    )
    db_path: Path = Field(
        default=Path("build/db/codeintel.duckdb"),
        description="DuckDB database path",
    )
    document_output_dir: Path | None = Field(
        default=None,
        description="Directory for final datasets (defaults to repo_root / 'Document Output')",
    )

    @field_validator("repo_root", "build_dir", "db_path", "document_output_dir", mode="before")
    @classmethod
    def _expand_user(cls, v: Path | str | None) -> Path | None:
        """
        Expand user home markers for any path-like CLI inputs.

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
    def _resolve_paths(self) -> PathsConfig:
        """
        Resolve relative paths against repo_root/build_dir and set defaults.

        Returns
        -------
        PathsConfig
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
        """
        Directory containing the DuckDB database file.

        Returns
        -------
        Path
            Parent directory of `db_path`.
        """
        return self.db_path.parent

    @property
    def logs_dir(self) -> Path:
        """
        Directory for pipeline log files under the build directory.

        Returns
        -------
        Path
            Absolute path to the logs directory.
        """
        return (self.build_dir / "logs").resolve()

    @property
    def scip_dir(self) -> Path:
        """
        Directory where SCIP index artifacts are stored.

        Returns
        -------
        Path
            Absolute path to the SCIP artifacts directory.
        """
        return (self.build_dir / "scip").resolve()


class ToolsConfig(BaseModel):
    """
    External tool configuration used by ingestion and analytics pipelines.

    The fields capture executable paths and report locations for SCIP index
    generation, static typing diagnostics, and coverage/test ingestion.
    """

    scip_python_bin: str = Field("scip-python", description="Path to scip-python binary")
    scip_bin: str = Field("scip", description="Path to scip binary")
    pyright_bin: str = Field("pyright", description="Path to pyright binary")

    coverage_file: Path | None = Field(
        default=None,
        description="Path to .coverage database (defaults to repo_root/.coverage at call site)",
    )
    pytest_report_path: Path | None = Field(
        default=None,
        description="Path to pytest JSON report (defaults to pytest-report.json under repo_root or build/)",
    )

    @field_validator("coverage_file", "pytest_report_path", mode="before")
    @classmethod
    def _normalize_optional_path(cls, v: Path | str | None) -> Path | None:
        if v is None:
            return None
        if isinstance(v, Path):
            return v.expanduser()
        return Path(str(v)).expanduser()


class CodeIntelConfig(BaseModel):
    """
    Top-level configuration used by the CLI.

    Typical construction:

        cfg = CodeIntelConfig(
            repo=RepoConfig(repo="myrepo", commit="deadbeef"),
            paths=PathsConfig(repo_root=".", ...),
        )
    """

    repo: RepoConfig
    paths: PathsConfig
    tools: ToolsConfig = Field(
        default_factory=lambda: ToolsConfig(
            scip_python_bin="scip-python",
            scip_bin="scip",
            pyright_bin="pyright",
        )
    )

    default_targets: list[str] = Field(
        default_factory=lambda: ["export_docs"],
        description="Default pipeline target(s) when none are specified",
    )

    @classmethod
    def from_cli_args(
        cls,
        *,
        repo_cfg: RepoConfig,
        paths_cfg: PathsConfig,
        tools_cfg: ToolsConfig | None = None,
        default_targets: list[str] | None = None,
    ) -> CodeIntelConfig:
        """
        Build a CodeIntelConfig from pre-parsed CLI models.

        Parameters
        ----------
        repo_cfg : RepoConfig
            Repository identity (slug and commit).
        paths_cfg : PathsConfig
            Normalized filesystem layout.
        tools_cfg : ToolsConfig | None
            Optional external tool configuration; defaults to ToolsConfig().
        default_targets : list[str] | None
            Optional override for default pipeline targets.

        Returns
        -------
        CodeIntelConfig
            Fully constructed configuration model.
        """
        return cls(
            repo=repo_cfg,
            paths=paths_cfg,
            tools=tools_cfg
            or ToolsConfig(
                scip_python_bin="scip-python",
                scip_bin="scip",
                pyright_bin="pyright",
            ),
            default_targets=default_targets or ["export_docs"],
        )

    @property
    def document_output_dir(self) -> Path:
        """
        Directory where documentation-ready artifacts are written.

        Returns
        -------
        Path
            Absolute path to the Document Output directory.

        Raises
        ------
        RuntimeError
            If the paths config was not resolved and document_output_dir is None.
        """
        doc_dir = self.paths.document_output_dir
        if doc_dir is None:
            message = "document_output_dir was not resolved; call PathsConfig._resolve_paths first."
            raise RuntimeError(message)
        return doc_dir
