"""
Configuration models used by the CodeIntel CLI and pipeline steps.

These Pydantic models normalize repository identity, filesystem layout, and tool
paths so downstream ingestion and analytics stages can rely on consistent
settings.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field, validator


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

    @validator("repo_root", "build_dir", "db_path", "document_output_dir", pre=True)
    def _expand_user(self, v: Path | str | None) -> Path | None:
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

    @validator("build_dir", always=True)
    def _normalize_build_dir(self, v: Path, values: dict[str, Path]) -> Path:
        """
        Resolve the build directory relative to the repository root when needed.

        Parameters
        ----------
        v : Path
            Candidate build directory path.
        values : dict
            Other parsed values, notably repo_root.

        Returns
        -------
        Path
            Absolute build directory location.
        """
        repo_root: Path = values.get("repo_root")  # type: ignore[assignment]
        if not v.is_absolute() and repo_root is not None:
            return (repo_root / v).resolve()
        return v

    @validator("db_path", always=True)
    def _normalize_db_path(self, v: Path, values: dict[str, Path]) -> Path:
        """
        Resolve the DuckDB path relative to the computed build directory.

        Parameters
        ----------
        v : Path
            Database path, possibly relative.
        values : dict
            Other parsed values, notably build_dir.

        Returns
        -------
        Path
            Absolute path to the DuckDB database file.
        """
        build_dir: Path = values.get("build_dir")  # type: ignore[assignment]
        if not v.is_absolute() and build_dir is not None:
            return (build_dir / v).resolve()
        return v

    @validator("document_output_dir", always=True)
    def _default_document_output_dir(self, v: Path | None, values: dict[str, Path]) -> Path | None:
        """
        Fill in the Document Output directory when omitted by the caller.

        Parameters
        ----------
        v : Path | None
            Optional override path.
        values : dict
            Other parsed values, notably repo_root.

        Returns
        -------
        Path | None
            Resolved path to the Document Output directory or None when the
            repository root is unknown.
        """
        if v is not None:
            return v
        repo_root: Path = values.get("repo_root")  # type: ignore[assignment]
        if repo_root is None:
            return None
        # README expects datasets under `Document Output/` at repo root. :contentReference[oaicite:2]{index=2}
        return (repo_root / "Document Output").resolve()

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

    @validator("coverage_file", "pytest_report_path", pre=True)
    def _normalize_optional_path(self, v: Path | str | None) -> Path | None:
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
    tools: ToolsConfig = Field(default_factory=ToolsConfig)

    default_targets: list[str] = Field(
        default_factory=lambda: ["export_docs"],
        description="Default pipeline target(s) when none are specified",
    )

    @classmethod
    def from_cli_args(
        cls,
        *,
        repo_root: Path,
        db_path: Path,
        build_dir: Path,
        document_output_dir: Path | None,
        repo: str,
        commit: str,
    ) -> CodeIntelConfig:
        """
        Build a CodeIntelConfig from CLI arguments.

        Parameters
        ----------
        repo_root : Path
            Repository root directory provided by the CLI.
        db_path : Path
            DuckDB path parsed from CLI flags.
        build_dir : Path
            Build directory used to derive other paths.
        document_output_dir : Path | None
            Optional override for the Document Output directory.
        repo : str
            Repository slug.
        commit : str
            Commit SHA for the snapshot under analysis.

        Returns
        -------
        CodeIntelConfig
            Fully constructed configuration object with normalized paths.
        """
        paths = PathsConfig(
            repo_root=repo_root,
            build_dir=build_dir,
            db_path=db_path,
            document_output_dir=document_output_dir,
        )
        repo_cfg = RepoConfig(repo=repo, commit=commit)
        return cls(repo=repo_cfg, paths=paths)

    @property
    def document_output_dir(self) -> Path:
        """
        Directory where documentation-ready artifacts are written.

        Returns
        -------
        Path
            Absolute path to the Document Output directory.
        """
        assert self.paths.document_output_dir is not None
        return self.paths.document_output_dir
