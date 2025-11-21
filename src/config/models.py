# src/codeintel/config/models.py

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

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
    document_output_dir: Optional[Path] = Field(
        default=None,
        description="Directory for final datasets (defaults to repo_root / 'Document Output')",
    )

    @validator("repo_root", "build_dir", "db_path", "document_output_dir", pre=True)
    def _expand_user(cls, v):
        if v is None:
            return None
        if isinstance(v, Path):
            return v.expanduser()
        return Path(str(v)).expanduser()

    @validator("build_dir", always=True)
    def _normalize_build_dir(cls, v, values):
        repo_root: Path = values.get("repo_root")  # type: ignore[assignment]
        if not v.is_absolute() and repo_root is not None:
            return (repo_root / v).resolve()
        return v

    @validator("db_path", always=True)
    def _normalize_db_path(cls, v, values):
        build_dir: Path = values.get("build_dir")  # type: ignore[assignment]
        if not v.is_absolute() and build_dir is not None:
            return (build_dir / v).resolve()
        return v

    @validator("document_output_dir", always=True)
    def _default_document_output_dir(cls, v, values):
        if v is not None:
            return v
        repo_root: Path = values.get("repo_root")  # type: ignore[assignment]
        if repo_root is None:
            return None
        # README expects datasets under `Document Output/` at repo root. :contentReference[oaicite:2]{index=2}
        return (repo_root / "Document Output").resolve()

    @property
    def db_dir(self) -> Path:
        return self.db_path.parent

    @property
    def logs_dir(self) -> Path:
        return (self.build_dir / "logs").resolve()

    @property
    def scip_dir(self) -> Path:
        return (self.build_dir / "scip").resolve()


class ToolsConfig(BaseModel):
    """
    External tool configuration used by ingestion/analytics:

      - scip-python / scip for SCIP index
      - pyright for static typing diagnostics
      - coverage.py / pytest-json-report artifacts for tests/coverage
    """

    scip_python_bin: str = Field("scip-python", description="Path to scip-python binary")
    scip_bin: str = Field("scip", description="Path to scip binary")
    pyright_bin: str = Field("pyright", description="Path to pyright binary")

    coverage_file: Optional[Path] = Field(
        default=None,
        description="Path to .coverage database (defaults to repo_root/.coverage at call site)",
    )
    pytest_report_path: Optional[Path] = Field(
        default=None,
        description="Path to pytest JSON report (defaults to pytest-report.json under repo_root or build/)",
    )

    @validator("coverage_file", "pytest_report_path", pre=True)
    def _normalize_optional_path(cls, v):
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

    default_targets: List[str] = Field(
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
        document_output_dir: Optional[Path],
        repo: str,
        commit: str,
    ) -> "CodeIntelConfig":
        """
        Convenience constructor used by the CLI to assemble config from
        command line arguments.
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
        assert self.paths.document_output_dir is not None
        return self.paths.document_output_dir
