"""
Configuration models used by the CodeIntel CLI and pipeline steps.

These Pydantic models normalize repository identity, filesystem layout, and tool
paths so downstream ingestion and analytics stages can rely on consistent
settings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator


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

    @property
    def document_output_dir(self) -> Path:
        """
        Document Output directory resolved in paths config.

        Raises
        ------
        RuntimeError
            If document_output_dir was not populated.
        """
        doc_dir = self.paths.document_output_dir
        if doc_dir is None:
            message = "document_output_dir was not resolved; ensure PathsConfig._resolve_paths ran."
            raise RuntimeError(message)
        return doc_dir

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


# ---------------------------------------------------------------------------
# Ingestion step configs
# ---------------------------------------------------------------------------


class RepoScanConfig(BaseModel):
    """Configuration for repository scanning into core.modules."""

    repo_root: Path
    repo: str
    commit: str
    tags_index_path: Path | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo_root: Path,
        repo: str,
        commit: str,
        tags_index_path: Path | None = None,
    ) -> Self:
        """
        Create a RepoScanConfig from repository context.

        Returns
        -------
        Self
            Normalized scan configuration.
        """
        return cls(repo_root=repo_root, repo=repo, commit=commit, tags_index_path=tags_index_path)

    @field_validator("repo_root", mode="before")
    @classmethod
    def _resolve_root(cls, value: Path | str) -> Path:
        return Path(value).resolve()

    @field_validator("tags_index_path", mode="before")
    @classmethod
    def _resolve_tags(cls, value: Path | str | None) -> Path | None:
        return Path(value).resolve() if value is not None else None


class CoverageIngestConfig(BaseModel):
    """Configuration for ingesting coverage lines."""

    repo_root: Path
    repo: str
    commit: str
    coverage_file: Path | None = Field(default=None)

    @classmethod
    def from_paths(
        cls,
        *,
        repo_root: Path,
        repo: str,
        commit: str,
        coverage_file: Path | None = None,
        tools: ToolsConfig | None = None,
    ) -> Self:
        """
        Build coverage ingestion settings using repo context and tool defaults.

        Returns
        -------
        Self
            Normalized coverage ingestion configuration.
        """
        resolved = coverage_file or (tools.coverage_file if tools else None)
        return cls(repo_root=repo_root, repo=repo, commit=commit, coverage_file=resolved)

    @field_validator("repo_root", mode="before")
    @classmethod
    def _resolve_repo_root(cls, value: Path | str) -> Path:
        return Path(value).resolve()

    @field_validator("coverage_file", mode="before")
    @classmethod
    def _default_coverage_file(
        cls,
        value: Path | str | None,
        info: ValidationInfo,
    ) -> Path | None:
        if value is None:
            repo_root = info.data.get("repo_root")
            return Path(repo_root) / ".coverage" if repo_root is not None else None
        return Path(value).resolve()


class TestsIngestConfig(BaseModel):
    """Configuration for pytest test catalog ingestion."""

    repo_root: Path
    repo: str
    commit: str
    pytest_report_path: Path | None = Field(default=None)

    @classmethod
    def from_paths(
        cls,
        *,
        repo_root: Path,
        repo: str,
        commit: str,
        pytest_report_path: Path | None = None,
        tools: ToolsConfig | None = None,
    ) -> Self:
        """
        Build tests ingestion settings using repo context and tool defaults.

        Returns
        -------
        Self
            Normalized tests ingestion configuration.
        """
        resolved = pytest_report_path or (tools.pytest_report_path if tools else None)
        return cls(repo_root=repo_root, repo=repo, commit=commit, pytest_report_path=resolved)

    @field_validator("repo_root", mode="before")
    @classmethod
    def _resolve_repo_root(cls, value: Path | str) -> Path:
        return Path(value).resolve()

    @field_validator("pytest_report_path", mode="before")
    @classmethod
    def _resolve_report(cls, value: Path | str | None) -> Path | None:
        return Path(value).resolve() if value is not None else None


class TypingIngestConfig(BaseModel):
    """Configuration for typedness and static diagnostic ingestion."""

    repo_root: Path
    repo: str
    commit: str

    @classmethod
    def from_paths(cls, *, repo_root: Path, repo: str, commit: str) -> Self:
        """
        Build typing ingestion settings from repository context.

        Returns
        -------
        Self
            Normalized typing ingestion configuration.
        """
        return cls(repo_root=repo_root, repo=repo, commit=commit)

    @field_validator("repo_root", mode="before")
    @classmethod
    def _resolve_repo_root(cls, value: Path | str) -> Path:
        return Path(value).resolve()


class ConfigIngestConfig(BaseModel):
    """Configuration for config-values ingestion."""

    repo_root: Path = Field(..., description="Repository root containing config files.")

    @classmethod
    def from_paths(cls, *, repo_root: Path) -> Self:
        """
        Build config-values ingestion settings for a repository root.

        Returns
        -------
        Self
            Normalized config ingestion configuration.
        """
        return cls(repo_root=repo_root)

    @field_validator("repo_root", mode="before")
    @classmethod
    def _resolve_repo_root(cls, value: Path | str) -> Path:
        return Path(value).resolve()


class PyAstIngestConfig(BaseModel):
    """Configuration for stdlib AST ingestion."""

    repo_root: Path
    repo: str
    commit: str

    @classmethod
    def from_paths(cls, *, repo_root: Path, repo: str, commit: str) -> Self:
        """
        Build AST ingestion settings from repository context.

        Returns
        -------
        Self
            Normalized AST ingestion configuration.
        """
        return cls(repo_root=repo_root, repo=repo, commit=commit)

    @field_validator("repo_root", mode="before")
    @classmethod
    def _resolve_repo_root(cls, value: Path | str) -> Path:
        return Path(value).resolve()


# ---------------------------------------------------------------------------
# Dataclass configs (graphs, analytics, ingestion steps using dataclasses)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScipIngestConfig:
    """Configuration for SCIP ingestion."""

    repo_root: Path
    repo: str
    commit: str
    build_dir: Path
    document_output_dir: Path
    scip_python_bin: str = "scip-python"
    scip_bin: str = "scip"

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        paths: PathsConfig,
        tools: ToolsConfig | None = None,
    ) -> Self:
        """
        Create SCIP ingest settings from shared path/tool configuration.

        Returns
        -------
        Self
            Normalized SCIP ingest configuration.
        """
        doc_dir = paths.document_output_dir or (paths.repo_root / "Document Output")
        return cls(
            repo_root=paths.repo_root,
            repo=repo,
            commit=commit,
            build_dir=paths.build_dir,
            document_output_dir=doc_dir,
            scip_python_bin=tools.scip_python_bin if tools else "scip-python",
            scip_bin=tools.scip_bin if tools else "scip",
        )


@dataclass(frozen=True)
class DocstringConfig:
    """Configuration for docstring ingestion."""

    repo_root: Path
    repo: str
    commit: str

    @classmethod
    def from_paths(cls, *, repo_root: Path, repo: str, commit: str) -> Self:
        """
        Build docstring ingestion settings from repository context.

        Returns
        -------
        Self
            Normalized docstring ingestion configuration.
        """
        return cls(repo_root=repo_root, repo=repo, commit=commit)


@dataclass(frozen=True)
class CallGraphConfig:
    """Configuration for constructing a call graph for a repo snapshot."""

    repo: str
    commit: str
    repo_root: Path

    @classmethod
    def from_paths(cls, *, repo: str, commit: str, repo_root: Path) -> Self:
        """
        Build call graph settings from repository context.

        Returns
        -------
        Self
            Normalized call graph configuration.
        """
        return cls(repo=repo, commit=commit, repo_root=repo_root)


@dataclass(frozen=True)
class CFGBuilderConfig:
    """Configuration for control/data-flow scaffolding."""

    repo: str
    commit: str
    repo_root: Path

    @classmethod
    def from_paths(cls, *, repo: str, commit: str, repo_root: Path) -> Self:
        """
        Build CFG/DFG settings from repository context.

        Returns
        -------
        Self
            Normalized CFG builder configuration.
        """
        return cls(repo=repo, commit=commit, repo_root=repo_root)


@dataclass(frozen=True)
class GoidBuilderConfig:
    """Configuration for building GOIDs."""

    repo: str
    commit: str
    language: str = "python"

    @classmethod
    def from_paths(cls, *, repo: str, commit: str, language: str = "python") -> Self:
        """
        Build GOID settings from repository context.

        Returns
        -------
        Self
            Normalized GOID builder configuration.
        """
        return cls(repo=repo, commit=commit, language=language)


@dataclass(frozen=True)
class ImportGraphConfig:
    """Configuration for constructing the import graph of a repo snapshot."""

    repo: str
    commit: str
    repo_root: Path

    @classmethod
    def from_paths(cls, *, repo: str, commit: str, repo_root: Path) -> Self:
        """
        Build import graph settings from repository context.

        Returns
        -------
        Self
            Normalized import graph configuration.
        """
        return cls(repo=repo, commit=commit, repo_root=repo_root)


@dataclass(frozen=True)
class SymbolUsesConfig:
    """Configuration for deriving symbol use edges."""

    repo_root: Path
    scip_json_path: Path
    repo: str
    commit: str

    @classmethod
    def from_paths(
        cls,
        *,
        repo_root: Path,
        repo: str,
        commit: str,
        scip_json_path: Path | None = None,
        build_dir: Path | None = None,
    ) -> Self:
        """
        Build symbol-uses settings, defaulting SCIP JSON under the build directory.

        Returns
        -------
        Self
            Normalized symbol uses configuration.
        """
        resolved_json = scip_json_path
        if resolved_json is None and build_dir is not None:
            resolved_json = build_dir / "scip" / "index.scip.json"
        if resolved_json is None:
            resolved_json = repo_root / "build" / "scip" / "index.scip.json"
        return cls(repo_root=repo_root, scip_json_path=resolved_json, repo=repo, commit=commit)


@dataclass(frozen=True)
class HotspotsConfig:
    """Configuration for file-level hotspot scoring."""

    repo_root: Path
    repo: str
    commit: str
    max_commits: int = 2000

    @classmethod
    def from_paths(
        cls,
        *,
        repo_root: Path,
        repo: str,
        commit: str,
        max_commits: int = 2000,
    ) -> Self:
        """
        Build hotspot analytics settings from repository context.

        Returns
        -------
        Self
            Normalized hotspot configuration.
        """
        return cls(repo_root=repo_root, repo=repo, commit=commit, max_commits=max_commits)


@dataclass(frozen=True)
class CoverageAnalyticsConfig:
    """Configuration for aggregating coverage into functions."""

    repo: str
    commit: str

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> Self:
        """
        Build coverage analytics settings from repository context.

        Returns
        -------
        Self
            Normalized coverage analytics configuration.
        """
        return cls(repo=repo, commit=commit)


@dataclass(frozen=True)
class FunctionAnalyticsOverrides:
    """Optional overrides for function analytics behavior."""

    fail_on_missing_spans: bool = False
    max_workers: int | None = None
    parser: str | None = None


@dataclass(frozen=True)
class FunctionAnalyticsConfig:
    """Configuration for function metrics and typedness analytics."""

    repo: str
    commit: str
    repo_root: Path
    fail_on_missing_spans: bool = False
    max_workers: int | None = None
    parser: str | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        overrides: FunctionAnalyticsOverrides | None = None,
    ) -> Self:
        """
        Build function analytics settings from repository context.

        Returns
        -------
        Self
            Normalized function analytics configuration.
        """
        applied = overrides or FunctionAnalyticsOverrides()
        return cls(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            fail_on_missing_spans=applied.fail_on_missing_spans,
            max_workers=applied.max_workers,
            parser=applied.parser,
        )


@dataclass(frozen=True)
class GraphMetricsConfig:
    """Configuration for graph metrics analytics."""

    repo: str
    commit: str
    max_betweenness_sample: int | None = 200

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        max_betweenness_sample: int | None = 200,
    ) -> Self:
        """
        Build graph metrics configuration from repository context.

        Returns
        -------
        Self
            Normalized graph metrics configuration.
        """
        return cls(
            repo=repo,
            commit=commit,
            max_betweenness_sample=max_betweenness_sample,
        )


@dataclass(frozen=True)
class ProfilesAnalyticsConfig:
    """Configuration for building function, file, and module profiles."""

    repo: str
    commit: str

    @classmethod
    def from_paths(cls, *, repo: str, commit: str) -> Self:
        """
        Build profile analytics settings from repository context.

        Returns
        -------
        Self
            Normalized profiles analytics configuration.
        """
        return cls(repo=repo, commit=commit)


@dataclass(frozen=True)
class SubsystemsOverrides:
    """Optional overrides for subsystem inference."""

    min_modules: int | None = None
    max_subsystems: int | None = None
    import_weight: float | None = None
    symbol_weight: float | None = None
    config_weight: float | None = None


def _coerce_int(value: float | None, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        message = f"{field_name} must be an integer, not bool"
        raise TypeError(message)
    if isinstance(value, int):
        return value
    message = f"{field_name} must be an integer, got {value!r}"
    raise TypeError(message)


def _coerce_float(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        message = f"{field_name} must be a float, not bool"
        raise TypeError(message)
    if isinstance(value, (int, float)):
        return float(value)
    message = f"{field_name} must be numeric, got {value!r}"
    raise TypeError(message)


@dataclass(frozen=True)
class SubsystemsConfig:
    """Configuration for subsystem inference."""

    repo: str
    commit: str
    min_modules: int = 3
    max_subsystems: int | None = None
    import_weight: float = 1.0
    symbol_weight: float = 0.5
    config_weight: float = 0.3

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        overrides: SubsystemsOverrides | None = None,
    ) -> Self:
        """
        Build subsystem inference configuration from repository context.

        Returns
        -------
        Self
            Normalized subsystem configuration.

        Raises
        ------
        TypeError
            If overrides contain non-numeric or non-integer values.
        """
        merged = overrides or SubsystemsOverrides()
        min_modules = _coerce_int(merged.min_modules, "min_modules") or cls.min_modules
        max_subsystems = merged.max_subsystems
        if max_subsystems is not None and not isinstance(max_subsystems, int):
            message = f"max_subsystems must be an integer or None, got {max_subsystems!r}"
            raise TypeError(message)
        import_weight = _coerce_float(merged.import_weight, "import_weight") or cls.import_weight
        symbol_weight = _coerce_float(merged.symbol_weight, "symbol_weight") or cls.symbol_weight
        config_weight = _coerce_float(merged.config_weight, "config_weight") or cls.config_weight
        return cls(
            repo=repo,
            commit=commit,
            min_modules=min_modules,
            max_subsystems=max_subsystems,
            import_weight=import_weight,
            symbol_weight=symbol_weight,
            config_weight=config_weight,
        )


@dataclass(frozen=True)
class TestCoverageConfig:
    """Configuration for deriving test coverage edges."""

    repo: str
    commit: str
    repo_root: Path
    coverage_file: Path | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        coverage_file: Path | None = None,
        tools: ToolsConfig | None = None,
    ) -> Self:
        """
        Build test coverage settings from repository context.

        Returns
        -------
        Self
            Normalized test coverage configuration.
        """
        resolved = coverage_file or (tools.coverage_file if tools else None)
        return cls(repo=repo, commit=commit, repo_root=repo_root, coverage_file=resolved)
