"""
Configuration models used by the CodeIntel CLI and pipeline steps.

These Pydantic models normalize repository identity, filesystem layout, and tool
paths so downstream ingestion and analytics stages can rely on consistent
settings.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from coverage import Coverage
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator, model_validator

from codeintel.config.parser_types import FunctionParserKind
from codeintel.models.rows import CallGraphEdgeRow, CFGBlockRow, CFGEdgeRow, DFGEdgeRow

if TYPE_CHECKING:
    from codeintel.ingestion.scip_ingest import ScipIngestResult
    from codeintel.ingestion.source_scanner import ScanProfile
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


class PathsConfig(BaseModel):
    """
    Filesystem layout for a single CodeIntel run.

    We assume the repo layout described in the architecture:

      repo_root/
        src/
        Document Output/
        build/
          db/codeintel_prefect.duckdb
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


@dataclass(frozen=True)
class GraphBackendConfig:
    """
    Configuration for selecting the NetworkX execution backend.

    Attributes
    ----------
    use_gpu : bool
        Whether to prefer a GPU-capable backend such as nx-cugraph when available.
    backend : Literal["auto", "cpu", "nx-cugraph"]
        Identifier for the preferred backend; "auto" defers to helper defaults.
    strict : bool
        If True, raise when the requested backend cannot be enabled instead of falling back.
    """

    use_gpu: bool = False
    backend: Literal["auto", "cpu", "nx-cugraph"] = "auto"
    strict: bool = False


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
        graph_backend: GraphBackendConfig | None = None,
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


# ---------------------------------------------------------------------------
# Ingestion step configs
# ---------------------------------------------------------------------------


class RepoScanConfig(BaseModel):
    """Configuration for repository scanning into core.modules."""

    repo_root: Path
    repo: str
    commit: str
    tags_index_path: Path | None = None
    tool_runner: object | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo_root: Path,
        repo: str,
        commit: str,
        tags_index_path: Path | None = None,
        tool_runner: object | None = None,
    ) -> Self:
        """
        Create a RepoScanConfig from repository context.

        Returns
        -------
        Self
            Normalized scan configuration.
        """
        return cls(
            repo_root=repo_root,
            repo=repo,
            commit=commit,
            tags_index_path=tags_index_path,
            tool_runner=tool_runner,
        )

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
    tool_runner: object | None = None

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
        if resolved is None:
            resolved = repo_root / "build" / "test-results" / "pytest-report.json"
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
    tool_runner: object | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo_root: Path,
        repo: str,
        commit: str,
        tool_runner: object | None = None,
    ) -> Self:
        """
        Build typing ingestion settings from repository context.

        Returns
        -------
        Self
            Normalized typing ingestion configuration.
        """
        return cls(repo_root=repo_root, repo=repo, commit=commit, tool_runner=tool_runner)

    @field_validator("repo_root", mode="before")
    @classmethod
    def _resolve_repo_root(cls, value: Path | str) -> Path:
        return Path(value).resolve()


class ConfigIngestConfig(BaseModel):
    """Configuration for config-values ingestion."""

    repo_root: Path = Field(..., description="Repository root containing config files.")
    repo: str
    commit: str

    @classmethod
    def from_paths(cls, *, repo_root: Path, repo: str, commit: str) -> Self:
        """
        Build config-values ingestion settings for a repository root.

        Returns
        -------
        Self
            Normalized config ingestion configuration.
        """
        return cls(repo_root=repo_root, repo=repo, commit=commit)

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
    scip_runner: Callable[..., ScipIngestResult] | None = None
    artifact_writer: Callable[[Path, Path, Path], None] | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        paths: PathsConfig,
        tools: ToolsConfig | None = None,
        artifact_writer: Callable[[Path, Path, Path], None] | None = None,
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
            artifact_writer=artifact_writer,
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
    cst_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
    ast_collector: Callable[..., list[CallGraphEdgeRow]] | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
    ) -> Self:
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
    cfg_builder: (
        Callable[..., tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]] | None
    ) = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
    ) -> Self:
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
class FunctionHistoryConfig:
    """Configuration for per-function git history aggregation."""

    repo: str
    commit: str
    repo_root: Path
    max_history_days: int | None = 365
    min_lines_threshold: int = 1
    default_branch: str = "HEAD"

    @dataclass(frozen=True)
    class Overrides:
        """Optional overrides for function history analytics."""

        max_history_days: int | None = None
        min_lines_threshold: int | None = None
        default_branch: str | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        overrides: Overrides | None = None,
    ) -> Self:
        """
        Build function history settings from repository context.

        Returns
        -------
        Self
            Normalized function history configuration.
        """
        applied = overrides or cls.Overrides()
        max_history_days = applied.max_history_days
        if max_history_days is None:
            max_history_days = cls.max_history_days
        min_lines_threshold = applied.min_lines_threshold
        if min_lines_threshold is None:
            min_lines_threshold = cls.min_lines_threshold
        default_branch = applied.default_branch or cls.default_branch
        return cls(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            max_history_days=max_history_days,
            min_lines_threshold=min_lines_threshold,
            default_branch=default_branch,
        )


@dataclass(frozen=True)
class HistoryTimeseriesConfig:
    """Configuration for cross-commit history aggregation."""

    repo: str
    repo_root: Path
    commits: tuple[str, ...]
    entity_kind: str = "function"
    max_entities: int = 500
    selection_strategy: str = "risk_score"

    @dataclass(frozen=True)
    class Overrides:
        """Optional overrides for history timeseries analytics."""

        entity_kind: str | None = None
        max_entities: int | None = None
        selection_strategy: str | None = None

    @classmethod
    def from_args(
        cls,
        *,
        repo: str,
        repo_root: Path,
        commits: Sequence[str],
        overrides: Overrides | None = None,
    ) -> Self:
        """
        Build history timeseries settings from CLI arguments.

        Returns
        -------
        Self
            Normalized history timeseries configuration.
        """
        applied = overrides or cls.Overrides()
        return cls(
            repo=repo,
            repo_root=repo_root,
            commits=tuple(commits),
            entity_kind=applied.entity_kind or cls.entity_kind,
            max_entities=applied.max_entities or cls.max_entities,
            selection_strategy=applied.selection_strategy or cls.selection_strategy,
        )


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
    parser: FunctionParserKind | None = None


@dataclass(frozen=True)
class FunctionAnalyticsConfig:
    """Configuration for function metrics and typedness analytics."""

    repo: str
    commit: str
    repo_root: Path
    fail_on_missing_spans: bool = False
    max_workers: int | None = None
    parser: FunctionParserKind | None = None

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
        parser_kind = applied.parser
        if isinstance(parser_kind, str):
            parser_kind = FunctionParserKind(parser_kind)
        return cls(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            fail_on_missing_spans=applied.fail_on_missing_spans,
            max_workers=applied.max_workers,
            parser=parser_kind,
        )


@dataclass(frozen=True)
class GraphMetricsConfig:
    """Configuration for graph metrics analytics."""

    repo: str
    commit: str
    max_betweenness_sample: int | None = 200
    eigen_max_iter: int = 200
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"
    seed: int = 0

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
    ) -> Self:
        """
        Build graph metrics configuration from repository context.

        Returns
        -------
        Self
            Normalized graph metrics configuration.
        """
        return cls(repo=repo, commit=commit)


@dataclass(frozen=True)
class DataModelsConfig:
    """Configuration for extracting data models."""

    repo: str
    commit: str
    repo_root: Path

    @classmethod
    def from_paths(cls, *, repo: str, commit: str, repo_root: Path) -> Self:
        """
        Build data model extraction settings from repository context.

        Returns
        -------
        Self
            Normalized data model configuration.
        """
        return cls(repo=repo, commit=commit, repo_root=repo_root.resolve())


@dataclass(frozen=True)
class DataModelUsageConfig:
    """Configuration for data model usage analytics."""

    repo: str
    commit: str
    repo_root: Path
    max_examples_per_usage: int = 5

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        max_examples_per_usage: int = 5,
    ) -> Self:
        """
        Build data model usage settings from repository context.

        Returns
        -------
        Self
            Normalized data model usage configuration.
        """
        return cls(
            repo=repo,
            commit=commit,
            repo_root=repo_root.resolve(),
            max_examples_per_usage=max_examples_per_usage,
        )


@dataclass(frozen=True)
class ConfigDataFlowConfig:
    """Configuration for config data flow analytics."""

    repo: str
    commit: str
    repo_root: Path
    max_paths_per_usage: int = 3
    max_path_length: int = 10

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        max_paths_per_usage: int = 3,
        max_path_length: int = 10,
    ) -> Self:
        """
        Build config data flow settings from repository context.

        Returns
        -------
        Self
            Normalized config data flow configuration.
        """
        return cls(
            repo=repo,
            commit=commit,
            repo_root=repo_root.resolve(),
            max_paths_per_usage=max_paths_per_usage,
            max_path_length=max_path_length,
        )


@dataclass(frozen=True)
class EntryPointDetectionToggles:
    """Toggle flags for entrypoint detection frameworks."""

    detect_fastapi: bool = True
    detect_flask: bool = True
    detect_click: bool = True
    detect_typer: bool = True
    detect_cron: bool = True
    detect_django: bool = True
    detect_celery: bool = True
    detect_airflow: bool = True
    detect_generic_routes: bool = True


@dataclass(frozen=True)
class EntryPointsConfig:
    """Configuration for entrypoint detection analytics."""

    repo: str
    commit: str
    repo_root: Path
    scan_profile: ScanProfile | None = None
    detect_fastapi: bool = True
    detect_flask: bool = True
    detect_click: bool = True
    detect_typer: bool = True
    detect_cron: bool = True
    detect_django: bool = True
    detect_celery: bool = True
    detect_airflow: bool = True
    detect_generic_routes: bool = True

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        scan_profile: ScanProfile | None = None,
        toggles: EntryPointDetectionToggles | None = None,
    ) -> Self:
        """
        Build entrypoint detection configuration from repository context.

        Returns
        -------
        Self
            Normalized entrypoint configuration.
        """
        resolved_toggles = toggles or EntryPointDetectionToggles()
        return cls(
            repo=repo,
            commit=commit,
            repo_root=repo_root.resolve(),
            scan_profile=scan_profile,
            detect_fastapi=resolved_toggles.detect_fastapi,
            detect_flask=resolved_toggles.detect_flask,
            detect_click=resolved_toggles.detect_click,
            detect_typer=resolved_toggles.detect_typer,
            detect_cron=resolved_toggles.detect_cron,
            detect_django=resolved_toggles.detect_django,
            detect_celery=resolved_toggles.detect_celery,
            detect_airflow=resolved_toggles.detect_airflow,
            detect_generic_routes=resolved_toggles.detect_generic_routes,
        )


@dataclass(frozen=True)
class DependencyPatternOptions:
    """Options for dependency pattern scanning."""

    language: str = "python"
    dependency_patterns_path: Path | None = None


@dataclass(frozen=True)
class ExternalDependenciesConfig:
    """Configuration for external dependency analytics."""

    repo: str
    commit: str
    repo_root: Path
    language: str = "python"
    dependency_patterns_path: Path | None = None
    scan_profile: ScanProfile | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        scan_profile: ScanProfile | None = None,
        options: DependencyPatternOptions | None = None,
    ) -> Self:
        """
        Build dependency analytics configuration from repository context.

        Returns
        -------
        Self
            Normalized dependency configuration.
        """
        resolved_options = options or DependencyPatternOptions()
        patterns = (
            resolved_options.dependency_patterns_path.resolve()
            if resolved_options.dependency_patterns_path
            else None
        )
        return cls(
            repo=repo,
            commit=commit,
            repo_root=repo_root.resolve(),
            language=resolved_options.language,
            dependency_patterns_path=patterns,
            scan_profile=scan_profile,
        )


def _default_io_apis() -> dict[str, list[str]]:
    return {
        "builtins": ["open", "print"],
        "pathlib": ["Path.open", "Path.write_text", "Path.write_bytes"],
        "logging": ["debug", "info", "warning", "error", "exception", "critical", "log"],
        "requests": ["get", "post", "put", "delete", "patch", "head", "options"],
        "httpx": ["get", "post", "put", "delete", "patch", "head", "options"],
    }


def _default_db_apis() -> dict[str, list[str]]:
    return {
        "sqlite3": ["connect"],
        "psycopg": ["connect"],
        "psycopg2": ["connect"],
        "asyncpg": ["connect", "create_pool"],
        "sqlalchemy": ["create_engine", "Session"],
    }


def _default_time_apis() -> dict[str, list[str]]:
    return {
        "time": ["sleep", "time"],
        "asyncio": ["sleep"],
        "datetime": ["datetime.now", "datetime.utcnow", "date.today"],
    }


def _default_random_apis() -> dict[str, list[str]]:
    return {
        "random": ["random", "randint", "choice", "randrange", "shuffle"],
        "secrets": ["token_hex", "token_urlsafe"],
        "uuid": ["uuid4", "uuid1"],
    }


def _default_threading_apis() -> dict[str, list[str]]:
    return {
        "threading": ["Thread", "Timer"],
        "multiprocessing": ["Process", "Pool"],
        "asyncio": ["create_task", "ensure_future", "gather"],
        "concurrent.futures": ["ThreadPoolExecutor", "ProcessPoolExecutor"],
    }


@dataclass(frozen=True)
class FunctionEffectsConfig:
    """Configuration for side-effect and purity detection."""

    repo: str
    commit: str
    repo_root: Path
    max_call_depth: int = 3
    require_all_callees_pure: bool = True
    io_apis: dict[str, list[str]] = field(default_factory=_default_io_apis)
    db_apis: dict[str, list[str]] = field(default_factory=_default_db_apis)
    time_apis: dict[str, list[str]] = field(default_factory=_default_time_apis)
    random_apis: dict[str, list[str]] = field(default_factory=_default_random_apis)
    threading_apis: dict[str, list[str]] = field(default_factory=_default_threading_apis)

    @classmethod
    def from_paths(cls, *, repo: str, commit: str, repo_root: Path) -> Self:
        """
        Build effects configuration from repository context.

        Returns
        -------
        Self
            Normalized configuration.
        """
        return cls(repo=repo, commit=commit, repo_root=repo_root)


@dataclass(frozen=True)
class FunctionContractsConfig:
    """Configuration for inferred contracts and nullability."""

    repo: str
    commit: str
    repo_root: Path
    max_conditions_per_func: int = 64

    @classmethod
    def from_paths(cls, *, repo: str, commit: str, repo_root: Path) -> Self:
        """
        Build contracts configuration from repository context.

        Returns
        -------
        Self
            Normalized configuration.
        """
        return cls(repo=repo, commit=commit, repo_root=repo_root)


@dataclass(frozen=True)
class SemanticRolesConfig:
    """Configuration for semantic role classification."""

    repo: str
    commit: str
    repo_root: Path
    enable_llm_refinement: bool = False

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        enable_llm_refinement: bool = False,
    ) -> Self:
        """
        Build semantic roles configuration from repository context.

        Returns
        -------
        Self
            Normalized roles configuration.
        """
        return cls(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            enable_llm_refinement=enable_llm_refinement,
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
    coverage_loader: Callable[[TestCoverageConfig], Coverage | None] | None = None

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


@dataclass(frozen=True)
class TestProfileConfig:
    """Define configuration for building analytics.test_profile."""

    repo: str
    commit: str
    repo_root: Path
    slow_test_threshold_ms: float = 2000.0
    io_spec: dict[str, object] | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        slow_test_threshold_ms: float = 2000.0,
        io_spec: dict[str, object] | None = None,
    ) -> Self:
        """
        Build test profile settings from repository context.

        Parameters
        ----------
        repo :
            Repository slug.
        commit :
            Commit identifier.
        repo_root :
            Repository root on disk for AST parsing.
        slow_test_threshold_ms :
            Threshold for marking tests as slow.
        io_spec :
            Optional IO classification overrides.

        Returns
        -------
        Self
            Normalized test profile configuration.
        """
        return cls(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            slow_test_threshold_ms=slow_test_threshold_ms,
            io_spec=io_spec,
        )


@dataclass(frozen=True)
class BehavioralCoverageConfig:
    """Define configuration for building analytics.behavioral_coverage."""

    repo: str
    commit: str
    repo_root: Path
    heuristic_version: str = "v1"
    enable_llm: bool = False
    llm_model: str | None = None

    @classmethod
    def from_paths(
        cls,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        enable_llm: bool = False,
        llm_model: str | None = None,
    ) -> Self:
        """
        Build behavioral coverage settings from repository context.

        Parameters
        ----------
        repo :
            Repository slug.
        commit :
            Commit identifier.
        repo_root :
            Repository root on disk for AST parsing.
        enable_llm :
            Whether to call the behavioral LLM to augment heuristic tags.
        llm_model :
            Optional model identifier forwarded to the LLM runner.

        Returns
        -------
        Self
            Normalized behavioral coverage configuration.
        """
        return cls(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            enable_llm=enable_llm,
            llm_model=llm_model,
        )
