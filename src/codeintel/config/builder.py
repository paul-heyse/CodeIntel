"""Unified configuration builder replacing 31+ `from_paths()` factory methods.

The ConfigBuilder provides a single point of construction for all pipeline step
configurations. Instead of each config class having its own factory method,
callers create a ConfigBuilder from a shared context and request specific
configs as needed.

Example
-------
    builder = ConfigBuilder.from_snapshot(
        repo="my-org/repo",
        commit="abc123",
        repo_root=Path("/path/to/repo"),
    )

    coverage_cfg = builder.coverage_ingest()
    graph_cfg = builder.graph_metrics(max_betweenness_sample=100)
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Self

from codeintel.config.primitives import (
    BuildPaths,
    GraphBackendConfig,
    ScanProfiles,
    SnapshotRef,
    ToolBinaries,
)

if TYPE_CHECKING:
    from coverage import Coverage

    from codeintel.config.parser_types import FunctionParserKind
    from codeintel.ingestion.scip_ingest import ScipIngestResult
    from codeintel.ingestion.source_scanner import ScanProfile
    from codeintel.models.rows import (
        CallGraphEdgeRow,
        CFGBlockRow,
        CFGEdgeRow,
        DFGEdgeRow,
    )


# ---------------------------------------------------------------------------
# Configuration toggles and options
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EntryPointToggles:
    """Toggle flags for entrypoint detection frameworks.

    Attributes
    ----------
    detect_fastapi : bool
        Enable FastAPI route detection.
    detect_flask : bool
        Enable Flask route detection.
    detect_click : bool
        Enable Click CLI detection.
    detect_typer : bool
        Enable Typer CLI detection.
    detect_cron : bool
        Enable cron/scheduler detection.
    detect_django : bool
        Enable Django route detection.
    detect_celery : bool
        Enable Celery task detection.
    detect_airflow : bool
        Enable Airflow DAG detection.
    detect_generic_routes : bool
        Enable generic HTTP route detection.
    """

    detect_fastapi: bool = True
    detect_flask: bool = True
    detect_click: bool = True
    detect_typer: bool = True
    detect_cron: bool = True
    detect_django: bool = True
    detect_celery: bool = True
    detect_airflow: bool = True
    detect_generic_routes: bool = True


# ---------------------------------------------------------------------------
# Step-specific config dataclasses using composition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScipIngestStepConfig:
    """Configuration for SCIP ingestion step."""

    snapshot: SnapshotRef
    paths: BuildPaths
    binaries: ToolBinaries
    scip_runner: Callable[..., ScipIngestResult] | None = None
    artifact_writer: Callable[[Path, Path, Path], None] | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root

    @property
    def build_dir(self) -> Path:
        """Build directory."""
        return self.paths.build_dir

    @property
    def document_output_dir(self) -> Path:
        """Document output directory."""
        return self.paths.document_output_dir

    @property
    def scip_python_bin(self) -> str:
        """Path to scip-python binary."""
        return self.binaries.scip_python_bin

    @property
    def scip_bin(self) -> str:
        """Path to scip binary."""
        return self.binaries.scip_bin


@dataclass(frozen=True)
class DocstringStepConfig:
    """Configuration for docstring ingestion step."""

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class CallGraphStepConfig:
    """Configuration for call graph construction step."""

    snapshot: SnapshotRef
    cst_collector: Callable[..., list[CallGraphEdgeRow]] | None = None
    ast_collector: Callable[..., list[CallGraphEdgeRow]] | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class CFGBuilderStepConfig:
    """Configuration for control/data-flow scaffolding step."""

    snapshot: SnapshotRef
    cfg_builder: (
        Callable[..., tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]] | None
    ) = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class GoidBuilderStepConfig:
    """Configuration for building GOIDs."""

    snapshot: SnapshotRef
    language: str = "python"

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit


@dataclass(frozen=True)
class ImportGraphStepConfig:
    """Configuration for import graph construction step."""

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class SymbolUsesStepConfig:
    """Configuration for deriving symbol use edges."""

    snapshot: SnapshotRef
    paths: BuildPaths
    scip_json_path: Path | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root

    @property
    def resolved_scip_json_path(self) -> Path:
        """Resolved path to SCIP JSON file."""
        if self.scip_json_path is not None:
            return self.scip_json_path
        return self.paths.scip_dir / "index.scip.json"


@dataclass(frozen=True)
class HotspotsStepConfig:
    """Configuration for file-level hotspot scoring."""

    snapshot: SnapshotRef
    max_commits: int = 2000

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class FunctionHistoryStepConfig:
    """Configuration for per-function git history aggregation."""

    snapshot: SnapshotRef
    max_history_days: int | None = 365
    min_lines_threshold: int = 1
    default_branch: str = "HEAD"

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class HistoryTimeseriesStepConfig:
    """Configuration for cross-commit history aggregation."""

    snapshot: SnapshotRef
    commits: tuple[str, ...]
    entity_kind: str = "function"
    max_entities: int = 500
    selection_strategy: str = "risk_score"

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class CoverageAnalyticsStepConfig:
    """Configuration for aggregating coverage into functions."""

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit


@dataclass(frozen=True)
class FunctionAnalyticsStepConfig:
    """Configuration for function metrics and typedness analytics."""

    snapshot: SnapshotRef
    fail_on_missing_spans: bool = False
    max_workers: int | None = None
    parser: FunctionParserKind | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class GraphMetricsStepConfig:
    """Configuration for graph metrics analytics."""

    snapshot: SnapshotRef
    max_betweenness_sample: int | None = 200
    eigen_max_iter: int = 200
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"
    seed: int = 0

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit


@dataclass(frozen=True)
class DataModelsStepConfig:
    """Configuration for extracting data models."""

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class DataModelUsageStepConfig:
    """Configuration for data model usage analytics."""

    snapshot: SnapshotRef
    max_examples_per_usage: int = 5

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class ConfigDataFlowStepConfig:
    """Configuration for config data flow analytics."""

    snapshot: SnapshotRef
    max_paths_per_usage: int = 3
    max_path_length: int = 10

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class EntryPointsStepConfig:
    """Configuration for entrypoint detection analytics."""

    snapshot: SnapshotRef
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

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class ExternalDependenciesStepConfig:
    """Configuration for external dependency analytics."""

    snapshot: SnapshotRef
    language: str = "python"
    dependency_patterns_path: Path | None = None
    scan_profile: ScanProfile | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class FunctionEffectsStepConfig:
    """Configuration for side-effect and purity detection."""

    snapshot: SnapshotRef
    max_call_depth: int = 3
    require_all_callees_pure: bool = True
    io_apis: dict[str, list[str]] = field(default_factory=dict)
    db_apis: dict[str, list[str]] = field(default_factory=dict)
    time_apis: dict[str, list[str]] = field(default_factory=dict)
    random_apis: dict[str, list[str]] = field(default_factory=dict)
    threading_apis: dict[str, list[str]] = field(default_factory=dict)

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class FunctionContractsStepConfig:
    """Configuration for inferred contracts and nullability."""

    snapshot: SnapshotRef
    max_conditions_per_func: int = 64

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class SemanticRolesStepConfig:
    """Configuration for semantic role classification."""

    snapshot: SnapshotRef
    enable_llm_refinement: bool = False

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class ProfilesAnalyticsStepConfig:
    """Configuration for building function, file, and module profiles."""

    snapshot: SnapshotRef

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit


@dataclass(frozen=True)
class SubsystemsStepConfig:
    """Configuration for subsystem inference."""

    snapshot: SnapshotRef
    min_modules: int = 3
    max_subsystems: int | None = None
    import_weight: float = 1.0
    symbol_weight: float = 0.5
    config_weight: float = 0.3

    def __post_init__(self) -> None:
        """Validate numeric fields to prevent silent coercion."""
        object.__setattr__(self, "min_modules", _require_int(self.min_modules, "min_modules") or 3)
        object.__setattr__(
            self, "max_subsystems", _require_int(self.max_subsystems, "max_subsystems")
        )
        object.__setattr__(
            self, "import_weight", _require_float(self.import_weight, "import_weight")
        )
        object.__setattr__(
            self, "symbol_weight", _require_float(self.symbol_weight, "symbol_weight")
        )
        object.__setattr__(
            self, "config_weight", _require_float(self.config_weight, "config_weight")
        )

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit


@dataclass(frozen=True)
class TestCoverageStepConfig:
    """Configuration for deriving test coverage edges."""

    snapshot: SnapshotRef
    coverage_file: Path | None = None
    coverage_loader: Callable[[TestCoverageStepConfig], Coverage | None] | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class TestProfileStepConfig:
    """Configuration for building analytics.test_profile."""

    snapshot: SnapshotRef
    slow_test_threshold_ms: float = 2000.0
    io_spec: dict[str, object] | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


@dataclass(frozen=True)
class BehavioralCoverageStepConfig:
    """Configuration for building analytics.behavioral_coverage."""

    snapshot: SnapshotRef
    heuristic_version: str = "v1"
    enable_llm: bool = False
    llm_model: str | None = None

    @property
    def repo(self) -> str:
        """Repository slug."""
        return self.snapshot.repo

    @property
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

    @property
    def repo_root(self) -> Path:
        """Repository root path."""
        return self.snapshot.repo_root


# ---------------------------------------------------------------------------
# Default API mappings for function effects detection
# ---------------------------------------------------------------------------


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


def _require_int(value: int | None, field_name: str) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        message = f"{field_name} must be an integer, not bool"
        raise TypeError(message)
    if isinstance(value, int):
        return value
    message = f"{field_name} must be an integer, got {value!r}"
    raise TypeError(message)


def _require_float(value: float | None, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        message = f"{field_name} must be a float, not bool"
        raise TypeError(message)
    if isinstance(value, (int, float)):
        return float(value)
    message = f"{field_name} must be numeric, got {value!r}"
    raise TypeError(message)


# ---------------------------------------------------------------------------
# ConfigBuilder: centralized factory replacing 31+ from_paths() methods
# ---------------------------------------------------------------------------


@dataclass
class ConfigBuilder:
    """Build specific step configs from a shared pipeline context.

    Replaces the 31+ `from_paths()` factory methods scattered across config
    classes with a single, unified builder pattern.

    Example
    -------
        builder = ConfigBuilder.from_snapshot(
            repo="my-org/repo",
            commit="abc123",
            repo_root=Path("/path/to/repo"),
        )

        # Build step configs on demand
        scip_cfg = builder.scip_ingest()
        graph_cfg = builder.graph_metrics(max_betweenness_sample=100)
    """

    snapshot: SnapshotRef
    paths: BuildPaths
    binaries: ToolBinaries = field(default_factory=ToolBinaries)
    profiles: ScanProfiles | None = None
    graph_backend: GraphBackendConfig = field(default_factory=GraphBackendConfig)

    @classmethod
    def from_snapshot(
        cls,
        repo: str,
        commit: str,
        repo_root: Path,
        *,
        build_dir: Path | None = None,
        branch: str | None = None,
    ) -> Self:
        """Create a builder from basic snapshot parameters.

        Parameters
        ----------
        repo
            Repository slug identifier.
        commit
            Commit SHA or identifier.
        repo_root
            Path to repository root.
        build_dir
            Optional override for build directory.
        branch
            Optional branch name.

        Returns
        -------
        Self
            ConfigBuilder ready to produce step configs.
        """
        snapshot = SnapshotRef.from_args(
            repo=repo,
            commit=commit,
            repo_root=repo_root,
            branch=branch,
        )
        paths = BuildPaths.from_repo_root(repo_root, build_dir=build_dir)
        return cls(snapshot=snapshot, paths=paths)

    @classmethod
    def from_primitives(
        cls,
        snapshot: SnapshotRef,
        paths: BuildPaths,
        *,
        binaries: ToolBinaries | None = None,
        profiles: ScanProfiles | None = None,
        graph_backend: GraphBackendConfig | None = None,
    ) -> Self:
        """Create a builder from pre-constructed primitives.

        Parameters
        ----------
        snapshot
            Pre-constructed snapshot reference.
        paths
            Pre-constructed build paths.
        binaries
            Optional tool binaries; defaults to ToolBinaries().
        profiles
            Optional scan profiles.
        graph_backend
            Optional graph backend configuration.

        Returns
        -------
        Self
            ConfigBuilder ready to produce step configs.
        """
        return cls(
            snapshot=snapshot,
            paths=paths,
            binaries=binaries or ToolBinaries(),
            profiles=profiles,
            graph_backend=graph_backend or GraphBackendConfig(),
        )

    # -----------------------------------------------------------------------
    # SCIP and Source Extraction Steps
    # -----------------------------------------------------------------------

    def scip_ingest(
        self,
        *,
        scip_runner: Callable[..., ScipIngestResult] | None = None,
        artifact_writer: Callable[[Path, Path, Path], None] | None = None,
    ) -> ScipIngestStepConfig:
        """Build SCIP ingestion configuration.

        Returns
        -------
        ScipIngestStepConfig
            Configuration for SCIP index generation.
        """
        return ScipIngestStepConfig(
            snapshot=self.snapshot,
            paths=self.paths,
            binaries=self.binaries,
            scip_runner=scip_runner,
            artifact_writer=artifact_writer,
        )

    def docstring(self) -> DocstringStepConfig:
        """Build docstring ingestion configuration.

        Returns
        -------
        DocstringStepConfig
            Configuration for docstring extraction.
        """
        return DocstringStepConfig(snapshot=self.snapshot)

    # -----------------------------------------------------------------------
    # Graph Construction Steps
    # -----------------------------------------------------------------------

    def call_graph(
        self,
        *,
        cst_collector: Callable[..., list[CallGraphEdgeRow]] | None = None,
        ast_collector: Callable[..., list[CallGraphEdgeRow]] | None = None,
    ) -> CallGraphStepConfig:
        """Build call graph construction configuration.

        Returns
        -------
        CallGraphStepConfig
            Configuration for call graph construction.
        """
        return CallGraphStepConfig(
            snapshot=self.snapshot,
            cst_collector=cst_collector,
            ast_collector=ast_collector,
        )

    def cfg_builder(
        self,
        *,
        cfg_builder: (
            Callable[..., tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]] | None
        ) = None,
    ) -> CFGBuilderStepConfig:
        """Build CFG/DFG scaffolding configuration.

        Returns
        -------
        CFGBuilderStepConfig
            Configuration for control/data flow graph construction.
        """
        return CFGBuilderStepConfig(
            snapshot=self.snapshot,
            cfg_builder=cfg_builder,
        )

    def goid_builder(self, *, language: str = "python") -> GoidBuilderStepConfig:
        """Build GOID construction configuration.

        Returns
        -------
        GoidBuilderStepConfig
            Configuration for GOID generation.
        """
        return GoidBuilderStepConfig(
            snapshot=self.snapshot,
            language=language,
        )

    def import_graph(self) -> ImportGraphStepConfig:
        """Build import graph construction configuration.

        Returns
        -------
        ImportGraphStepConfig
            Configuration for import graph construction.
        """
        return ImportGraphStepConfig(snapshot=self.snapshot)

    def symbol_uses(
        self,
        *,
        scip_json_path: Path | None = None,
    ) -> SymbolUsesStepConfig:
        """Build symbol uses derivation configuration.

        Returns
        -------
        SymbolUsesStepConfig
            Configuration for symbol usage extraction.
        """
        return SymbolUsesStepConfig(
            snapshot=self.snapshot,
            paths=self.paths,
            scip_json_path=scip_json_path,
        )

    # -----------------------------------------------------------------------
    # History and Hotspot Steps
    # -----------------------------------------------------------------------

    def hotspots(self, *, max_commits: int = 2000) -> HotspotsStepConfig:
        """Build hotspot scoring configuration.

        Returns
        -------
        HotspotsStepConfig
            Configuration for hotspot analysis.
        """
        return HotspotsStepConfig(
            snapshot=self.snapshot,
            max_commits=max_commits,
        )

    def function_history(
        self,
        *,
        max_history_days: int | None = 365,
        min_lines_threshold: int = 1,
        default_branch: str = "HEAD",
    ) -> FunctionHistoryStepConfig:
        """Build function history aggregation configuration.

        Returns
        -------
        FunctionHistoryStepConfig
            Configuration for function history aggregation.
        """
        return FunctionHistoryStepConfig(
            snapshot=self.snapshot,
            max_history_days=max_history_days,
            min_lines_threshold=min_lines_threshold,
            default_branch=default_branch,
        )

    def history_timeseries(
        self,
        commits: Sequence[str],
        *,
        entity_kind: str = "function",
        max_entities: int = 500,
        selection_strategy: str = "risk_score",
    ) -> HistoryTimeseriesStepConfig:
        """Build cross-commit history timeseries configuration.

        Returns
        -------
        HistoryTimeseriesStepConfig
            Configuration for history timeseries analysis.
        """
        return HistoryTimeseriesStepConfig(
            snapshot=self.snapshot,
            commits=tuple(commits),
            entity_kind=entity_kind,
            max_entities=max_entities,
            selection_strategy=selection_strategy,
        )

    # -----------------------------------------------------------------------
    # Coverage and Test Steps
    # -----------------------------------------------------------------------

    def coverage_analytics(self) -> CoverageAnalyticsStepConfig:
        """Build coverage aggregation configuration.

        Returns
        -------
        CoverageAnalyticsStepConfig
            Configuration for coverage analytics.
        """
        return CoverageAnalyticsStepConfig(snapshot=self.snapshot)

    def test_coverage(
        self,
        *,
        coverage_file: Path | None = None,
        coverage_loader: Callable[[TestCoverageStepConfig], Coverage | None] | None = None,
    ) -> TestCoverageStepConfig:
        """Build test coverage edges configuration.

        Returns
        -------
        TestCoverageStepConfig
            Configuration for test coverage edge extraction.
        """
        return TestCoverageStepConfig(
            snapshot=self.snapshot,
            coverage_file=coverage_file,
            coverage_loader=coverage_loader,
        )

    def test_profile(
        self,
        *,
        slow_test_threshold_ms: float = 2000.0,
        io_spec: dict[str, object] | None = None,
    ) -> TestProfileStepConfig:
        """Build test profile configuration.

        Returns
        -------
        TestProfileStepConfig
            Configuration for test profile analysis.
        """
        return TestProfileStepConfig(
            snapshot=self.snapshot,
            slow_test_threshold_ms=slow_test_threshold_ms,
            io_spec=io_spec,
        )

    def behavioral_coverage(
        self,
        *,
        enable_llm: bool = False,
        llm_model: str | None = None,
    ) -> BehavioralCoverageStepConfig:
        """Build behavioral coverage configuration.

        Returns
        -------
        BehavioralCoverageStepConfig
            Configuration for behavioral coverage analysis.
        """
        return BehavioralCoverageStepConfig(
            snapshot=self.snapshot,
            enable_llm=enable_llm,
            llm_model=llm_model,
        )

    # -----------------------------------------------------------------------
    # Function Analytics Steps
    # -----------------------------------------------------------------------

    def function_analytics(
        self,
        *,
        fail_on_missing_spans: bool = False,
        max_workers: int | None = None,
        parser: FunctionParserKind | None = None,
    ) -> FunctionAnalyticsStepConfig:
        """Build function metrics configuration.

        Returns
        -------
        FunctionAnalyticsStepConfig
            Configuration for function metrics analysis.
        """
        return FunctionAnalyticsStepConfig(
            snapshot=self.snapshot,
            fail_on_missing_spans=fail_on_missing_spans,
            max_workers=max_workers,
            parser=parser,
        )

    def function_effects(
        self,
        *,
        max_call_depth: int = 3,
        require_all_callees_pure: bool = True,
    ) -> FunctionEffectsStepConfig:
        """Build function effects detection configuration.

        Returns
        -------
        FunctionEffectsStepConfig
            Configuration for side-effect detection.
        """
        return FunctionEffectsStepConfig(
            snapshot=self.snapshot,
            max_call_depth=max_call_depth,
            require_all_callees_pure=require_all_callees_pure,
            io_apis=_default_io_apis(),
            db_apis=_default_db_apis(),
            time_apis=_default_time_apis(),
            random_apis=_default_random_apis(),
            threading_apis=_default_threading_apis(),
        )

    def function_contracts(
        self,
        *,
        max_conditions_per_func: int = 64,
    ) -> FunctionContractsStepConfig:
        """Build function contracts configuration.

        Returns
        -------
        FunctionContractsStepConfig
            Configuration for contract inference.
        """
        return FunctionContractsStepConfig(
            snapshot=self.snapshot,
            max_conditions_per_func=max_conditions_per_func,
        )

    def semantic_roles(
        self,
        *,
        enable_llm_refinement: bool = False,
    ) -> SemanticRolesStepConfig:
        """Build semantic roles configuration.

        Returns
        -------
        SemanticRolesStepConfig
            Configuration for semantic role classification.
        """
        return SemanticRolesStepConfig(
            snapshot=self.snapshot,
            enable_llm_refinement=enable_llm_refinement,
        )

    # -----------------------------------------------------------------------
    # Graph Analytics Steps
    # -----------------------------------------------------------------------

    def graph_metrics(
        self,
        *,
        max_betweenness_sample: int | None = 200,
        eigen_max_iter: int = 200,
        pagerank_weight: str | None = "weight",
        betweenness_weight: str | None = "weight",
        seed: int = 0,
    ) -> GraphMetricsStepConfig:
        """Build graph metrics analytics configuration.

        Returns
        -------
        GraphMetricsStepConfig
            Configuration for graph centrality metrics.
        """
        return GraphMetricsStepConfig(
            snapshot=self.snapshot,
            max_betweenness_sample=max_betweenness_sample,
            eigen_max_iter=eigen_max_iter,
            pagerank_weight=pagerank_weight,
            betweenness_weight=betweenness_weight,
            seed=seed,
        )

    # -----------------------------------------------------------------------
    # Data Model Steps
    # -----------------------------------------------------------------------

    def data_models(self) -> DataModelsStepConfig:
        """Build data model extraction configuration.

        Returns
        -------
        DataModelsStepConfig
            Configuration for data model extraction.
        """
        return DataModelsStepConfig(snapshot=self.snapshot)

    def data_model_usage(
        self,
        *,
        max_examples_per_usage: int = 5,
    ) -> DataModelUsageStepConfig:
        """Build data model usage analytics configuration.

        Returns
        -------
        DataModelUsageStepConfig
            Configuration for data model usage analysis.
        """
        return DataModelUsageStepConfig(
            snapshot=self.snapshot,
            max_examples_per_usage=max_examples_per_usage,
        )

    def config_data_flow(
        self,
        *,
        max_paths_per_usage: int = 3,
        max_path_length: int = 10,
    ) -> ConfigDataFlowStepConfig:
        """Build config data flow analytics configuration.

        Returns
        -------
        ConfigDataFlowStepConfig
            Configuration for config data flow analysis.
        """
        return ConfigDataFlowStepConfig(
            snapshot=self.snapshot,
            max_paths_per_usage=max_paths_per_usage,
            max_path_length=max_path_length,
        )

    # -----------------------------------------------------------------------
    # Entrypoint and Dependency Steps
    # -----------------------------------------------------------------------

    def entrypoints(
        self,
        *,
        scan_profile: ScanProfile | None = None,
        toggles: EntryPointToggles | None = None,
    ) -> EntryPointsStepConfig:
        """Build entrypoint detection configuration.

        Parameters
        ----------
        scan_profile
            Optional scan profile for file discovery.
        toggles
            Optional detection toggles; defaults to all frameworks enabled.

        Returns
        -------
        EntryPointsStepConfig
            Configuration for entrypoint detection.
        """
        t = toggles or EntryPointToggles()
        return EntryPointsStepConfig(
            snapshot=self.snapshot,
            scan_profile=scan_profile or (self.profiles.code if self.profiles else None),
            detect_fastapi=t.detect_fastapi,
            detect_flask=t.detect_flask,
            detect_click=t.detect_click,
            detect_typer=t.detect_typer,
            detect_cron=t.detect_cron,
            detect_django=t.detect_django,
            detect_celery=t.detect_celery,
            detect_airflow=t.detect_airflow,
            detect_generic_routes=t.detect_generic_routes,
        )

    def external_dependencies(
        self,
        *,
        language: str = "python",
        dependency_patterns_path: Path | None = None,
        scan_profile: ScanProfile | None = None,
    ) -> ExternalDependenciesStepConfig:
        """Build external dependencies configuration.

        Returns
        -------
        ExternalDependenciesStepConfig
            Configuration for dependency analysis.
        """
        return ExternalDependenciesStepConfig(
            snapshot=self.snapshot,
            language=language,
            dependency_patterns_path=dependency_patterns_path,
            scan_profile=scan_profile or (self.profiles.code if self.profiles else None),
        )

    # -----------------------------------------------------------------------
    # Aggregation and Profile Steps
    # -----------------------------------------------------------------------

    def profiles_analytics(self) -> ProfilesAnalyticsStepConfig:
        """Build profiles analytics configuration.

        Returns
        -------
        ProfilesAnalyticsStepConfig
            Configuration for profiles analysis.
        """
        return ProfilesAnalyticsStepConfig(snapshot=self.snapshot)

    def subsystems(
        self,
        *,
        min_modules: int = 3,
        max_subsystems: int | None = None,
        import_weight: float = 1.0,
        symbol_weight: float = 0.5,
        config_weight: float = 0.3,
    ) -> SubsystemsStepConfig:
        """Build subsystems inference configuration.

        Returns
        -------
        SubsystemsStepConfig
            Configuration for subsystem inference.
        """
        return SubsystemsStepConfig(
            snapshot=self.snapshot,
            min_modules=min_modules,
            max_subsystems=max_subsystems,
            import_weight=import_weight,
            symbol_weight=symbol_weight,
            config_weight=config_weight,
        )


__all__ = [
    "BehavioralCoverageStepConfig",
    "CFGBuilderStepConfig",
    "CallGraphStepConfig",
    "ConfigBuilder",
    "ConfigDataFlowStepConfig",
    "CoverageAnalyticsStepConfig",
    "DataModelUsageStepConfig",
    "DataModelsStepConfig",
    "DocstringStepConfig",
    "EntryPointToggles",
    "EntryPointsStepConfig",
    "ExternalDependenciesStepConfig",
    "FunctionAnalyticsStepConfig",
    "FunctionContractsStepConfig",
    "FunctionEffectsStepConfig",
    "FunctionHistoryStepConfig",
    "GoidBuilderStepConfig",
    "GraphMetricsStepConfig",
    "HistoryTimeseriesStepConfig",
    "HotspotsStepConfig",
    "ImportGraphStepConfig",
    "ProfilesAnalyticsStepConfig",
    "ScipIngestStepConfig",
    "SemanticRolesStepConfig",
    "SubsystemsStepConfig",
    "SymbolUsesStepConfig",
    "TestCoverageStepConfig",
    "TestProfileStepConfig",
]
