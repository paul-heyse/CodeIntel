"""Analytics step configuration models and builder."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from codeintel.config.parser_types import FunctionParserKind
from codeintel.config.primitives import ScanProfiles, SnapshotRef

if TYPE_CHECKING:
    from coverage import Coverage

    from codeintel.ingestion.source_scanner import ScanProfile


@dataclass(frozen=True)
class EntryPointToggles:
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
class HotspotsStepConfig:
    """Configuration for hotspot scoring."""

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
    """Configuration for function history aggregation."""

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
    """Configuration for history timeseries aggregation."""

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
    def commit(self) -> str:
        """Commit identifier."""
        return self.snapshot.commit

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


class _AnalyticsOwner(Protocol):
    snapshot: SnapshotRef
    profiles: ScanProfiles | None


class AnalyticsStepBuilder:
    """Analytics config builders composed by ConfigBuilder."""

    def __init__(self, owner: _AnalyticsOwner) -> None:
        self._owner = owner

    @property
    def snapshot(self) -> SnapshotRef:
        """Snapshot reference shared across analytics configs."""
        return self._owner.snapshot

    @property
    def profiles(self) -> ScanProfiles | None:
        """Optional scan profiles supplied by the owning builder."""
        return getattr(self._owner, "profiles", None)

    def hotspots(self, *, max_commits: int = 2000) -> HotspotsStepConfig:
        """
        Build hotspot scoring configuration.

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
        """
        Build function history aggregation configuration.

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
        """
        Build cross-commit history timeseries configuration.

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

    def coverage_analytics(self) -> CoverageAnalyticsStepConfig:
        """
        Build coverage aggregation configuration.

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
        """
        Build test coverage edges configuration.

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
        """
        Build test profile configuration.

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
        """
        Build behavioral coverage configuration.

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

    def function_analytics(
        self,
        *,
        fail_on_missing_spans: bool = False,
        max_workers: int | None = None,
        parser: FunctionParserKind | None = None,
    ) -> FunctionAnalyticsStepConfig:
        """
        Build function metrics configuration.

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
        """
        Build function effects detection configuration.

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
        """
        Build function contracts configuration.

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
        """
        Build semantic roles configuration.

        Returns
        -------
        SemanticRolesStepConfig
            Configuration for semantic role classification.
        """
        return SemanticRolesStepConfig(
            snapshot=self.snapshot,
            enable_llm_refinement=enable_llm_refinement,
        )

    def data_models(self) -> DataModelsStepConfig:
        """
        Build data model extraction configuration.

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
        """
        Build data model usage analytics configuration.

        Returns
        -------
        DataModelUsageStepConfig
            Configuration for data model usage analysis.
        """
        return DataModelUsageStepConfig(
            snapshot=self.snapshot,
            max_examples_per_usage=max_examples_per_usage,
        )

    def entrypoints(
        self,
        *,
        scan_profile: ScanProfile | None = None,
        toggles: EntryPointToggles | None = None,
    ) -> EntryPointsStepConfig:
        """
        Build entrypoint detection configuration.

        Returns
        -------
        EntryPointsStepConfig
            Configuration for entrypoint detection.
        """
        profile = scan_profile
        if profile is None and self.profiles is not None:
            profile = self.profiles.code
        t = toggles or EntryPointToggles()
        return EntryPointsStepConfig(
            snapshot=self.snapshot,
            scan_profile=profile,
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

    def profiles_analytics(self) -> ProfilesAnalyticsStepConfig:
        """
        Build profiles analytics configuration.

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
        """
        Build subsystems inference configuration.

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
    "AnalyticsStepBuilder",
    "BehavioralCoverageStepConfig",
    "CoverageAnalyticsStepConfig",
    "DataModelUsageStepConfig",
    "DataModelsStepConfig",
    "EntryPointToggles",
    "EntryPointsStepConfig",
    "FunctionAnalyticsStepConfig",
    "FunctionContractsStepConfig",
    "FunctionEffectsStepConfig",
    "FunctionHistoryStepConfig",
    "HistoryTimeseriesStepConfig",
    "HotspotsStepConfig",
    "ProfilesAnalyticsStepConfig",
    "SemanticRolesStepConfig",
    "SubsystemsStepConfig",
    "TestCoverageStepConfig",
    "TestProfileStepConfig",
]
