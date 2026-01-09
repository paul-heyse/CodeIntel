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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Self

from codeintel.core.env import get_bool, get_int, is_set
from codeintel.core.tools.config import ToolBinaries

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.ingestion.infrastructure.scanning import ScanProfile


def _assert_unique_paths(paths: dict[str, Path]) -> None:
    """Ensure no two logical paths resolve to the same location.

    Raises
    ------
    ValueError
        If any two logical paths point to the same resolved location.
    """
    seen: dict[Path, str] = {}
    for name, path in paths.items():
        existing = seen.get(path)
        if existing is not None:
            message = f"Paths for {existing} and {name} both resolve to {path}"
            raise ValueError(message)
        seen[path] = name


@dataclass(frozen=True)
class SnapshotInit:
    """Snapshot constructor inputs before normalization."""

    repo: str
    commit: str
    repo_root: Path
    branch: str | None = None

    def to_snapshot_ref(self) -> SnapshotRef:
        """Convert inputs into a normalized SnapshotRef.

        Returns
        -------
        SnapshotRef
            Snapshot reference with resolved paths.
        """
        return SnapshotRef(
            repo=self.repo,
            commit=self.commit,
            repo_root=self.repo_root,
            branch=self.branch,
        )


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
class BuildLayoutOptions:
    """Optional layout overrides when deriving build paths from a repo root."""

    build_dir: Path | None = None
    db_path: Path | None = None
    document_output_dir: Path | None = None
    dataset_root_dir: Path | None = None
    log_db_path: Path | None = None

    def materialize(self, repo_root: Path, *, check_collisions: bool = False) -> BuildPaths:
        """Construct BuildPaths using the provided repo root and overrides.

        Parameters
        ----------
        repo_root
            Base repository root to anchor derived paths.
        check_collisions
            When True, detect collisions between resolved paths and raise a ValueError.

        Returns
        -------
        BuildPaths
            Concrete build path bundle derived from this layout.
        """
        return BuildPaths.from_layout(
            repo_root=repo_root,
            overrides=self,
            check_collisions=check_collisions,
        )


@dataclass(frozen=True)
class BuildPathOverrides:
    """Optional overrides for explicit build path construction."""

    db_path: Path | None = None
    document_output_dir: Path | None = None
    dataset_root_dir: Path | None = None
    scip_dir: Path | None = None
    pytest_report: Path | None = None
    tool_cache: Path | None = None
    log_db_path: Path | None = None

    def validate(self, build_dir: Path) -> None:
        """Validate overrides for collisions and path normalization.

        Parameters
        ----------
        build_dir
            Base build directory to resolve relative overrides.

        Raises
        ------
        ValueError
            If two override targets resolve to the same path.
        """
        normalized = self.resolved(build_dir)
        paths = {
            name: path
            for name, path in (
                ("db_path", normalized.db_path),
                ("document_output_dir", normalized.document_output_dir),
                ("dataset_root_dir", normalized.dataset_root_dir),
                ("scip_dir", normalized.scip_dir),
                ("pytest_report", normalized.pytest_report),
                ("tool_cache", normalized.tool_cache),
                ("log_db_path", normalized.log_db_path),
            )
            if path is not None
        }
        try:
            _assert_unique_paths(paths)
        except ValueError as error:
            raise ValueError(str(error)) from error

    def resolved(self, build_dir: Path) -> BuildPathOverrides:
        """Resolve relative override paths against the build directory.

        Parameters
        ----------
        build_dir
            Base build directory used to resolve relative overrides.

        Returns
        -------
        BuildPathOverrides
            Overrides normalized to absolute paths where provided.
        """
        resolved_build = build_dir.resolve()
        return BuildPathOverrides(
            db_path=self._resolve_optional(self.db_path, resolved_build),
            document_output_dir=self._resolve_optional(self.document_output_dir, resolved_build),
            dataset_root_dir=self._resolve_optional(self.dataset_root_dir, resolved_build),
            scip_dir=self._resolve_optional(self.scip_dir, resolved_build),
            pytest_report=self._resolve_optional(self.pytest_report, resolved_build),
            tool_cache=self._resolve_optional(self.tool_cache, resolved_build),
            log_db_path=self._resolve_optional(self.log_db_path, resolved_build),
        )

    @staticmethod
    def _resolve_optional(path: Path | None, anchor: Path) -> Path | None:
        if path is None:
            return None
        if path.is_absolute():
            return path
        return (anchor / path).resolve()


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
    dataset_root_dir : Path
        Directory for Arrow dataset outputs.
    scip_dir : Path
        Directory for SCIP index artifacts.
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
    dataset_root_dir: Path
    scip_dir: Path
    pytest_report: Path
    tool_cache: Path
    log_db_path: Path

    def __post_init__(self) -> None:
        """Resolve all paths to absolute."""
        for field_name in (
            "build_dir",
            "db_path",
            "document_output_dir",
            "dataset_root_dir",
            "scip_dir",
            "pytest_report",
            "tool_cache",
            "log_db_path",
        ):
            path = getattr(self, field_name)
            if not path.is_absolute():
                object.__setattr__(self, field_name, path.resolve())

    @classmethod
    def from_repo_root(
        cls: type[Self],
        repo_root: Path,
        build_dir: Path | None = None,
    ) -> Self:
        """Derive all paths from repo root with sensible defaults.

        Parameters
        ----------
        cls
            BuildPaths class to construct.
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
        document_output_dir = resolved_build / "document_output"
        dataset_root_dir = _default_dataset_root_dir(resolved_build)
        return cls(
            build_dir=resolved_build,
            db_path=resolved_build / "db" / "codeintel.duckdb",
            document_output_dir=document_output_dir,
            dataset_root_dir=dataset_root_dir,
            scip_dir=resolved_build / "scip",
            pytest_report=resolved_build / "test-results" / "pytest-report.json",
            tool_cache=resolved_build / ".tool_cache",
            log_db_path=resolved_build / "db" / "codeintel_logs.duckdb",
        )

    @classmethod
    def from_explicit(
        cls: type[Self],
        *,
        build_dir: Path,
        overrides: BuildPathOverrides | None = None,
    ) -> Self:
        """Construct BuildPaths with explicit overrides for specific paths.

        Parameters
        ----------
        cls
            BuildPaths class to construct.
        build_dir
            Root build directory (required).
        overrides
            Optional bundle of path overrides.

        Returns
        -------
        Self
            BuildPaths with specified overrides applied.
        """
        resolved_build = build_dir.resolve()
        override_bundle = overrides or BuildPathOverrides()
        override_bundle.validate(resolved_build)
        normalized = override_bundle.resolved(resolved_build)
        document_output_dir = (
            normalized.document_output_dir or resolved_build / "document_output"
        ).resolve()
        dataset_root_dir = _default_dataset_root_dir(resolved_build)
        return cls(
            build_dir=resolved_build,
            db_path=(normalized.db_path or resolved_build / "db" / "codeintel.duckdb").resolve(),
            document_output_dir=document_output_dir,
            dataset_root_dir=(normalized.dataset_root_dir or dataset_root_dir).resolve(),
            scip_dir=(normalized.scip_dir or resolved_build / "scip").resolve(),
            pytest_report=(
                normalized.pytest_report or resolved_build / "test-results" / "pytest-report.json"
            ).resolve(),
            tool_cache=(normalized.tool_cache or resolved_build / ".tool_cache").resolve(),
            log_db_path=(
                normalized.log_db_path or resolved_build / "db" / "codeintel_logs.duckdb"
            ).resolve(),
        )

    @classmethod
    def from_layout(
        cls: type[Self],
        *,
        repo_root: Path,
        overrides: BuildLayoutOptions | None = None,
        check_collisions: bool = False,
    ) -> Self:
        """Construct BuildPaths from a repo-centric layout.

        Parameters
        ----------
        cls
            BuildPaths class to construct.
        repo_root
            Root directory of the repository.
        overrides
            Optional layout overrides.
        check_collisions
            When True, detect collisions between resolved paths and raise a ValueError.

        Returns
        -------
        Self
            Build paths resolved against the repository layout.
        """
        layout = overrides or BuildLayoutOptions()
        resolved_root = repo_root.resolve()
        resolved_build = (layout.build_dir or resolved_root / "build").resolve()
        document_output_dir = (
            layout.document_output_dir or resolved_build / "document_output"
        ).resolve()
        dataset_root_dir = _default_dataset_root_dir(resolved_build)
        paths = cls(
            build_dir=resolved_build,
            db_path=(layout.db_path or resolved_build / "db" / "codeintel.duckdb").resolve(),
            document_output_dir=document_output_dir,
            dataset_root_dir=(layout.dataset_root_dir or dataset_root_dir).resolve(),
            scip_dir=(resolved_build / "scip").resolve(),
            pytest_report=(resolved_build / "test-results" / "pytest-report.json").resolve(),
            tool_cache=(resolved_build / ".tool_cache").resolve(),
            log_db_path=(
                layout.log_db_path or resolved_build / "db" / "codeintel_logs.duckdb"
            ).resolve(),
        )
        if check_collisions:
            _assert_unique_paths(
                {
                    "db_path": paths.db_path,
                    "document_output_dir": paths.document_output_dir,
                    "dataset_root_dir": paths.dataset_root_dir,
                    "scip_dir": paths.scip_dir,
                    "pytest_report": paths.pytest_report,
                    "tool_cache": paths.tool_cache,
                    "log_db_path": paths.log_db_path,
                }
            )
        return paths


def _default_dataset_root_dir(build_dir: Path) -> Path:
    """Return the default dataset root directory under the build directory.

    Returns
    -------
    Path
        Default dataset root directory for the build directory.
    """
    return build_dir / "datasets"


@dataclass(frozen=True)
class ScanProfiles:
    """Bundle of code and config scan profiles for a build run.

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
class GraphBackendConfig:
    """Configuration for selecting the graph execution backend.

    Attributes
    ----------
    use_gpu : bool
        Prefer GPU-capable backend when available (ignored; rustworkx is CPU-only).
    backend : str
        Backend identifier: "auto" or "cpu" (rustworkx only).
    strict : bool
        Raise when the requested backend cannot be enabled.
    engine : str
        Graph engine implementation: "rustworkx".
    """

    use_gpu: bool = False
    backend: Literal["auto", "cpu"] = "cpu"
    strict: bool = False
    engine: Literal["rustworkx"] = "rustworkx"


@dataclass(frozen=True)
class GraphOutputToggles:
    """Toggle flags for graph output surfaces."""

    core_metadata: bool = True
    graph_stats: bool = True
    node_payloads: bool = True
    edge_payloads: bool = True
    algorithms_basic: bool = True
    algorithms_advanced: bool = False
    serialization_exports: bool = False
    materialized_tables: bool = True

    @classmethod
    def from_env(cls) -> GraphOutputToggles | None:
        """Construct GraphOutputToggles from CODEINTEL_GRAPH_OUTPUT_* variables.

        Returns
        -------
        GraphOutputToggles | None
            Parsed toggles when any output env vars are set.
        """
        keys = (
            "CODEINTEL_GRAPH_OUTPUT_CORE_METADATA",
            "CODEINTEL_GRAPH_OUTPUT_GRAPH_STATS",
            "CODEINTEL_GRAPH_OUTPUT_NODE_PAYLOADS",
            "CODEINTEL_GRAPH_OUTPUT_EDGE_PAYLOADS",
            "CODEINTEL_GRAPH_OUTPUT_ALGORITHMS_BASIC",
            "CODEINTEL_GRAPH_OUTPUT_ALGORITHMS_ADVANCED",
            "CODEINTEL_GRAPH_OUTPUT_SERIALIZATION_EXPORTS",
            "CODEINTEL_GRAPH_OUTPUT_MATERIALIZED_TABLES",
        )
        if not any(is_set(key) for key in keys):
            return None

        defaults = cls()

        def _resolve_flag(key: str, *, default: bool) -> bool:
            value = get_bool(key, default=default)
            return default if value is None else value

        return cls(
            core_metadata=_resolve_flag(keys[0], default=defaults.core_metadata),
            graph_stats=_resolve_flag(keys[1], default=defaults.graph_stats),
            node_payloads=_resolve_flag(keys[2], default=defaults.node_payloads),
            edge_payloads=_resolve_flag(keys[3], default=defaults.edge_payloads),
            algorithms_basic=_resolve_flag(keys[4], default=defaults.algorithms_basic),
            algorithms_advanced=_resolve_flag(keys[5], default=defaults.algorithms_advanced),
            serialization_exports=_resolve_flag(keys[6], default=defaults.serialization_exports),
            materialized_tables=_resolve_flag(keys[7], default=defaults.materialized_tables),
        )


@dataclass(frozen=True)
class GraphFeatureFlags:
    """Optional feature toggles for graph runtime behaviors."""

    eager_hydration: bool | None = None
    community_detection_limit: int | None = None
    validation_strict: bool | None = None
    cfg_normalization: bool | None = None
    dfg_normalization: bool | None = None
    graph_outputs: GraphOutputToggles | None = None

    def __post_init__(self) -> None:
        """Normalize nested graph output toggles from mappings.

        Raises
        ------
        TypeError
            If a provided graph_outputs mapping is invalid.
        """
        if isinstance(self.graph_outputs, Mapping):
            try:
                resolved = GraphOutputToggles(**self.graph_outputs)
            except TypeError as exc:
                message = "Invalid graph_outputs mapping for GraphOutputToggles"
                raise TypeError(message) from exc
            object.__setattr__(self, "graph_outputs", resolved)

    @classmethod
    def from_env(cls) -> Self:
        """Construct GraphFeatureFlags from CODEINTEL_GRAPH_* environment variables.

        Returns
        -------
        Self
            Parsed feature flags derived from environment variables.

        Raises
        ------
        ValueError
            If an environment value is invalid (for example, a non-integer limit).
        """
        try:
            eager = (
                get_bool("CODEINTEL_GRAPH_EAGER", default=None)
                if is_set("CODEINTEL_GRAPH_EAGER")
                else None
            )
            community_limit = (
                get_int("CODEINTEL_GRAPH_COMMUNITY_LIMIT", default=None, min_value=1)
                if is_set("CODEINTEL_GRAPH_COMMUNITY_LIMIT")
                else None
            )
            validation_strict = (
                get_bool("CODEINTEL_GRAPH_VALIDATION_STRICT", default=None)
                if is_set("CODEINTEL_GRAPH_VALIDATION_STRICT")
                else None
            )
            cfg_normalization = (
                get_bool("CODEINTEL_GRAPH_CFG_NORMALIZE", default=None)
                if is_set("CODEINTEL_GRAPH_CFG_NORMALIZE")
                else None
            )
            dfg_normalization = (
                get_bool("CODEINTEL_GRAPH_DFG_NORMALIZE", default=None)
                if is_set("CODEINTEL_GRAPH_DFG_NORMALIZE")
                else None
            )
            graph_outputs = GraphOutputToggles.from_env()
        except ValueError as exc:
            message = "Invalid graph feature flag environment configuration"
            raise ValueError(message) from exc

        return cls(
            eager_hydration=eager,
            community_detection_limit=community_limit,
            validation_strict=validation_strict,
            cfg_normalization=cfg_normalization,
            dfg_normalization=dfg_normalization,
            graph_outputs=graph_outputs,
        )

    def validate(self) -> None:
        """
        Validate flag values for correctness.

        Raises
        ------
        ValueError
            If community_detection_limit is non-positive.
        """
        if self.community_detection_limit is not None and self.community_detection_limit <= 0:
            message = "community_detection_limit must be positive when provided"
            raise ValueError(message)
        if self.graph_outputs is not None and not isinstance(
            self.graph_outputs,
            GraphOutputToggles,
        ):
            message = "graph_outputs must be GraphOutputToggles when provided"
            raise ValueError(message)


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


__all__ = [
    "BuildLayoutOptions",
    "BuildPathOverrides",
    "BuildPaths",
    "EntryPointToggles",
    "GraphBackendConfig",
    "GraphFeatureFlags",
    "GraphOutputToggles",
    "ScanProfiles",
    "SnapshotInit",
    "SnapshotRef",
    "ToolBinaries",
]
