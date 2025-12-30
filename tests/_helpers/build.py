"""Helpers for build-system unit tests.

Use catalog outputs derived from saver tags; tests should not declare output
inventories outside the catalog.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.build.config import CONFIG_FILE_NAME, BuildConfig
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.targets import TargetDescriptor
from codeintel.core.build_manifest import OutputManifest
from codeintel.core.config.settings import (
    BuildSettings,
    ExportAuditSettings,
    IcebergSettings,
)
from codeintel.core.runtime.loader import load_runtime_settings
from tests._helpers.catalog import build_catalog, make_target_descriptor
from tests._helpers.fakes.configs import create_test_build_paths, create_test_snapshot
from tests._helpers.fakes.fake_providers import (
    FakeCoverageCollector,
    FakeGitHistoryProvider,
    FakeScipIndexer,
    FakeTestReporter,
    FakeToolRunner,
    FakeTypeChecker,
)
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT

if TYPE_CHECKING:
    from codeintel.config.primitives import BuildPaths, SnapshotRef

_RUNTIME_BUILD_SETTINGS = load_runtime_settings().build


def iceberg_settings_for_paths(build_paths: BuildPaths) -> IcebergSettings:
    """Create Iceberg settings rooted at a build directory.

    Parameters
    ----------
    build_paths
        Build paths to root the Iceberg catalog/warehouse.

    Returns
    -------
    IcebergSettings
        Iceberg settings configured for local test catalogs.
    """
    catalog_dir = build_paths.build_dir / "iceberg"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = (catalog_dir / "catalog.duckdb").resolve()
    warehouse_path = (catalog_dir / "warehouse").resolve()
    warehouse_path.mkdir(parents=True, exist_ok=True)
    return IcebergSettings(
        read_enabled=True,
        write_enabled=True,
        catalog_type="sql",
        catalog_uri=f"duckdb:///{catalog_path}",
        catalog_warehouse=str(warehouse_path),
    )


def build_settings_for_paths(
    build_paths: BuildPaths,
    *,
    engine_version: str = "test",
) -> BuildSettings:
    """Return BuildSettings configured for a specific build directory.

    Parameters
    ----------
    build_paths
        Build paths used to scope Iceberg catalogs and warehouses.
    engine_version
        Engine version identifier to embed in hashes.

    Returns
    -------
    BuildSettings
        Build settings configured for the provided build paths.
    """
    return BuildSettings(
        engine_version=engine_version,
        export_audit=_RUNTIME_BUILD_SETTINGS.export_audit or ExportAuditSettings(),
        iceberg=iceberg_settings_for_paths(build_paths),
    )


TEST_BUILD_SETTINGS = BuildSettings(
    engine_version="test",
    export_audit=_RUNTIME_BUILD_SETTINGS.export_audit or ExportAuditSettings(),
)


def make_build_settings(
    engine_version: str = "test",
    *,
    build_paths: BuildPaths | None = None,
) -> BuildSettings:
    """Return BuildSettings for tests.

    Parameters
    ----------
    engine_version
        Engine version identifier to embed in hashes.
    build_paths
        Optional build paths used to scope Iceberg catalogs and warehouses.

    Returns
    -------
    BuildSettings
        Build settings for tests.
    """
    resolved_paths = build_paths or create_test_build_paths(Path.cwd())
    return build_settings_for_paths(resolved_paths, engine_version=engine_version)


def make_snapshot(
    tmp_path: Path | None = None,
    *,
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
) -> SnapshotRef:
    """Create a SnapshotRef with sensible defaults.

    Parameters
    ----------
    tmp_path
        Temporary path for repo_root. If None, uses current directory.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    SnapshotRef
        Snapshot reference configured for tests.
    """
    effective_path = tmp_path if tmp_path is not None else Path.cwd()
    return create_test_snapshot(effective_path, repo=repo, commit=commit)


def make_build_paths(tmp_path: Path | None = None) -> BuildPaths:
    """Create BuildPaths rooted at tmp_path (or current directory).

    Parameters
    ----------
    tmp_path
        Temporary path for build directory. If None, uses current directory.

    Returns
    -------
    BuildPaths
        Build paths configured for tests.
    """
    effective_path = tmp_path if tmp_path is not None else Path.cwd()
    return create_test_build_paths(effective_path)


def make_build_config(
    data: Mapping[str, Any] | None = None,
    *,
    config_path: Path | None = None,
) -> BuildConfig:
    """Create a BuildConfig from a mapping.

    Returns
    -------
    BuildConfig
        Parsed configuration.
    """
    return (
        BuildConfig.from_dict(dict(data), config_path=config_path)
        if data is not None
        else BuildConfig.empty()
    )


def write_build_config(project_root: Path, sections: Mapping[str, Mapping[str, Any]]) -> Path:
    """Render a minimal TOML config file under project_root.

    Returns
    -------
    Path
        Path to the written configuration file.
    """
    lines: list[str] = []
    for section, values in sections.items():
        lines.append(f"[{section}]")
        for key, value in values.items():
            lines.append(f"{key} = {_format_toml_value(value)}")
        lines.append("")
    config_path = project_root / CONFIG_FILE_NAME
    config_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return config_path


def sample_target_graph(
    targets: Sequence[TargetDescriptor] | None = None,
    *,
    table_keys_by_target: Mapping[str, Sequence[str]] | None = None,
) -> DagCatalog:
    """Build a small DAG catalog for tests.

    Returns
    -------
    DagCatalog
        Catalog populated with default targets unless provided.
    """
    if targets is None:
        targets = _default_targets()
        table_keys_by_target = table_keys_by_target or _DEFAULT_TABLE_KEYS_BY_TARGET
    return build_catalog(targets=targets, table_keys_by_target=table_keys_by_target)


@dataclass(frozen=True)
class ManifestParams:
    """Parameters for constructing manifest fixtures."""

    repo: str = DEFAULT_VARIANT.repo
    commit: str = DEFAULT_VARIANT.commit
    impl_kind: str = "native"
    input_hash: str = "input-hash"
    computed_at: datetime | None = None
    duration_ms: float = 1.0
    row_count: int | None = None
    output_hash: str | None = None
    options_hash: str | None = None


def sample_manifest(target: str, params: ManifestParams | None = None) -> OutputManifest:
    """Create an OutputManifest with defaulted timestamps and hashes.

    Parameters
    ----------
    target
        Target name for the manifest.
    params
        Optional manifest parameters; defaults will be used otherwise.

    Returns
    -------
    OutputManifest
        Manifest ready for insertion into tracking tables.
    """
    cfg = params or ManifestParams()
    return OutputManifest(
        target=target,
        repo=cfg.repo,
        commit=cfg.commit,
        impl_kind=cfg.impl_kind,
        computed_at=cfg.computed_at or datetime.now(tz=UTC),
        duration_ms=cfg.duration_ms,
        input_hash=cfg.input_hash,
        output_hash=cfg.output_hash,
        row_count=cfg.row_count,
        options_hash=cfg.options_hash,
    )


@dataclass
class RecordingProviders:
    """Container of fake providers with recording hooks."""

    tool_runner: FakeToolRunner = field(default_factory=FakeToolRunner)
    scip_indexer: FakeScipIndexer = field(default_factory=FakeScipIndexer)
    type_checker: FakeTypeChecker = field(default_factory=FakeTypeChecker)
    coverage_collector: FakeCoverageCollector = field(default_factory=FakeCoverageCollector)
    test_reporter: FakeTestReporter = field(default_factory=FakeTestReporter)
    git_history: FakeGitHistoryProvider = field(default_factory=FakeGitHistoryProvider)

    def as_dict(self) -> dict[str, object]:
        """Return providers as a mapping keyed by provider name.

        Returns
        -------
        dict[str, object]
            Mapping of provider name to fake implementation.
        """
        return {
            "tool_runner": self.tool_runner,
            "scip_indexer": self.scip_indexer,
            "type_checker": self.type_checker,
            "coverage_collector": self.coverage_collector,
            "test_reporter": self.test_reporter,
            "git_history": self.git_history,
        }


def _format_toml_value(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, Sequence) and not isinstance(value, (bytes, str)):
        return "[" + ", ".join(_format_toml_value(v) for v in value) + "]"
    message = f"Unsupported TOML value type: {type(value).__name__}"
    raise TypeError(message)


_DEFAULT_TABLE_KEYS_BY_TARGET: Mapping[str, Sequence[str]] = {
    "modules": ("core.modules",),
    "ast": ("core.ast_nodes",),
    "goids": ("core.goids",),
    "function_metrics": ("analytics.function_metrics",),
}


def _default_targets() -> tuple[TargetDescriptor, ...]:
    ingestion_modules = make_target_descriptor(
        name="modules",
        module="ingestion",
        description="Repository module index",
    )
    ast_target = make_target_descriptor(
        name="ast",
        module="ingestion",
        dependencies=("modules",),
        description="AST extraction",
    )
    goids_target = make_target_descriptor(
        name="goids",
        module="graphs",
        dependencies=("ast",),
        description="GOID construction",
    )
    metrics_target = make_target_descriptor(
        name="function_metrics",
        module="analytics",
        dependencies=("goids",),
        description="Function metrics",
    )
    return (ingestion_modules, ast_target, goids_target, metrics_target)


__all__ = [
    "ManifestParams",
    "RecordingProviders",
    "make_build_config",
    "make_build_paths",
    "make_snapshot",
    "sample_manifest",
    "sample_target_graph",
    "write_build_config",
]
