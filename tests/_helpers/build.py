"""Helpers for build-system unit tests.

Use catalog outputs derived from saver tags; tests should not declare output
inventories outside the catalog.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.build.config import CONFIG_FILE_NAME, BuildConfig
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.targets import TargetDescriptor
from codeintel.core.config.settings import BuildSettings, ExportAuditSettings
from codeintel.core.runtime.loader import load_runtime_settings
from tests._helpers.catalog import build_catalog, make_target_descriptor
from tests._helpers.fakes.configs import create_test_build_paths, create_test_snapshot
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT

if TYPE_CHECKING:
    from codeintel.config.primitives import BuildPaths, SnapshotRef

_RUNTIME_BUILD_SETTINGS = load_runtime_settings().build
TEST_BUILD_SETTINGS = BuildSettings(
    engine_version="test",
    export_audit=_RUNTIME_BUILD_SETTINGS.export_audit or ExportAuditSettings(),
)


def make_build_settings(engine_version: str = "test") -> BuildSettings:
    """Return BuildSettings for tests.

    Parameters
    ----------
    engine_version
        Engine version identifier to embed in hashes.

    Returns
    -------
    BuildSettings
        Build settings for tests.
    """
    return BuildSettings(
        engine_version=engine_version,
        export_audit=_RUNTIME_BUILD_SETTINGS.export_audit or ExportAuditSettings(),
    )


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


def make_build_paths(repo_root: Path | None = None) -> BuildPaths:
    """Create BuildPaths rooted at repo_root (or current directory).

    Parameters
    ----------
    repo_root
        Repository root for build outputs. If None, uses current directory.

    Returns
    -------
    BuildPaths
        Build paths configured for tests.
    """
    resolved_root = repo_root if repo_root is not None else Path.cwd()
    return create_test_build_paths(resolved_root)


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
    "function_types": ("analytics.function_types",),
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
    types_target = make_target_descriptor(
        name="function_types",
        module="analytics",
        dependencies=("goids",),
        description="Function typing metadata",
    )
    return (ingestion_modules, ast_target, goids_target, types_target)


__all__ = [
    "make_build_config",
    "make_build_paths",
    "make_snapshot",
    "sample_target_graph",
    "write_build_config",
]
