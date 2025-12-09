"""Shared helpers for ingestion plugin tests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.ingestion import (
    BuildToolAdapter,
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.plugins.helpers import get_module_paths, paths_to_modules
from tests._helpers.fakes.contexts import (
    EnvOverrides,
    ExecutionContextBuilder,
    TargetResourceOverrides,
    make_test_output_target,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.build.context import TargetExecutionContext
    from codeintel.build.plugin import TargetPlugin
    from codeintel.build.providers import Providers
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.storage.gateway import StorageGateway

__all__ = [
    "TargetContextConfig",
    "build_ingestion_adapters",
    "build_target_context_for_plugin",
    "module_paths_from_context",
    "module_records_for_paths",
    "write_coverage_file",
    "write_pytest_report",
]


@dataclass(frozen=True)
class TargetContextConfig:
    """Configuration for building a TargetExecutionContext."""

    repo_root: Path | None = None
    modules: tuple[str, ...] | None = None
    providers: Providers | None = None
    gateway: StorageGateway | None = None
    resources: TargetResourceOverrides | None = None


def build_target_context_for_plugin(
    plugin: TargetPlugin,
    tmp_path: Path,
    *,
    config: TargetContextConfig | None = None,
) -> TargetExecutionContext:
    """Construct a TargetExecutionContext wired to real adapters.

    Returns
    -------
    TargetExecutionContext
        Context with gateway, resources, and target metadata.
    """
    cfg = config or TargetContextConfig()
    effective_repo_root = cfg.repo_root or (tmp_path / "repo")
    effective_repo_root.mkdir(parents=True, exist_ok=True)

    overrides = EnvOverrides(tmp_path=effective_repo_root, gateway=cfg.gateway)
    builder = ExecutionContextBuilder.create(tmp_path, env_overrides=overrides)
    target = make_test_output_target(plugin)
    resource_overrides = cfg.resources or TargetResourceOverrides(
        providers=cfg.providers,
        modules=cfg.modules or (),
    )
    return builder.build_target_context(target=target, resources=resource_overrides)


def build_ingestion_adapters(
    ctx: TargetExecutionContext,
) -> tuple[
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
    BuildToolAdapter,
]:
    """Create ingestion adapters aligned with a target context.

    Returns
    -------
    tuple
        storage, discovery, change detection, and tool adapters.
    """
    storage = DuckDBStorageAdapter(ctx.gateway)
    discovery = FilesystemDiscoveryAdapter(ctx.repo_root)
    change_detection = HashChangeDetectionAdapter(storage)
    tools = BuildToolAdapter(
        type_checker=ctx.resources.type_checker,
        coverage_collector=ctx.resources.coverage_collector,
        scip_indexer=ctx.resources.scip_indexer,
        test_reporter=ctx.resources.test_reporter,
    )
    return storage, discovery, change_detection, tools


def write_coverage_file(
    build_dir: Path,
    *,
    filename: str = "coverage.json",
    content: str | None = None,
) -> Path:
    """Write a coverage artifact into the build directory.

    Returns
    -------
    Path
        Path to the written coverage file.
    """
    coverage_path = build_dir / filename
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    coverage_path.write_text(content or "{}", encoding="utf-8")
    return coverage_path


def write_pytest_report(
    build_dir: Path,
    *,
    tests: list[dict[str, object]] | None = None,
    summary: dict[str, object] | None = None,
    filename: str = "pytest-report.json",
) -> Path:
    """Render a pytest JSON report under the build/test-results directory.

    Returns
    -------
    Path
        Path to the written report file.
    """
    report_dir = build_dir / "test-results"
    report_dir.mkdir(parents=True, exist_ok=True)
    payload = {"tests": tests or [], "summary": summary or {}}
    report_path = report_dir / filename
    report_path.write_text(json.dumps(payload), encoding="utf-8")
    return report_path


def module_paths_from_context(ctx: TargetExecutionContext) -> list[str]:
    """Fetch module paths using the shared helper.

    Returns
    -------
    list[str]
        Module paths for the target context.
    """
    return get_module_paths(ctx)


def module_records_for_paths(
    paths: Sequence[str],
    repo_root: Path,
) -> list[ModuleRecord]:
    """Convert relative paths to ModuleRecord objects with metadata.

    Returns
    -------
    list[ModuleRecord]
        Module records containing module names and file paths.
    """
    return paths_to_modules(paths, repo_root)
