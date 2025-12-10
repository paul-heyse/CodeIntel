"""Shared helpers for ingestion plugin tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.contracts import OutputContract
from codeintel.build.targets import OutputTarget
from codeintel.config.models import ToolsConfig
from codeintel.config.datasets.primitives import TableSchema
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
from tests._helpers.gateway import GatewayFactory
from tests._helpers.fakes.tools import write_dummy_scip_files as write_dummy_scip_files
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.plugins.tests_plugin import TestsIngestPlugin

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
    "build_repo_target",
    "build_target_context_for_plugin",
    "build_scip_ingest_context",
    "ScipIngestContext",
    "module_paths_from_context",
    "module_records_for_paths",
    "seed_modules_and_repo_map",
    "write_coverage_file",
    "write_pytest_report",
    "write_scip_index",
    "write_dummy_scip_files",
]


@dataclass(frozen=True)
class TargetContextConfig:
    """Configuration for building a TargetExecutionContext."""

    repo_root: Path | None = None
    modules: tuple[str, ...] | None = None
    providers: Providers | None = None
    gateway: StorageGateway | None = None
    gateway_factory: GatewayFactory | None = None
    resources: TargetResourceOverrides | None = None


def build_target_context_for_plugin(
    plugin: TargetPlugin,
    tmp_path: Path,
    *,
    config: TargetContextConfig | None = None,
    target: OutputTarget | None = None,
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

    gateway = cfg.gateway
    if gateway is None:
        factory = cfg.gateway_factory or GatewayFactory().with_macros()
        gateway = factory.open()

    overrides = EnvOverrides(tmp_path=effective_repo_root, gateway=gateway)
    builder = ExecutionContextBuilder.create(tmp_path, env_overrides=overrides)
    effective_target = target or make_test_output_target(plugin)
    resource_overrides = cfg.resources or TargetResourceOverrides(
        providers=cfg.providers,
        modules=cfg.modules or (),
    )
    return builder.build_target_context(target=effective_target, resources=resource_overrides)


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
    content: str | Mapping[str, object] | None = None,
) -> Path:
    """Write a coverage artifact into the build directory.

    Returns
    -------
    Path
        Path to the written coverage file.
    """
    coverage_path = build_dir / filename
    coverage_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(content) if isinstance(content, Mapping) else content or "{}"
    coverage_path.write_text(payload, encoding="utf-8")
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


def write_scip_index(
    build_dir: Path,
    documents: Sequence[Mapping[str, object]],
    *,
    filename: str = "index.json",
) -> Path:
    """
    Write a SCIP index JSON artifact under ``build/scip``.

    Parameters
    ----------
    build_dir
        Base build directory for the target context.
    documents
        Iterable of SCIP document payloads.
    filename
        Optional filename override (defaults to ``index.json``).

    Returns
    -------
    Path
        Path to the written index file.
    """
    scip_dir = build_dir / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)
    index_path = scip_dir / filename
    typed_documents = [dict(doc) for doc in documents]
    index_path.write_text(json.dumps({"documents": typed_documents}), encoding="utf-8")
    return index_path


@dataclass(frozen=True)
class ScipIngestContext:
    """Bundle of SCIP ingest fixtures reused across tests."""

    repo_root: Path
    gateway: StorageGateway
    storage: DuckDBStorageAdapter
    tools: BuildToolAdapter
    build_dir: Path


def build_scip_ingest_context(tmp_path: Path) -> ScipIngestContext:
    """Create a repo, gateway, and adapters for SCIP ingest tests.

    Returns
    -------
    ScipIngestContext
        Context containing repo_root, gateway, adapters, and build_dir.
    """
    repo_root = build_repo_tree(
        tmp_path / "repo",
        {"pkg/__init__.py": "", "pkg/mod.py": "def foo(x: int) -> int:\n    return x + 1\n"},
    )
    build_dir = repo_root / "build"
    gateway = GatewayFactory().with_macros().open()
    plugin = TestsIngestPlugin()
    ctx = build_target_context_for_plugin(
        plugin,
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, gateway=gateway),
    )
    storage, discovery, change_detection, _ = build_ingestion_adapters(ctx)
    _ = discovery, change_detection
    tools_config = ToolsConfig.default()
    runner = ToolRunner(tools_config=tools_config, cache_dir=build_dir / ".tool_cache")
    service = ToolService(runner, tools_config)
    tool_adapter = ToolRunnerAdapter(service)
    return ScipIngestContext(
        repo_root=repo_root,
        gateway=gateway,
        storage=storage,
        tools=tool_adapter,
        build_dir=build_dir,
    )


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


def seed_modules_and_repo_map(
    ctx: TargetExecutionContext,
    paths: Sequence[str],
) -> None:
    """
    Insert module rows and a repo_map entry for the provided paths.

    Parameters
    ----------
    ctx
        Target execution context with gateway and repo metadata.
    paths
        Iterable of module-relative file paths to seed.
    """
    records = module_records_for_paths(paths, ctx.repo_root)
    con = ctx.gateway.con
    con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    )
    con.execute(
        "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
        [ctx.repo, ctx.commit],
    )
    con.executemany(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, 'python', '[]', '[]')
        """,
        [
            (
                record.module_name,
                record.file_path.relative_to(ctx.repo_root).as_posix(),
                ctx.repo,
                ctx.commit,
            )
            for record in records
        ],
    )
    modules_json = {
        record.module_name: record.file_path.relative_to(ctx.repo_root).as_posix()
        for record in records
    }
    con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
        VALUES (?, ?, ?, '{}', CURRENT_TIMESTAMP)
        """,
        [ctx.repo, ctx.commit, json.dumps(modules_json)],
    )


def build_repo_target(plugin: TargetPlugin, tables: tuple[TableSchema, ...]) -> OutputTarget:
    """
    Construct an OutputTarget for repo-oriented plugins.

    Parameters
    ----------
    plugin
        Plugin instance providing name/description metadata.
    tables
        Contract tables expected to be produced.

    Returns
    -------
    OutputTarget
        Target wired with an OutputContract for the supplied tables.
    """
    return OutputTarget(
        name=plugin.plugin_name,
        module="ingestion",
        plugin=plugin.plugin_name,
        contract=OutputContract(tables=tables),
        description=plugin.plugin_description,
    )
