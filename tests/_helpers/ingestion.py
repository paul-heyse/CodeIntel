"""Shared helpers for ingestion plugin tests."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

from codeintel.build.context import TargetExecutionContext
from codeintel.build.contracts import OutputContract
from codeintel.build.targets import OutputTarget
from codeintel.config.datasets.primitives import TableSchema
from codeintel.config.models import ToolsConfig
from codeintel.ingestion import (
    BuildToolAdapter,
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.compute import DocstringsExtractStep, RepoScanStep
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.infrastructure.scanning import ScanProfile
from codeintel.ingestion.plugins.helpers import get_module_paths, paths_to_modules
from codeintel.ingestion.plugins.repo_scan import RepoScanPlugin
from codeintel.ingestion.plugins.tests_plugin import TestsIngestPlugin
from tests._helpers import build_repo_tree
from tests._helpers.fakes.contexts import (
    EnvOverrides,
    ExecutionContextBuilder,
    TargetResourceOverrides,
    make_test_output_target,
)
from tests._helpers.fakes.tools import write_dummy_scip_files
from tests._helpers.gateway import GatewayFactory

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.build.plugin import TargetPlugin
    from codeintel.build.providers import Providers
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.storage.gateway import StorageGateway

__all__ = [
    "ScipIngestContext",
    "TargetContextConfig",
    "build_ingestion_adapters",
    "build_repo_tree",
    "build_repo_target",
    "build_repo_with_configs",
    "build_scan_profile",
    "build_scip_ingest_context",
    "build_target_context_for_plugin",
    "create_scan_and_docstring_steps",
    "create_scan_step",
    "module_paths_from_context",
    "module_records_for_paths",
    "seed_foreign_key_tables",
    "seed_inventory_from_paths",
    "seed_modules_and_repo_map",
    "seed_numeric_table",
    "seed_varchar_table",
    "write_coverage_file",
    "write_dummy_scip_files",
    "write_pytest_report",
    "write_scip_index",
]


@dataclass(frozen=True)
class TargetContextConfig:
    """Configuration for building a TargetExecutionContext."""

    repo_root: Path | None = None
    modules: tuple[str, ...] | None = None
    providers: Providers | None = None
    gateway: StorageGateway | None = None
    gateway_factory: GatewayFactory | None = None
    snapshot: tuple[str, str] | None = None
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

    overrides = EnvOverrides(
        snapshot=cfg.snapshot,
        tmp_path=effective_repo_root,
        gateway=gateway,
    )
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


def build_repo_with_configs(
    tmp_path: Path,
    *,
    include_invalid: bool = False,
) -> tuple[Path, tuple[str, ...]]:
    """Create a realistic repo layout with optional invalid config.

    Returns
    -------
    tuple[Path, tuple[str, ...]]
        Repo root and the Python module paths created.
    """
    structure: dict[str, str] = {
        "pkg/__init__.py": "",
        "pkg/service.py": "def add(x: int, y: int) -> int:\n    return x + y\n",
        "config/app.yaml": "service:\n  retries: 3\n  hosts:\n    - api.local\n",
        "config/settings.toml": 'feature = true\n[db]\nurl = "duckdb:///tmp.db"\n',
        "config/app.ini": "[service]\ntimeout=30\n",
    }
    if include_invalid:
        structure["config/broken.yml"] = ":\n  - invalid\n"

    repo_root = build_repo_tree(tmp_path / "repo", structure)
    modules = tuple(path for path in structure if path.endswith(".py"))
    return repo_root, modules


def build_scan_profile(
    repo_root: Path,
    *,
    include_globs: Sequence[str] = ("*.py",),
    ignore_dirs: Sequence[str] = (),
    log_every: int | None = None,
    log_interval: float | None = None,
) -> ScanProfile:
    """Create a ScanProfile with shared defaults.

    Returns
    -------
    ScanProfile
        Profile covering the repo root with optional ignore patterns.
    """
    if log_every is None and log_interval is None:
        return ScanProfile(
            repo_root=repo_root,
            source_roots=(repo_root,),
            include_globs=tuple(include_globs),
            ignore_dirs=tuple(ignore_dirs),
        )
    if log_every is None and log_interval is not None:
        return ScanProfile(
            repo_root=repo_root,
            source_roots=(repo_root,),
            include_globs=tuple(include_globs),
            ignore_dirs=tuple(ignore_dirs),
            log_interval=log_interval,
        )
    if log_interval is None and log_every is not None:
        return ScanProfile(
            repo_root=repo_root,
            source_roots=(repo_root,),
            include_globs=tuple(include_globs),
            ignore_dirs=tuple(ignore_dirs),
            log_every=log_every,
        )
    return ScanProfile(
        repo_root=repo_root,
        source_roots=(repo_root,),
        include_globs=tuple(include_globs),
        ignore_dirs=tuple(ignore_dirs),
        log_every=log_every or 0,
        log_interval=log_interval or 0.0,
    )


def create_scan_step(
    gateway: StorageGateway,
    repo_root: Path,
    tmp_path: Path,
) -> tuple[RepoScanStep, DuckDBStorageAdapter, FilesystemDiscoveryAdapter]:
    """Create scan step and adapters for a repository.

    Returns
    -------
    tuple[RepoScanStep, DuckDBStorageAdapter, FilesystemDiscoveryAdapter]
        Repo scan step plus the storage and discovery adapters backing it.
    """
    ctx = build_target_context_for_plugin(
        RepoScanPlugin(),
        tmp_path,
        config=TargetContextConfig(repo_root=repo_root, gateway=gateway),
    )
    storage, discovery, change_detection, _ = build_ingestion_adapters(ctx)
    scan_step = RepoScanStep(
        storage=storage, discovery=discovery, change_detection=change_detection
    )
    return scan_step, storage, discovery


def create_scan_and_docstring_steps(
    gateway: StorageGateway,
    repo_root: Path,
    tmp_path: Path,
) -> tuple[RepoScanStep, DocstringsExtractStep]:
    """Create scan and docstring steps from gateway and repo root.

    Returns
    -------
    tuple[RepoScanStep, DocstringsExtractStep]
        Configured repo scan and docstring extraction steps.
    """
    scan_step, storage, discovery = create_scan_step(gateway, repo_root, tmp_path)
    doc_step = DocstringsExtractStep(storage=storage, discovery=discovery)
    return scan_step, doc_step


@dataclass(frozen=True)
class ScipIngestContext:
    """Bundle of SCIP ingest fixtures reused across tests."""

    repo_root: Path
    gateway: StorageGateway
    storage: DuckDBStorageAdapter
    tools: ToolRunnerAdapter
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
    db_path = build_dir / "db" / "codeintel.duckdb"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    gateway = GatewayFactory().file_backed(db_path).with_macros().open()
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


def seed_inventory_from_paths(
    repo_root: Path,
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    paths: Sequence[str],
) -> None:
    """Seed core.modules and repo_map tables from relative paths."""
    dummy_context = SimpleNamespace(
        gateway=gateway,
        repo=repo,
        commit=commit,
        repo_root=repo_root,
    )
    seed_modules_and_repo_map(cast("TargetExecutionContext", dummy_context), paths)


def seed_foreign_key_tables(
    gateway: StorageGateway,
    *,
    parent_table: str,
    child_table: str,
    parent_rows: Sequence[tuple[int, str]],
    child_rows: Sequence[tuple[int, int | None]],
) -> None:
    """
    Create parent/child tables and populate rows for orphan-ref tests.

    Raises
    ------
    ValueError
        If the provided table pair is not supported.
    """
    sql_lookup: dict[tuple[str, str], tuple[str, str, str, str, str, str]] = {
        (
            "core.test_parent",
            "core.test_child",
        ): (
            "CREATE TABLE IF NOT EXISTS core.test_parent (id INTEGER PRIMARY KEY, name VARCHAR)",
            "CREATE TABLE IF NOT EXISTS core.test_child (id INTEGER, parent_id INTEGER)",
            "DELETE FROM core.test_parent",
            "DELETE FROM core.test_child",
            "INSERT INTO core.test_parent (id, name) VALUES (?, ?)",
            "INSERT INTO core.test_child (id, parent_id) VALUES (?, ?)",
        ),
        (
            "core.test_parent2",
            "core.test_child2",
        ): (
            "CREATE TABLE IF NOT EXISTS core.test_parent2 (id INTEGER PRIMARY KEY, name VARCHAR)",
            "CREATE TABLE IF NOT EXISTS core.test_child2 (id INTEGER, parent_id INTEGER)",
            "DELETE FROM core.test_parent2",
            "DELETE FROM core.test_child2",
            "INSERT INTO core.test_parent2 (id, name) VALUES (?, ?)",
            "INSERT INTO core.test_child2 (id, parent_id) VALUES (?, ?)",
        ),
        (
            "core.test_parent3",
            "core.test_child3",
        ): (
            "CREATE TABLE IF NOT EXISTS core.test_parent3 (id INTEGER PRIMARY KEY, name VARCHAR)",
            "CREATE TABLE IF NOT EXISTS core.test_child3 (id INTEGER, parent_id INTEGER)",
            "DELETE FROM core.test_parent3",
            "DELETE FROM core.test_child3",
            "INSERT INTO core.test_parent3 (id, name) VALUES (?, ?)",
            "INSERT INTO core.test_child3 (id, parent_id) VALUES (?, ?)",
        ),
    }
    queries = sql_lookup.get((parent_table, child_table))
    if queries is None:
        message = f"Unsupported foreign key test tables: {parent_table}, {child_table}"
        raise ValueError(message)

    (
        create_parent_sql,
        create_child_sql,
        delete_parent_sql,
        delete_child_sql,
        insert_parent_sql,
        insert_child_sql,
    ) = queries
    gateway.con.execute(create_parent_sql)
    gateway.con.execute(create_child_sql)
    gateway.con.execute(delete_parent_sql)
    gateway.con.execute(delete_child_sql)
    if parent_rows:
        gateway.con.executemany(
            insert_parent_sql,
            list(parent_rows),
        )
    if child_rows:
        gateway.con.executemany(
            insert_child_sql,
            list(child_rows),
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


def seed_numeric_table(gateway: StorageGateway, table: str, values: Sequence[float]) -> None:
    """
    Create or truncate a numeric table and insert provided values.

    Raises
    ------
    ValueError
        If an unsupported table name is provided.
    """
    sql_lookup: dict[str, tuple[str, str, str]] = {
        "core.test_numeric": (
            "CREATE TABLE IF NOT EXISTS core.test_numeric (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_numeric",
            "INSERT INTO core.test_numeric (id, value) VALUES (?, ?)",
        ),
        "core.test_numeric2": (
            "CREATE TABLE IF NOT EXISTS core.test_numeric2 (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_numeric2",
            "INSERT INTO core.test_numeric2 (id, value) VALUES (?, ?)",
        ),
        "core.test_empty_num": (
            "CREATE TABLE IF NOT EXISTS core.test_empty_num (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_empty_num",
            "INSERT INTO core.test_empty_num (id, value) VALUES (?, ?)",
        ),
        "core.test_empty_num2": (
            "CREATE TABLE IF NOT EXISTS core.test_empty_num2 (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_empty_num2",
            "INSERT INTO core.test_empty_num2 (id, value) VALUES (?, ?)",
        ),
        "core.test_pos": (
            "CREATE TABLE IF NOT EXISTS core.test_pos (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_pos",
            "INSERT INTO core.test_pos (id, value) VALUES (?, ?)",
        ),
        "core.test_all_pos": (
            "CREATE TABLE IF NOT EXISTS core.test_all_pos (id INTEGER, value DOUBLE)",
            "DELETE FROM core.test_all_pos",
            "INSERT INTO core.test_all_pos (id, value) VALUES (?, ?)",
        ),
    }
    queries = sql_lookup.get(table)
    if queries is None:
        message = f"Unsupported numeric table for tests: {table}"
        raise ValueError(message)

    create_sql, delete_sql, insert_sql = queries
    gateway.con.execute(create_sql)
    gateway.con.execute(delete_sql)
    params = [(idx, value) for idx, value in enumerate(values, start=1)]
    if params:
        gateway.con.executemany(insert_sql, params)


def seed_varchar_table(
    gateway: StorageGateway,
    table: str,
    values: Sequence[tuple[int, str | None]],
) -> None:
    """
    Create or truncate a VARCHAR table and insert provided rows.

    Raises
    ------
    ValueError
        If an unsupported table name is provided.
    """
    sql_lookup: dict[str, tuple[str, str, str]] = {
        "core.test_nulls": (
            "CREATE TABLE IF NOT EXISTS core.test_nulls (id INTEGER, value VARCHAR)",
            "DELETE FROM core.test_nulls",
            "INSERT INTO core.test_nulls (id, value) VALUES (?, ?)",
        ),
        "core.test_dupes": (
            "CREATE TABLE IF NOT EXISTS core.test_dupes (id INTEGER, name VARCHAR)",
            "DELETE FROM core.test_dupes",
            "INSERT INTO core.test_dupes (id, name) VALUES (?, ?)",
        ),
        "core.test_unique": (
            "CREATE TABLE IF NOT EXISTS core.test_unique (id INTEGER, name VARCHAR)",
            "DELETE FROM core.test_unique",
            "INSERT INTO core.test_unique (id, name) VALUES (?, ?)",
        ),
        "core.test_frac1": (
            "CREATE TABLE IF NOT EXISTS core.test_frac1 (id INTEGER, value VARCHAR)",
            "DELETE FROM core.test_frac1",
            "INSERT INTO core.test_frac1 (id, value) VALUES (?, ?)",
        ),
        "core.test_frac2": (
            "CREATE TABLE IF NOT EXISTS core.test_frac2 (id INTEGER, value VARCHAR)",
            "DELETE FROM core.test_frac2",
            "INSERT INTO core.test_frac2 (id, value) VALUES (?, ?)",
        ),
        "core.test_frac3": (
            "CREATE TABLE IF NOT EXISTS core.test_frac3 (id INTEGER, value VARCHAR)",
            "DELETE FROM core.test_frac3",
            "INSERT INTO core.test_frac3 (id, value) VALUES (?, ?)",
        ),
        "core.test_frac_empty": (
            "CREATE TABLE IF NOT EXISTS core.test_frac_empty (id INTEGER, value VARCHAR)",
            "DELETE FROM core.test_frac_empty",
            "INSERT INTO core.test_frac_empty (id, value) VALUES (?, ?)",
        ),
    }
    queries = sql_lookup.get(table)
    if queries is None:
        message = f"Unsupported varchar table for tests: {table}"
        raise ValueError(message)

    create_sql, delete_sql, insert_sql = queries
    gateway.con.execute(create_sql)
    gateway.con.execute(delete_sql)
    if values:
        gateway.con.executemany(insert_sql, list(values))
