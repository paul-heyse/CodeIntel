"""Provisioning helpers for production-parity gateway-backed tests.

This module provides functions for setting up test environments with
real database schemas, ingestion pipelines, and tooling configurations.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.graphs import compute_graph_metrics
from codeintel.build.context import ContextResources
from codeintel.config import ConfigBuilder, SnapshotInit
from codeintel.config.primitives import BuildPathOverrides, BuildPaths, SnapshotRef
from codeintel.graphs.plugins.builders.callgraph import CallGraphPlugin
from codeintel.ingestion import (
    CoverageIngestStep,
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
    ModuleRecord,
    RepoScanStep,
    ToolRunnerAdapter,
    TypingIngestStep,
)
from codeintel.ingestion.engine.infrastructure import ToolName, ToolRunner
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway
from codeintel.storage.macros import ensure_ingest_macros, list_ingest_macros
from codeintel.storage.metadata import INGEST_MACROS
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    GoidRow,
    RepoMapRow,
    SymbolUseEdgeRow,
    insert_rows,
)
from tests._helpers.configs import (
    DEFAULT_COMMIT,
    DEFAULT_REPO,
    CallgraphFixtureOptions,
    GatewayOptions,
    GraphMetricsGatewayOptions,
    ProvisionedGateway,
    ProvisioningConfig,
    ProvisioningSetup,
    ProvisionOptions,
    RepoContext,
)
from tests._helpers.context import TestContext
from tests._helpers.fakes import utcnow
from tests._helpers.gateway import gateway_with_macros
from tests._helpers.orchestration.repo_writers import (
    write_callgraph_alias_repo,
    write_coverage_driver,
    write_graph_metrics_repo,
    write_sample_repo,
)
from tests._helpers.orchestration.seeding import seed_callgraph_goids, seed_cfg_dfg_for_metrics
from tests._helpers.orchestration.seeding_docs import seed_docs_export_minimal
from tests._helpers.orchestration.tooling import make_tools_config
from tests._helpers.plugin_execution import PluginTestContext, execute_target_plugin

if TYPE_CHECKING:
    from codeintel.config.models import ToolsConfig
    from codeintel.storage.gateway import DuckDBConnection


def _assert_ingest_macros_present(con: DuckDBConnection) -> None:
    """Fail fast if ingest macros are missing for a connection.

    Raises
    ------
    RuntimeError
        When any ingest macro is missing.
    """
    macros = list_ingest_macros(con)
    missing = {m.lower() for m in INGEST_MACROS.values() if m.lower() not in macros}
    if missing:
        message = f"Missing ingest macros on gateway: {sorted(missing)}"
        raise RuntimeError(message)


def make_repo_context(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    db_path: Path | None = None,
) -> RepoContext:
    """Build a RepoContext with derived build/document paths.

    Parameters
    ----------
    repo_root
        Root directory of the repository.
    repo
        Repository identifier.
    commit
        Commit hash.
    db_path
        Optional explicit database path.

    Returns
    -------
    RepoContext
        Derived paths and identifiers for the repo.
    """
    build_dir = repo_root / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    document_output_dir = repo_root / "Document Output"
    db = db_path or build_dir / "db" / "codeintel.duckdb"
    db.parent.mkdir(parents=True, exist_ok=True)
    return RepoContext(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=build_dir,
        db_path=db,
        document_output_dir=document_output_dir,
    )


def _generate_coverage_payload(
    repo_root: Path,
    *,
    coverage_file: Path,
    files: list[Path],
    runner: ToolRunner,
) -> None:
    """Execute coverage run against the sample repo using the real binary.

    Raises
    ------
    RuntimeError
        If the coverage tool exits with a non-zero status.
    """
    driver_path = write_coverage_driver(repo_root, files)
    result = runner.run(
        ToolName.COVERAGE,
        ["run", "--data-file", str(coverage_file), str(driver_path)],
        cwd=repo_root,
    )
    if not result.ok:
        message = f"coverage run failed: code={result.returncode} stderr={result.stderr}"
        raise RuntimeError(message)


def _make_runner(
    repo_root: Path,
    files: list[Path],
    *,
    coverage_file: Path,
    tools_cfg: ToolsConfig,
) -> ToolRunner:
    """Build a ToolRunner seeded with real tool binaries and coverage data.

    Returns
    -------
    ToolRunner
        Runner configured with real tooling and a populated coverage DB.
    """
    cache_dir = repo_root / "build" / ".tool_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner(tools_config=tools_cfg, cache_dir=cache_dir)
    _generate_coverage_payload(
        repo_root,
        coverage_file=coverage_file,
        files=files,
        runner=runner,
    )
    return runner


def _open_gateway_from_context(ctx: RepoContext, opts: GatewayOptions) -> StorageGateway:
    """Open a gateway from a RepoContext with the given options.

    Parameters
    ----------
    ctx
        Repository context with paths and identifiers.
    opts
        Gateway configuration options.

    Returns
    -------
    StorageGateway
        Configured gateway instance.
    """
    effective_ensure_views = opts.ensure_views or opts.strict_schema
    effective_validate_schema = opts.validate_schema or opts.strict_schema
    if opts.file_backed:
        cfg = StorageConfig(
            db_path=ctx.db_path,
            read_only=False,
            apply_schema=opts.apply_schema,
            ensure_views=effective_ensure_views,
            validate_schema=effective_validate_schema,
            repo=ctx.repo,
            commit=ctx.commit,
        )
        gateway = open_gateway(cfg)
        ensure_ingest_macros(gateway.con)
    else:
        gateway = gateway_with_macros(
            apply_schema=opts.apply_schema,
            ensure_views=effective_ensure_views,
            validate_schema=effective_validate_schema,
            repo=ctx.repo,
            commit=ctx.commit,
        )
    _assert_ingest_macros_present(gateway.con)
    return gateway


def _build_provisioning_setup(
    repo_root: Path,
    files: list[Path],
    opts: ProvisionOptions,
    repo: str,
    commit: str,
) -> ProvisioningSetup:
    """Build all components needed for repo provisioning.

    Parameters
    ----------
    repo_root
        Root path of the repository.
    files
        List of Python files discovered in the repo.
    opts
        Provisioning options.
    repo
        Repository identifier.
    commit
        Commit hash.

    Returns
    -------
    ProvisioningSetup
        Container with all provisioning components.
    """
    ctx = make_repo_context(repo_root, repo=repo, commit=commit, db_path=opts.db_path)
    build_paths = BuildPaths.from_explicit(
        build_dir=ctx.build_dir,
        overrides=BuildPathOverrides(
            db_path=ctx.db_path,
            document_output_dir=ctx.document_output_dir,
            coverage_json=repo_root / ".coverage",
            pytest_report=ctx.build_dir / "test-results" / "pytest-report.json",
            scip_dir=ctx.build_dir / "scip",
            tool_cache=ctx.build_dir / ".tool_cache",
            log_db_path=ctx.build_dir / "db" / "codeintel_logs.duckdb",
        ),
    )
    coverage_file = build_paths.coverage_json

    tools_cfg = make_tools_config()
    runner = _make_runner(repo_root, files, coverage_file=coverage_file, tools_cfg=tools_cfg)
    tool_service = ToolService(runner, tools_cfg)

    gateway_opts = GatewayOptions(file_backed=opts.file_backed)
    gateway = _open_gateway_from_context(ctx, gateway_opts)
    ensure_ingest_macros(gateway.con)
    _assert_ingest_macros_present(gateway.con)

    storage = DuckDBStorageAdapter(gateway)
    discovery = FilesystemDiscoveryAdapter(repo_root)
    change_detection = HashChangeDetectionAdapter(storage)
    tool_adapter = ToolRunnerAdapter(tool_service)

    return ProvisioningSetup(
        ctx=ctx,
        build_paths=build_paths,
        coverage_file=coverage_file,
        tools_cfg=tools_cfg,
        runner=runner,
        tool_service=tool_service,
        gateway=gateway,
        storage=storage,
        discovery=discovery,
        change_detection=change_detection,
        tool_adapter=tool_adapter,
    )


def _run_ingestion_steps(
    setup: ProvisioningSetup,
    modules: list[ModuleRecord],
    opts: ProvisionOptions,
    repo: str,
    commit: str,
) -> None:
    """Run optional ingestion steps based on provision options.

    Parameters
    ----------
    setup
        Provisioning setup container.
    modules
        List of modules discovered during repo scan.
    opts
        Provisioning options.
    repo
        Repository identifier.
    commit
        Commit hash.
    """
    if opts.include_typing:
        typing_step = TypingIngestStep(
            storage=setup.storage,
            discovery=setup.discovery,
            tools=setup.tool_adapter,
        )
        asyncio.run(
            typing_step.execute_async(
                list(modules),
                repo=repo,
                commit=commit,
                repo_root=str(setup.ctx.repo_root),
            )
        )
    if opts.include_coverage:
        coverage_step = CoverageIngestStep(storage=setup.storage, tools=setup.tool_adapter)
        asyncio.run(
            coverage_step.execute_async(
                [],
                repo=repo,
                commit=commit,
                repo_root=setup.ctx.repo_root,
                coverage_file=setup.coverage_file,
            )
        )
    if opts.build_graph_metrics:
        seed_cfg_dfg_for_metrics(setup.gateway, rel_path="pkg/mod.py")
        compute_cfg_metrics(setup.gateway, repo=repo, commit=commit)
        compute_dfg_metrics(setup.gateway, repo=repo, commit=commit)


@contextmanager
def provisioned_gateway(
    repo_root: Path,
    config: ProvisioningConfig | None = None,
) -> Iterator[ProvisionedGateway]:
    """Context manager wrapping gateway provisioning and cleanup.

    Parameters
    ----------
    repo_root
        Root directory for the repository under test.
    config
        Provisioning configuration; defaults mirror ProvisioningConfig.

    Yields
    ------
    ProvisionedGateway
        Provisioned gateway scoped to the repo root.
    """
    cfg = config or ProvisioningConfig()
    if cfg.run_ingestion:
        ctx = provision_ingested_repo(
            repo_root,
            repo=cfg.repo,
            commit=cfg.commit,
            options=cfg.provision_options,
        )
    else:
        ctx = provision_gateway_with_repo(
            repo_root,
            repo=cfg.repo,
            commit=cfg.commit,
            options=cfg.gateway_options,
        )
    try:
        yield ctx
    finally:
        ctx.close()


def provision_ingested_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    options: ProvisionOptions | None = None,
) -> ProvisionedGateway:
    """Build a sample repo, run ingestion steps, and return a provisioned gateway.

    The gateway uses real schemas/views and populates:
    - core.modules/core.repo_map via ingest_repo
    - analytics.typedness/static_diagnostics via ingest_typing_signals
    - analytics.coverage_lines via ingest_coverage_lines

    Parameters
    ----------
    repo_root
        Root directory for the repository.
    repo
        Repository identifier.
    commit
        Commit hash.
    options
        Provisioning options.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway plus filesystem context for tests.
    """
    opts = options or ProvisionOptions()
    repo_root.mkdir(parents=True, exist_ok=True)

    files = write_sample_repo(repo_root)
    setup = _build_provisioning_setup(repo_root, files, opts, repo, commit)
    code_profile = default_code_profile(repo_root)

    scan_step = RepoScanStep(
        storage=setup.storage,
        discovery=setup.discovery,
        change_detection=setup.change_detection,
    )
    _, modules, _ = scan_step.execute(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        profile=code_profile,
    )

    # Insert repo_map entry needed for serving layer verification
    modules_map = {mod.rel_path: mod.module_name for mod in modules}
    insert_rows(
        setup.gateway,
        [
            RepoMapRow(
                repo=repo,
                commit=commit,
                modules=modules_map,
                overlays={},
            )
        ],
    )

    _run_ingestion_steps(setup, list(modules), opts, repo, commit)

    _assert_ingest_macros_present(setup.gateway.con)
    return ProvisionedGateway(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=setup.ctx.build_dir,
        db_path=setup.ctx.db_path,
        document_output_dir=setup.ctx.document_output_dir,
        coverage_file=setup.coverage_file,
        gateway=setup.gateway,
        runner=setup.runner,
    )


def provision_existing_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    options: ProvisionOptions | None = None,
) -> ProvisionedGateway:
    """Run ingestion over an existing repo tree using production entry points.

    Mirrors `provision_ingested_repo` but assumes callers have already written the
    desired repo contents to disk.

    Parameters
    ----------
    repo_root
        Root directory for the repository.
    repo
        Repository identifier.
    commit
        Commit hash.
    options
        Provisioning options.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway plus repo context.
    """
    opts = options or ProvisionOptions()
    repo_root.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in repo_root.rglob("*.py") if path.is_file())
    setup = _build_provisioning_setup(repo_root, files, opts, repo, commit)
    code_profile = default_code_profile(repo_root)

    scan_step = RepoScanStep(
        storage=setup.storage,
        discovery=setup.discovery,
        change_detection=setup.change_detection,
    )
    _, modules, _ = scan_step.execute(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        profile=code_profile,
    )

    _run_ingestion_steps(setup, list(modules), opts, repo, commit)

    return ProvisionedGateway(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=setup.ctx.build_dir,
        db_path=setup.ctx.db_path,
        document_output_dir=setup.ctx.document_output_dir,
        coverage_file=setup.coverage_file,
        gateway=setup.gateway,
        runner=setup.runner,
    )


def provision_gateway_with_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    options: GatewayOptions | None = None,
) -> ProvisionedGateway:
    """Open a gateway anchored to repo paths without running ingestion.

    Useful when tests need to seed custom rows (including invalid ones) but want
    the canonical schemas applied.

    Parameters
    ----------
    repo_root
        Root directory for the repository.
    repo
        Repository identifier.
    commit
        Commit hash.
    options
        Gateway options.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway with filesystem context.
    """
    opts = options or GatewayOptions()
    repo_root.mkdir(parents=True, exist_ok=True)
    ctx = make_repo_context(repo_root, repo=repo, commit=commit, db_path=opts.db_path)
    coverage_file = repo_root / ".coverage"
    coverage_file.touch()
    tools_cfg = make_tools_config()
    runner = _make_runner(
        repo_root,
        [],
        coverage_file=coverage_file,
        tools_cfg=tools_cfg,
    )
    gateway = _open_gateway_from_context(ctx, opts)
    if opts.apply_schema and (opts.ensure_views or opts.strict_schema):
        apply_all_schemas(gateway.con)
    return ProvisionedGateway(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=ctx.build_dir,
        db_path=ctx.db_path,
        document_output_dir=ctx.document_output_dir,
        coverage_file=coverage_file,
        gateway=gateway,
        runner=runner,
    )


def provision_docs_export_ready(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    db_path: Path | None = None,
    file_backed: bool = True,
) -> ProvisionedGateway:
    """Provision a gateway with minimal data for docs export smoke/validation tests.

    Parameters
    ----------
    repo_root
        Root directory for the repository.
    repo
        Repository identifier.
    commit
        Commit hash.
    db_path
        Optional explicit database path.
    file_backed
        Whether to use file-backed storage.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway populated with docs export seeds.
    """
    ctx = provision_gateway_with_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=GatewayOptions(
            db_path=db_path,
            apply_schema=True,
            ensure_views=True,
            validate_schema=True,
            file_backed=file_backed,
        ),
    )
    seed_docs_export_minimal(ctx.gateway, repo=repo, commit=commit)
    return ctx


def provision_graph_ready_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    options: ProvisionOptions | None = None,
) -> ProvisionedGateway:
    """Provision a repo with graph metrics ready (modules + CFG/DFG metrics seeded).

    Inserts a minimal GOID row for pkg.mod.func so graph tests can bind to it.

    Parameters
    ----------
    repo_root
        Root directory for the repository.
    repo
        Repository identifier.
    commit
        Commit hash.
    options
        Provisioning options.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway with seeded CFG/DFG state.
    """
    opts = options or ProvisionOptions(build_graph_metrics=True)
    ctx = provision_ingested_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=ProvisionOptions(
            include_typing=opts.include_typing,
            include_coverage=opts.include_coverage,
            build_graph_metrics=True,
            file_backed=opts.file_backed,
            db_path=opts.db_path,
            include_seed_goid=opts.include_seed_goid,
        ),
    )
    if opts.include_seed_goid:
        con = ctx.gateway.con
        con.execute(
            """
            INSERT INTO core.goids (
                goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
                start_line, end_line, created_at
            )
            VALUES (
                1, 'urn:pkg.mod:func', ?, ?, 'pkg/mod.py', 'python', 'function',
                'pkg.mod.func', 1, 2, CURRENT_TIMESTAMP
            )
            """,
            [repo, commit],
        )
    return ctx


def graph_metrics_ready_gateway(
    repo_root: Path,
    options: GraphMetricsGatewayOptions | None = None,
) -> ProvisionedGateway:
    """Provision a gateway with callgraph/import data and run graph metrics end-to-end.

    Parameters
    ----------
    repo_root
        Root directory for the repository.
    options
        Graph metrics gateway options.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway with graph metrics populated.
    """
    repo_root.mkdir(parents=True, exist_ok=True)
    opts = options or GraphMetricsGatewayOptions()
    write_graph_metrics_repo(repo_root)
    ctx = provision_existing_repo(
        repo_root,
        repo=opts.repo,
        commit=opts.commit,
        options=ProvisionOptions(
            include_typing=False,
            include_coverage=False,
            build_graph_metrics=False,
            file_backed=opts.file_backed,
            db_path=opts.db_path,
            include_seed_goid=False,
        ),
    )
    gateway = ctx.gateway
    if opts.run_metrics:
        # Clear any prior seeds for these deterministic ids/paths to avoid PK clashes.
        gateway.con.execute("DELETE FROM core.goids WHERE goid_h128 IN (1001, 1002)")
        gateway.con.execute(
            "DELETE FROM core.modules WHERE path IN ('pkg/mod_a.py', 'pkg/mod_b.py')"
        )
        gateway.con.execute("DELETE FROM graph.call_graph_nodes WHERE goid_h128 IN (1001, 1002)")
        insert_rows(
            gateway,
            [
                GoidRow(
                    goid_h128=1001,
                    urn="urn:pkg.mod_a.a",
                    repo=opts.repo,
                    commit=opts.commit,
                    rel_path="pkg/mod_a.py",
                    kind="function",
                    qualname="pkg.mod_a.a",
                    start_line=1,
                    end_line=4,
                    created_at=utcnow(),
                ),
                GoidRow(
                    goid_h128=1002,
                    urn="urn:pkg.mod_b.b",
                    repo=opts.repo,
                    commit=opts.commit,
                    rel_path="pkg/mod_b.py",
                    kind="function",
                    qualname="pkg.mod_b.b",
                    start_line=1,
                    end_line=3,
                    created_at=utcnow(),
                ),
            ],
        )
        insert_rows(
            gateway,
            [
                CallGraphNodeRow(
                    1001,
                    "python",
                    "function",
                    0,
                    is_public=True,
                    rel_path="pkg/mod_a.py",
                ),
                CallGraphNodeRow(
                    1002,
                    "python",
                    "function",
                    0,
                    is_public=True,
                    rel_path="pkg/mod_b.py",
                ),
            ],
        )
        insert_rows(
            gateway,
            [
                CallGraphEdgeRow(
                    opts.repo,
                    opts.commit,
                    1001,
                    1002,
                    "pkg/mod_a.py",
                    3,
                    0,
                    "python",
                    "direct",
                    "local_name",
                    1.0,
                )
            ],
        )
    if opts.build_callgraph_enabled and not opts.run_metrics:
        # Note: Callgraph building via plugin system has been migrated to TargetPlugin.
        # Use BuildExecutor with CallGraphPlugin for full callgraph construction.
        # For now, this code path is a no-op until test infrastructure is updated.
        _ = CallGraphPlugin()  # Suppress unused import warning
    if opts.include_symbol_edges:
        insert_rows(
            gateway,
            [
                SymbolUseEdgeRow(
                    symbol="sym",
                    def_path="pkg/mod_b.py",
                    use_path="pkg/mod_a.py",
                    same_file=False,
                    same_module=False,
                )
            ],
        )
    if opts.run_metrics:
        cfg = (
            opts.graph_cfg
            or ConfigBuilder.from_snapshot(
                snapshot=SnapshotInit(repo=opts.repo, commit=opts.commit, repo_root=repo_root),
            ).graph_metrics()
        )
        compute_graph_metrics(gateway, cfg)
    return ctx


def docs_views_ready_gateway(
    repo_root: Path,
    *,
    repo: str = DEFAULT_REPO,
    commit: str = DEFAULT_COMMIT,
    file_backed: bool = False,
    db_path: Path | None = None,
) -> ProvisionedGateway:
    """Provision a gateway ready for docs views/tests with realistic seeds.

    Parameters
    ----------
    repo_root
        Root directory for the repository.
    repo
        Repository identifier.
    commit
        Commit hash.
    file_backed
        Whether to use file-backed storage.
    db_path
        Optional explicit database path.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway with repo_map/modules/goids, coverage, and risk factors.
    """
    ctx = provision_ingested_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=ProvisionOptions(
            include_typing=True,
            include_coverage=True,
            build_graph_metrics=True,
            file_backed=file_backed,
            db_path=db_path,
            include_seed_goid=True,
        ),
    )
    seed_docs_export_minimal(ctx.gateway, repo=repo, commit=commit)
    return ctx


def build_callgraph_fixture_repo(
    repo_root: Path,
    options: CallgraphFixtureOptions | None = None,
) -> ProvisionedGateway:
    """Create the alias/relative-import callgraph repo and build callgraph via production APIs.

    Parameters
    ----------
    repo_root
        Root directory for the repository.
    options
        Callgraph fixture options.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway after callgraph build.

    Raises
    ------
    RuntimeError
        If the CallGraphPlugin fails execution.
    """
    write_callgraph_alias_repo(repo_root)
    opts = options or CallgraphFixtureOptions()
    ctx = provision_existing_repo(
        repo_root,
        repo=opts.repo,
        commit=opts.commit,
        options=ProvisionOptions(
            include_typing=False,
            include_coverage=False,
            build_graph_metrics=False,
            file_backed=opts.file_backed,
            db_path=opts.db_path,
            include_seed_goid=False,
        ),
    )
    gateway = ctx.gateway
    if opts.goid_entries:
        gateway.con.execute("DELETE FROM core.goids WHERE goid_h128 IN (1001, 1002, 1003, 1004)")
        gateway.con.execute("DELETE FROM core.modules WHERE path IN ('pkg/a.py', 'pkg/b.py')")
        seed_callgraph_goids(gateway, repo=opts.repo, commit=opts.commit, entries=opts.goid_entries)

    # Build callgraph using the plugin system
    snapshot = SnapshotRef(
        repo=opts.repo,
        commit=opts.commit,
        repo_root=repo_root,
    )
    build_dir = repo_root / ".build"
    paths = BuildPaths(
        build_dir=build_dir,
        db_path=build_dir / "db" / "codeintel.duckdb",
        document_output_dir=build_dir / "output",
        scip_dir=build_dir / "scip",
        coverage_json=build_dir / "coverage" / "coverage.json",
        pytest_report=build_dir / "test-results" / "pytest-report.json",
        tool_cache=build_dir / ".tool_cache",
        log_db_path=build_dir / "db" / "codeintel_logs.duckdb",
    )
    resources = ContextResources(gateway=gateway)
    test_ctx = PluginTestContext(
        gateway=gateway,
        snapshot=snapshot,
        paths=paths,
        resources=resources,
    )

    result = execute_target_plugin(CallGraphPlugin(), test_ctx)
    if not result.success:
        msg = f"CallGraphPlugin failed: {result.error_message}"
        raise RuntimeError(msg)

    return ctx


# =============================================================================
# Provisioning Builder
# =============================================================================


class ProvisioningBuilder:
    """Fluent builder for test environment provisioning.

    Provide a composable interface for creating test contexts with various
    provisioning configurations.

    Example
    -------
    >>> ctx = ProvisioningBuilder(repo_root).with_typing().with_coverage().build()
    >>> ctx = ProvisioningBuilder(repo_root).for_docs_export().build()
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        repo: str = DEFAULT_REPO,
        commit: str = DEFAULT_COMMIT,
    ) -> None:
        """Initialize builder with repository paths.

        Parameters
        ----------
        repo_root
            Path to repository root.
        repo
            Repository identifier.
        commit
            Commit hash.
        """
        self._repo_root = repo_root
        self._repo = repo
        self._commit = commit
        self._options = ProvisionOptions()

    def with_typing(self) -> ProvisioningBuilder:
        """Enable typing ingestion.

        Returns
        -------
        ProvisioningBuilder
            Self for chaining.
        """
        self._options = ProvisionOptions(
            include_typing=True,
            include_coverage=self._options.include_coverage,
            build_graph_metrics=self._options.build_graph_metrics,
            file_backed=self._options.file_backed,
            db_path=self._options.db_path,
        )
        return self

    def with_coverage(self) -> ProvisioningBuilder:
        """Enable coverage ingestion.

        Returns
        -------
        ProvisioningBuilder
            Self for chaining.
        """
        self._options = ProvisionOptions(
            include_typing=self._options.include_typing,
            include_coverage=True,
            build_graph_metrics=self._options.build_graph_metrics,
            file_backed=self._options.file_backed,
            db_path=self._options.db_path,
        )
        return self

    def with_graph_metrics(self) -> ProvisioningBuilder:
        """Enable graph metrics computation.

        Returns
        -------
        ProvisioningBuilder
            Self for chaining.
        """
        self._options = ProvisionOptions(
            include_typing=self._options.include_typing,
            include_coverage=self._options.include_coverage,
            build_graph_metrics=True,
            file_backed=self._options.file_backed,
            db_path=self._options.db_path,
        )
        return self

    def file_backed(self, db_path: Path | None = None) -> ProvisioningBuilder:
        """Use file-backed database.

        Parameters
        ----------
        db_path
            Optional path to database file.

        Returns
        -------
        ProvisioningBuilder
            Self for chaining.
        """
        self._options = ProvisionOptions(
            include_typing=self._options.include_typing,
            include_coverage=self._options.include_coverage,
            build_graph_metrics=self._options.build_graph_metrics,
            file_backed=True,
            db_path=db_path,
        )
        return self

    def for_docs_export(self) -> ProvisioningBuilder:
        """Configure for docs export testing.

        Returns
        -------
        ProvisioningBuilder
            Self for chaining.
        """
        return self.with_typing().with_coverage()

    def build(self) -> TestContext:
        """Build the provisioned TestContext.

        Returns
        -------
        TestContext
            Configured test context with provisioned gateway.
        """
        provisioned = provision_ingested_repo(
            self._repo_root,
            repo=self._repo,
            commit=self._commit,
            options=self._options,
        )
        return TestContext.from_provisioned(provisioned)


__all__ = [
    "ProvisioningBuilder",
    "build_callgraph_fixture_repo",
    "docs_views_ready_gateway",
    "graph_metrics_ready_gateway",
    "make_repo_context",
    "provision_docs_export_ready",
    "provision_existing_repo",
    "provision_gateway_with_repo",
    "provision_graph_ready_repo",
    "provision_ingested_repo",
    "provisioned_gateway",
]
