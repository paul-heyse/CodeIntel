"""Provisioning helpers for production-parity gateway-backed tests.

This module provides functions for setting up test environments with
real database schemas, ingestion pipelines, and tooling configurations.
"""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import contextmanager
from typing import TYPE_CHECKING

from coverage import Coverage

from codeintel.analytics.graphs.graph_metrics import build_graph_metrics_rows
from codeintel.analytics.graphs.graph_metrics_ext import build_graph_metrics_functions_ext_rows
from codeintel.analytics.graphs.graph_stats import build_graph_stats_rows
from codeintel.analytics.graphs.module_graph_metrics_ext import build_graph_metrics_modules_ext_rows
from codeintel.build.config import BuildConfig
from codeintel.build.providers import create_default_providers
from codeintel.config.primitives import BuildPathOverrides, BuildPaths, SnapshotRef
from codeintel.graphs.runtime import GraphMetricsOptions
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.compute.coverage_ingest import CoverageIngestStep
from codeintel.ingestion.compute.repo_scan import RepoScanStep
from codeintel.ingestion.compute.typing_ingest import TypingIngestStep
from codeintel.ingestion.engine.infrastructure import ToolName, ToolRunner, ToolRunOptions
from codeintel.ingestion.engine.infrastructure.runner import ToolNotFoundError
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.infrastructure.scanning import default_code_profile
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.assertions import assert_target_ok
from tests._helpers.assertions.modules import ModulesAssertions, compute_file_state_hash_from_table
from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    GoidRow,
    SubsystemModuleRow,
    SubsystemRow,
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
    provisioning_gateway_options,
)
from tests._helpers.context import TestContext
from tests._helpers.fakes import utcnow
from tests._helpers.gateway import GatewayFactory
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness
from tests._helpers.ingestion import materialize_repo_scan_result, materialize_rows_for_snapshot
from tests._helpers.modules_expectations import module_paths_expected_from_repo_tree
from tests._helpers.orchestration.repo_writers import (
    write_callgraph_alias_repo,
    write_coverage_driver,
    write_graph_metrics_repo,
    write_sample_repo,
)
from tests._helpers.orchestration.seeding import seed_callgraph_goids, seed_cfg_dfg_for_metrics
from tests._helpers.orchestration.seeding_docs import seed_docs_export_minimal
from tests._helpers.orchestration.tooling import make_tools_config

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.storage.gateway import StorageGateway


log = logging.getLogger(__name__)


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
    """Execute coverage run; on failure, write an empty coverage database."""
    driver_path = write_coverage_driver(repo_root, files)
    log = logging.getLogger(__name__)
    try:
        result = runner.run(
            ToolName.COVERAGE,
            ["run", "--data-file", str(coverage_file), str(driver_path)],
            options=ToolRunOptions(cwd=repo_root),
        )
    except ToolNotFoundError as exc:
        log.warning(
            "coverage binary missing (%s); writing empty coverage data",
            exc,
        )
        result = None

    if result is not None and result.ok:
        return

    if result is not None:
        log.warning(
            "coverage run failed: code=%s stderr=%s; writing empty coverage data",
            result.returncode,
            result.stderr,
        )

    coverage_file.parent.mkdir(parents=True, exist_ok=True)
    cov = Coverage(data_file=str(coverage_file))
    cov.start()
    cov.stop()
    cov.save()


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
    else:
        factory = GatewayFactory()
        if not opts.apply_schema:
            factory = factory.without_schema()
        if effective_ensure_views:
            factory = factory.with_views()
        if not effective_validate_schema:
            factory = factory.without_validation()
        if not opts.strict_schema:
            factory = factory.relaxed()
        factory = factory.with_snapshot(ctx.repo, ctx.commit)
        gateway = factory.open()
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
            discovery=setup.discovery,
            tools=setup.tool_adapter,
        )
        typing_result = asyncio.run(
            typing_step.execute_async(
                list(modules),
                repo=repo,
                commit=commit,
                repo_root=str(setup.ctx.repo_root),
            )
        )
        if typing_result.result.success:
            snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=setup.ctx.repo_root)
            materialize_rows_for_snapshot(
                setup.gateway,
                "analytics.typedness",
                typing_result.typedness_rows,
                snapshot=snapshot,
            )
            materialize_rows_for_snapshot(
                setup.gateway,
                "analytics.static_diagnostics",
                typing_result.diagnostic_rows,
                snapshot=snapshot,
            )
        else:
            log.warning(
                "Typing ingest failed during provisioning: %s",
                typing_result.result.error or "unknown",
            )
    if opts.include_coverage:
        coverage_step = CoverageIngestStep(tools=setup.tool_adapter)
        coverage_result = asyncio.run(
            coverage_step.execute_async(
                [],
                repo=repo,
                commit=commit,
                repo_root=setup.ctx.repo_root,
                coverage_file=setup.coverage_file,
            )
        )
        if coverage_result.result.success:
            snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=setup.ctx.repo_root)
            materialize_rows_for_snapshot(
                setup.gateway,
                "analytics.coverage_lines",
                coverage_result.rows,
                snapshot=snapshot,
            )
        else:
            log.warning(
                "Coverage ingest failed during provisioning: %s",
                coverage_result.result.error or "unknown",
            )
    if opts.build_graph_metrics:
        seed_cfg_dfg_for_metrics(setup.gateway, rel_path="pkg/mod.py")
        # CFG/DFG metrics computation now happens via Hamilton native modules
        # See codeintel.build.hamilton.native.analytics.cfg_dfg for full context


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
        discovery=setup.discovery,
        change_detection=setup.change_detection,
    )
    scan_result = scan_step.execute(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        profile=code_profile,
    )
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=repo_root)
    materialize_repo_scan_result(
        setup.gateway,
        scan_result,
        snapshot=snapshot,
    )
    ModulesAssertions(setup.gateway, snapshot).inventory_consistent()

    _run_ingestion_steps(setup, list(scan_result.modules), opts, repo, commit)

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

    module_paths = module_paths_expected_from_repo_tree(repo_root)
    files = [repo_root / path for path in module_paths]
    setup = _build_provisioning_setup(repo_root, files, opts, repo, commit)
    code_profile = default_code_profile(repo_root)

    scan_step = RepoScanStep(
        discovery=setup.discovery,
        change_detection=setup.change_detection,
    )
    scan_result = scan_step.execute(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        profile=code_profile,
    )

    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=repo_root)
    materialize_repo_scan_result(
        setup.gateway,
        scan_result,
        snapshot=snapshot,
    )
    ModulesAssertions(setup.gateway, snapshot).inventory_consistent()

    _run_ingestion_steps(setup, list(scan_result.modules), opts, repo, commit)

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
    opts = options or provisioning_gateway_options()
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
    seed_docs_export_minimal(ctx.gateway, repo=repo, commit=commit, repo_root=repo_root)
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
    # Note: call_graph target should be executed via Hamilton if needed
    # The build_callgraph_enabled flag is legacy and can be ignored
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
        snapshot = SnapshotRef(repo=opts.repo, commit=opts.commit, repo_root=repo_root)
        metric_options = opts.metrics_options or GraphMetricsOptions()
        metrics_rows = build_graph_metrics_rows(gateway, snapshot, options=metric_options)
        backend = gateway.policy
        if metrics_rows.function_rows:
            backend.delete_for_snapshot(
                "analytics.graph_metrics_functions",
                repo=snapshot.repo,
                commit=snapshot.commit,
            )
            backend.bulk_insert_mappings(
                "analytics.graph_metrics_functions",
                metrics_rows.function_rows,
            )
        if metrics_rows.module_rows:
            backend.delete_for_snapshot(
                "analytics.graph_metrics_modules",
                repo=snapshot.repo,
                commit=snapshot.commit,
            )
            backend.bulk_insert_mappings(
                "analytics.graph_metrics_modules",
                metrics_rows.module_rows,
            )

        functions_ext_rows = build_graph_metrics_functions_ext_rows(
            gateway,
            repo=snapshot.repo,
            commit=snapshot.commit,
        )
        if functions_ext_rows:
            backend.delete_for_snapshot(
                "analytics.graph_metrics_functions_ext",
                repo=snapshot.repo,
                commit=snapshot.commit,
            )
            backend.bulk_insert_mappings(
                "analytics.graph_metrics_functions_ext",
                functions_ext_rows,
            )

        modules_ext_rows = build_graph_metrics_modules_ext_rows(
            gateway,
            repo=snapshot.repo,
            commit=snapshot.commit,
        )
        if modules_ext_rows:
            backend.delete_for_snapshot(
                "analytics.graph_metrics_modules_ext",
                repo=snapshot.repo,
                commit=snapshot.commit,
            )
            backend.bulk_insert_mappings(
                "analytics.graph_metrics_modules_ext",
                modules_ext_rows,
            )

        graph_stats_rows = build_graph_stats_rows(
            gateway,
            repo=snapshot.repo,
            commit=snapshot.commit,
        )
        if graph_stats_rows:
            backend.delete_for_snapshot(
                "analytics.graph_stats",
                repo=snapshot.repo,
                commit=snapshot.commit,
            )
            backend.bulk_insert("analytics.graph_stats", graph_stats_rows)
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
    seed_docs_export_minimal(ctx.gateway, repo=repo, commit=commit, repo_root=repo_root)
    _seed_minimal_subsystems(ctx.gateway, repo=repo, commit=commit)
    ctx.gateway.policy.ensure_all_views(overwrite=True, strict=True)
    return ctx


def _seed_minimal_subsystems(gateway: StorageGateway, *, repo: str, commit: str) -> None:
    """Seed subsystem rows for docs/CLI tests when missing."""
    con = gateway.con
    exists = con.execute(
        "SELECT 1 FROM analytics.subsystems WHERE repo = ? AND commit = ? LIMIT 1",
        [repo, commit],
    ).fetchone()
    if exists is not None:
        return

    module_rows = con.execute(
        "SELECT module FROM core.modules WHERE repo = ? AND commit = ? ORDER BY module",
        [repo, commit],
    ).fetchall()
    modules = [str(row[0]) for row in module_rows if row and row[0] is not None]
    if not modules:
        return

    entrypoint_module = "pkg.mod" if "pkg.mod" in modules else modules[0]
    ordered_modules = [entrypoint_module, *[m for m in modules if m != entrypoint_module]]
    subsystem_id = "subsysdemo"

    insert_rows(
        gateway,
        [
            SubsystemModuleRow(
                repo=repo,
                commit=commit,
                subsystem_id=subsystem_id,
                module=mod,
                role="entrypoint" if mod == entrypoint_module else "internal",
            )
            for mod in ordered_modules
        ],
    )
    insert_rows(
        gateway,
        [
            SubsystemRow(
                repo=repo,
                commit=commit,
                subsystem_id=subsystem_id,
                name="Subsystem Demo",
                description="Seeded subsystem for docs/CLI tests",
                module_count=len(ordered_modules),
                modules_json=json.dumps(ordered_modules),
                entrypoints_json=json.dumps([entrypoint_module]),
                internal_edge_count=0,
                external_edge_count=0,
                fan_in=0,
                fan_out=0,
                function_count=0,
                avg_risk_score=None,
                max_risk_score=None,
                high_risk_function_count=0,
                risk_level="low",
                created_at=utcnow(),
            )
        ],
    )


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
        If call graph extraction fails execution.
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

    snapshot = SnapshotRef(
        repo=opts.repo,
        commit=opts.commit,
        repo_root=repo_root,
    )
    build_dir = repo_root / ".build"
    build_dir.mkdir(parents=True, exist_ok=True)
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
    providers = create_default_providers(make_tools_config())
    ctx_runtime = TestContext(
        snapshot=snapshot,
        gateway=gateway,
        build_paths=paths,
    )
    harness = HamiltonBuildHarness.wrap(
        ctx_runtime,
        providers=providers,
        build_config=BuildConfig.empty(),
    )
    harness.with_force_targets("modules")
    artifacts = harness.artifacts
    artifacts.write_pytest_report()
    file_state_hash = compute_file_state_hash_from_table(gateway, snapshot)
    harness.priming.prime_modules_manifest(
        file_state_hash=file_state_hash,
        change_delta={"state_hash": file_state_hash},
    )
    result = harness.run_targets(["call_graph"])
    record = harness.record("call_graph", result=result)
    try:
        assert_target_ok(record)
    except AssertionError as exc:
        message = f"call_graph extraction failed: {record.error}"
        raise RuntimeError(message) from exc

    return ctx


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
