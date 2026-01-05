"""Provisioning helpers for production-parity gateway-backed tests.

This module provides functions for setting up test environments with
real database schemas, ingestion pipelines, and tooling configurations.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import networkx as nx

from codeintel.build.analytics.compute.row_builders import (
    build_symbol_module_edges,
    component_metadata_from_import_rows,
)
from codeintel.build.analytics.graphs.config_graph_metrics import build_config_module_bipartite
from codeintel.build.analytics.graphs.graph_metrics import (
    GraphMetricsInputs,
    SymbolModuleEdges,
    build_call_graph_from_rows,
    build_graph_metric_filters_from_sets,
    build_graph_metrics_rows,
    build_import_graph_from_rows,
)
from codeintel.build.analytics.graphs.graph_metrics_ext import (
    build_graph_metrics_functions_ext_rows,
)
from codeintel.build.analytics.graphs.graph_stats import (
    GraphStatsInputs,
    build_graph_stats_rows,
)
from codeintel.build.analytics.graphs.module_graph_metrics_ext import (
    build_graph_metrics_modules_ext_rows,
)
from codeintel.build.analytics.graphs.symbol_graph_metrics import (
    build_symbol_function_graph,
    build_symbol_module_graph,
)
from codeintel.build.config import BuildConfig
from codeintel.build.graphs.runtime import GraphMetricsOptions, GraphRuntimeOptions
from codeintel.build.providers import create_default_providers
from codeintel.config.primitives import BuildPathOverrides, BuildPaths, SnapshotRef
from codeintel.core.data_models.ids import normalize_decimal_id
from codeintel.ingestion.adapters import (
    DuckDBStorageAdapter,
    FilesystemDiscoveryAdapter,
    HashChangeDetectionAdapter,
)
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.compute.typing_ingest import TypingIngestStep
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.ingestion.engine.service import ToolService
from codeintel.runtime.runtime_bundle import RuntimeBundle
from codeintel.storage.gateway import StorageConfig, open_gateway
from codeintel.storage.schema import apply_all_schemas
from tests._helpers.assertions import assert_target_ok
from tests._helpers.assertions.modules import ModulesAssertions, compute_file_state_hash_from_table
from tests._helpers.configs import (
    DEFAULT_VARIANT,
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
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.docs_views import materialize_view_plans
from tests._helpers.env_options import EnvOptions
from tests._helpers.fakes import utcnow
from tests._helpers.fixtures.repos import (
    write_callgraph_alias_repo,
    write_graph_metrics_repo,
    write_sample_repo,
)
from tests._helpers.fixtures.rows import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    GoidRow,
    SubsystemModuleRow,
    SubsystemRow,
    SymbolUseEdgeRow,
    insert_rows,
)
from tests._helpers.fixtures.snapshots import SnapshotVariant
from tests._helpers.gateway import GatewayFactory, seed_contract_catalog
from tests._helpers.harnesses.hamilton_build import (
    HamiltonBuildHarness,
    HarnessConfig,
)
from tests._helpers.ingestion import materialize_rows_for_snapshot
from tests._helpers.orchestration.seeding import seed_callgraph_goids, seed_cfg_dfg_for_metrics
from tests._helpers.orchestration.seeding_docs import seed_docs_export_minimal
from tests._helpers.orchestration.tooling import make_tools_config
from tests._helpers.schemas import ensure_schema_service

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.config.models import ToolsConfig
    from codeintel.ingestion.ports.discovery import ModuleRecord
    from codeintel.storage.gateway import StorageGateway


log = logging.getLogger(__name__)


def _matches_optional_scope(value: object, expected: str) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and not value.strip():
        return True
    return str(value) == expected


def _fetch_rows(
    gateway: StorageGateway,
    sql: str,
    columns: tuple[str, ...],
    params: Sequence[object],
) -> list[dict[str, object]]:
    rows = gateway.con.execute(sql, list(params)).fetchall()
    return [dict(zip(columns, row, strict=True)) for row in rows]


def _seed_graph_metrics_inputs(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> None:
    gateway.con.execute("DELETE FROM core.goids WHERE goid_h128 IN (1001, 1002)")
    gateway.con.execute("DELETE FROM core.modules WHERE path IN ('pkg/mod_a.py', 'pkg/mod_b.py')")
    gateway.con.execute("DELETE FROM graph.call_graph_nodes WHERE goid_h128 IN (1001, 1002)")
    insert_rows(
        gateway,
        [
            GoidRow(
                goid_h128=1001,
                urn="urn:pkg.mod_a.a",
                repo=repo,
                commit=commit,
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
                repo=repo,
                commit=commit,
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
                repo,
                commit,
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


def _module_inputs_for_graph_metrics(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> tuple[dict[str, str], set[str]]:
    modules_rows = _fetch_rows(
        gateway,
        "SELECT module, path, repo, commit FROM core.modules",
        ("module", "path", "repo", "commit"),
        [],
    )
    module_by_path: dict[str, str] = {}
    module_names: set[str] = set()
    for row in modules_rows:
        module = row.get("module")
        if module is None:
            continue
        if not _matches_optional_scope(row.get("repo"), snapshot.repo):
            continue
        if not _matches_optional_scope(row.get("commit"), snapshot.commit):
            continue
        module_name = str(module)
        module_names.add(module_name)
        path = row.get("path")
        if path is not None:
            module_by_path[str(path)] = module_name
    return module_by_path, module_names


def _function_goids_for_graph_metrics(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> set[int]:
    goid_rows = _fetch_rows(
        gateway,
        "SELECT goid_h128, kind FROM core.goids WHERE repo = ? AND commit = ?",
        ("goid_h128", "kind"),
        [snapshot.repo, snapshot.commit],
    )
    function_goids: set[int] = set()
    for row in goid_rows:
        if row.get("kind") not in {"function", "method"}:
            continue
        goid = normalize_decimal_id(row.get("goid_h128"))
        if goid is not None:
            function_goids.add(goid)
    return function_goids


def _subsystem_ids_for_graph_metrics(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> set[str]:
    subsystem_rows = _fetch_rows(
        gateway,
        "SELECT subsystem_id FROM analytics.subsystem_modules WHERE repo = ? AND commit = ?",
        ("subsystem_id",),
        [snapshot.repo, snapshot.commit],
    )
    subsystem_ids: set[str] = set()
    for row in subsystem_rows:
        subsystem_id = row.get("subsystem_id")
        if subsystem_id is not None:
            subsystem_ids.add(str(subsystem_id))
    return subsystem_ids


def _call_graph_for_graph_metrics(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> nx.DiGraph:
    call_edge_rows = _fetch_rows(
        gateway,
        "SELECT caller_goid_h128, callee_goid_h128 "
        "FROM graph.call_graph_edges WHERE repo = ? AND commit = ?",
        ("caller_goid_h128", "callee_goid_h128"),
        [snapshot.repo, snapshot.commit],
    )
    call_node_rows = _fetch_rows(
        gateway,
        "SELECT goid_h128, kind FROM graph.call_graph_nodes",
        ("goid_h128", "kind"),
        [],
    )
    return build_call_graph_from_rows(call_edge_rows, call_node_rows)


def _import_graph_for_graph_metrics(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> tuple[nx.DiGraph, Mapping[str, Mapping[str, int | bool]] | None]:
    import_edge_rows = _fetch_rows(
        gateway,
        "SELECT src_module, dst_module, module_layer "
        "FROM graph.import_graph_edges WHERE repo = ? AND commit = ?",
        ("src_module", "dst_module", "module_layer"),
        [snapshot.repo, snapshot.commit],
    )
    import_module_rows = _fetch_rows(
        gateway,
        "SELECT module, scc_id, component_size, layer "
        "FROM graph.import_modules WHERE repo = ? AND commit = ?",
        ("module", "scc_id", "component_size", "layer"),
        [snapshot.repo, snapshot.commit],
    )
    import_graph = build_import_graph_from_rows(import_edge_rows, import_module_rows)
    component_meta = component_metadata_from_import_rows(import_module_rows)
    return import_graph, component_meta


def _symbol_graph_inputs_for_graph_metrics(
    gateway: StorageGateway,
    module_by_path: Mapping[str, str],
) -> tuple[SymbolModuleEdges, nx.Graph, nx.Graph]:
    symbol_rows = _fetch_rows(
        gateway,
        "SELECT def_path, use_path, def_goid_h128, use_goid_h128 FROM graph.symbol_use_edges",
        ("def_path", "use_path", "def_goid_h128", "use_goid_h128"),
        [],
    )
    return (
        build_symbol_module_edges(symbol_rows, module_by_path),
        build_symbol_module_graph(symbol_rows, module_by_path),
        build_symbol_function_graph(symbol_rows),
    )


def _config_bipartite_for_graph_metrics(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    module_names: set[str],
) -> nx.Graph:
    config_rows = _fetch_rows(
        gateway,
        "SELECT repo, commit, key, reference_modules "
        "FROM analytics.config_values WHERE repo = ? AND commit = ?",
        ("repo", "commit", "key", "reference_modules"),
        [snapshot.repo, snapshot.commit],
    )
    return build_config_module_bipartite(
        config_rows,
        allowed_modules=module_names,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )


@dataclass(frozen=True)
class _GraphMetricArtifacts:
    graph_inputs: GraphMetricsInputs
    symbol_module_graph: nx.Graph
    symbol_function_graph: nx.Graph


def _graph_metrics_artifacts_for_gateway(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    metric_options: GraphMetricsOptions,
) -> _GraphMetricArtifacts:
    module_by_path, module_names = _module_inputs_for_graph_metrics(gateway, snapshot)
    function_goids = _function_goids_for_graph_metrics(gateway, snapshot)
    subsystem_ids = _subsystem_ids_for_graph_metrics(gateway, snapshot)
    filters = build_graph_metric_filters_from_sets(
        function_goids=function_goids,
        modules=module_names,
        subsystems=subsystem_ids,
    )
    call_graph = _call_graph_for_graph_metrics(gateway, snapshot)
    import_graph, component_meta = _import_graph_for_graph_metrics(gateway, snapshot)
    (
        symbol_module_edges,
        symbol_module_graph,
        symbol_function_graph,
    ) = _symbol_graph_inputs_for_graph_metrics(gateway, module_by_path)
    graph_inputs = GraphMetricsInputs(
        snapshot=snapshot,
        call_graph=call_graph,
        import_graph=import_graph,
        symbol_module_edges=symbol_module_edges,
        module_names=module_names,
        component_meta=component_meta,
        filters=filters,
        options=metric_options,
    )
    return _GraphMetricArtifacts(
        graph_inputs=graph_inputs,
        symbol_module_graph=symbol_module_graph,
        symbol_function_graph=symbol_function_graph,
    )


def _write_graph_metrics_base(
    gateway: StorageGateway,
    graph_inputs: GraphMetricsInputs,
) -> None:
    metrics_rows = build_graph_metrics_rows(graph_inputs)
    backend = gateway.policy
    if metrics_rows.function_rows:
        backend.delete_for_snapshot(
            "analytics.graph_metrics_functions",
            repo=graph_inputs.snapshot.repo,
            commit=graph_inputs.snapshot.commit,
        )
        backend.bulk_insert_mappings(
            "analytics.graph_metrics_functions",
            metrics_rows.function_rows,
        )
    if metrics_rows.module_rows:
        backend.delete_for_snapshot(
            "analytics.graph_metrics_modules",
            repo=graph_inputs.snapshot.repo,
            commit=graph_inputs.snapshot.commit,
        )
        backend.bulk_insert_mappings(
            "analytics.graph_metrics_modules",
            metrics_rows.module_rows,
        )


def _write_graph_metrics_ext(
    gateway: StorageGateway,
    graph_inputs: GraphMetricsInputs,
    runtime_options: GraphRuntimeOptions,
) -> None:
    backend = gateway.policy
    functions_ext_rows = build_graph_metrics_functions_ext_rows(
        repo=graph_inputs.snapshot.repo,
        commit=graph_inputs.snapshot.commit,
        call_graph=graph_inputs.call_graph,
        runtime=runtime_options,
        filters=graph_inputs.filters,
    )
    if functions_ext_rows:
        backend.delete_for_snapshot(
            "analytics.graph_metrics_functions_ext",
            repo=graph_inputs.snapshot.repo,
            commit=graph_inputs.snapshot.commit,
        )
        backend.bulk_insert_mappings(
            "analytics.graph_metrics_functions_ext",
            functions_ext_rows,
        )

    modules_ext_rows = build_graph_metrics_modules_ext_rows(
        repo=graph_inputs.snapshot.repo,
        commit=graph_inputs.snapshot.commit,
        import_graph=graph_inputs.import_graph,
        runtime=runtime_options,
        filters=graph_inputs.filters,
    )
    if modules_ext_rows:
        backend.delete_for_snapshot(
            "analytics.graph_metrics_modules_ext",
            repo=graph_inputs.snapshot.repo,
            commit=graph_inputs.snapshot.commit,
        )
        backend.bulk_insert_mappings(
            "analytics.graph_metrics_modules_ext",
            modules_ext_rows,
        )


def _write_graph_stats(
    gateway: StorageGateway,
    artifacts: _GraphMetricArtifacts,
    runtime_options: GraphRuntimeOptions,
) -> None:
    module_names = {str(name) for name in artifacts.graph_inputs.module_names}
    config_bipartite = _config_bipartite_for_graph_metrics(
        gateway,
        artifacts.graph_inputs.snapshot,
        module_names,
    )
    graph_stats_rows = build_graph_stats_rows(
        GraphStatsInputs(
            repo=artifacts.graph_inputs.snapshot.repo,
            commit=artifacts.graph_inputs.snapshot.commit,
            call_graph=artifacts.graph_inputs.call_graph,
            import_graph=artifacts.graph_inputs.import_graph,
            symbol_module_graph=artifacts.symbol_module_graph,
            symbol_function_graph=artifacts.symbol_function_graph,
            config_module_bipartite=config_bipartite,
            use_gpu=runtime_options.use_gpu,
        )
    )
    if graph_stats_rows:
        gateway.policy.delete_for_snapshot(
            "analytics.graph_stats",
            repo=artifacts.graph_inputs.snapshot.repo,
            commit=artifacts.graph_inputs.snapshot.commit,
        )
        gateway.policy.bulk_insert("analytics.graph_stats", graph_stats_rows)


def _run_graph_metrics_for_gateway(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    metric_options: GraphMetricsOptions,
    runtime_options: GraphRuntimeOptions,
) -> None:
    artifacts = _graph_metrics_artifacts_for_gateway(
        gateway,
        snapshot,
        metric_options=metric_options,
    )
    _write_graph_metrics_base(gateway, artifacts.graph_inputs)
    _write_graph_metrics_ext(gateway, artifacts.graph_inputs, runtime_options)
    _write_graph_stats(gateway, artifacts, runtime_options)


def make_repo_context(
    repo_root: Path,
    *,
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
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
    document_output_dir = repo_root / "build" / "document_output"
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


def _make_runner(
    repo_root: Path,
    *,
    tools_cfg: ToolsConfig,
) -> ToolRunner:
    """Build a ToolRunner seeded with real tool binaries.

    Returns
    -------
    ToolRunner
        Runner configured with real tooling.
    """
    cache_dir = repo_root / "build" / ".tool_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return ToolRunner(tools_config=tools_cfg, cache_dir=cache_dir)


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
        gateway = open_gateway(cfg, seed_contract_catalog=seed_contract_catalog)
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
    opts: ProvisionOptions,
    repo: str,
    commit: str,
) -> ProvisioningSetup:
    """Build all components needed for repo provisioning.

    Parameters
    ----------
    repo_root
        Root path of the repository.
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
            pytest_report=ctx.build_dir / "test-results" / "pytest-report.json",
            scip_dir=ctx.build_dir / "scip",
            tool_cache=ctx.build_dir / ".tool_cache",
            log_db_path=ctx.build_dir / "db" / "codeintel_logs.duckdb",
        ),
    )

    tools_cfg = make_tools_config()
    runner = _make_runner(repo_root, tools_cfg=tools_cfg)
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
                "analytics.static_diagnostics",
                typing_result.diagnostic_rows,
                snapshot=snapshot,
            )
        else:
            log.warning(
                "Typing ingest failed during provisioning: %s",
                typing_result.result.error or "unknown",
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
    variant = cfg.snapshot_variant
    if cfg.run_ingestion:
        ctx = provision_hamilton_repo(
            repo_root,
            repo=variant.repo,
            commit=variant.commit,
            options=cfg.provision_options,
        )
    else:
        ctx = provision_gateway_with_repo(
            repo_root,
            repo=variant.repo,
            commit=variant.commit,
            options=cfg.gateway_options,
        )
    try:
        yield ctx
    finally:
        ctx.close()


def _collect_repo_files(repo_root: Path) -> list[Path]:
    return sorted([path for path in repo_root.rglob("*.py") if path.is_file()])


def provision_hamilton_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
    options: ProvisionOptions | None = None,
    repo_writer: Callable[[Path], list[Path]] | None = None,
) -> ProvisionedGateway:
    """Provision a repo by running Hamilton ingestion targets.

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
    repo_writer
        Optional callback to populate repo_root.

    Returns
    -------
    ProvisionedGateway
        Provisioned gateway with filesystem context.
    """
    opts = options or ProvisionOptions()
    repo_root.mkdir(parents=True, exist_ok=True)
    if repo_writer is not None:
        repo_writer(repo_root)

    db_path = opts.db_path or (repo_root / "build" / "db" / "codeintel.duckdb")
    tools_cfg = make_tools_config()
    runner = _make_runner(repo_root, tools_cfg=tools_cfg)

    env_opts = EnvOptions(
        file_backed=opts.file_backed,
        repo_root=repo_root,
        build_dir=repo_root / "build",
        db_path=db_path,
        snapshot_variant=SnapshotVariant(repo=repo, commit=commit, run_id=DEFAULT_VARIANT.run_id),
    )
    gateway_opts = GatewayOptions(
        file_backed=opts.file_backed,
        db_path=db_path,
        apply_schema=True,
        ensure_views=True,
        validate_schema=True,
        strict_schema=True,
        repo=repo,
        commit=commit,
    )
    ctx = create_test_context(repo_root.parent, options=env_opts, gateway_options=gateway_opts)

    harness = HamiltonBuildHarness.wrap(
        ctx,
        harness=HarnessConfig(repo=repo, commit=commit, file_backed_db=opts.file_backed),
        tools_config=tools_cfg,
    )
    targets = ["modules"]
    if opts.include_typing:
        targets.append("typing")

    result = harness.run_targets(targets)
    for target in targets:
        assert_target_ok(harness.record(target, result=result))
    ModulesAssertions(ctx.gateway, ctx.snapshot).inventory_consistent()

    if opts.build_graph_metrics:
        seed_cfg_dfg_for_metrics(ctx.gateway, rel_path="pkg/mod.py")

    return ProvisionedGateway(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=ctx.build_paths.build_dir,
        db_path=ctx.build_paths.db_path,
        document_output_dir=ctx.build_paths.document_output_dir,
        gateway=ctx.gateway,
        runner=runner,
    )


def provision_ingested_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
    options: ProvisionOptions | None = None,
) -> ProvisionedGateway:
    """Build a sample repo, run Hamilton ingestion, and return a provisioned gateway.

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
    return provision_hamilton_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=options,
        repo_writer=write_sample_repo,
    )


def provision_existing_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
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
    return provision_hamilton_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=options,
        repo_writer=None,
    )


def provision_gateway_with_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
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
    tools_cfg = make_tools_config()
    runner = _make_runner(repo_root, tools_cfg=tools_cfg)
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
        gateway=gateway,
        runner=runner,
    )


def provision_docs_export_ready(
    repo_root: Path,
    *,
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
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
    ensure_schema_service()
    seed_docs_export_minimal(ctx.gateway, repo=repo, commit=commit, repo_root=repo_root)
    return ctx


def provision_graph_ready_repo(
    repo_root: Path,
    *,
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
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
            build_graph_metrics=True,
            file_backed=opts.file_backed,
            db_path=opts.db_path,
            include_seed_goid=opts.include_seed_goid,
        ),
    )
    ensure_schema_service()
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
            build_graph_metrics=False,
            file_backed=opts.file_backed,
            db_path=opts.db_path,
            include_seed_goid=False,
        ),
    )
    gateway = ctx.gateway
    if opts.run_metrics:
        _seed_graph_metrics_inputs(gateway, repo=opts.repo, commit=opts.commit)
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
        runtime_options = GraphRuntimeOptions(snapshot=snapshot)
        _run_graph_metrics_for_gateway(
            gateway,
            snapshot,
            metric_options=metric_options,
            runtime_options=runtime_options,
        )
    return ctx


def docs_views_ready_gateway(
    repo_root: Path,
    *,
    repo: str = DEFAULT_VARIANT.repo,
    commit: str = DEFAULT_VARIANT.commit,
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
        Provisioned gateway with repo_map/modules/goids and risk factors.
    """
    ctx = provision_ingested_repo(
        repo_root,
        repo=repo,
        commit=commit,
        options=ProvisionOptions(
            include_typing=True,
            build_graph_metrics=True,
            file_backed=file_backed,
            db_path=db_path,
            include_seed_goid=True,
        ),
    )
    ensure_schema_service()
    seed_docs_export_minimal(ctx.gateway, repo=repo, commit=commit, repo_root=repo_root)
    _seed_minimal_subsystems(ctx.gateway, repo=repo, commit=commit)
    materialize_view_plans(ctx.gateway.con)
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
                modules_json=ordered_modules,
                entrypoints_json=[entrypoint_module],
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
    *,
    runtime: RuntimeBundle,
) -> ProvisionedGateway:
    """Create the alias/relative-import callgraph repo and build callgraph via production APIs.

    Parameters
    ----------
    repo_root
        Root directory for the repository.
    options
        Callgraph fixture options.
    runtime
        Runtime bundle providing build configuration.

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
        document_output_dir=build_dir / "document_output",
        dataset_root_dir=build_dir / "datasets",
        scip_dir=build_dir / "scip",
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
    harness.with_runtime(runtime)
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
    >>> ctx = ProvisioningBuilder(repo_root).with_typing().build()
    >>> ctx = ProvisioningBuilder(repo_root).for_docs_export().build()
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        repo: str = DEFAULT_VARIANT.repo,
        commit: str = DEFAULT_VARIANT.commit,
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
        return self.with_typing()

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
    "provision_hamilton_repo",
    "provision_ingested_repo",
    "provisioned_gateway",
]
