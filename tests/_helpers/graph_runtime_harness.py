"""Shared graph runtime harness and pipeline helpers for analytics tests."""

from __future__ import annotations

import ast
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.build.analytics.compute.row_builders import (
    build_symbol_module_edges,
    component_metadata_from_import_rows,
)
from codeintel.build.analytics.graphs.config_data_flow import (
    ConfigDataFlowInputs,
    compute_config_data_flow_result,
)
from codeintel.build.analytics.graphs.config_graph_metrics import (
    build_config_module_bipartite,
    compute_config_graph_metrics_result,
)
from codeintel.build.analytics.graphs.graph_metrics import (
    GraphMetricsInputs,
    build_graph_metric_filters_from_sets,
    build_graph_metrics_rows,
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
from codeintel.build.analytics.graphs.subsystem_agreement import (
    SubsystemAgreementInputs,
    build_subsystem_agreement_rows,
)
from codeintel.build.analytics.graphs.subsystem_graph_metrics import (
    SubsystemGraphMetricInputs,
    build_subsystem_graph_metrics_rows,
)
from codeintel.build.analytics.graphs.symbol_graph_metrics import (
    build_symbol_graph_metrics_function_rows,
    build_symbol_graph_metrics_module_rows,
)
from codeintel.build.analytics.parsing.ast_cache import FunctionAst
from codeintel.build.graphs.runtime import GraphRuntime, GraphRuntimeOptions
from codeintel.config.primitives import SnapshotRef
from codeintel.core.catalog import FunctionCatalog
from codeintel.storage.query_results import records_from_relation
from tests._helpers.catalogs import seed_goids_for_snapshot
from tests._helpers.fakes.graph_runtime import (
    CountingGraphEngineAdapter,
    build_graph_engine_double,
)
from tests._helpers.fakes.graph_runtime import (
    GraphRuntimeDouble as GraphStubEngine,
)
from tests._helpers.fixtures.graphs import build_sample_graphs
from tests._helpers.fixtures.repos import (
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_FQN,
    MOD_A_PATH,
    MOD_B_FQN,
    MOD_B_PATH,
    MOD_C_FQN,
    MOD_C_PATH,
    MOD_UTIL_FQN,
    MOD_UTIL_PATH,
    write_canonical_repo,
)
from tests._helpers.fixtures.rows import (
    ConfigValueRow,
    ModuleRow,
    SubsystemModuleRow,
    SymbolEdgeOptions,
    function_meta,
    insert_rows,
    insert_symbol_use_edges,
    make_symbol_use_edge_row,
)
from tests._helpers.gateway import GatewayFactory
from tests._helpers.seeds import AST_METRICS_PACK, CORE_PACK

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from codeintel.build.analytics.graphs.graph_metrics import GraphMetricFilters
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.context import TestContext
    from tests._helpers.fixtures.graphs import GraphFixtures


def build_source_files(repo_root: Path) -> dict[str, Path]:
    """Write sample modules and return their paths keyed by module name.

    Parameters
    ----------
    repo_root
        Repository root used for writing the sample package.

    Returns
    -------
    dict[str, Path]
        Mapping of module name to written file path.
    """
    canonical = write_canonical_repo(repo_root)
    return {module: repo_root / rel_path for module, rel_path in canonical.module_paths.items()}


def build_canonical_ast_lookup(repo_root: Path) -> dict[int, FunctionAst]:
    """Build FunctionAst lookup for the canonical sample repo.

    Returns
    -------
    dict[int, FunctionAst]
        Mapping of GOID to parsed AST metadata.
    """
    paths = {
        MOD_A_FQN: repo_root / MOD_A_PATH,
        MOD_B_FQN: repo_root / MOD_B_PATH,
        MOD_C_FQN: repo_root / MOD_C_PATH,
        MOD_UTIL_FQN: repo_root / MOD_UTIL_PATH,
    }
    goids = {
        "func_a": GOID_FUNC_A,
        "func_b": GOID_FUNC_B,
        "func_c": GOID_FUNC_C,
        "helper": GOID_HELPER,
    }
    target_names = {
        MOD_A_FQN: "func_a",
        MOD_B_FQN: "func_b",
        MOD_C_FQN: "func_c",
        MOD_UTIL_FQN: "helper",
    }
    return build_ast_map(paths, goids, repo_root, target_names=target_names)


def canonical_ast_map(ctx: TestContext) -> dict[int, FunctionAst]:
    """Return canonical AST map for a context already seeded with AST metrics.

    Returns
    -------
    dict[int, FunctionAst]
        Mapping from GOID to parsed FunctionAst.
    """
    return build_canonical_ast_lookup(ctx.repo_root)


@dataclass(frozen=True)
class CanonicalAstArtifacts:
    """Bundle canonical FunctionCatalog plus AST map for analytics tests."""

    catalog: FunctionCatalog
    ast_map: dict[int, FunctionAst]


def canonical_ast_artifacts(ctx: TestContext) -> CanonicalAstArtifacts:
    """Ensure core/AST packs are applied and return catalog + AST map.

    Returns
    -------
    CanonicalAstArtifacts
        Bundled catalog and AST map for canonical fixtures.
    """
    canonical = ctx.ensure_canonical_repo()
    ctx.require(CORE_PACK, AST_METRICS_PACK)
    functions = [
        function_meta(
            goid=meta.goid,
            rel_path=meta.rel_path,
            qualname=meta.qualname,
            snapshot=(ctx.repo, ctx.commit),
            line_span=(meta.start_line, meta.end_line),
        )
        for meta in canonical.functions.values()
    ]
    module_by_path = {path: module for module, path in canonical.module_paths.items()}
    catalog = FunctionCatalog(functions=functions, module_by_path=module_by_path)
    return CanonicalAstArtifacts(catalog=catalog, ast_map=canonical_ast_map(ctx))


def _function_node(
    tree: ast.AST,
    target: str,
) -> ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef:
    """Find a function/class node in an AST by fully qualified name suffix.

    Returns
    -------
    ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef
        Matching function or class node.

    Raises
    ------
    ValueError
        If no matching node is found.
    """
    target_name = target.rsplit(".", maxsplit=1)[-1]
    for node in ast.walk(tree):
        if (
            isinstance(
                node,
                (
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                    ast.ClassDef,
                ),
            )
            and node.name == target_name
        ):
            return node
    message = f"Function {target} not found"
    raise ValueError(message)


def build_ast_map(
    paths: Mapping[str, Path],
    goids: Mapping[str, int],
    repo_root: Path,
    *,
    target_names: Mapping[str, str | Sequence[str]] | None = None,
) -> dict[int, FunctionAst]:
    """
    Build FunctionAst mapping for known targets.

    Parameters
    ----------
    paths
        Mapping of module names to file paths.
    goids
        GOID mapping keyed by target name.
    repo_root
        Repository root for computing relative paths.
    target_names
        Optional override for target names per module.

    Returns
    -------
    dict[int, FunctionAst]
        Mapping of GOID to parsed function/class AST metadata.

    Raises
    ------
    ValueError
        If a requested target cannot be found in the provided source or GOID mapping.
    """
    ast_by_goid: dict[int, FunctionAst] = {}
    target_lookup = target_names or {
        "pkg.api": "api_handler",
        "pkg.service": "process",
        "pkg.utils": "calc",
        "pkg.mod_a": "func_a",
        "pkg.mod_b": "func_b",
        "pkg.mod_c": "func_c",
        "pkg.util": ("helper", "util_func", "func_b"),
    }
    for module, path in paths.items():
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        targets_raw = target_lookup[module]
        targets = (targets_raw,) if isinstance(targets_raw, str) else tuple(targets_raw)
        for target in targets:
            goid = goids.get(target)
            if goid is None:
                message = f"Function {target} not found"
                raise ValueError(message)
            func_node = _function_node(tree, target)
            start_line = getattr(func_node, "lineno", 0)
            end_line = getattr(func_node, "end_lineno", start_line)
            ast_by_goid[goid] = FunctionAst(
                goid=goid,
                rel_path=path.relative_to(repo_root).as_posix(),
                qualname=target,
                start_line=start_line,
                end_line=end_line,
                node=func_node,
                lines=list(source.splitlines()),
            )
    return ast_by_goid


def insert_modules(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    paths: Mapping[str, Path],
) -> None:
    """Insert module rows for provided paths."""
    rows = [
        ModuleRow(
            module=module,
            path=path.relative_to(snapshot.repo_root).as_posix(),
            repo=snapshot.repo,
            commit=snapshot.commit,
        )
        for module, path in paths.items()
    ]
    insert_rows(gateway, rows)


def insert_goids(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    ast_by_goid: Mapping[int, FunctionAst],
    *,
    _now: datetime,
) -> None:
    """Insert GOID rows for provided FunctionAst map using catalog-based seeding."""
    kinds = {
        func_ast.goid: "class" if isinstance(func_ast.node, ast.ClassDef) else "function"
        for func_ast in ast_by_goid.values()
    }
    functions = [
        function_meta(
            goid=func_ast.goid,
            rel_path=func_ast.rel_path,
            qualname=func_ast.qualname,
            snapshot=(snapshot.repo, snapshot.commit),
            line_span=(func_ast.start_line, func_ast.end_line),
        )
        for func_ast in ast_by_goid.values()
    ]
    catalog = FunctionCatalog(functions=functions, module_by_path={})
    seed_goids_for_snapshot(gateway, snapshot, catalog, kinds=kinds)


def insert_config_values(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    goids: Mapping[str, int],
    ast_by_goid: Mapping[int, FunctionAst],
) -> None:
    """Seed config_values rows for API_TOKEN and FEATURE_FLAG."""
    insert_rows(
        gateway,
        [
            ConfigValueRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                config_path="config/settings.yml",
                format="yaml",
                key="API_TOKEN",
                reference_paths=[
                    ast_by_goid[goids["func_a"]].rel_path,
                    ast_by_goid[goids["func_b"]].rel_path,
                ],
                reference_modules=["pkg.mod_a", "pkg.mod_b"],
                reference_count=2,
            ),
            ConfigValueRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                config_path="config/settings.yml",
                format="yaml",
                key="FEATURE_FLAG",
                reference_paths=[ast_by_goid[goids["func_b"]].rel_path],
                reference_modules=["pkg.mod_b"],
                reference_count=1,
            ),
        ],
    )


def insert_entrypoints(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    goids: Mapping[str, int],
    ast_by_goid: Mapping[int, FunctionAst],
    *,
    now: datetime,
) -> None:
    """Seed analytics.entrypoints with a single FastAPI handler."""
    gateway.policy.ensure_schemas_preserve()
    gateway.con.execute(
        """
        INSERT INTO analytics.entrypoints (
            repo, commit, entrypoint_id, kind, framework,
            handler_goid_h128, handler_urn, handler_rel_path, handler_module,
            handler_qualname, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot.repo,
            snapshot.commit,
            "api.get_items",
            "http",
            "fastapi",
            goids["func_a"],
            "urn:pkg.mod_a.func_a",
            ast_by_goid[goids["func_a"]].rel_path,
            "pkg.mod_a",
            "func_a",
            now,
        ),
    )


def insert_subsystems(gateway: StorageGateway, snapshot: SnapshotRef) -> None:
    """Seed subsystem-module memberships."""
    insert_rows(
        gateway,
        [
            SubsystemModuleRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                subsystem_id="api",
                module="pkg.api",
                role="edge",
            ),
            SubsystemModuleRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                subsystem_id="core",
                module="pkg.service",
                role="service",
            ),
            SubsystemModuleRow(
                repo=snapshot.repo,
                commit=snapshot.commit,
                subsystem_id="core",
                module="pkg.utils",
                role="library",
            ),
        ],
    )


def insert_symbol_edges(
    gateway: StorageGateway,
    goids: Mapping[str, int],
    ast_by_goid: Mapping[int, FunctionAst],
) -> None:
    """Seed symbol use edges between api/service/utils."""
    insert_symbol_use_edges(
        gateway,
        [
            make_symbol_use_edge_row(
                "pkg.mod_b.func_b",
                ast_by_goid[goids["func_b"]].rel_path,
                ast_by_goid[goids["func_a"]].rel_path,
                options=SymbolEdgeOptions(
                    same_file=False,
                    same_module=False,
                ),
            ),
            make_symbol_use_edge_row(
                "pkg.mod_c.func_c",
                ast_by_goid[goids["func_c"]].rel_path,
                ast_by_goid[goids["func_b"]].rel_path,
                options=SymbolEdgeOptions(
                    same_file=False,
                    same_module=False,
                ),
            ),
        ],
    )


def build_module_map(
    ast_by_goid: Mapping[int, FunctionAst],
    goid_to_module: Mapping[int, str],
) -> dict[str, str]:
    """Map relative paths to module names for given GOIDs.

    Parameters
    ----------
    ast_by_goid
        Mapping of GOID to FunctionAst metadata.
    goid_to_module
        Mapping of GOID to module name.

    Returns
    -------
    dict[str, str]
        Mapping of relative source paths to module names.
    """
    module_map: dict[str, str] = {}
    for goid, module in goid_to_module.items():
        ast_obj = ast_by_goid[goid]
        module_map[ast_obj.rel_path] = module
    return module_map


@dataclass
class GraphRuntimeHarness:
    """Reusable graph runtime harness with seeded analytics tables."""

    snapshot: SnapshotRef
    gateway: StorageGateway
    cache_dir: Path
    fixtures: GraphFixtures
    ast_by_goid: dict[int, FunctionAst]
    goids: dict[str, int]
    module_map: dict[str, str]
    runtime_options: GraphRuntimeOptions

    def build_engine(self) -> CountingGraphEngineAdapter:
        """Create a counting graph engine backed by seeded fixtures.

        Returns
        -------
        CountingGraphEngineAdapter
            Graph engine double configured with seeded fixtures.
        """
        runtime = GraphStubEngine.from_fixtures(
            self.fixtures,
            gateway=self.gateway,
            snapshot=self.snapshot,
        )
        return CountingGraphEngineAdapter(runtime, gateway=self.gateway, snapshot=self.snapshot)

    def build_runtime(
        self,
        *,
        engine: CountingGraphEngineAdapter | None = None,
        cache_dir: Path | None = None,
    ) -> GraphRuntime:
        """Construct a GraphRuntime bound to this harness.

        Returns
        -------
        GraphRuntime
            Runtime configured with seeded graph data.
        """
        options = GraphRuntimeOptions(
            snapshot=self.snapshot,
            graph_cache_dir=cache_dir or self.cache_dir,
        )
        return GraphRuntime(options=options, engine=engine or self.build_engine())

    def close(self) -> None:
        """Close the underlying gateway."""
        self.gateway.close()


def build_graph_runtime_harness(tmp_path: Path) -> GraphRuntimeHarness:
    """Seed canonical repo/goids and build a graph runtime harness.

    Parameters
    ----------
    tmp_path
        Temporary path provided by pytest for writing files.

    Returns
    -------
    GraphRuntimeHarness
        Harness containing seeded gateway, ASTs, and graph fixtures.
    """
    snapshot = SnapshotRef(repo="demo/repo", commit="abc123", repo_root=tmp_path / "repo")
    paths = build_source_files(snapshot.repo_root)
    gateway = GatewayFactory().with_snapshot(snapshot.repo, snapshot.commit).open()
    gateway.policy.ensure_schemas_preserve()
    now = datetime.now(tz=UTC)

    goids = {
        "func_a": GOID_FUNC_A,
        "func_b": GOID_FUNC_B,
        "func_c": GOID_FUNC_C,
        "helper": GOID_HELPER,
    }
    target_names = {
        MOD_A_FQN: "func_a",
        MOD_B_FQN: "func_b",
        MOD_C_FQN: "func_c",
        "pkg.util": "helper",
    }
    insert_modules(gateway, snapshot, paths)
    ast_by_goid = build_ast_map(paths, goids, snapshot.repo_root, target_names=target_names)
    insert_goids(gateway, snapshot, ast_by_goid, _now=now)
    insert_config_values(gateway, snapshot, goids, ast_by_goid)
    insert_entrypoints(gateway, snapshot, goids, ast_by_goid, now=now)
    insert_subsystems(gateway, snapshot)
    insert_symbol_edges(gateway, goids, ast_by_goid)

    fixtures = build_sample_graphs(goids)
    engine = build_graph_engine_double(
        gateway,
        snapshot,
        call_graph=fixtures.call_graph,
        import_graph=fixtures.import_graph,
        config_graph=fixtures.config_graph,
        symbol_module_graph=fixtures.symbol_module_graph,
        symbol_function_graph=fixtures.symbol_function_graph,
        cfg_graph=fixtures.cfg_graph,
    )
    runtime_options = GraphRuntimeOptions(
        snapshot=snapshot,
        engine=engine,
        graph_cache_dir=tmp_path,
    )

    module_map = build_module_map(
        ast_by_goid,
        {
            goids["func_a"]: MOD_A_FQN,
            goids["func_b"]: MOD_B_FQN,
            goids["func_c"]: MOD_C_FQN,
        },
    )

    return GraphRuntimeHarness(
        snapshot=snapshot,
        gateway=gateway,
        cache_dir=tmp_path,
        fixtures=fixtures,
        ast_by_goid=ast_by_goid,
        goids=goids,
        module_map=module_map,
        runtime_options=runtime_options,
    )


def _write_tuple_rows(
    ctx: GraphRuntimeHarness,
    table_key: str,
    rows: Sequence[tuple[object, ...]] | None,
) -> None:
    if not rows:
        return
    ctx.gateway.policy.delete_for_snapshot(
        table_key,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
    )
    ctx.gateway.policy.bulk_insert(table_key, rows)


def _write_mapping_rows(
    ctx: GraphRuntimeHarness,
    table_key: str,
    rows: Sequence[Mapping[str, object]] | None,
) -> None:
    if not rows:
        return
    ctx.gateway.policy.delete_for_snapshot(
        table_key,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
    )
    ctx.gateway.policy.bulk_insert_mappings(table_key, rows)


def _config_value_rows(ctx: GraphRuntimeHarness) -> Sequence[Mapping[str, object]]:
    return records_from_relation(
        ctx.gateway.relation_from_table_key("analytics.config_values").select(
            "repo",
            "commit",
            "config_path",
            "key",
            "reference_paths",
            "reference_modules",
        )
    )


def _entrypoint_rows(ctx: GraphRuntimeHarness) -> Sequence[Mapping[str, object]]:
    return records_from_relation(
        ctx.gateway.relation_from_table_key("analytics.entrypoints").select(
            "repo",
            "commit",
            "handler_goid_h128",
        )
    )


def _subsystem_rows(ctx: GraphRuntimeHarness) -> Sequence[Mapping[str, object]]:
    return records_from_relation(
        ctx.gateway.relation_from_table_key("analytics.subsystem_modules").select(
            "repo",
            "commit",
            "subsystem_id",
            "module",
        )
    )


def _graph_metric_filters(
    ctx: GraphRuntimeHarness,
    module_names: set[str],
    filters: GraphMetricFilters | None,
) -> GraphMetricFilters:
    return filters or build_graph_metric_filters_from_sets(
        function_goids=set(ctx.goids.values()),
        modules=module_names,
        subsystems=None,
    )


def _write_config_metrics(
    ctx: GraphRuntimeHarness,
    *,
    config_value_rows: Sequence[Mapping[str, object]],
    entrypoint_rows: Sequence[Mapping[str, object]],
    module_names: set[str],
) -> None:
    data_flow_result = compute_config_data_flow_result(
        ConfigDataFlowInputs(
            snapshot=ctx.snapshot,
            config_value_rows=config_value_rows,
            entrypoint_rows=entrypoint_rows,
            call_graph=ctx.fixtures.call_graph,
            ast_by_goid=ctx.ast_by_goid,
        )
    )
    _write_tuple_rows(ctx, "analytics.config_data_flow", data_flow_result.rows)

    config_metrics_result = compute_config_graph_metrics_result(
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        config_value_rows=config_value_rows,
        allowed_modules=module_names,
        runtime=ctx.runtime_options,
    )
    _write_tuple_rows(ctx, "analytics.config_graph_metrics_keys", config_metrics_result.key_rows)
    _write_tuple_rows(
        ctx, "analytics.config_graph_metrics_modules", config_metrics_result.module_rows
    )
    _write_tuple_rows(
        ctx,
        "analytics.config_projection_key_edges",
        config_metrics_result.key_edge_rows,
    )
    _write_tuple_rows(
        ctx,
        "analytics.config_projection_module_edges",
        config_metrics_result.module_edge_rows,
    )


def _write_graph_metrics(
    ctx: GraphRuntimeHarness,
    *,
    module_names: set[str],
    active_filters: GraphMetricFilters,
) -> Sequence[Mapping[str, object]]:
    import_module_rows = records_from_relation(
        ctx.gateway.relation_from_table_key("graph.import_modules").select(
            "module",
            "scc_id",
            "component_size",
            "layer",
        )
    )
    component_meta = component_metadata_from_import_rows(import_module_rows)
    symbol_rows = records_from_relation(
        ctx.gateway.relation_from_table_key("graph.symbol_use_edges").select(
            "def_path",
            "use_path",
        )
    )
    symbol_module_edges = build_symbol_module_edges(symbol_rows, ctx.module_map)

    metrics_rows = build_graph_metrics_rows(
        GraphMetricsInputs(
            snapshot=ctx.snapshot,
            call_graph=ctx.fixtures.call_graph,
            import_graph=ctx.fixtures.import_graph,
            symbol_module_edges=symbol_module_edges,
            module_names=module_names,
            component_meta=component_meta,
            filters=active_filters,
        )
    )
    _write_mapping_rows(ctx, "analytics.graph_metrics_functions", metrics_rows.function_rows)
    _write_mapping_rows(ctx, "analytics.graph_metrics_modules", metrics_rows.module_rows)

    functions_ext_rows = build_graph_metrics_functions_ext_rows(
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        call_graph=ctx.fixtures.call_graph,
        runtime=ctx.runtime_options,
        filters=active_filters,
    )
    _write_mapping_rows(ctx, "analytics.graph_metrics_functions_ext", functions_ext_rows)

    modules_ext_rows = build_graph_metrics_modules_ext_rows(
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        import_graph=ctx.fixtures.import_graph,
        runtime=ctx.runtime_options,
        filters=active_filters,
    )
    _write_mapping_rows(ctx, "analytics.graph_metrics_modules_ext", modules_ext_rows)
    return modules_ext_rows


def _write_graph_stats(
    ctx: GraphRuntimeHarness,
    *,
    config_value_rows: Sequence[Mapping[str, object]],
    module_names: set[str],
) -> None:
    config_bipartite = build_config_module_bipartite(
        config_value_rows,
        allowed_modules=module_names,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
    )
    graph_stats_rows = build_graph_stats_rows(
        GraphStatsInputs(
            repo=ctx.snapshot.repo,
            commit=ctx.snapshot.commit,
            call_graph=ctx.fixtures.call_graph,
            import_graph=ctx.fixtures.import_graph,
            symbol_module_graph=ctx.fixtures.symbol_module_graph,
            symbol_function_graph=ctx.fixtures.symbol_function_graph,
            config_module_bipartite=config_bipartite,
            use_gpu=ctx.runtime_options.use_gpu,
        )
    )
    _write_tuple_rows(ctx, "analytics.graph_stats", graph_stats_rows)


def _write_subsystem_metrics(
    ctx: GraphRuntimeHarness,
    *,
    subsystem_rows: Sequence[Mapping[str, object]],
    active_filters: GraphMetricFilters,
) -> None:
    subsystem_graph_metrics_rows = build_subsystem_graph_metrics_rows(
        SubsystemGraphMetricInputs(
            repo=ctx.snapshot.repo,
            commit=ctx.snapshot.commit,
            import_graph=ctx.fixtures.import_graph,
            membership_rows=subsystem_rows,
            runtime=ctx.runtime_options,
            filters=active_filters,
        )
    )
    _write_tuple_rows(ctx, "analytics.subsystem_graph_metrics", subsystem_graph_metrics_rows)


def _write_subsystem_agreement(
    ctx: GraphRuntimeHarness,
    *,
    subsystem_rows: Sequence[Mapping[str, object]],
    modules_ext_rows: Sequence[Mapping[str, object]],
) -> None:
    subsystem_agreement_rows = build_subsystem_agreement_rows(
        SubsystemAgreementInputs(
            repo=ctx.snapshot.repo,
            commit=ctx.snapshot.commit,
            subsystem_module_rows=subsystem_rows,
            graph_metrics_module_rows=modules_ext_rows,
        )
    )
    _write_tuple_rows(ctx, "analytics.subsystem_agreement", subsystem_agreement_rows)


def _write_symbol_graph_metrics(
    ctx: GraphRuntimeHarness,
    *,
    module_names: set[str],
) -> None:
    symbol_module_rows = build_symbol_graph_metrics_module_rows(
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        graph=ctx.fixtures.symbol_module_graph,
        known_modules=module_names or None,
        runtime=ctx.runtime_options,
    )
    _write_tuple_rows(ctx, "analytics.symbol_graph_metrics_modules", symbol_module_rows)

    symbol_function_rows = build_symbol_graph_metrics_function_rows(
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        graph=ctx.fixtures.symbol_function_graph,
        known_functions=set(ctx.goids.values()) or None,
        runtime=ctx.runtime_options,
    )
    _write_tuple_rows(ctx, "analytics.symbol_graph_metrics_functions", symbol_function_rows)


def run_graph_metrics_pipeline(
    ctx: GraphRuntimeHarness,
    *,
    filters: GraphMetricFilters | None = None,
) -> None:
    """Run the full analytics graph metrics pipeline for a harness."""
    module_names = set(ctx.module_map.values())
    config_value_rows = _config_value_rows(ctx)
    entrypoint_rows = _entrypoint_rows(ctx)
    subsystem_rows = _subsystem_rows(ctx)
    active_filters = _graph_metric_filters(ctx, module_names, filters)
    _write_config_metrics(
        ctx,
        config_value_rows=config_value_rows,
        entrypoint_rows=entrypoint_rows,
        module_names=module_names,
    )
    modules_ext_rows = _write_graph_metrics(
        ctx,
        module_names=module_names,
        active_filters=active_filters,
    )
    _write_graph_stats(
        ctx,
        config_value_rows=config_value_rows,
        module_names=module_names,
    )
    _write_subsystem_metrics(
        ctx,
        subsystem_rows=subsystem_rows,
        active_filters=active_filters,
    )
    _write_subsystem_agreement(
        ctx,
        subsystem_rows=subsystem_rows,
        modules_ext_rows=modules_ext_rows,
    )
    _write_symbol_graph_metrics(ctx, module_names=module_names)


__all__ = [
    "CanonicalAstArtifacts",
    "GraphRuntimeHarness",
    "build_ast_map",
    "build_canonical_ast_lookup",
    "build_graph_runtime_harness",
    "build_module_map",
    "build_source_files",
    "canonical_ast_artifacts",
    "canonical_ast_map",
    "insert_config_values",
    "insert_entrypoints",
    "insert_goids",
    "insert_modules",
    "insert_subsystems",
    "insert_symbol_edges",
    "run_graph_metrics_pipeline",
]
