"""Shared graph runtime harness and pipeline helpers for analytics tests."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.analytics.graphs.config_data_flow import compute_config_data_flow
from codeintel.analytics.graphs.config_graph_metrics import compute_config_graph_metrics
from codeintel.analytics.graphs.graph_metrics import (
    GraphMetricsDeps,
    compute_graph_metrics,
)
from codeintel.analytics.graphs.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.graphs.graph_stats import compute_graph_stats
from codeintel.analytics.graphs.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.analytics.graphs.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.graphs.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.analytics.graphs.symbol_graph_metrics import (
    compute_symbol_graph_metrics_functions,
    compute_symbol_graph_metrics_modules,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.runtime import GraphRuntime, GraphRuntimeOptions
from tests._helpers.fakes.graph_runtime import (
    CountingGraphEngineAdapter,
    build_graph_engine_double,
)
from tests._helpers.fakes.graph_runtime import (
    GraphRuntimeDouble as GraphStubEngine,
)
from tests._helpers.gateway import GatewayFactory
from tests._helpers.graphs import (
    build_ast_map,
    build_module_map,
    build_sample_graphs,
    build_source_files,
    insert_config_values,
    insert_entrypoints,
    insert_goids,
    insert_modules,
    insert_subsystems,
    insert_symbol_edges,
)
from tests._helpers.repo import (
    GOID_FUNC_A,
    GOID_FUNC_B,
    GOID_FUNC_C,
    GOID_HELPER,
    MOD_A_FQN,
    MOD_B_FQN,
    MOD_C_FQN,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.analytics.graphs.graph_metrics import (
        GraphMetricFilters,
    )
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.storage.gateway import StorageGateway
    from tests._helpers.graphs import (
        GraphFixtures,
    )


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
    insert_goids(gateway, snapshot, ast_by_goid, now=now)
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


def run_graph_metrics_pipeline(
    ctx: GraphRuntimeHarness,
    *,
    filters: GraphMetricFilters | None = None,
) -> None:
    """Run the full analytics graph metrics pipeline for a harness."""
    compute_config_data_flow(
        ctx.gateway,
        ctx.snapshot,
        call_graph=ctx.fixtures.call_graph,
        ast_by_goid=ctx.ast_by_goid,
    )
    compute_config_graph_metrics(
        ctx.gateway,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        runtime=ctx.runtime_options,
    )
    compute_graph_metrics(
        ctx.gateway,
        ctx.snapshot,
        deps=GraphMetricsDeps(
            runtime=ctx.runtime_options,
            filters=filters,
            module_by_path=ctx.module_map,
        ),
    )
    compute_graph_metrics_functions_ext(
        ctx.gateway,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        runtime=ctx.runtime_options,
        filters=filters,
    )
    compute_graph_metrics_modules_ext(
        ctx.gateway,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        runtime=ctx.runtime_options,
        filters=filters,
    )
    compute_graph_stats(
        ctx.gateway,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        runtime=ctx.runtime_options,
    )
    compute_subsystem_graph_metrics(
        ctx.gateway,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        runtime=ctx.runtime_options,
        filters=filters,
    )
    compute_subsystem_agreement(
        ctx.gateway,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
    )
    compute_symbol_graph_metrics_modules(
        ctx.gateway,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        runtime=ctx.runtime_options,
    )
    compute_symbol_graph_metrics_functions(
        ctx.gateway,
        repo=ctx.snapshot.repo,
        commit=ctx.snapshot.commit,
        runtime=ctx.runtime_options,
    )


__all__ = [
    "GraphRuntimeHarness",
    "build_graph_runtime_harness",
    "run_graph_metrics_pipeline",
]
