"""Golden dataset fixtures for realistic graph testing.

This module provides production-realistic graph data structures for testing
graph algorithms, validation checks, and analytics plugins. The datasets
simulate a multi-package Python codebase with:

- Complex call patterns including recursion and hub functions
- Layered import architecture with intentional cycles
- Cross-module symbol sharing
- Config references spanning multiple modules

Usage
-----
>>> from tests._helpers.seeds.golden_graphs import seed_golden_graphs
>>> seed_golden_graphs(gateway, snapshot_variant=SnapshotVariant(repo="test/repo", commit="abc123"))
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

import networkx as nx

from tests._helpers.fixtures.graphs import GOLDEN_CALL, GOLDEN_IMPORT, GraphFixtureFactory
from tests._helpers.fixtures.rows import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    ConfigValueRow,
    GoidRow,
    ImportGraphEdgeRow,
    ModuleRow,
    SymbolUseEdgeRow,
    dataclass_row,
    insert_rows,
    insert_symbol_use_edges,
)
from tests._helpers.fixtures.snapshots import GOLDEN_VARIANT, SnapshotVariant

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


GOLDEN_MODULES: Final = {
    "core/__init__.py": "core",
    "core/utils.py": "core.utils",
    "core/types.py": "core.types",
    "core/errors.py": "core.errors",
    "core/config.py": "core.config",
    "services/__init__.py": "services",
    "services/auth.py": "services.auth",
    "services/cache.py": "services.cache",
    "services/database.py": "services.database",
    "services/queue.py": "services.queue",
    "services/storage.py": "services.storage",
    "handlers/__init__.py": "handlers",
    "handlers/user.py": "handlers.user",
    "handlers/product.py": "handlers.product",
    "handlers/order.py": "handlers.order",
    "handlers/payment.py": "handlers.payment",
    "api/__init__.py": "api",
    "api/routes.py": "api.routes",
    "api/middleware.py": "api.middleware",
    "api/schemas.py": "api.schemas",
    "utils/logging.py": "utils.logging",
    "utils/metrics.py": "utils.metrics",
    "utils/helpers.py": "utils.helpers",
    "tests/__init__.py": "tests",
    "tests/test_user.py": "tests.test_user",
    "tests/test_auth.py": "tests.test_auth",
    "tests/conftest.py": "tests.conftest",
}


GOLDEN_MODULE_COUNT: Final = len(GOLDEN_MODULES)
GOLDEN_FUNCTION_COUNT: Final = 60
GOLDEN_CALL_EDGE_COUNT: Final = 80


@dataclass(frozen=True)
class GoldenGraphStats:
    """Statistics about the seeded golden graphs.

    Attributes
    ----------
    module_count
        Number of modules seeded.
    function_count
        Number of function GOIDs seeded.
    call_edge_count
        Number of call graph edges seeded.
    import_edge_count
        Number of import graph edges seeded.
    symbol_edge_count
        Number of symbol use edges seeded.
    config_key_count
        Number of config keys seeded.
    """

    module_count: int
    function_count: int
    call_edge_count: int
    import_edge_count: int
    symbol_edge_count: int
    config_key_count: int


def _map_nodes_to_goids(graph: nx.Graph, goids: list[GoidRow]) -> dict[object, GoidRow]:
    function_goids = [goid for goid in goids if goid.kind == "function"]
    nodes = list(graph.nodes())
    if len(nodes) > len(function_goids):
        message = "Golden graph has more nodes than available function GOIDs"
        raise ValueError(message)
    return {node: function_goids[idx] for idx, node in enumerate(nodes)}


def _map_nodes_to_modules(graph: nx.Graph) -> dict[object, str]:
    modules = list(GOLDEN_MODULES.values())
    nodes = list(graph.nodes())
    if len(nodes) > len(modules):
        message = "Golden graph has more nodes than available modules"
        raise ValueError(message)
    return {node: modules[idx] for idx, node in enumerate(nodes)}


def _cycle_nodes(graph: nx.DiGraph) -> set[object]:
    cycle_nodes: set[object] = set()
    for component in nx.strongly_connected_components(graph):
        if len(component) > 1:
            cycle_nodes.update(component)
    return cycle_nodes


def _build_modules(repo: str, commit: str) -> list[ModuleRow]:
    """Build module rows for golden dataset.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    list[ModuleRow]
        Module rows for insertion.
    """
    return [
        dataclass_row(ModuleRow, module=module, path=path, repo=repo, commit=commit)
        for path, module in GOLDEN_MODULES.items()
    ]


def _build_goids(repo: str, commit: str) -> list[GoidRow]:
    """Build GOID rows for golden dataset with realistic function distribution.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    list[GoidRow]
        GOID rows for insertion.
    """
    goids: list[GoidRow] = []
    goid_counter = 1000
    now = datetime.now(UTC)

    core_functions = [
        (
            "core/utils.py",
            "core.utils",
            ["format_string", "parse_json", "validate_input", "hash_value", "encode_base64"],
        ),
        ("core/types.py", "core.types", ["TypeA", "TypeB", "TypeC"]),
        ("core/errors.py", "core.errors", ["ValidationError", "NotFoundError", "AuthError"]),
        ("core/config.py", "core.config", ["load_config", "get_setting", "validate_config"]),
    ]

    service_functions = [
        (
            "services/auth.py",
            "services.auth",
            ["authenticate", "authorize", "refresh_token", "validate_session", "hash_password"],
        ),
        (
            "services/cache.py",
            "services.cache",
            ["get_cached", "set_cached", "invalidate", "clear_all"],
        ),
        (
            "services/database.py",
            "services.database",
            ["connect", "query", "execute", "transaction", "close"],
        ),
        ("services/queue.py", "services.queue", ["enqueue", "dequeue", "peek", "process_batch"]),
        ("services/storage.py", "services.storage", ["upload", "download", "delete", "list_files"]),
    ]

    handler_functions = [
        (
            "handlers/user.py",
            "handlers.user",
            ["create_user", "get_user", "update_user", "delete_user", "list_users"],
        ),
        (
            "handlers/product.py",
            "handlers.product",
            ["create_product", "get_product", "search_products"],
        ),
        (
            "handlers/order.py",
            "handlers.order",
            ["create_order", "process_order", "cancel_order", "get_order_status"],
        ),
        (
            "handlers/payment.py",
            "handlers.payment",
            ["process_payment", "refund", "verify_payment"],
        ),
    ]

    api_functions = [
        ("api/routes.py", "api.routes", ["register_routes", "handle_request", "error_handler"]),
        (
            "api/middleware.py",
            "api.middleware",
            ["auth_middleware", "logging_middleware", "rate_limit"],
        ),
        ("api/schemas.py", "api.schemas", ["validate_request", "serialize_response"]),
    ]

    util_functions = [
        (
            "utils/logging.py",
            "utils.logging",
            ["setup_logger", "log_error", "log_info", "log_debug"],
        ),
        ("utils/metrics.py", "utils.metrics", ["record_metric", "get_stats", "reset_counters"]),
        ("utils/helpers.py", "utils.helpers", ["retry", "timeout", "memoize"]),
    ]

    all_functions = (
        core_functions + service_functions + handler_functions + api_functions + util_functions
    )

    for rel_path, module, funcs in all_functions:
        line = 10
        for func_name in funcs:
            kind = "class" if func_name[0].isupper() else "function"
            goids.append(
                dataclass_row(
                    GoidRow,
                    goid_h128=goid_counter,
                    urn=f"urn:goid:{repo}:{commit}:{module}.{func_name}",
                    repo=repo,
                    commit=commit,
                    rel_path=rel_path,
                    kind=kind,
                    qualname=f"{module}.{func_name}",
                    start_line=line,
                    end_line=line + 15,
                    language="python",
                    created_at=now,
                )
            )
            goid_counter += 1
            line += 20

    return goids


def _build_call_graph_nodes(goids: list[GoidRow]) -> list[CallGraphNodeRow]:
    """Build call graph nodes from GOIDs.

    Parameters
    ----------
    goids
        GOID rows to create nodes from.

    Returns
    -------
    list[CallGraphNodeRow]
        Call graph node rows.
    """
    return [
        dataclass_row(
            CallGraphNodeRow,
            goid_h128=goid.goid_h128,
            language=goid.language,
            kind=goid.kind,
            arity=2,
            is_public=not goid.qualname.startswith("_"),
            rel_path=goid.rel_path,
        )
        for goid in goids
        if goid.kind == "function"
    ]


def _build_call_graph_edges(repo: str, commit: str, goids: list[GoidRow]) -> list[CallGraphEdgeRow]:
    """Build call graph edges based on the golden graph fixture.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.
    goids
        GOID rows to create edges between.

    Returns
    -------
    list[CallGraphEdgeRow]
        Call graph edge rows.
    """
    graph = cast("nx.DiGraph", GraphFixtureFactory.build(GOLDEN_CALL))
    node_to_goid = _map_nodes_to_goids(graph, goids)
    edges: list[CallGraphEdgeRow] = []

    edge_pairs = cast("list[tuple[object, object]]", list(graph.edges()))
    for idx, edge in enumerate(edge_pairs):
        caller_node = edge[0]
        callee_node = edge[1]
        caller_row = node_to_goid[caller_node]
        callee_row = node_to_goid[callee_node]
        edges.append(
            dataclass_row(
                CallGraphEdgeRow,
                repo=repo,
                commit=commit,
                caller_goid_h128=caller_row.goid_h128,
                callee_goid_h128=callee_row.goid_h128,
                callsite_path=caller_row.rel_path,
                callsite_line=caller_row.start_line + (idx % 5),
                callsite_col=4,
                language="python",
                kind="direct",
                resolved_via="graph_fixture",
                confidence=0.95,
            )
        )

    return edges


def _build_import_edges(repo: str, commit: str) -> list[ImportGraphEdgeRow]:
    """Build import graph edges based on the golden graph fixture.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    list[ImportGraphEdgeRow]
        Import graph edge rows.
    """
    graph = cast("nx.DiGraph", GraphFixtureFactory.build(GOLDEN_IMPORT))
    node_to_module = _map_nodes_to_modules(graph)
    cycle_nodes = _cycle_nodes(graph)

    edges: list[ImportGraphEdgeRow] = []
    fan_out_counts: dict[str, int] = {}
    fan_in_counts: dict[str, int] = {}
    edge_pairs = cast("list[tuple[object, object]]", list(graph.edges()))
    for edge in edge_pairs:
        src_node = edge[0]
        dst_node = edge[1]
        src_module = node_to_module[src_node]
        dst_module = node_to_module[dst_node]
        fan_out_counts[src_module] = fan_out_counts.get(src_module, 0) + 1
        fan_in_counts[dst_module] = fan_in_counts.get(dst_module, 0) + 1

    for edge in edge_pairs:
        src_node = edge[0]
        dst_node = edge[1]
        src_module = node_to_module[src_node]
        dst_module = node_to_module[dst_node]
        cycle_group = 1 if src_node in cycle_nodes and dst_node in cycle_nodes else 0
        edges.append(
            dataclass_row(
                ImportGraphEdgeRow,
                repo=repo,
                commit=commit,
                src_module=src_module,
                dst_module=dst_module,
                src_fan_out=fan_out_counts.get(src_module, 1),
                dst_fan_in=fan_in_counts.get(dst_module, 1),
                cycle_group=cycle_group,
            )
        )

    return edges


def _build_symbol_use_edges(goids: list[GoidRow]) -> list[SymbolUseEdgeRow]:
    """Build symbol use edges showing cross-module dependencies.

    Parameters
    ----------
    goids
        GOID rows for symbol references.

    Returns
    -------
    list[SymbolUseEdgeRow]
        Symbol use edge rows.
    """
    edges: list[SymbolUseEdgeRow] = []
    goid_by_qualname: dict[str, GoidRow] = {g.qualname: g for g in goids}

    symbol_uses = [
        ("ValidationError", "core.errors", ["services.auth", "handlers.user", "handlers.payment"]),
        ("NotFoundError", "core.errors", ["handlers.user", "handlers.product", "handlers.order"]),
        ("AuthError", "core.errors", ["services.auth", "api.middleware"]),
        ("format_string", "core.utils", ["services.auth", "services.database", "handlers.user"]),
        ("validate_input", "core.utils", ["services.storage", "handlers.product"]),
        ("TypeA", "core.types", ["api.schemas", "handlers.user"]),
        ("TypeB", "core.types", ["api.schemas", "handlers.order"]),
    ]

    for symbol, def_module, use_modules in symbol_uses:
        def_qualname = f"{def_module}.{symbol}"
        def_goid = goid_by_qualname.get(def_qualname)
        if def_goid is None:
            continue

        for use_module in use_modules:
            use_funcs = [g for g in goids if g.qualname.startswith(f"{use_module}.")]
            if not use_funcs:
                continue

            use_goid = use_funcs[0]
            edges.append(
                dataclass_row(
                    SymbolUseEdgeRow,
                    symbol=symbol,
                    def_path=def_goid.rel_path,
                    use_path=use_goid.rel_path,
                    same_file=False,
                    same_module=def_module.split(".")[0] == use_module.split(".")[0],
                    def_goid_h128=def_goid.goid_h128,
                    use_goid_h128=use_goid.goid_h128,
                )
            )

    return edges


def _build_config_values(repo: str, commit: str) -> list[ConfigValueRow]:
    """Build config values representing config key usage across modules.

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    list[ConfigValueRow]
        Config value rows.
    """
    return [
        dataclass_row(
            ConfigValueRow,
            repo=repo,
            commit=commit,
            config_path="config/app.yaml",
            format="yaml",
            key="database.connection_string",
            reference_paths=["services/database.py", "services/cache.py"],
            reference_modules=["services.database", "services.cache"],
            reference_count=2,
        ),
        dataclass_row(
            ConfigValueRow,
            repo=repo,
            commit=commit,
            config_path="config/app.yaml",
            format="yaml",
            key="auth.secret_key",
            reference_paths=["services/auth.py", "api/middleware.py"],
            reference_modules=["services.auth", "api.middleware"],
            reference_count=2,
        ),
        dataclass_row(
            ConfigValueRow,
            repo=repo,
            commit=commit,
            config_path="config/app.yaml",
            format="yaml",
            key="cache.ttl",
            reference_paths=["services/cache.py", "handlers/user.py", "handlers/product.py"],
            reference_modules=["services.cache", "handlers.user", "handlers.product"],
            reference_count=3,
        ),
        dataclass_row(
            ConfigValueRow,
            repo=repo,
            commit=commit,
            config_path="config/app.yaml",
            format="yaml",
            key="logging.level",
            reference_paths=[
                "utils/logging.py",
                "services/auth.py",
                "services/database.py",
                "handlers/user.py",
                "handlers/order.py",
                "api/routes.py",
            ],
            reference_modules=[
                "utils.logging",
                "services.auth",
                "services.database",
                "handlers.user",
                "handlers.order",
                "api.routes",
            ],
            reference_count=6,
        ),
        dataclass_row(
            ConfigValueRow,
            repo=repo,
            commit=commit,
            config_path="config/app.yaml",
            format="yaml",
            key="api.rate_limit",
            reference_paths=["api/middleware.py"],
            reference_modules=["api.middleware"],
            reference_count=1,
        ),
    ]


def seed_golden_graphs(
    gateway: StorageGateway,
    *,
    snapshot_variant: SnapshotVariant = GOLDEN_VARIANT,
) -> GoldenGraphStats:
    """Seed a gateway with the golden graph dataset.

    This function populates the gateway with realistic graph data suitable
    for testing graph algorithms, validation, and analytics plugins.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    snapshot_variant
        Snapshot variant for the golden dataset.

    Returns
    -------
    GoldenGraphStats
        Statistics about the seeded data.
    """
    repo = snapshot_variant.repo
    commit = snapshot_variant.commit
    modules = _build_modules(repo, commit)
    goids = _build_goids(repo, commit)
    call_nodes = _build_call_graph_nodes(goids)
    call_edges = _build_call_graph_edges(repo, commit, goids)
    import_edges = _build_import_edges(repo, commit)
    symbol_edges = _build_symbol_use_edges(goids)
    config_values = _build_config_values(repo, commit)

    insert_rows(gateway, modules)
    insert_rows(gateway, goids)
    insert_rows(gateway, call_nodes)
    insert_rows(gateway, call_edges)
    insert_rows(gateway, import_edges)
    insert_symbol_use_edges(gateway, symbol_edges)
    insert_rows(gateway, config_values)

    return GoldenGraphStats(
        module_count=len(modules),
        function_count=len([g for g in goids if g.kind == "function"]),
        call_edge_count=len(call_edges),
        import_edge_count=len(import_edges),
        symbol_edge_count=len(symbol_edges),
        config_key_count=len(config_values),
    )


def get_golden_call_graph_goids(
    snapshot_variant: SnapshotVariant = GOLDEN_VARIANT,
) -> list[GoidRow]:
    """Get GOID rows without seeding (for unit tests that build graphs manually).

    Parameters
    ----------
    snapshot_variant
        Snapshot variant for the golden dataset.

    Returns
    -------
    list[GoidRow]
        GOID rows for the golden dataset.
    """
    return _build_goids(snapshot_variant.repo, snapshot_variant.commit)


def get_golden_import_edges(
    snapshot_variant: SnapshotVariant = GOLDEN_VARIANT,
) -> list[ImportGraphEdgeRow]:
    """Get import edges without seeding (for unit tests).

    Parameters
    ----------
    snapshot_variant
        Snapshot variant for the golden dataset.

    Returns
    -------
    list[ImportGraphEdgeRow]
        Import edge rows for the golden dataset.
    """
    return _build_import_edges(snapshot_variant.repo, snapshot_variant.commit)


__all__ = [
    "GOLDEN_FUNCTION_COUNT",
    "GOLDEN_MODULE_COUNT",
    "GoldenGraphStats",
    "get_golden_call_graph_goids",
    "get_golden_import_edges",
    "seed_golden_graphs",
]
