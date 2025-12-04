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
>>> seed_golden_graphs(gateway, repo="test/repo", commit="abc123")
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    ConfigValueRow,
    GoidRow,
    ImportGraphEdgeRow,
    ModuleRow,
    SymbolUseEdgeRow,
    insert_rows,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GOLDEN_REPO: Final = "golden/repo"
GOLDEN_COMMIT: Final = "golden123"

# Module structure simulating a layered architecture
# Layer 0: core utilities (no internal deps)
# Layer 1: services (depend on core)
# Layer 2: handlers (depend on services, core)
# Layer 3: api (depend on handlers, services)
GOLDEN_MODULES: Final = {
    # Layer 0 - Core utilities
    "core/__init__.py": "core",
    "core/utils.py": "core.utils",
    "core/types.py": "core.types",
    "core/errors.py": "core.errors",
    "core/config.py": "core.config",
    # Layer 1 - Services
    "services/__init__.py": "services",
    "services/auth.py": "services.auth",
    "services/cache.py": "services.cache",
    "services/database.py": "services.database",
    "services/queue.py": "services.queue",
    "services/storage.py": "services.storage",
    # Layer 2 - Handlers
    "handlers/__init__.py": "handlers",
    "handlers/user.py": "handlers.user",
    "handlers/product.py": "handlers.product",
    "handlers/order.py": "handlers.order",
    "handlers/payment.py": "handlers.payment",
    # Layer 3 - API
    "api/__init__.py": "api",
    "api/routes.py": "api.routes",
    "api/middleware.py": "api.middleware",
    "api/schemas.py": "api.schemas",
    # Standalone utilities
    "utils/logging.py": "utils.logging",
    "utils/metrics.py": "utils.metrics",
    "utils/helpers.py": "utils.helpers",
    # Tests (separate package)
    "tests/__init__.py": "tests",
    "tests/test_user.py": "tests.test_user",
    "tests/test_auth.py": "tests.test_auth",
    "tests/conftest.py": "tests.conftest",
}

# Number of golden nodes for reference
GOLDEN_MODULE_COUNT: Final = len(GOLDEN_MODULES)
GOLDEN_FUNCTION_COUNT: Final = 60  # Approximate
GOLDEN_CALL_EDGE_COUNT: Final = 80  # Approximate


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
        ModuleRow(module=module, path=path, repo=repo, commit=commit)
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
    goid_counter = 1000  # Start at 1000 for easy identification
    now = datetime.now(UTC)

    # Core utilities - many small functions
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

    # Services - medium complexity
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

    # Handlers - business logic
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

    # API layer
    api_functions = [
        ("api/routes.py", "api.routes", ["register_routes", "handle_request", "error_handler"]),
        (
            "api/middleware.py",
            "api.middleware",
            ["auth_middleware", "logging_middleware", "rate_limit"],
        ),
        ("api/schemas.py", "api.schemas", ["validate_request", "serialize_response"]),
    ]

    # Utilities
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
                GoidRow(
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
        CallGraphNodeRow(
            goid_h128=goid.goid_h128,
            language=goid.language,
            kind=goid.kind,
            arity=2,  # Typical function arity
            is_public=not goid.qualname.startswith("_"),
            rel_path=goid.rel_path,
        )
        for goid in goids
        if goid.kind == "function"
    ]


def _build_call_graph_edges(repo: str, commit: str, goids: list[GoidRow]) -> list[CallGraphEdgeRow]:
    """Build realistic call graph edges with patterns.

    Creates edges representing:
    - Hub functions (auth.authenticate called by many handlers)
    - Recursive patterns (queue.process_batch)
    - Utility chains (format_string -> validate_input)

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
    edges: list[CallGraphEdgeRow] = []
    goid_by_name: dict[str, int] = {g.qualname.split(".")[-1]: g.goid_h128 for g in goids}
    goid_by_qualname: dict[str, GoidRow] = {g.qualname: g for g in goids}

    # Define call patterns: (caller_func, callee_func, callsite_offset)
    call_patterns = [
        # Handlers call auth
        ("create_user", "authenticate", 5),
        ("get_user", "authenticate", 3),
        ("update_user", "authenticate", 4),
        ("delete_user", "authenticate", 3),
        ("create_order", "authenticate", 5),
        ("process_order", "authenticate", 3),
        ("create_product", "authenticate", 4),
        # Handlers call database
        ("create_user", "query", 10),
        ("get_user", "query", 8),
        ("update_user", "execute", 12),
        ("delete_user", "execute", 8),
        ("create_order", "transaction", 15),
        ("process_order", "transaction", 10),
        # Auth calls cache
        ("authenticate", "get_cached", 5),
        ("authenticate", "set_cached", 15),
        ("validate_session", "get_cached", 3),
        # Handlers call cache
        ("get_user", "get_cached", 5),
        ("get_product", "get_cached", 4),
        ("search_products", "get_cached", 3),
        # Core utility chains
        ("validate_input", "format_string", 5),
        ("parse_json", "validate_input", 8),
        ("load_config", "parse_json", 3),
        ("get_setting", "load_config", 5),
        # Queue processing (recursive pattern - calls itself indirectly)
        ("process_batch", "dequeue", 5),
        ("process_batch", "enqueue", 20),  # Re-enqueue failures
        # Storage calls
        ("upload", "validate_input", 3),
        ("download", "get_cached", 5),
        # API calls handlers
        ("handle_request", "create_user", 10),
        ("handle_request", "get_user", 15),
        ("handle_request", "create_order", 20),
        ("handle_request", "process_payment", 25),
        # Middleware chains
        ("auth_middleware", "validate_session", 5),
        ("rate_limit", "get_cached", 3),
        # Logging calls (hub - called by many)
        ("authenticate", "log_info", 20),
        ("authorize", "log_info", 15),
        ("process_order", "log_info", 25),
        ("process_payment", "log_info", 20),
        ("create_user", "log_info", 18),
        ("upload", "log_info", 12),
        ("download", "log_info", 10),
        # Metrics (another hub)
        ("authenticate", "record_metric", 22),
        ("process_order", "record_metric", 28),
        ("process_payment", "record_metric", 25),
        ("query", "record_metric", 15),
        ("execute", "record_metric", 12),
        # Error handling
        ("authenticate", "log_error", 25),
        ("query", "log_error", 20),
        ("process_payment", "log_error", 30),
        # Helper usage
        ("query", "retry", 5),
        ("upload", "retry", 8),
        ("download", "timeout", 3),
    ]

    for caller_name, callee_name, offset in call_patterns:
        caller_goid = goid_by_name.get(caller_name)
        callee_goid = goid_by_name.get(callee_name)

        if caller_goid is None or callee_goid is None:
            continue

        # Find caller's path for callsite
        caller_qualname = next((q for q in goid_by_qualname if q.endswith(f".{caller_name}")), None)
        if caller_qualname is None:
            continue

        caller_row = goid_by_qualname[caller_qualname]
        edges.append(
            CallGraphEdgeRow(
                repo=repo,
                commit=commit,
                caller_goid_h128=caller_goid,
                callee_goid_h128=callee_goid,
                callsite_path=caller_row.rel_path,
                callsite_line=caller_row.start_line + offset,
                callsite_col=4,
                language="python",
                kind="direct",
                resolved_via="local_name",
                confidence=0.95,
            )
        )

    return edges


def _build_import_edges(repo: str, commit: str) -> list[ImportGraphEdgeRow]:
    """Build realistic import graph edges with layered architecture.

    Creates import relationships:
    - Layer adherence (handlers -> services -> core)
    - Cross-cutting utilities (everyone imports utils)
    - Intentional cycles (for testing cycle detection)

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
    edges: list[ImportGraphEdgeRow] = []

    # Layer 0 -> nothing (core has no internal deps)
    # Layer 1 (services) -> Layer 0 (core)
    service_to_core = [
        ("services.auth", "core.utils"),
        ("services.auth", "core.errors"),
        ("services.auth", "core.config"),
        ("services.cache", "core.utils"),
        ("services.cache", "core.config"),
        ("services.database", "core.utils"),
        ("services.database", "core.errors"),
        ("services.database", "core.config"),
        ("services.queue", "core.utils"),
        ("services.storage", "core.utils"),
        ("services.storage", "core.errors"),
    ]

    # Layer 2 (handlers) -> Layer 1 (services) and Layer 0 (core)
    handler_imports = [
        ("handlers.user", "services.auth"),
        ("handlers.user", "services.database"),
        ("handlers.user", "services.cache"),
        ("handlers.user", "core.errors"),
        ("handlers.product", "services.database"),
        ("handlers.product", "services.cache"),
        ("handlers.product", "core.errors"),
        ("handlers.order", "services.database"),
        ("handlers.order", "services.queue"),
        ("handlers.order", "services.auth"),
        ("handlers.payment", "services.database"),
        ("handlers.payment", "services.auth"),
        ("handlers.payment", "core.errors"),
    ]

    # Layer 3 (api) -> Layer 2 (handlers) and below
    api_imports = [
        ("api.routes", "handlers.user"),
        ("api.routes", "handlers.product"),
        ("api.routes", "handlers.order"),
        ("api.routes", "handlers.payment"),
        ("api.routes", "api.schemas"),
        ("api.middleware", "services.auth"),
        ("api.middleware", "services.cache"),
        ("api.schemas", "core.types"),
    ]

    # Utils are cross-cutting (imported by many)
    utils_imports = [
        ("services.auth", "utils.logging"),
        ("services.database", "utils.logging"),
        ("handlers.user", "utils.logging"),
        ("handlers.order", "utils.logging"),
        ("api.routes", "utils.logging"),
        ("services.auth", "utils.metrics"),
        ("services.database", "utils.metrics"),
        ("handlers.payment", "utils.metrics"),
    ]

    # Intentional cycle for testing (services.auth <-> services.cache)
    cycle_imports = [
        ("services.auth", "services.cache"),  # Auth uses cache for sessions
        ("services.cache", "services.auth"),  # Cache validates with auth (cycle!)
    ]

    all_imports = service_to_core + handler_imports + api_imports + utils_imports + cycle_imports

    # Assign fan-out/fan-in based on import counts
    fan_out_counts: dict[str, int] = {}
    fan_in_counts: dict[str, int] = {}

    for src, dst in all_imports:
        fan_out_counts[src] = fan_out_counts.get(src, 0) + 1
        fan_in_counts[dst] = fan_in_counts.get(dst, 0) + 1

    # Detect cycle group (services.auth <-> services.cache)
    cycle_modules = {"services.auth", "services.cache"}

    for src, dst in all_imports:
        cycle_group = 1 if src in cycle_modules and dst in cycle_modules else 0
        edges.append(
            ImportGraphEdgeRow(
                repo=repo,
                commit=commit,
                src_module=src,
                dst_module=dst,
                src_fan_out=fan_out_counts.get(src, 1),
                dst_fan_in=fan_in_counts.get(dst, 1),
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

    # Define symbol uses: (symbol_name, def_module, use_modules)
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
            # Find a function in the use_module
            use_funcs = [g for g in goids if g.qualname.startswith(f"{use_module}.")]
            if not use_funcs:
                continue

            use_goid = use_funcs[0]
            edges.append(
                SymbolUseEdgeRow(
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
        ConfigValueRow(
            repo=repo,
            commit=commit,
            config_path="config/app.yaml",
            format="yaml",
            key="database.connection_string",
            reference_paths=["services/database.py", "services/cache.py"],
            reference_modules=["services.database", "services.cache"],
            reference_count=2,
        ),
        ConfigValueRow(
            repo=repo,
            commit=commit,
            config_path="config/app.yaml",
            format="yaml",
            key="auth.secret_key",
            reference_paths=["services/auth.py", "api/middleware.py"],
            reference_modules=["services.auth", "api.middleware"],
            reference_count=2,
        ),
        ConfigValueRow(
            repo=repo,
            commit=commit,
            config_path="config/app.yaml",
            format="yaml",
            key="cache.ttl",
            reference_paths=["services/cache.py", "handlers/user.py", "handlers/product.py"],
            reference_modules=["services.cache", "handlers.user", "handlers.product"],
            reference_count=3,
        ),
        ConfigValueRow(
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
        ConfigValueRow(
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
    repo: str = GOLDEN_REPO,
    commit: str = GOLDEN_COMMIT,
) -> GoldenGraphStats:
    """Seed a gateway with the golden graph dataset.

    This function populates the gateway with realistic graph data suitable
    for testing graph algorithms, validation, and analytics plugins.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    GoldenGraphStats
        Statistics about the seeded data.
    """
    # Build all data
    modules = _build_modules(repo, commit)
    goids = _build_goids(repo, commit)
    call_nodes = _build_call_graph_nodes(goids)
    call_edges = _build_call_graph_edges(repo, commit, goids)
    import_edges = _build_import_edges(repo, commit)
    symbol_edges = _build_symbol_use_edges(goids)
    config_values = _build_config_values(repo, commit)

    # Insert data
    insert_rows(gateway, modules)
    insert_rows(gateway, goids)
    insert_rows(gateway, call_nodes)
    insert_rows(gateway, call_edges)
    insert_rows(gateway, import_edges)
    insert_rows(gateway, symbol_edges)
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
    repo: str = GOLDEN_REPO, commit: str = GOLDEN_COMMIT
) -> list[GoidRow]:
    """Get GOID rows without seeding (for unit tests that build graphs manually).

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    list[GoidRow]
        GOID rows for the golden dataset.
    """
    return _build_goids(repo, commit)


def get_golden_import_edges(
    repo: str = GOLDEN_REPO, commit: str = GOLDEN_COMMIT
) -> list[ImportGraphEdgeRow]:
    """Get import edges without seeding (for unit tests).

    Parameters
    ----------
    repo
        Repository identifier.
    commit
        Commit identifier.

    Returns
    -------
    list[ImportGraphEdgeRow]
        Import edge rows for the golden dataset.
    """
    return _build_import_edges(repo, commit)


__all__ = [
    "GOLDEN_COMMIT",
    "GOLDEN_FUNCTION_COUNT",
    "GOLDEN_MODULE_COUNT",
    "GOLDEN_REPO",
    "GoldenGraphStats",
    "get_golden_call_graph_goids",
    "get_golden_import_edges",
    "seed_golden_graphs",
]
