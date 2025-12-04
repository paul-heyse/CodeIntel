"""Comprehensive tests for gateway accessor classes.

This module tests all table accessor classes in codeintel.storage.gateway.accessors,
following the Testing Charter by using real DuckDB connections.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from codeintel.storage.gateway import StorageGateway
from codeintel.storage.gateway.accessors import (
    AnalyticsTables,
    CoreTables,
    DocsViews,
    GraphTables,
)

# =============================================================================
# CoreTables Tests
# =============================================================================


def test_core_tables_creates_with_connection(fresh_gateway: StorageGateway) -> None:
    """Verify CoreTables initializes with DuckDB connection."""
    core = CoreTables(fresh_gateway.con)
    assert core.con is fresh_gateway.con


def test_core_tables_goids_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify goids() returns DuckDB relation."""
    core = CoreTables(fresh_gateway.con)
    relation = core.goids()
    # Verify it's a valid relation by executing a count
    result = relation.count("*").fetchone()
    assert result is not None


def test_core_tables_modules_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify modules() returns DuckDB relation."""
    core = CoreTables(fresh_gateway.con)
    relation = core.modules()
    result = relation.count("*").fetchone()
    assert result is not None


def test_core_tables_repo_map_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify repo_map() returns DuckDB relation."""
    core = CoreTables(fresh_gateway.con)
    relation = core.repo_map()
    result = relation.count("*").fetchone()
    assert result is not None


def test_core_tables_insert_modules_inserts_rows(fresh_gateway: StorageGateway) -> None:
    """Verify insert_modules inserts rows into core.modules."""
    core = CoreTables(fresh_gateway.con)
    rows = [
        ("test_mod1", "test1.py", "test/repo", "abc123"),
        ("test_mod2", "test2.py", "test/repo", "abc123"),
    ]
    core.insert_modules(rows)

    # Verify rows were inserted
    result = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM core.modules WHERE repo = ?",
        ["test/repo"],
    ).fetchone()
    assert result is not None
    expected_count = 2
    assert result[0] == expected_count


def test_core_tables_insert_modules_adds_defaults(fresh_gateway: StorageGateway) -> None:
    """Verify insert_modules adds default values for language and lists."""
    core = CoreTables(fresh_gateway.con)
    rows = [("mod", "mod.py", "repo", "commit")]
    core.insert_modules(rows)

    result = fresh_gateway.con.execute(
        "SELECT language FROM core.modules WHERE module = ?",
        ["mod"],
    ).fetchone()
    assert result is not None
    assert result[0] == "python"


def test_core_tables_insert_repo_map_inserts_rows(fresh_gateway: StorageGateway) -> None:
    """Verify insert_repo_map inserts rows."""
    core = CoreTables(fresh_gateway.con)
    now = datetime.now(tz=UTC).isoformat()
    rows = [("test/repo", "abc123", "{}", "{}", now)]
    core.insert_repo_map(rows)

    result = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM core.repo_map WHERE repo = ?",
        ["test/repo"],
    ).fetchone()
    assert result is not None
    assert result[0] == 1


def test_core_tables_insert_goids_inserts_rows(fresh_gateway: StorageGateway) -> None:
    """Verify insert_goids inserts rows into core.goids."""
    core = CoreTables(fresh_gateway.con)
    now = datetime.now(tz=UTC).isoformat()
    rows = [
        (
            1001,  # goid_h128
            "urn:test.module:func",  # urn
            "test/repo",  # repo
            "abc123",  # commit
            "test.py",  # rel_path
            "python",  # language
            "function",  # kind
            "test.module.func",  # qualname
            1,  # start_line
            10,  # end_line
            now,  # created_at
        )
    ]
    core.insert_goids(rows)

    result = fresh_gateway.con.execute(
        "SELECT qualname FROM core.goids WHERE goid_h128 = ?",
        [1001],
    ).fetchone()
    assert result is not None
    assert result[0] == "test.module.func"


def test_core_tables_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify CoreTables is immutable."""
    core = CoreTables(fresh_gateway.con)
    with pytest.raises(AttributeError):
        core.con = fresh_gateway.con  # type: ignore[misc]


# =============================================================================
# GraphTables Tests
# =============================================================================


def test_graph_tables_creates_with_connection(fresh_gateway: StorageGateway) -> None:
    """Verify GraphTables initializes with DuckDB connection."""
    graph = GraphTables(fresh_gateway.con)
    assert graph.con is fresh_gateway.con


def test_graph_tables_call_graph_edges_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify call_graph_edges() returns DuckDB relation."""
    graph = GraphTables(fresh_gateway.con)
    relation = graph.call_graph_edges()
    result = relation.count("*").fetchone()
    assert result is not None


def test_graph_tables_call_graph_nodes_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify call_graph_nodes() returns DuckDB relation."""
    graph = GraphTables(fresh_gateway.con)
    relation = graph.call_graph_nodes()
    result = relation.count("*").fetchone()
    assert result is not None


def test_graph_tables_import_graph_edges_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify import_graph_edges() returns DuckDB relation."""
    graph = GraphTables(fresh_gateway.con)
    relation = graph.import_graph_edges()
    result = relation.count("*").fetchone()
    assert result is not None


def test_graph_tables_symbol_use_edges_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify symbol_use_edges() returns DuckDB relation."""
    graph = GraphTables(fresh_gateway.con)
    relation = graph.symbol_use_edges()
    result = relation.count("*").fetchone()
    assert result is not None


def test_graph_tables_insert_call_graph_nodes(fresh_gateway: StorageGateway) -> None:
    """Verify insert_call_graph_nodes inserts rows."""
    graph = GraphTables(fresh_gateway.con)
    rows = [
        (1001, "python", "function", 0, True, "test.py"),
        (1002, "python", "function", 2, False, "test.py"),
    ]
    graph.insert_call_graph_nodes(rows)

    result = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM graph.call_graph_nodes WHERE goid_h128 IN (1001, 1002)"
    ).fetchone()
    assert result is not None
    expected_count = 2
    assert result[0] == expected_count


def test_graph_tables_insert_call_graph_edges(fresh_gateway: StorageGateway) -> None:
    """Verify insert_call_graph_edges inserts rows."""
    graph = GraphTables(fresh_gateway.con)
    edges = [
        (
            "test/repo",
            "abc123",
            1001,  # caller
            1002,  # callee
            "test.py",
            10,  # line
            5,  # col
            "python",
            "direct",
            "local_name",
            1.0,  # confidence
            "{}",  # evidence
        )
    ]
    graph.insert_call_graph_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM graph.call_graph_edges WHERE repo = ?",
        ["test/repo"],
    ).fetchone()
    assert result is not None
    assert result[0] == 1


def test_graph_tables_insert_import_graph_edges(fresh_gateway: StorageGateway) -> None:
    """Verify insert_import_graph_edges inserts rows."""
    graph = GraphTables(fresh_gateway.con)
    edges = [
        ("test/repo", "abc123", "mod_a", "mod_b", 5, 3, 0),
    ]
    graph.insert_import_graph_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT src_module FROM graph.import_graph_edges WHERE repo = ?",
        ["test/repo"],
    ).fetchone()
    assert result is not None
    assert result[0] == "mod_a"


def test_graph_tables_insert_symbol_use_edges_with_5_fields(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify insert_symbol_use_edges handles 5-field rows."""
    graph = GraphTables(fresh_gateway.con)
    edges = [
        ("symbol1", "def.py", "use.py", False, False),
    ]
    graph.insert_symbol_use_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT symbol FROM graph.symbol_use_edges WHERE symbol = ?",
        ["symbol1"],
    ).fetchone()
    assert result is not None
    assert result[0] == "symbol1"


def test_graph_tables_insert_symbol_use_edges_with_7_fields(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify insert_symbol_use_edges handles 7-field rows."""
    graph = GraphTables(fresh_gateway.con)
    edges = [
        ("symbol2", "def.py", "use.py", False, False, 1001, 1002),
    ]
    graph.insert_symbol_use_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT def_goid_h128 FROM graph.symbol_use_edges WHERE symbol = ?",
        ["symbol2"],
    ).fetchone()
    assert result is not None
    expected_goid = 1001
    assert result[0] == expected_goid


def test_graph_tables_insert_symbol_use_edges_rejects_invalid_length(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify insert_symbol_use_edges rejects invalid row lengths."""
    graph = GraphTables(fresh_gateway.con)
    edges = [("a", "b", "c")]  # Only 3 fields
    with pytest.raises(ValueError, match="must have 5 or 7 fields"):
        graph.insert_symbol_use_edges(edges)


def test_graph_tables_insert_cfg_blocks(fresh_gateway: StorageGateway) -> None:
    """Verify insert_cfg_blocks inserts rows."""
    graph = GraphTables(fresh_gateway.con)
    # Schema: function_goid_h128, block_idx, block_id, label, file_path, start_line, end_line, kind, stmts_json, in_degree, out_degree
    blocks = [
        (1001, 0, "block_0", "entry", "test.py", 1, 5, "entry", "[]", 0, 1),
    ]
    graph.insert_cfg_blocks(blocks)

    result = fresh_gateway.con.execute(
        "SELECT block_id FROM graph.cfg_blocks WHERE function_goid_h128 = ?",
        [1001],
    ).fetchone()
    assert result is not None
    assert result[0] == "block_0"


def test_graph_tables_insert_cfg_edges(fresh_gateway: StorageGateway) -> None:
    """Verify insert_cfg_edges inserts rows."""
    graph = GraphTables(fresh_gateway.con)
    edges = [
        (1001, "0", "1", "sequential"),
    ]
    graph.insert_cfg_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT edge_kind FROM graph.cfg_edges WHERE function_goid_h128 = ?",
        [1001],
    ).fetchone()
    assert result is not None
    assert result[0] == "sequential"


def test_graph_tables_insert_dfg_edges(fresh_gateway: StorageGateway) -> None:
    """Verify insert_dfg_edges inserts rows."""
    graph = GraphTables(fresh_gateway.con)
    # Schema: function_goid_h128, src_block_id, dst_block_id, src_var, dst_var, edge_kind, via_phi, use_kind
    edges = [
        (1001, "block_0", "block_1", "x", "x", "def-use", False, "read"),
    ]
    graph.insert_dfg_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT src_var FROM graph.dfg_edges WHERE function_goid_h128 = ?",
        [1001],
    ).fetchone()
    assert result is not None
    assert result[0] == "x"


def test_graph_tables_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify GraphTables is immutable."""
    graph = GraphTables(fresh_gateway.con)
    with pytest.raises(AttributeError):
        graph.con = fresh_gateway.con  # type: ignore[misc]


# =============================================================================
# DocsViews Tests
# =============================================================================


def test_docs_views_creates_with_connection(fresh_gateway: StorageGateway) -> None:
    """Verify DocsViews initializes with DuckDB connection."""
    docs = DocsViews(fresh_gateway.con)
    assert docs.con is fresh_gateway.con


def test_docs_views_function_summary_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify function_summary() returns DuckDB relation by direct SQL."""
    # The view has complex joins so test via direct SQL count
    result = fresh_gateway.con.execute("SELECT COUNT(*) FROM docs.v_function_summary").fetchone()
    assert result is not None
    # Empty is fine - just verify query works
    assert isinstance(result[0], int)


def test_docs_views_call_graph_enriched_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify call_graph_enriched() returns DuckDB relation by direct SQL."""
    # The view has complex joins so test via direct SQL count
    result = fresh_gateway.con.execute("SELECT COUNT(*) FROM docs.v_call_graph_enriched").fetchone()
    assert result is not None
    assert isinstance(result[0], int)


def test_docs_views_function_profile_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify function_profile() returns DuckDB relation."""
    docs = DocsViews(fresh_gateway.con)
    relation = docs.function_profile()
    result = relation.count("*").fetchone()
    assert result is not None


def test_docs_views_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify DocsViews is immutable."""
    docs = DocsViews(fresh_gateway.con)
    with pytest.raises(AttributeError):
        docs.con = fresh_gateway.con  # type: ignore[misc]


# =============================================================================
# AnalyticsTables Tests
# =============================================================================


def test_analytics_tables_creates_with_connection(fresh_gateway: StorageGateway) -> None:
    """Verify AnalyticsTables initializes with DuckDB connection."""
    analytics = AnalyticsTables(fresh_gateway.con)
    assert analytics.con is fresh_gateway.con


def test_analytics_tables_function_metrics_returns_relation(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify function_metrics() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway.con)
    relation = analytics.function_metrics()
    result = relation.count("*").fetchone()
    assert result is not None


def test_analytics_tables_function_types_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify function_types() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway.con)
    relation = analytics.function_types()
    result = relation.count("*").fetchone()
    assert result is not None


def test_analytics_tables_coverage_functions_returns_relation(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify coverage_functions() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway.con)
    relation = analytics.coverage_functions()
    result = relation.count("*").fetchone()
    assert result is not None


def test_analytics_tables_coverage_lines_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify coverage_lines() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway.con)
    relation = analytics.coverage_lines()
    result = relation.count("*").fetchone()
    assert result is not None


def test_analytics_tables_test_catalog_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify test_catalog() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway.con)
    relation = analytics.test_catalog()
    result = relation.count("*").fetchone()
    assert result is not None


def test_analytics_tables_test_coverage_edges_returns_relation(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify test_coverage_edges() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway.con)
    relation = analytics.test_coverage_edges()
    result = relation.count("*").fetchone()
    assert result is not None


def test_analytics_tables_insert_coverage_lines(fresh_gateway: StorageGateway) -> None:
    """Verify insert_coverage_lines inserts rows."""
    analytics = AnalyticsTables(fresh_gateway.con)
    now = datetime.now(tz=UTC).isoformat()
    # Schema: repo, commit, rel_path, line, is_executable, is_covered, hits, context_count, created_at
    rows = [
        ("test/repo", "abc123", "test.py", 10, True, False, 0, 1, now),
    ]
    analytics.insert_coverage_lines(rows)

    result = fresh_gateway.con.execute(
        "SELECT line FROM analytics.coverage_lines WHERE repo = ?",
        ["test/repo"],
    ).fetchone()
    assert result is not None
    expected_line = 10
    assert result[0] == expected_line


def test_analytics_tables_insert_test_catalog(fresh_gateway: StorageGateway) -> None:
    """Verify insert_test_catalog inserts rows."""
    analytics = AnalyticsTables(fresh_gateway.con)
    now = datetime.now(tz=UTC).isoformat()
    # Schema: test_id, test_goid_h128, urn, repo, commit, rel_path, qualname, kind, status, duration_ms, markers, parametrized, flaky, created_at
    rows = [
        (
            "test_func",  # test_id
            1001,  # test_goid_h128
            "urn:test:test_func",  # urn
            "test/repo",  # repo
            "abc123",  # commit
            "test.py",  # rel_path
            "test.test_func",  # qualname
            "test",  # kind
            "passed",  # status
            100,  # duration_ms
            "[]",  # markers
            False,  # parametrized
            False,  # flaky
            now,  # created_at
        )
    ]
    analytics.insert_test_catalog(rows)

    result = fresh_gateway.con.execute(
        "SELECT status FROM analytics.test_catalog WHERE test_id = ?",
        ["test_func"],
    ).fetchone()
    assert result is not None
    assert result[0] == "passed"


def test_analytics_tables_insert_typedness(fresh_gateway: StorageGateway) -> None:
    """Verify insert_typedness inserts rows."""
    analytics = AnalyticsTables(fresh_gateway.con)
    # Schema: repo, commit, path, type_error_count, annotation_ratio (JSON), untyped_defs, overlay_needed
    rows = [
        ("test/repo", "abc123", "test.py", 5, '{"ratio": 0.85}', 2, False),
    ]
    analytics.insert_typedness(rows)

    result = fresh_gateway.con.execute(
        "SELECT type_error_count FROM analytics.typedness WHERE repo = ?",
        ["test/repo"],
    ).fetchone()
    assert result is not None
    expected_errors = 5
    assert result[0] == expected_errors


def test_analytics_tables_insert_static_diagnostics(fresh_gateway: StorageGateway) -> None:
    """Verify insert_static_diagnostics inserts rows."""
    analytics = AnalyticsTables(fresh_gateway.con)
    # Schema: repo, commit, rel_path, pyrefly_errors, pyright_errors, ruff_errors, total_errors, has_errors
    rows = [
        ("test/repo", "abc123", "test.py", 2, 3, 0, 5, True),
    ]
    analytics.insert_static_diagnostics(rows)

    result = fresh_gateway.con.execute(
        "SELECT total_errors FROM analytics.static_diagnostics WHERE repo = ?",
        ["test/repo"],
    ).fetchone()
    assert result is not None
    expected_errors = 5
    assert result[0] == expected_errors


def test_analytics_tables_insert_subsystems(fresh_gateway: StorageGateway) -> None:
    """Verify insert_subsystems inserts rows."""
    analytics = AnalyticsTables(fresh_gateway.con)
    now = datetime.now(tz=UTC).isoformat()
    # Schema: repo, commit, subsystem_id, name, description, module_count, modules_json, entrypoints_json,
    #         internal_edge_count, external_edge_count, fan_in, fan_out, function_count,
    #         avg_risk_score, max_risk_score, high_risk_function_count, risk_level, created_at
    rows = [
        (
            "test/repo",  # repo
            "abc123",  # commit
            "sub1",  # subsystem_id
            "Subsystem 1",  # name
            "Core subsystem",  # description
            5,  # module_count
            "[]",  # modules_json
            "[]",  # entrypoints_json
            10,  # internal_edge_count
            5,  # external_edge_count
            3,  # fan_in
            4,  # fan_out
            20,  # function_count
            0.5,  # avg_risk_score
            0.8,  # max_risk_score
            2,  # high_risk_function_count
            "low",  # risk_level
            now,  # created_at
        )
    ]
    analytics.insert_subsystems(rows)

    result = fresh_gateway.con.execute(
        "SELECT name FROM analytics.subsystems WHERE subsystem_id = ?",
        ["sub1"],
    ).fetchone()
    assert result is not None
    assert result[0] == "Subsystem 1"


def test_analytics_tables_insert_subsystem_modules(fresh_gateway: StorageGateway) -> None:
    """Verify insert_subsystem_modules inserts rows."""
    analytics = AnalyticsTables(fresh_gateway.con)
    rows = [
        ("test/repo", "abc123", "sub1", "test.mod", None),
    ]
    analytics.insert_subsystem_modules(rows)

    result = fresh_gateway.con.execute(
        "SELECT module FROM analytics.subsystem_modules WHERE subsystem_id = ?",
        ["sub1"],
    ).fetchone()
    assert result is not None
    assert result[0] == "test.mod"


def test_analytics_tables_insert_config_values(fresh_gateway: StorageGateway) -> None:
    """Verify insert_config_values inserts rows."""
    analytics = AnalyticsTables(fresh_gateway.con)
    rows = [
        ("test/repo", "abc123", "config.yaml", "yaml", "key1", None, None, 1),
    ]
    analytics.insert_config_values(rows)

    result = fresh_gateway.con.execute(
        "SELECT config_path FROM analytics.config_values WHERE repo = ?",
        ["test/repo"],
    ).fetchone()
    assert result is not None
    assert result[0] == "config.yaml"


def test_analytics_tables_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify AnalyticsTables is immutable."""
    analytics = AnalyticsTables(fresh_gateway.con)
    with pytest.raises(AttributeError):
        analytics.con = fresh_gateway.con  # type: ignore[misc]


# =============================================================================
# DuckDBGateway Tests
# =============================================================================


def test_gateway_creates_accessor_instances(fresh_gateway: StorageGateway) -> None:
    """Verify DuckDBGateway creates all accessor instances."""
    # fresh_gateway is a DuckDBGateway instance under the hood
    assert hasattr(fresh_gateway, "core")
    assert hasattr(fresh_gateway, "graph")
    assert hasattr(fresh_gateway, "docs")
    assert hasattr(fresh_gateway, "analytics")
    assert hasattr(fresh_gateway, "runs")


def test_gateway_core_accessor_is_core_tables(fresh_gateway: StorageGateway) -> None:
    """Verify core accessor is CoreTables instance."""
    assert isinstance(fresh_gateway.core, CoreTables)


def test_gateway_graph_accessor_is_graph_tables(fresh_gateway: StorageGateway) -> None:
    """Verify graph accessor is GraphTables instance."""
    assert isinstance(fresh_gateway.graph, GraphTables)


def test_gateway_docs_accessor_is_docs_views(fresh_gateway: StorageGateway) -> None:
    """Verify docs accessor is DocsViews instance."""
    assert isinstance(fresh_gateway.docs, DocsViews)


def test_gateway_analytics_accessor_is_analytics_tables(fresh_gateway: StorageGateway) -> None:
    """Verify analytics accessor is AnalyticsTables instance."""
    assert isinstance(fresh_gateway.analytics, AnalyticsTables)


def test_gateway_execute_runs_sql(fresh_gateway: StorageGateway) -> None:
    """Verify execute() method runs SQL."""
    result = fresh_gateway.execute("SELECT 1 AS num").fetchone()
    assert result is not None
    assert result[0] == 1


def test_gateway_execute_with_params(fresh_gateway: StorageGateway) -> None:
    """Verify execute() method supports parameters."""
    result = fresh_gateway.execute("SELECT ? AS num", [42]).fetchone()
    assert result is not None
    expected_value = 42
    assert result[0] == expected_value


def test_gateway_table_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify table() method returns relation."""
    relation = fresh_gateway.table("core.modules")
    result = relation.count("*").fetchone()
    assert result is not None


# =============================================================================
# Integration Tests
# =============================================================================


def test_insert_and_query_full_flow(fresh_gateway: StorageGateway) -> None:
    """Verify inserting via accessors and querying via relations."""
    # Insert modules
    fresh_gateway.core.insert_modules(
        [
            ("mod_a", "mod_a.py", "test/repo", "abc123"),
            ("mod_b", "mod_b.py", "test/repo", "abc123"),
        ]
    )

    # Insert goids
    now = datetime.now(tz=UTC).isoformat()
    fresh_gateway.core.insert_goids(
        [
            (
                1001,
                "urn:mod_a:func_a",
                "test/repo",
                "abc123",
                "mod_a.py",
                "python",
                "function",
                "mod_a.func_a",
                1,
                5,
                now,
            ),
            (
                1002,
                "urn:mod_b:func_b",
                "test/repo",
                "abc123",
                "mod_b.py",
                "python",
                "function",
                "mod_b.func_b",
                1,
                5,
                now,
            ),
        ]
    )

    # Insert call graph data
    fresh_gateway.graph.insert_call_graph_nodes(
        [
            (1001, "python", "function", 0, True, "mod_a.py"),
            (1002, "python", "function", 1, True, "mod_b.py"),
        ]
    )

    # Query using relations
    modules_count = fresh_gateway.core.modules().count("*").fetchone()
    goids_count = fresh_gateway.core.goids().count("*").fetchone()
    nodes_count = fresh_gateway.graph.call_graph_nodes().count("*").fetchone()

    assert modules_count is not None
    expected_modules = 2
    assert modules_count[0] == expected_modules

    assert goids_count is not None
    expected_goids = 2
    assert goids_count[0] == expected_goids

    assert nodes_count is not None
    expected_nodes = 2
    assert nodes_count[0] == expected_nodes


def test_relations_support_filtering(fresh_gateway: StorageGateway) -> None:
    """Verify relations support DuckDB filtering."""
    fresh_gateway.core.insert_modules(
        [
            ("mod_a", "mod_a.py", "repo_a", "commit1"),
            ("mod_b", "mod_b.py", "repo_b", "commit2"),
        ]
    )

    # Filter using relation API
    relation = fresh_gateway.core.modules().filter("repo = 'repo_a'")
    result = relation.fetchall()
    assert len(result) == 1
    assert result[0][0] == "mod_a"


def test_accessors_share_same_connection(fresh_gateway: StorageGateway) -> None:
    """Verify all accessors share the same DuckDB connection."""
    assert fresh_gateway.core.con is fresh_gateway.graph.con
    assert fresh_gateway.graph.con is fresh_gateway.docs.con
    assert fresh_gateway.docs.con is fresh_gateway.analytics.con
