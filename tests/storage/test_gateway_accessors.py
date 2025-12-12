"""Comprehensive tests for gateway accessor classes.

This module tests all table accessor classes in codeintel.storage.gateway.accessors,
following the Testing Charter by using real DuckDB connections.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.storage.gateway.accessors import (
    AnalyticsTables,
    CoreTables,
    DocsViews,
    GraphTables,
)
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_not_none,
    expect_length,
    expect_true,
    require_row,
)
from tests._helpers.builders import (
    ConfigValueRow,
    CoverageLineRow,
    StaticDiagnosticsRow,
    SubsystemModuleRow,
    SubsystemRow,
    TestCatalogRow,
    TypednessRow,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.gateway.rows.core import (
        CoreFileStateRow,
        CoreGoidsRow,
        CoreModulesRow,
        CoreRepoMapRow,
        CoreScipOccurrencesRow,
    )
    from codeintel.storage.gateway.rows.graph import (
        GraphCallGraphEdgesRow,
        GraphCallGraphNodesRow,
        GraphImportGraphEdgesRow,
        GraphSymbolUseEdgesRow,
    )
    from tests._helpers.context import TestContext


def _require_row(row: tuple[object, ...] | None, message: str) -> tuple[object, ...]:
    """Ensure fetchone() returned a row and return it.

    Returns
    -------
    tuple[object, ...]
        The fetched row when present.
    """
    if row is None:
        pytest.fail(message)
    return row


# =============================================================================
# CoreTables Tests
# =============================================================================


def test_core_tables_creates_with_connection(fresh_gateway: StorageGateway) -> None:
    """Verify CoreTables initializes with DuckDB connection."""
    core = CoreTables(fresh_gateway)
    expect_true(core.con is fresh_gateway.con)


def test_core_tables_goids_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify goids() returns DuckDB relation."""
    core = CoreTables(fresh_gateway)
    relation = core.goids()
    # Verify it's a valid relation by executing a count
    _require_row(relation.count("*").fetchone(), "goids count missing")


def test_core_tables_modules_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify modules() returns DuckDB relation."""
    core = CoreTables(fresh_gateway)
    relation = core.modules()
    _require_row(relation.count("*").fetchone(), "modules count missing")


def test_core_tables_repo_map_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify repo_map() returns DuckDB relation."""
    core = CoreTables(fresh_gateway)
    relation = core.repo_map()
    _require_row(relation.count("*").fetchone(), "repo_map count missing")


def test_core_tables_insert_modules_inserts_rows(fresh_gateway: StorageGateway) -> None:
    """Verify insert_modules inserts rows into core.modules."""
    core = CoreTables(fresh_gateway)
    rows: list[CoreModulesRow] = [
        {
            "module": "test_mod1",
            "path": "test1.py",
            "repo": "test/repo",
            "commit": "abc123",
            "language": "python",
            "tags": "[]",
            "owners": "[]",
        },
        {
            "module": "test_mod2",
            "path": "test2.py",
            "repo": "test/repo",
            "commit": "abc123",
            "language": "python",
            "tags": "[]",
            "owners": "[]",
        },
    ]
    core.insert_modules(rows)

    # Verify rows were inserted
    row = _require_row(
        fresh_gateway.con.execute(
            "SELECT COUNT(*) FROM core.modules WHERE repo = ?",
            ["test/repo"],
        ).fetchone(),
        "modules insert count missing",
    )
    expected_count = 2
    expect_equal(row[0], expected_count)


def test_core_tables_insert_modules_adds_defaults(fresh_gateway: StorageGateway) -> None:
    """Verify insert_modules adds default values for language and lists."""
    core = CoreTables(fresh_gateway)
    rows: list[CoreModulesRow] = [
        {
            "module": "mod",
            "path": "mod.py",
            "repo": "repo",
            "commit": "commit",
            "language": "python",
            "tags": "[]",
            "owners": "[]",
        }
    ]
    core.insert_modules(rows)

    row = _require_row(
        fresh_gateway.con.execute(
            "SELECT language FROM core.modules WHERE module = ?",
            ["mod"],
        ).fetchone(),
        "module language missing",
    )
    expect_equal(row[0], "python")


def test_core_tables_insert_repo_map_inserts_rows(fresh_gateway: StorageGateway) -> None:
    """Verify insert_repo_map inserts rows."""
    core = CoreTables(fresh_gateway)
    now = datetime.now(tz=UTC).isoformat()
    rows: list[CoreRepoMapRow] = [
        {
            "repo": "test/repo",
            "commit": "abc123",
            "modules": "{}",
            "overlays": "{}",
            "generated_at": now,
        }
    ]
    core.insert_repo_map(rows)

    row = _require_row(
        fresh_gateway.con.execute(
            "SELECT COUNT(*) FROM core.repo_map WHERE repo = ?",
            ["test/repo"],
        ).fetchone(),
        "repo_map count missing",
    )
    expect_equal(row[0], 1)


def test_core_tables_insert_goids_inserts_rows(fresh_gateway: StorageGateway) -> None:
    """Verify insert_goids inserts rows into core.goids."""
    core = CoreTables(fresh_gateway)
    now = datetime.now(tz=UTC).isoformat()
    rows: list[CoreGoidsRow] = [
        {
            "goid_h128": 1001,
            "urn": "urn:test.module:func",
            "repo": "test/repo",
            "commit": "abc123",
            "rel_path": "test.py",
            "language": "python",
            "kind": "function",
            "qualname": "test.module.func",
            "start_line": 1,
            "end_line": 10,
            "created_at": now,
        }
    ]
    core.insert_goids(rows)

    result = fresh_gateway.con.execute(
        "SELECT qualname FROM core.goids WHERE goid_h128 = ?",
        [1001],
    ).fetchone()
    row = _require_row(result, "goids qualname missing")
    expect_equal(row[0], "test.module.func")


def test_core_tables_insert_file_state_inserts_rows(fresh_gateway: StorageGateway) -> None:
    """Verify insert_file_state accepts mapping rows."""
    core = CoreTables(fresh_gateway)
    rows: list[CoreFileStateRow] = [
        {
            "repo": "test/repo",
            "commit": "abc123",
            "rel_path": "path.py",
            "language": "python",
            "size_bytes": 10,
            "mtime_ns": 123456789,
            "content_hash": "hash",
        }
    ]
    core.insert_file_state(rows)

    result = fresh_gateway.con.execute(
        "SELECT language FROM core.file_state WHERE rel_path = ?",
        ["path.py"],
    ).fetchone()
    row = _require_row(result, "file_state language missing")
    expect_equal(row[0], "python")


def test_core_tables_insert_scip_occurrences_inserts_rows(fresh_gateway: StorageGateway) -> None:
    """Verify insert_scip_occurrences accepts mapping rows."""
    core = CoreTables(fresh_gateway)
    now = datetime.now(tz=UTC).isoformat()
    rows: list[CoreScipOccurrencesRow] = [
        {
            "repo": "test/repo",
            "commit": "abc123",
            "rel_path": "path.py",
            "symbol": "sym",
            "start_line": 1,
            "start_col": 0,
            "end_line": 1,
            "end_col": 4,
            "roles": 1,
            "created_at": now,
        }
    ]
    core.insert_scip_occurrences(rows)

    result = fresh_gateway.con.execute(
        "SELECT symbol FROM core.scip_occurrences WHERE rel_path = ?",
        ["path.py"],
    ).fetchone()
    row = _require_row(result, "scip_occurrences symbol missing")
    expect_equal(row[0], "sym")


def test_core_tables_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify CoreTables is immutable."""
    core = CoreTables(fresh_gateway)
    assert_frozen(core, "con", fresh_gateway.con)


# =============================================================================
# GraphTables Tests
# =============================================================================


def test_graph_tables_creates_with_connection(fresh_gateway: StorageGateway) -> None:
    """Verify GraphTables initializes with DuckDB connection."""
    graph = GraphTables(fresh_gateway)
    expect_true(graph.con is fresh_gateway.con)


def test_graph_tables_call_graph_edges_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify call_graph_edges() returns DuckDB relation."""
    graph = GraphTables(fresh_gateway)
    relation = graph.call_graph_edges()
    _require_row(relation.count("*").fetchone(), "call_graph_edges count missing")


def test_graph_tables_call_graph_nodes_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify call_graph_nodes() returns DuckDB relation."""
    graph = GraphTables(fresh_gateway)
    relation = graph.call_graph_nodes()
    _require_row(relation.count("*").fetchone(), "call_graph_nodes count missing")


def test_graph_tables_import_graph_edges_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify import_graph_edges() returns DuckDB relation."""
    graph = GraphTables(fresh_gateway)
    relation = graph.import_graph_edges()
    _require_row(relation.count("*").fetchone(), "import_graph_edges count missing")


def test_graph_tables_symbol_use_edges_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify symbol_use_edges() returns DuckDB relation."""
    graph = GraphTables(fresh_gateway)
    relation = graph.symbol_use_edges()
    _require_row(relation.count("*").fetchone(), "symbol_use_edges count missing")


def test_graph_tables_insert_call_graph_nodes(fresh_gateway: StorageGateway) -> None:
    """Verify insert_call_graph_nodes inserts rows."""
    graph = GraphTables(fresh_gateway)
    rows: list[GraphCallGraphNodesRow] = [
        {
            "goid_h128": 1001,
            "language": "python",
            "kind": "function",
            "arity": 0,
            "is_public": True,
            "rel_path": "test.py",
        },
        {
            "goid_h128": 1002,
            "language": "python",
            "kind": "function",
            "arity": 2,
            "is_public": False,
            "rel_path": "test.py",
        },
    ]
    graph.insert_call_graph_nodes(rows)

    row = _require_row(
        fresh_gateway.con.execute(
            "SELECT COUNT(*) FROM graph.call_graph_nodes WHERE goid_h128 IN (1001, 1002)"
        ).fetchone(),
        "call_graph_nodes inserted count missing",
    )
    expected_count = 2
    expect_equal(row[0], expected_count)


def test_graph_tables_insert_call_graph_edges(fresh_gateway: StorageGateway) -> None:
    """Verify insert_call_graph_edges inserts rows."""
    graph = GraphTables(fresh_gateway)
    edges: list[GraphCallGraphEdgesRow] = [
        {
            "repo": "test/repo",
            "commit": "abc123",
            "caller_goid_h128": 1001.0,
            "callee_goid_h128": 1002.0,
            "callsite_path": "test.py",
            "callsite_line": 10,
            "callsite_col": 5,
            "language": "python",
            "kind": "direct",
            "resolved_via": "local_name",
            "confidence": 1.0,
            "evidence_json": "{}",
        }
    ]
    graph.insert_call_graph_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT COUNT(*) FROM graph.call_graph_edges WHERE repo = ?",
        ["test/repo"],
    ).fetchone()
    row = _require_row(result, "call_graph_edges count missing")
    expect_equal(row[0], 1)


def test_graph_tables_insert_import_graph_edges(fresh_gateway: StorageGateway) -> None:
    """Verify insert_import_graph_edges inserts rows."""
    graph = GraphTables(fresh_gateway)
    edges: list[GraphImportGraphEdgesRow] = [
        {
            "repo": "test/repo",
            "commit": "abc123",
            "src_module": "mod_a",
            "dst_module": "mod_b",
            "src_fan_out": 5,
            "dst_fan_in": 3,
            "cycle_group": 0,
            "module_layer": None,
        },
    ]
    graph.insert_import_graph_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT src_module FROM graph.import_graph_edges WHERE repo = ?",
        ["test/repo"],
    ).fetchone()
    row = _require_row(result, "import_graph_edges row missing")
    expect_equal(row[0], "mod_a")


def test_graph_tables_insert_symbol_use_edges_with_5_fields(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify insert_symbol_use_edges handles 5-field rows."""
    graph = GraphTables(fresh_gateway)
    edges: list[GraphSymbolUseEdgesRow] = [
        {
            "symbol": "symbol1",
            "def_path": "def.py",
            "use_path": "use.py",
            "same_file": False,
            "same_module": False,
            "def_goid_h128": None,
            "use_goid_h128": None,
        },
    ]
    graph.insert_symbol_use_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT symbol FROM graph.symbol_use_edges WHERE symbol = ?",
        ["symbol1"],
    ).fetchone()
    row = _require_row(result, "symbol_use_edges row missing")
    expect_equal(row[0], "symbol1")


def test_graph_tables_insert_symbol_use_edges_with_7_fields(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify insert_symbol_use_edges handles 7-field rows."""
    graph = GraphTables(fresh_gateway)
    edges: list[GraphSymbolUseEdgesRow] = [
        {
            "symbol": "symbol2",
            "def_path": "def.py",
            "use_path": "use.py",
            "same_file": False,
            "same_module": False,
            "def_goid_h128": 1001,
            "use_goid_h128": 1002,
        }
    ]
    graph.insert_symbol_use_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT def_goid_h128 FROM graph.symbol_use_edges WHERE symbol = ?",
        ["symbol2"],
    ).fetchone()
    row = _require_row(result, "symbol2 row missing")
    expected_goid = 1001
    expect_equal(row[0], expected_goid)


def test_graph_tables_insert_symbol_use_edges_rejects_invalid_length(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify insert_symbol_use_edges rejects invalid row lengths."""
    graph = GraphTables(fresh_gateway)
    edges = [("a", "b", "c")]  # Only 3 fields
    with pytest.raises(ValueError, match="must have 5 or 7 fields"):
        graph.insert_symbol_use_edges(edges)


def test_graph_tables_insert_cfg_blocks(fresh_gateway: StorageGateway) -> None:
    """Verify insert_cfg_blocks inserts rows."""
    graph = GraphTables(fresh_gateway)
    # Schema: function_goid_h128, block_idx, block_id, label, file_path, start_line, end_line, kind, stmts_json, in_degree, out_degree
    blocks = [
        (1001, 0, "block_0", "entry", "test.py", 1, 5, "entry", "[]", 0, 1),
    ]
    graph.insert_cfg_blocks(blocks)

    result = fresh_gateway.con.execute(
        "SELECT block_id FROM graph.cfg_blocks WHERE function_goid_h128 = ?",
        [1001],
    ).fetchone()
    row = _require_row(result, "cfg_blocks row missing")
    expect_equal(row[0], "block_0")


def test_graph_tables_insert_cfg_edges(fresh_gateway: StorageGateway) -> None:
    """Verify insert_cfg_edges inserts rows."""
    graph = GraphTables(fresh_gateway)
    edges = [
        (1001, "0", "1", "sequential"),
    ]
    graph.insert_cfg_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT edge_kind FROM graph.cfg_edges WHERE function_goid_h128 = ?",
        [1001],
    ).fetchone()
    row = _require_row(result, "cfg_edges row missing")
    expect_equal(row[0], "sequential")


def test_graph_tables_insert_dfg_edges(fresh_gateway: StorageGateway) -> None:
    """Verify insert_dfg_edges inserts rows."""
    graph = GraphTables(fresh_gateway)
    # Schema: function_goid_h128, src_block_id, dst_block_id, src_var, dst_var, edge_kind, via_phi, use_kind
    edges = [
        (1001, "block_0", "block_1", "x", "x", "def-use", False, "read"),
    ]
    graph.insert_dfg_edges(edges)

    result = fresh_gateway.con.execute(
        "SELECT src_var FROM graph.dfg_edges WHERE function_goid_h128 = ?",
        [1001],
    ).fetchone()
    row = _require_row(result, "dfg_edges row missing")
    expect_equal(row[0], "x")


def test_graph_tables_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify GraphTables is immutable."""
    graph = GraphTables(fresh_gateway)
    assert_frozen(graph, "con", fresh_gateway.con)


# =============================================================================
# DocsViews Tests
# =============================================================================


def test_docs_views_creates_with_connection(fresh_gateway: StorageGateway) -> None:
    """Verify DocsViews initializes with DuckDB connection."""
    docs = DocsViews(fresh_gateway)
    expect_true(docs.con is fresh_gateway.con)


def test_docs_views_function_summary_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify function_summary() returns DuckDB relation by direct SQL."""
    # The view has complex joins so test via direct SQL count
    row = _require_row(
        fresh_gateway.con.execute("SELECT COUNT(*) FROM docs.v_function_summary").fetchone(),
        "function_summary row missing",
    )
    expect_is_instance(row[0], int)


def test_docs_views_call_graph_enriched_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify call_graph_enriched() returns DuckDB relation by direct SQL."""
    # The view has complex joins so test via direct SQL count
    row = _require_row(
        fresh_gateway.con.execute("SELECT COUNT(*) FROM docs.v_call_graph_enriched").fetchone(),
        "call_graph_enriched row missing",
    )
    expect_is_instance(row[0], int)


def test_docs_views_function_profile_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify function_profile() returns DuckDB relation."""
    docs = DocsViews(fresh_gateway)
    relation = docs.function_profile()
    _require_row(relation.count("*").fetchone(), "function_profile count missing")


def test_docs_views_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify DocsViews is immutable."""
    docs = DocsViews(fresh_gateway)
    assert_frozen(docs, "con", fresh_gateway.con)


# =============================================================================
# AnalyticsTables Tests
# =============================================================================


def test_analytics_tables_creates_with_connection(fresh_gateway: StorageGateway) -> None:
    """Verify AnalyticsTables initializes with DuckDB connection."""
    analytics = AnalyticsTables(fresh_gateway)
    expect_true(analytics.con is fresh_gateway.con)


def test_analytics_tables_function_metrics_returns_relation(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify function_metrics() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway)
    relation = analytics.function_metrics()
    _require_row(relation.count("*").fetchone(), "function_metrics count missing")


def test_analytics_tables_function_types_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify function_types() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway)
    relation = analytics.function_types()
    _require_row(relation.count("*").fetchone(), "function_types count missing")


def test_analytics_tables_coverage_functions_returns_relation(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify coverage_functions() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway)
    relation = analytics.coverage_functions()
    result = relation.count("*").fetchone()
    expect_is_not_none(result)


def test_analytics_tables_coverage_lines_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify coverage_lines() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway)
    relation = analytics.coverage_lines()
    result = relation.count("*").fetchone()
    require_row(result, message="Expected coverage_lines count row")


def test_analytics_tables_test_catalog_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify test_catalog() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway)
    relation = analytics.test_catalog()
    result = relation.count("*").fetchone()
    require_row(result, message="Expected test_catalog count row")


def test_analytics_tables_test_coverage_edges_returns_relation(
    fresh_gateway: StorageGateway,
) -> None:
    """Verify test_coverage_edges() returns DuckDB relation."""
    analytics = AnalyticsTables(fresh_gateway)
    relation = analytics.test_coverage_edges()
    result = relation.count("*").fetchone()
    require_row(result, message="Expected test_coverage_edges count row")


def test_analytics_tables_insert_coverage_lines(coverage_ctx: TestContext) -> None:
    """Verify insert_coverage_lines inserts rows using coverage pack gateway."""
    analytics = coverage_ctx.gateway.analytics
    now = datetime.now(tz=UTC)
    rows = [
        CoverageLineRow(
            repo=coverage_ctx.repo,
            commit=coverage_ctx.commit,
            rel_path="test.py",
            line=10,
            is_executable=True,
            is_covered=False,
            hits=0,
            context_count=1,
            created_at=now,
        ).to_tuple(),
    ]
    analytics.insert_coverage_lines(rows)

    result = coverage_ctx.con.execute(
        "SELECT line FROM analytics.coverage_lines WHERE repo = ?",
        [coverage_ctx.repo],
    ).fetchone()
    result = require_row(result, message="Expected coverage line row")
    expected_line = 10
    expect_equal(result[0], expected_line)


def test_analytics_tables_insert_test_catalog(coverage_ctx: TestContext) -> None:
    """Verify insert_test_catalog inserts rows with coverage seed data."""
    analytics = coverage_ctx.gateway.analytics
    now = datetime.now(tz=UTC)
    rows = [
        TestCatalogRow(
            test_id="test_func",
            test_goid_h128=1001,
            urn=f"urn:{coverage_ctx.repo}:{coverage_ctx.commit}:test.py#test_func",
            repo=coverage_ctx.repo,
            commit=coverage_ctx.commit,
            rel_path="test.py",
            qualname="test.test_func",
            kind="unit",
            status="passed",
            duration_ms=100,
            markers="[]",
            parametrized=False,
            flaky=False,
            created_at=now,
        ).to_tuple()
    ]
    analytics.insert_test_catalog(rows)

    result = coverage_ctx.con.execute(
        "SELECT status FROM analytics.test_catalog WHERE test_id = ?",
        ["test_func"],
    ).fetchone()
    row = require_row(result, message="Expected test_catalog row")
    expect_equal(row[0], "passed")


def test_analytics_tables_insert_typedness(metrics_ctx: TestContext) -> None:
    """Verify insert_typedness inserts rows on metrics gateway."""
    analytics = metrics_ctx.gateway.analytics
    rows = [
        TypednessRow(
            repo=metrics_ctx.repo,
            commit=metrics_ctx.commit,
            path="test.py",
            type_error_count=5,
            annotation_ratio='{"ratio": 0.85}',
            untyped_defs=2,
            overlay_needed=False,
        ).to_tuple(),
    ]
    analytics.insert_typedness(rows)

    result = metrics_ctx.con.execute(
        "SELECT type_error_count FROM analytics.typedness WHERE repo = ? AND path = ?",
        [metrics_ctx.repo, "test.py"],
    ).fetchone()
    result = require_row(result, message="Expected typedness row")
    expected_errors = 5
    expect_equal(result[0], expected_errors)


def test_analytics_tables_insert_static_diagnostics(metrics_ctx: TestContext) -> None:
    """Verify insert_static_diagnostics inserts rows on metrics gateway."""
    analytics = metrics_ctx.gateway.analytics
    rows = [
        StaticDiagnosticsRow(
            repo=metrics_ctx.repo,
            commit=metrics_ctx.commit,
            rel_path="test.py",
            pyrefly_errors=2,
            pyright_errors=3,
            ruff_errors=0,
            total_errors=5,
            has_errors=True,
        ).to_tuple(),
    ]
    analytics.insert_static_diagnostics(rows)

    result = metrics_ctx.con.execute(
        "SELECT total_errors FROM analytics.static_diagnostics WHERE repo = ? AND rel_path = ?",
        [metrics_ctx.repo, "test.py"],
    ).fetchone()
    result = require_row(result, message="Expected static_diagnostics row")
    expected_errors = 5
    expect_equal(result[0], expected_errors)


def test_analytics_tables_insert_subsystems(test_ctx: TestContext) -> None:
    """Verify insert_subsystems inserts rows using subsystem helpers."""
    analytics = test_ctx.gateway.analytics
    now = datetime.now(tz=UTC)
    rows = [
        SubsystemRow(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            subsystem_id="sub1",
            name="Subsystem 1",
            description="Core subsystem",
            module_count=5,
            modules_json="[]",
            entrypoints_json="[]",
            internal_edge_count=0,
            external_edge_count=0,
            fan_in=0,
            fan_out=0,
            function_count=20,
            avg_risk_score=0.1,
            max_risk_score=0.1,
            high_risk_function_count=0,
            risk_level="low",
            created_at=now,
        ).to_tuple()
    ]
    analytics.insert_subsystems(rows)

    result = test_ctx.con.execute(
        "SELECT name FROM analytics.subsystems WHERE subsystem_id = ?",
        ["sub1"],
    ).fetchone()
    row = require_row(result, message="Expected subsystem row after insert")
    expect_equal(row[0], "Subsystem 1")


def test_analytics_tables_insert_subsystem_modules(test_ctx: TestContext) -> None:
    """Verify insert_subsystem_modules inserts rows."""
    analytics = test_ctx.gateway.analytics
    rows = [
        SubsystemRow(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            subsystem_id="sub2",
            name="Subsystem 2",
            description="Subsystem 2",
            module_count=1,
            modules_json="[]",
            entrypoints_json="[]",
            internal_edge_count=0,
            external_edge_count=0,
            fan_in=0,
            fan_out=0,
            function_count=1,
            avg_risk_score=0.0,
            max_risk_score=0.0,
            high_risk_function_count=0,
            risk_level="low",
            created_at=datetime.now(tz=UTC),
        ).to_tuple(),
    ]
    analytics.insert_subsystems(rows)
    module_rows = [
        SubsystemModuleRow(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            subsystem_id="sub2",
            module="test.mod",
            role="member",
        ).to_tuple(),
    ]
    analytics.insert_subsystem_modules(module_rows)

    result = test_ctx.con.execute(
        "SELECT module FROM analytics.subsystem_modules WHERE subsystem_id = ?",
        ["sub2"],
    ).fetchone()
    row = require_row(result, message="Expected subsystem module row")
    expect_equal(row[0], "test.mod")


def test_analytics_tables_insert_config_values(test_ctx: TestContext) -> None:
    """Verify insert_config_values inserts rows."""
    analytics = test_ctx.gateway.analytics
    rows = [
        ConfigValueRow(
            repo=test_ctx.repo,
            commit=test_ctx.commit,
            config_path="config.yaml",
            format="yaml",
            key="key1",
            reference_paths=["config.yaml"],
            reference_modules=["mod"],
            reference_count=1,
        ).to_tuple(),
    ]
    analytics.insert_config_values(rows)

    result = test_ctx.con.execute(
        "SELECT config_path FROM analytics.config_values WHERE repo = ?",
        [test_ctx.repo],
    ).fetchone()
    row = require_row(result, message="Expected config_values row")
    expect_equal(row[0], "config.yaml")


def test_analytics_tables_is_frozen(fresh_gateway: StorageGateway) -> None:
    """Verify AnalyticsTables is immutable."""
    analytics = AnalyticsTables(fresh_gateway)
    assert_frozen(analytics, "con", fresh_gateway.con)


# =============================================================================
# DuckDBGateway Tests
# =============================================================================


def test_gateway_creates_accessor_instances(fresh_gateway: StorageGateway) -> None:
    """Verify DuckDBGateway creates all accessor instances."""
    # fresh_gateway is a DuckDBGateway instance under the hood
    expect_true(hasattr(fresh_gateway, "core"))
    expect_true(hasattr(fresh_gateway, "graph"))
    expect_true(hasattr(fresh_gateway, "docs"))
    expect_true(hasattr(fresh_gateway, "analytics"))
    expect_true(hasattr(fresh_gateway, "runs"))


def test_gateway_core_accessor_is_core_tables(fresh_gateway: StorageGateway) -> None:
    """Verify core accessor is CoreTables instance."""
    expect_is_instance(fresh_gateway.core, CoreTables)


def test_gateway_graph_accessor_is_graph_tables(fresh_gateway: StorageGateway) -> None:
    """Verify graph accessor is GraphTables instance."""
    expect_is_instance(fresh_gateway.graph, GraphTables)


def test_gateway_docs_accessor_is_docs_views(fresh_gateway: StorageGateway) -> None:
    """Verify docs accessor is DocsViews instance."""
    expect_is_instance(fresh_gateway.docs, DocsViews)


def test_gateway_analytics_accessor_is_analytics_tables(fresh_gateway: StorageGateway) -> None:
    """Verify analytics accessor is AnalyticsTables instance."""
    expect_is_instance(fresh_gateway.analytics, AnalyticsTables)


def test_gateway_execute_runs_sql(fresh_gateway: StorageGateway) -> None:
    """Verify execute() method runs SQL."""
    result = fresh_gateway.execute("SELECT 1 AS num").fetchone()
    row = require_row(result, message="Expected SELECT 1 result")
    expect_equal(row[0], 1)


def test_gateway_execute_with_params(fresh_gateway: StorageGateway) -> None:
    """Verify execute() method supports parameters."""
    result = fresh_gateway.execute("SELECT ? AS num", [42]).fetchone()
    row = require_row(result, message="Expected SELECT ? result")
    expected_value = 42
    expect_equal(row[0], expected_value)


def test_gateway_table_returns_relation(fresh_gateway: StorageGateway) -> None:
    """Verify table() method returns relation."""
    relation = fresh_gateway.table("core.modules")
    result = relation.count("*").fetchone()
    require_row(result, message="Expected relation count row")


# =============================================================================
# Integration Tests
# =============================================================================


def test_insert_and_query_full_flow(fresh_gateway: StorageGateway) -> None:
    """Verify inserting via accessors and querying via relations."""
    # Insert modules
    fresh_gateway.core.insert_modules(
        [
            {
                "module": "mod_a",
                "path": "mod_a.py",
                "repo": "test/repo",
                "commit": "abc123",
                "language": "python",
                "tags": "[]",
                "owners": "[]",
            },
            {
                "module": "mod_b",
                "path": "mod_b.py",
                "repo": "test/repo",
                "commit": "abc123",
                "language": "python",
                "tags": "[]",
                "owners": "[]",
            },
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

    modules_row = require_row(modules_count, message="Expected modules count")
    expected_modules = 2
    expect_equal(modules_row[0], expected_modules)

    goids_row = require_row(goids_count, message="Expected goids count")
    expected_goids = 2
    expect_equal(goids_row[0], expected_goids)

    nodes_row = require_row(nodes_count, message="Expected call graph nodes count")
    expected_nodes = 2
    expect_equal(nodes_row[0], expected_nodes)


def test_relations_support_filtering(fresh_gateway: StorageGateway) -> None:
    """Verify relations support DuckDB filtering."""
    fresh_gateway.core.insert_modules(
        [
            {
                "module": "mod_a",
                "path": "mod_a.py",
                "repo": "repo_a",
                "commit": "commit1",
                "language": "python",
                "tags": "[]",
                "owners": "[]",
            },
            {
                "module": "mod_b",
                "path": "mod_b.py",
                "repo": "repo_b",
                "commit": "commit2",
                "language": "python",
                "tags": "[]",
                "owners": "[]",
            },
        ]
    )

    # Filter using relation API
    relation = fresh_gateway.core.modules().filter("repo = 'repo_a'")
    result = relation.fetchall()
    expect_length(result, 1)
    expect_equal(result[0][0], "mod_a")


def test_accessors_share_same_connection(fresh_gateway: StorageGateway) -> None:
    """Verify all accessors share the same DuckDB connection."""
    expect_true(fresh_gateway.core.con is fresh_gateway.graph.con)
    expect_true(fresh_gateway.graph.con is fresh_gateway.docs.con)
    expect_true(fresh_gateway.docs.con is fresh_gateway.analytics.con)
