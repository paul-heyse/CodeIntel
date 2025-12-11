"""Tests for symbol_use_edges helper utilities."""

from __future__ import annotations

from decimal import Decimal
from typing import TYPE_CHECKING

import pytest

from codeintel.graphs.engine import views as nx_views
from tests._helpers.assertions import (
    expect_equal,
    expect_false,
    expect_true,
)
from tests._helpers.builders import insert_symbol_use_edges, make_symbol_use_edge_row
from tests._helpers.seeds.core import MOD_A_FQN, MOD_A_PATH, MOD_B_FQN, MOD_B_PATH

if TYPE_CHECKING:
    from tests._helpers.context import TestContext


def test_insert_symbol_use_edges_coerces_five_field_rows(test_ctx: TestContext) -> None:
    """Five-field tuples are extended with NULL GOIDs during insertion."""
    con = test_ctx.gateway.con
    con.execute("DELETE FROM graph.symbol_use_edges")

    inserted = insert_symbol_use_edges(
        test_ctx.gateway,
        [("sym", "a.py", "a.py", True, True)],
    )
    expect_equal(inserted, 1)
    row = con.execute(
        """
        SELECT symbol, def_path, use_path, same_file, same_module, def_goid_h128, use_goid_h128
        FROM graph.symbol_use_edges
        """
    ).fetchone()
    expect_equal(row, ("sym", "a.py", "a.py", True, True, None, None))


def test_insert_symbol_use_edges_respects_explicit_goids(test_ctx: TestContext) -> None:
    """Seven-field tuples pass through GOID fields unchanged."""
    con = test_ctx.gateway.con
    con.execute("DELETE FROM graph.symbol_use_edges")

    inserted = insert_symbol_use_edges(
        test_ctx.gateway,
        [
            ("sym", "a.py", "b.py", False, False, 10, 20),
        ],
    )
    expect_equal(inserted, 1)
    row = con.execute(
        """
        SELECT def_goid_h128, use_goid_h128 FROM graph.symbol_use_edges WHERE symbol = 'sym'
        """
    ).fetchone()
    expect_equal(row, (10, 20))


def test_make_symbol_use_edge_row_infers_same_flags() -> None:
    """same_file and same_module default to path-derived values when omitted."""
    edge = make_symbol_use_edge_row("sym", "pkg/a.py", "pkg/a.py")
    expect_true(edge.same_file is True)
    expect_true(edge.same_module is True)


def test_insert_symbol_use_edges_invalid_shape_raises(test_ctx: TestContext) -> None:
    """Invalid-length sequences raise a clear ValueError."""
    with pytest.raises(ValueError, match="symbol_use_edges rows must have 5 or 7 fields"):
        insert_symbol_use_edges(test_ctx.gateway, [("sym", "a", "b", False, False, None)])


def test_load_symbol_module_graph_smoke(core_ctx: TestContext) -> None:
    """Helpers seed symbol edges compatible with graph view loaders."""
    con = core_ctx.gateway.con
    con.execute("DELETE FROM graph.symbol_use_edges")

    insert_symbol_use_edges(
        core_ctx.gateway,
        [
            ("sym", MOD_A_PATH, MOD_B_PATH, False, False),
        ],
    )

    graph = nx_views.load_symbol_module_graph(core_ctx.gateway, core_ctx.repo, core_ctx.commit)
    expect_true(graph.has_edge(MOD_B_FQN, MOD_A_FQN))


def test_load_symbol_function_graph_smoke(test_ctx: TestContext) -> None:
    """load_symbol_function_graph normalizes GOIDs and skips invalid/self edges."""
    con = test_ctx.gateway.con
    con.execute("DELETE FROM graph.symbol_use_edges")

    insert_symbol_use_edges(
        test_ctx.gateway,
        [
            ("s1", "a.py", "b.py", False, False, Decimal("10"), 20),
            ("s2", "a.py", "b.py", False, False, Decimal("10"), 20),
            ("self", "a.py", "a.py", True, True, 30, 30),
            ("bad", "a.py", "c.py", False, False, None, 40),
        ],
    )

    graph = nx_views.load_symbol_function_graph(test_ctx.gateway, test_ctx.repo, test_ctx.commit)

    expect_true(graph.has_edge(10, 20))
    expect_equal(graph[10][20]["weight"], 2)
    expect_false(graph.has_node(30))
    expect_false(graph.has_node(40))
