"""Integration tests for graph feature summarization."""

from __future__ import annotations

from datetime import UTC, datetime

import duckdb
import pytest

from codeintel.analytics.profiles.graph_features import summarize_graph_for_function_profile
from codeintel.analytics.profiles.types import FunctionProfileInputs

FAN_OUT_TWO = 2
FAN_IN_ZERO = 0
FAN_IN_ONE = 1


def _inputs(con: duckdb.DuckDBPyConnection) -> FunctionProfileInputs:
    return FunctionProfileInputs(
        con=con,
        repo="r",
        commit="c",
        created_at=datetime.now(tz=UTC),
        slow_test_threshold_ms=1000.0,
    )


def _setup_graph() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database=":memory:")
    con.execute("CREATE SCHEMA graph")
    con.execute(
        """
        CREATE TABLE graph.call_graph_edges (
            caller_goid_h128 BIGINT,
            callee_goid_h128 BIGINT,
            repo VARCHAR,
            commit VARCHAR
        )
        """
    )
    con.execute(
        """
        CREATE TABLE graph.call_graph_nodes (
            goid_h128 BIGINT,
            is_public BOOLEAN
        )
        """
    )
    con.executemany(
        "INSERT INTO graph.call_graph_edges VALUES (?, ?, ?, ?)",
        [
            (1, 2, "r", "c"),
            (1, 3, "r", "c"),
            (4, 2, "r", "c"),
        ],
    )
    con.executemany(
        "INSERT INTO graph.call_graph_nodes VALUES (?, ?)",
        [
            (1, True),
            (2, False),
            (3, True),
            (4, False),
        ],
    )
    return con


def test_summarize_graph_for_function_profile_contract() -> None:
    """Graph feature summary should return fan-in/out and role flags per function."""
    con = _setup_graph()
    features = summarize_graph_for_function_profile(_inputs(con))

    expected = {
        1: {
            "fan_in": FAN_IN_ZERO,
            "fan_out": FAN_OUT_TWO,
            "edge_in": FAN_IN_ZERO,
            "edge_out": FAN_OUT_TWO,
            "leaf": False,
            "entry": True,
            "public": True,
        },
        2: {
            "fan_in": FAN_OUT_TWO,
            "fan_out": FAN_IN_ZERO,
            "edge_in": FAN_OUT_TWO,
            "edge_out": FAN_IN_ZERO,
            "leaf": True,
            "entry": False,
            "public": False,
        },
        3: {
            "fan_in": FAN_IN_ONE,
            "fan_out": FAN_IN_ZERO,
            "edge_in": FAN_IN_ONE,
            "edge_out": FAN_IN_ZERO,
            "leaf": True,
            "entry": False,
            "public": True,
        },
        4: {
            "fan_in": FAN_IN_ZERO,
            "fan_out": FAN_IN_ONE,
            "edge_in": FAN_IN_ZERO,
            "edge_out": FAN_IN_ONE,
            "leaf": False,
            "entry": True,
            "public": False,
        },
    }

    if set(features) != set(expected):
        msg = "Function graph features missing expected GOIDs."
        pytest.fail(msg)

    for goid, exp in expected.items():
        feat = features[goid]
        if feat.call_fan_in != exp["fan_in"] or feat.call_fan_out != exp["fan_out"]:
            msg = f"Fan-in/out mismatch for {goid}."
            pytest.fail(msg)
        if feat.call_edge_in_count != exp["edge_in"] or feat.call_edge_out_count != exp["edge_out"]:
            msg = f"Edge counts mismatch for {goid}."
            pytest.fail(msg)
        if feat.call_is_leaf is not exp["leaf"] or feat.call_is_entrypoint is not exp["entry"]:
            msg = f"Role flags incorrect for {goid}."
            pytest.fail(msg)
        if feat.call_is_public is not exp["public"]:
            msg = f"Public flag incorrect for {goid}."
            pytest.fail(msg)
