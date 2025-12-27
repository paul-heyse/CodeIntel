"""Integration tests for graph feature summarization."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.profiles.graph_features import summarize_graph_for_function_profile
from codeintel.analytics.profiles.types import FunctionProfileInputs
from codeintel.storage.gateway.factory import MemoryGatewayOptions, open_memory_gateway
from tests._helpers.gateway import seed_contract_catalog
from tests._helpers.schemas import ensure_production_schemas

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway


FAN_OUT_TWO = 2
FAN_IN_ZERO = 0
FAN_IN_ONE = 1
GOID_1 = 1
GOID_2 = 2
GOID_3 = 3
GOID_4 = 4
SLOW_TEST_THRESHOLD_MS = 1000.0


def _inputs(gateway: StorageGateway) -> FunctionProfileInputs:
    return FunctionProfileInputs(
        con=gateway.con,
        gateway=gateway,
        repo="r",
        commit="c",
        created_at=datetime.now(tz=UTC),
        slow_test_threshold_ms=SLOW_TEST_THRESHOLD_MS,
    )


def _setup_graph() -> StorageGateway:
    """Create a minimal in-memory gateway with test graph tables.

    This test uses a simplified schema with only the columns needed for
    summarize_graph_for_function_profile, rather than the full production
    schema. This allows focused testing of the graph feature logic without
    requiring all the production column constraints.

    Returns
    -------
    StorageGateway
        Gateway with minimal test graph tables.
    """
    gateway = open_memory_gateway(
        options=MemoryGatewayOptions(
            apply_schema=False,
            ensure_views=False,
            validate_schema=False,
            repo="r",
            commit="c",
        ),
        seed_contract_catalog=seed_contract_catalog,
    )
    con = gateway.con
    ensure_production_schemas(con)
    con.execute("DELETE FROM graph.call_graph_edges")
    con.execute("DELETE FROM graph.call_graph_nodes")
    con.executemany(
        """
        INSERT INTO graph.call_graph_edges (
            repo,
            commit,
            caller_goid_h128,
            callee_goid_h128,
            callsite_path,
            callsite_line,
            callsite_col,
            language,
            kind,
            resolved_via,
            confidence,
            evidence_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            ("r", "c", GOID_1, GOID_2, "pkg/a.py", 1, 1, "python", "call", None, None, None),
            ("r", "c", GOID_1, GOID_3, "pkg/a.py", 2, 1, "python", "call", None, None, None),
            ("r", "c", GOID_4, GOID_2, "pkg/b.py", 1, 1, "python", "call", None, None, None),
        ],
    )
    con.executemany(
        """
        INSERT INTO graph.call_graph_nodes (
            goid_h128,
            language,
            kind,
            arity,
            is_public,
            rel_path
        )
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (GOID_1, "python", "function", 0, True, "pkg/a.py"),
            (GOID_2, "python", "function", 1, False, "pkg/b.py"),
            (GOID_3, "python", "function", 0, True, "pkg/a.py"),
            (GOID_4, "python", "function", 2, False, "pkg/b.py"),
        ],
    )
    return gateway


def test_summarize_graph_for_function_profile_contract() -> None:
    """Graph feature summary should return fan-in/out and role flags per function."""
    gateway = _setup_graph()
    try:
        features = summarize_graph_for_function_profile(_inputs(gateway))
        expected = {
            GOID_1: {
                "fan_in": FAN_IN_ZERO,
                "fan_out": FAN_OUT_TWO,
                "edge_in": FAN_IN_ZERO,
                "edge_out": FAN_OUT_TWO,
                "leaf": False,
                "entry": True,
                "public": True,
            },
            GOID_2: {
                "fan_in": FAN_OUT_TWO,
                "fan_out": FAN_IN_ZERO,
                "edge_in": FAN_OUT_TWO,
                "edge_out": FAN_IN_ZERO,
                "leaf": True,
                "entry": False,
                "public": False,
            },
            GOID_3: {
                "fan_in": FAN_IN_ONE,
                "fan_out": FAN_IN_ZERO,
                "edge_in": FAN_IN_ONE,
                "edge_out": FAN_IN_ZERO,
                "leaf": True,
                "entry": False,
                "public": True,
            },
            GOID_4: {
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
            if (
                feat.call_edge_in_count != exp["edge_in"]
                or feat.call_edge_out_count != exp["edge_out"]
            ):
                msg = f"Edge counts mismatch for {goid}."
                pytest.fail(msg)
            if feat.call_is_leaf is not exp["leaf"] or feat.call_is_entrypoint is not exp["entry"]:
                msg = f"Role flags incorrect for {goid}."
                pytest.fail(msg)
            if feat.call_is_public is not exp["public"]:
                msg = f"Public flag incorrect for {goid}."
                pytest.fail(msg)
    finally:
        gateway.close()
