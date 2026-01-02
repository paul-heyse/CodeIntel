"""Integration tests for graph feature summarization."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from codeintel.build.analytics.profiles.graph_features import summarize_graph_for_function_profile
from codeintel.build.analytics.profiles.types import FunctionProfileInputs

FAN_OUT_TWO = 2
FAN_IN_ZERO = 0
FAN_IN_ONE = 1
GOID_1 = 1
GOID_2 = 2
GOID_3 = 3
GOID_4 = 4
SLOW_TEST_THRESHOLD_MS = 1000.0


def _call_graph_edges_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "repo": "r",
                "commit": "c",
                "caller_goid_h128": GOID_1,
                "callee_goid_h128": GOID_2,
                "callsite_path": "pkg/a.py",
                "callsite_line": 1,
                "callsite_col": 1,
                "language": "python",
                "kind": "call",
                "resolved_via": None,
                "confidence": None,
                "evidence_json": None,
            },
            {
                "repo": "r",
                "commit": "c",
                "caller_goid_h128": GOID_1,
                "callee_goid_h128": GOID_3,
                "callsite_path": "pkg/a.py",
                "callsite_line": 2,
                "callsite_col": 1,
                "language": "python",
                "kind": "call",
                "resolved_via": None,
                "confidence": None,
                "evidence_json": None,
            },
            {
                "repo": "r",
                "commit": "c",
                "caller_goid_h128": GOID_4,
                "callee_goid_h128": GOID_2,
                "callsite_path": "pkg/b.py",
                "callsite_line": 1,
                "callsite_col": 1,
                "language": "python",
                "kind": "call",
                "resolved_via": None,
                "confidence": None,
                "evidence_json": None,
            },
        ]
    )


def _call_graph_nodes_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "goid_h128": GOID_1,
                "language": "python",
                "kind": "function",
                "arity": 0,
                "is_public": True,
                "rel_path": "pkg/a.py",
            },
            {
                "goid_h128": GOID_2,
                "language": "python",
                "kind": "function",
                "arity": 1,
                "is_public": False,
                "rel_path": "pkg/b.py",
            },
            {
                "goid_h128": GOID_3,
                "language": "python",
                "kind": "function",
                "arity": 0,
                "is_public": True,
                "rel_path": "pkg/a.py",
            },
            {
                "goid_h128": GOID_4,
                "language": "python",
                "kind": "function",
                "arity": 2,
                "is_public": False,
                "rel_path": "pkg/b.py",
            },
        ]
    )


def _inputs() -> FunctionProfileInputs:
    empty = pl.DataFrame()
    return FunctionProfileInputs(
        repo="r",
        commit="c",
        created_at=datetime.now(tz=UTC),
        slow_test_threshold_ms=SLOW_TEST_THRESHOLD_MS,
        function_metrics=empty,
        function_types=empty,
        modules=empty,
        typedness=empty,
        diagnostics=empty,
        goid_risk_factors=empty,
        graph_metrics_functions=empty,
        function_effects=empty,
        function_contracts=empty,
        semantic_roles_functions=empty,
        docstrings=empty,
        hotspots=empty,
        call_graph_edges=_call_graph_edges_frame(),
        call_graph_nodes=_call_graph_nodes_frame(),
    )


def test_summarize_graph_for_function_profile_contract() -> None:
    """Graph feature summary should return fan-in/out and role flags per function."""
    features = summarize_graph_for_function_profile(_inputs())
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
        if feat.call_edge_in_count != exp["edge_in"] or feat.call_edge_out_count != exp["edge_out"]:
            msg = f"Edge counts mismatch for {goid}."
            pytest.fail(msg)
        if feat.call_is_leaf is not exp["leaf"] or feat.call_is_entrypoint is not exp["entry"]:
            msg = f"Role flags incorrect for {goid}."
            pytest.fail(msg)
        if feat.call_is_public is not exp["public"]:
            msg = f"Public flag incorrect for {goid}."
            pytest.fail(msg)
