"""Tests for decision trace serialization."""

from __future__ import annotations

from codeintel.build.hamilton.decision_trace import build_decision_trace_payload
from codeintel.build.hamilton.planner import HamiltonBuildPlan, PlanEntry


def test_decision_trace_payload_sorts_dep_hashes() -> None:
    """Sort dependency hashes within decision trace payloads."""
    entry = PlanEntry(
        target="alpha",
        node="t__alpha",
        module="analytics",
        status="compute",
        reason="forced",
        input_hash="hash-alpha",
        options_hash="opts-alpha",
        prior_input_hash=None,
        dependencies=("beta", "alpha"),
        table_keys=("analytics.alpha",),
        artifact_keys=(),
        dep_hashes={"beta": "2", "alpha": "1"},
        prior_dep_hashes={"zeta": "9", "alpha": "0"},
        impl_kind="native",
    )
    plan = HamiltonBuildPlan(
        requested=("alpha",),
        closure=("alpha",),
        entries=(entry,),
    )

    payload = build_decision_trace_payload(plan)
    dep_hashes = payload[0]["dep_hashes"]
    prior_dep_hashes = payload[0]["prior_dep_hashes"]

    assert list(dep_hashes.keys()) == ["alpha", "beta"]
    assert list(prior_dep_hashes.keys()) == ["alpha", "zeta"]


def test_decision_trace_payload_preserves_entry_order() -> None:
    """Preserve entry order in decision trace payloads."""
    entry_a = PlanEntry(
        target="alpha",
        node="t__alpha",
        module="analytics",
        status="compute",
        reason="forced",
        input_hash="hash-alpha",
        options_hash="opts-alpha",
        prior_input_hash=None,
        dependencies=(),
        table_keys=("analytics.alpha",),
        artifact_keys=(),
        dep_hashes={},
        prior_dep_hashes={},
        impl_kind="native",
    )
    entry_b = PlanEntry(
        target="beta",
        node="t__beta",
        module="graphs",
        status="skip",
        reason="up_to_date",
        input_hash="hash-beta",
        options_hash="opts-beta",
        prior_input_hash="hash-beta",
        dependencies=(),
        table_keys=("graph.beta",),
        artifact_keys=(),
        dep_hashes={},
        prior_dep_hashes={},
        impl_kind="native",
    )
    plan = HamiltonBuildPlan(
        requested=("beta", "alpha"),
        closure=("beta", "alpha"),
        entries=(entry_b, entry_a),
    )

    payload = build_decision_trace_payload(plan)

    assert payload[0]["target"] == "beta"
    assert payload[0]["index"] == 0
    assert payload[1]["target"] == "alpha"
    assert payload[1]["index"] == 1
