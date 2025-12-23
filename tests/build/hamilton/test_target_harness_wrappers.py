"""Sanity checks for target-family harness wrappers."""

from __future__ import annotations

from pathlib import Path

from tests._helpers.assertions import assert_target_ok, expect_true
from tests._helpers.hamilton_manifest_priming import ManifestPriming
from tests._helpers.harnesses.analytics_harness import AnalyticsTargetHarness
from tests._helpers.harnesses.graph_harness import GraphTargetHarness
from tests._helpers.harnesses.serving_harness import ServingTargetHarness
from tests._helpers.manifests import assert_skipped, compute_input_hash


def test_graph_target_harness_runs(graph_target_harness: GraphTargetHarness) -> None:
    """Run graph targets using the graph harness.

    Parameters
    ----------
    graph_target_harness
        Graph target harness fixture.
    """
    records = graph_target_harness.run_targets()
    record = records["call_graph"]
    assert_target_ok(record)
    graph_target_harness.assert_call_graph_datasets(record)
    graph_target_harness.assert_graph_tables(min_rows=1)


def test_graph_target_harness_skips_with_primed_manifest(
    graph_target_harness: GraphTargetHarness,
) -> None:
    """Prime call_graph manifest so the target skips cleanly."""
    harness = graph_target_harness.harness
    input_hash, options_hash = compute_input_hash(harness, "call_graph")
    harness.priming.prime_manifest(
        ManifestPriming.ManifestSpec(
            target="call_graph",
            input_hash=input_hash,
            options_hash=options_hash,
            row_count=0,
        )
    )
    records = harness.run_targets(["call_graph"])
    record = harness.record("call_graph", result=records)
    assert_skipped(record)


def test_analytics_target_harness_runs(
    analytics_target_harness: AnalyticsTargetHarness,
) -> None:
    """Run analytics targets using the analytics harness.

    Parameters
    ----------
    analytics_target_harness
        Analytics target harness fixture.
    """
    records = analytics_target_harness.run_targets()
    record = records["function_metrics"]
    assert_target_ok(record)
    analytics_target_harness.assert_function_metrics(min_rows=1)


def test_serving_target_harness_publishes_snapshot(
    serving_target_harness: ServingTargetHarness,
) -> None:
    """Publish a serving snapshot using the serving harness.

    Parameters
    ----------
    serving_target_harness
        Serving target harness fixture.
    """
    records = serving_target_harness.run_targets()
    record = records["serving_artifacts"]
    assert_target_ok(record)
    serving_target_harness.assert_artifacts_exist()
    manifest = serving_target_harness.publish_snapshot(run_id="test-serving")
    expect_true(Path(manifest.db_path).is_file(), message="Expected serving DB path to exist.")
    expect_true(
        Path(manifest.semantic_registry_path).is_file(),
        message="Expected semantic registry path to exist.",
    )


def test_serving_target_harness_skips_with_primed_manifest(
    serving_target_harness: ServingTargetHarness,
) -> None:
    """Prime serving_artifacts manifest so the target skips cleanly."""
    harness = serving_target_harness.harness
    input_hash, options_hash = compute_input_hash(harness, "serving_artifacts")
    harness.priming.prime_manifest(
        ManifestPriming.ManifestSpec(
            target="serving_artifacts",
            input_hash=input_hash,
            options_hash=options_hash,
            row_count=0,
        )
    )
    records = harness.run_targets(["serving_artifacts"])
    record = harness.record("serving_artifacts", result=records)
    assert_skipped(record)
