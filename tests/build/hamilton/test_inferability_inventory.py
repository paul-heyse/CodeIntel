"""Tests for inferability inventory guardrails."""

from __future__ import annotations

from codeintel.build.schemas.inference_service import inferability_inventory
from codeintel.core.schemas.output_registry import NON_INFERABLE_OUTPUT_KEYS
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle
from tests._helpers.assertions.expectation_assertions import expect_true


def _saver_nodes(runtime: HamiltonRuntimeBundle) -> set[str]:
    return {output.saver_node for output in runtime.catalog.table_outputs.values()}


def test_inferability_inventory_marks_inferable_outputs(
    hamilton_runtime: HamiltonRuntimeBundle,
) -> None:
    """Inferability inventory should flag all inferable outputs as inferable."""
    records = inferability_inventory(
        driver=hamilton_runtime.driver,
        catalog=hamilton_runtime.catalog,
    )
    failures = [
        (record.table_key, record.reason)
        for record in records
        if record.table_key not in NON_INFERABLE_OUTPUT_KEYS and record.status != "inferable"
    ]
    expect_true(
        not failures,
        message=f"Non-inferable outputs detected: {failures}",
    )


def test_inferable_outputs_do_not_depend_on_saver_nodes(
    hamilton_runtime: HamiltonRuntimeBundle,
) -> None:
    """Inferable outputs should not use saver nodes as compute dependencies."""
    records = inferability_inventory(
        driver=hamilton_runtime.driver,
        catalog=hamilton_runtime.catalog,
    )
    saver_nodes = _saver_nodes(hamilton_runtime)
    violations: list[str] = []
    for record in records:
        if record.table_key in NON_INFERABLE_OUTPUT_KEYS:
            continue
        compute_node = record.compute_node
        if compute_node is None:
            violations.append(f"{record.table_key}: missing compute node")
            continue
        if compute_node in saver_nodes:
            violations.append(f"{record.table_key}: compute node is a saver ({compute_node})")
            continue
        node = hamilton_runtime.catalog.nodes.get(compute_node)
        if node is None:
            violations.append(f"{record.table_key}: compute node missing ({compute_node})")
            continue
        upstream_savers = [dep for dep in node.deps if dep in saver_nodes]
        if upstream_savers:
            joined = ", ".join(sorted(upstream_savers))
            violations.append(f"{record.table_key}: depends on saver nodes ({joined})")
    expect_true(
        not violations,
        message=f"Saver node dependencies detected: {violations}",
    )
