"""Tests for PR-74: Auto mode helper nodes for native outputs.

PR-74 ensures that auto-mode driver composition emits d__/q__/df__/a__ helper nodes
for native targets, enabling native->native composition through the same loader
conventions used by wrapper targets.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.impl_kind import native_target_names
from codeintel.build.hamilton.naming import artifact_node, dataset_node, query_node

_MAX_MISSING_LINES: int = 100


def test_auto_mode_native_outputs_have_helpers() -> None:
    """Verify auto mode exposes helper nodes for native outputs."""
    runtime = build_driver()
    node_names = set(runtime.dr.graph.nodes.keys())

    missing: list[str] = []
    for target_name in sorted(native_target_names(runtime)):
        target = runtime.graph.get(target_name)
        if target is None:
            pytest.fail(f"Target not found in runtime graph: {target_name}")

        for table_key in target.contract.table_keys:
            d_name = dataset_node(table_key)
            q_name = query_node(table_key)
            if d_name not in node_names:
                missing.append(f"{target_name}: missing dataset node {d_name} for {table_key}")
            if q_name not in node_names:
                missing.append(f"{target_name}: missing loader node {q_name} for {table_key}")

        for artifact_name in target.contract.artifact_names:
            a_name = artifact_node(artifact_name)
            if a_name not in node_names:
                missing.append(f"{target_name}: missing artifact node {a_name} for {artifact_name}")

    if missing:
        summary = "\n".join(missing[:_MAX_MISSING_LINES])
        extra = (
            ""
            if len(missing) <= _MAX_MISSING_LINES
            else f"\n... +{len(missing) - _MAX_MISSING_LINES} more"
        )
        pytest.fail(
            f"Auto mode is missing required helper nodes for native outputs:\n{summary}{extra}"
        )
