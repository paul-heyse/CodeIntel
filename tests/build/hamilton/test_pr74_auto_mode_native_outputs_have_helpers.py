"""Tests for PR-74: Auto mode helper nodes for native outputs.

PR-74 ensures that auto-mode driver composition emits d__/q__/a__ helper nodes
for native targets, enabling native->native composition through consistent loader
conventions.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.impl_kind import native_target_names
from codeintel.build.hamilton.naming import artifact_node, dataset_node, query_node
from codeintel.runtime.runtime_bundle import RuntimeBundle

_MAX_MISSING_LINES: int = 100


def test_auto_mode_native_outputs_have_helpers(hamilton_runtime: RuntimeBundle) -> None:
    """Verify auto mode exposes helper nodes for native outputs."""
    node_names = set(hamilton_runtime.dr.graph.nodes.keys())

    missing: list[str] = []
    for target_name in sorted(native_target_names(hamilton_runtime)):
        target = hamilton_runtime.catalog.get_target(target_name)
        if target is None:
            pytest.fail(f"Target not found in catalog: {target_name}")

        for output in hamilton_runtime.catalog.table_outputs_by_target.get(target_name, ()):
            table_key = output.key
            d_name = dataset_node(table_key)
            q_name = query_node(table_key)
            if d_name not in node_names:
                missing.append(f"{target_name}: missing dataset node {d_name} for {table_key}")
            if q_name not in node_names:
                missing.append(f"{target_name}: missing loader node {q_name} for {table_key}")

        for output in hamilton_runtime.catalog.artifact_outputs_by_target.get(target_name, ()):
            a_name = artifact_node(output.key)
            if a_name not in node_names:
                missing.append(f"{target_name}: missing artifact node {a_name} for {output.key}")

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
