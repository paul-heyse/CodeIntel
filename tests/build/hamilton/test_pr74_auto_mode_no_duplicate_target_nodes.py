"""Tests for PR-74: Generated module excludes native target nodes in auto mode.

In auto mode, native modules provide the `t__<target>` materialize nodes. The
generated wrapper module must *not* generate those same `t__<target>` nodes to
avoid collisions, while still generating helper nodes for native outputs.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.naming import target_node
from codeintel.build.hamilton.native.registry import native_target_names
from codeintel.build.hamilton.nodes.node_factory import (
    GenerationOptions,
    clear_generated_module_cache,
    get_generated_module,
)


def test_generated_module_skips_target_nodes_for_native_targets() -> None:
    """Verify generated module does not define t__ nodes for native targets."""
    clear_generated_module_cache()

    native = native_target_names()
    module = get_generated_module(
        options=GenerationOptions(exclude_target_nodes_for_targets=native)
    )

    unexpected: list[str] = []
    for target_name in sorted(native):
        node_name = target_node(target_name)
        if hasattr(module, node_name):
            unexpected.append(node_name)

    if unexpected:
        lines = "\n".join(f"- {name}" for name in unexpected)
        pytest.fail(
            f"Generated module unexpectedly defined native target nodes (collision risk):\n{lines}"
        )
