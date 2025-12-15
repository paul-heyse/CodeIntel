"""PR-64: Node tags are consistent (`node_type`)."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver


def test_pr64_all_nodes_have_node_type_tag() -> None:
    """All Hamilton nodes should have a canonical node_type tag."""
    runtime = build_driver(mode="auto")

    missing: list[str] = []
    for node_name, node in runtime.dr.graph.nodes.items():
        if node.user_defined:
            continue
        tags = node.tags
        node_type = tags.get("node_type")
        if not isinstance(node_type, str) or not node_type:
            missing.append(node_name)
        if "node_kind" in tags:
            missing.append(node_name)

    if missing:
        pytest.fail(f"Nodes missing canonical node_type tags: {sorted(set(missing))}")
