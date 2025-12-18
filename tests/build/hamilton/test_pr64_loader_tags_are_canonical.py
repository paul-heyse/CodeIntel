"""PR-64: Loader node tags use canonical `node_type` values."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.core.hamilton.tags import NODE_TYPE_LOADER_DATAFRAME, NODE_TYPE_LOADER_QUERY


def test_pr64_loader_tags_are_canonical() -> None:
    """q__/df__ nodes should be tagged with loader.* node types."""
    runtime = build_driver()

    for node_name, node in runtime.dr.graph.nodes.items():
        tags = node.tags
        node_type = tags.get("node_type")

        if node_name.startswith("q__"):
            if node_type != NODE_TYPE_LOADER_QUERY:
                pytest.fail(
                    f"{node_name} expected node_type={NODE_TYPE_LOADER_QUERY}, got {node_type}"
                )
            if not isinstance(tags.get("table_key"), str):
                pytest.fail(f"{node_name} missing table_key tag")
        elif node_name.startswith("df__"):
            if node_type != NODE_TYPE_LOADER_DATAFRAME:
                pytest.fail(
                    f"{node_name} expected node_type={NODE_TYPE_LOADER_DATAFRAME}, got {node_type}"
                )
            if not isinstance(tags.get("table_key"), str):
                pytest.fail(f"{node_name} missing table_key tag")
