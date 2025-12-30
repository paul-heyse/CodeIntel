"""PR-64: Loader node tags use canonical `node_type` values."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.naming import query_node
from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.tags import NODE_TYPE_LOADER_QUERY
from codeintel.runtime.runtime_bundle import RuntimeBundle


def _variable_name(variable: object) -> str:
    if isinstance(variable, str):
        return variable
    name = getattr(variable, "name", None)
    return str(name) if name is not None else str(variable)


def test_pr64_loader_tags_are_canonical(hamilton_runtime: RuntimeBundle) -> None:
    """q__ nodes should be tagged with loader.* node types."""
    variables = list(hamilton_runtime.dr.list_available_variables())
    q_nodes = {_variable_name(var) for var in variables if _variable_name(var).startswith("q__")}
    expected_tagged = {
        query_node(table_key) for table_key in hamilton_runtime.catalog.table_outputs
    } & q_nodes

    q_vars = hamilton_runtime.tag_query.query({ht.TAG_NODE_TYPE: NODE_TYPE_LOADER_QUERY})

    q_tagged = {_variable_name(var) for var in q_vars}

    if expected_tagged != q_tagged:
        missing = sorted(expected_tagged - q_tagged)
        extra = sorted(q_tagged - expected_tagged)
        pytest.fail(f"q__ node tag mismatch missing={missing} extra={extra}")

    for variable in q_vars:
        tags = getattr(variable, "tags", None)
        if not isinstance(tags, dict) or not isinstance(tags.get(ht.TAG_TABLE_KEY), str):
            pytest.fail(f"{_variable_name(variable)} missing table_key tag")
