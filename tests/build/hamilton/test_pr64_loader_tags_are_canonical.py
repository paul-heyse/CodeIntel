"""PR-64: Loader node tags use canonical `node_type` values."""

from __future__ import annotations

import pytest

from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.tags import NODE_TYPE_LOADER_QUERY
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle


def _variable_name(variable: object) -> str:
    if isinstance(variable, str):
        return variable
    name = getattr(variable, "name", None)
    return str(name) if name is not None else str(variable)


def test_pr64_loader_tags_are_canonical(hamilton_runtime: HamiltonRuntimeBundle) -> None:
    """q__ nodes should be tagged with loader.* node types."""
    all_names = {_variable_name(var) for var in hamilton_runtime.dr.list_available_variables()}
    q_nodes = {name for name in all_names if name.startswith("q__")}

    q_vars = hamilton_runtime.tag_query.query({ht.TAG_NODE_TYPE: NODE_TYPE_LOADER_QUERY})

    q_tagged = {_variable_name(var) for var in q_vars}

    if q_nodes != q_tagged:
        missing = sorted(q_nodes - q_tagged)
        extra = sorted(q_tagged - q_nodes)
        pytest.fail(f"q__ node tag mismatch missing={missing} extra={extra}")

    for variable in q_vars:
        tags = getattr(variable, "tags", None)
        if not isinstance(tags, dict) or not isinstance(tags.get(ht.TAG_TABLE_KEY), str):
            pytest.fail(f"{_variable_name(variable)} missing table_key tag")
