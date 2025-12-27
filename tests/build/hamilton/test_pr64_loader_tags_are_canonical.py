"""PR-64: Loader node tags use canonical `node_type` values."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.tags import NODE_TYPE_LOADER_DATAFRAME, NODE_TYPE_LOADER_QUERY


def _variable_name(variable: object) -> str:
    if isinstance(variable, str):
        return variable
    name = getattr(variable, "name", None)
    return str(name) if name is not None else str(variable)


def test_pr64_loader_tags_are_canonical() -> None:
    """q__/df__ nodes should be tagged with loader.* node types."""
    runtime = build_driver()

    all_names = {_variable_name(var) for var in runtime.dr.list_available_variables()}
    q_nodes = {name for name in all_names if name.startswith("q__")}
    df_nodes = {name for name in all_names if name.startswith("df__")}

    q_vars = runtime.tag_query.query({ht.TAG_NODE_TYPE: NODE_TYPE_LOADER_QUERY})
    df_vars = runtime.tag_query.query({ht.TAG_NODE_TYPE: NODE_TYPE_LOADER_DATAFRAME})

    q_tagged = {_variable_name(var) for var in q_vars}
    df_tagged = {_variable_name(var) for var in df_vars}

    if q_nodes != q_tagged:
        missing = sorted(q_nodes - q_tagged)
        extra = sorted(q_tagged - q_nodes)
        pytest.fail(f"q__ node tag mismatch missing={missing} extra={extra}")

    if df_nodes != df_tagged:
        missing = sorted(df_nodes - df_tagged)
        extra = sorted(df_tagged - df_nodes)
        pytest.fail(f"df__ node tag mismatch missing={missing} extra={extra}")

    for variable in q_vars:
        tags = getattr(variable, "tags", None)
        if not isinstance(tags, dict) or not isinstance(tags.get(ht.TAG_TABLE_KEY), str):
            pytest.fail(f"{_variable_name(variable)} missing table_key tag")

    for variable in df_vars:
        tags = getattr(variable, "tags", None)
        if not isinstance(tags, dict) or not isinstance(tags.get(ht.TAG_TABLE_KEY), str):
            pytest.fail(f"{_variable_name(variable)} missing table_key tag")
