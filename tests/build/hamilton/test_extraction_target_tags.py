"""Tests for template-generated extraction target nodes."""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.native.ingestion.extraction_targets import (
    AST_METRICS_TABLE_KEY,
    AST_NODES_TABLE_KEY,
    CST_NODES_TABLE_KEY,
    DOCSTRINGS_TABLE_KEY,
)
from codeintel.core.hamilton import tags as ht
from codeintel.runtime.runtime_bundle import RuntimeBundle


def _variable_name(variable: object) -> str:
    name = getattr(variable, "name", None)
    return str(name) if name is not None else str(variable)


def test_extraction_table_nodes_have_dataset_tags(
    hamilton_runtime: RuntimeBundle,
) -> None:
    """Ensure extraction table nodes are tagged as datasets with correct table keys."""
    variables = {
        _variable_name(variable): variable
        for variable in hamilton_runtime.dr.list_available_variables()
    }
    expected = {
        "ast__node_rows": AST_NODES_TABLE_KEY,
        "ast__metric_rows": AST_METRICS_TABLE_KEY,
        "cst__node_rows": CST_NODES_TABLE_KEY,
        "docstrings__rows": DOCSTRINGS_TABLE_KEY,
    }
    missing = [name for name in expected if name not in variables]
    if missing:
        pytest.fail("Missing extraction table nodes: " + ", ".join(sorted(missing)))

    for node_name, table_key in expected.items():
        tags = getattr(variables[node_name], "tags", None)
        if not isinstance(tags, dict):
            pytest.fail(f"{node_name} missing tag metadata")
        if tags.get(ht.TAG_NODE_TYPE) != ht.NODE_TYPE_DATASET:
            pytest.fail(f"{node_name} missing node_type={ht.NODE_TYPE_DATASET}")
        if tags.get(ht.TAG_TABLE_KEY) != table_key:
            pytest.fail(f"{node_name} missing table_key={table_key}")
