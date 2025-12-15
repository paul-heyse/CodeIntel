"""PR-62: Row model caching keyed by schema signature."""

from __future__ import annotations

import pytest

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.row_models import row_model_for_table_schema


def test_pr62_row_model_cache_keys_on_schema_signature() -> None:
    """Row model generation should cache by schema signature and table identity."""
    schema_v1 = TableSchema(
        schema="analytics",
        name="risk_factors",
        columns=[
            Column(name="repo", type="VARCHAR"),
            Column(name="risk_score", type="DOUBLE"),
        ],
    )
    schema_v2 = TableSchema(
        schema="analytics",
        name="risk_factors",
        columns=[
            Column(name="repo", type="VARCHAR"),
            Column(name="risk_score", type="DOUBLE"),
            Column(name="risk_level", type="VARCHAR"),
        ],
    )

    model1a = row_model_for_table_schema(table_schema=schema_v1)
    model1b = row_model_for_table_schema(table_schema=schema_v1)
    model2 = row_model_for_table_schema(table_schema=schema_v2)

    if model1a is not model1b:
        pytest.fail("Expected identical schema inputs to produce the same cached row model")
    if model1a is model2:
        pytest.fail("Expected schema signature change to produce a different row model")
