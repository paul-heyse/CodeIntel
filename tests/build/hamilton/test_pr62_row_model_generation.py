"""PR-62: Row model generation from TableSchema."""

from __future__ import annotations

import pytest

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.row_models import row_model_for_table_schema


def test_pr62_row_model_generation_has_expected_fields_and_name() -> None:
    """Generated row models should have a stable name and field set."""
    table_schema = TableSchema(
        schema="analytics",
        name="risk_factors",
        columns=[
            Column(name="repo", type="VARCHAR", nullable=False),
            Column(name="commit", type="VARCHAR", nullable=False),
            Column(name="risk_score", type="DOUBLE"),
        ],
    )

    model = row_model_for_table_schema(table_schema=table_schema)

    if model.__name__ != "Analytics__risk_factors__Row":
        pytest.fail(f"Unexpected row model name: {model.__name__}")
    fields = set(model.__annotations__.keys())
    expected_fields = {"repo", "commit", "risk_score"}
    if fields != expected_fields:
        pytest.fail(f"Unexpected row model fields: {fields} != {expected_fields}")
