"""PR-61: Pandera schema generation from TableSchema."""

from __future__ import annotations

import pandas as pd
import pytest

from codeintel.core.schemas.pandera_gen import pandera_schema_from_table_schema
from codeintel.core.schemas.primitives import Column, TableSchema


def test_pr61_pandera_generated_from_table_schema_validates() -> None:
    """Generated Pandera schemas should validate and preserve column order."""
    table_schema = TableSchema(
        schema="test",
        name="example",
        columns=[
            Column(name="repo", type="VARCHAR", nullable=False),
            Column(name="count", type="INTEGER"),
            Column(name="score", type="DOUBLE"),
            Column(name="flag", type="BOOLEAN"),
            Column(name="goid", type="DECIMAL(38,0)"),
        ],
        primary_key=("repo",),
    )
    schema = pandera_schema_from_table_schema(table_key="test.example", table_schema=table_schema)

    df = pd.DataFrame(
        {
            "repo": ["r1"],
            "count": [1],
            "score": [0.5],
            "flag": [True],
            "goid": [123],
        }
    )
    validated = schema.validate(df, lazy=False)
    columns = list(validated.columns)
    expected = ["repo", "count", "score", "flag", "goid"]
    if columns != expected:
        pytest.fail(f"Unexpected validated column order: {columns} != {expected}")
