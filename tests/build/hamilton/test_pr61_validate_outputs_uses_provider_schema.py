"""PR-61: Pandera hook uses SchemaProvider fallback."""

from __future__ import annotations

import pandas as pd
import pytest

from codeintel.build.hamilton.contracts.pandera_hook import get_pandera_schema
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider


def test_pr61_get_pandera_schema_falls_back_to_schema_provider() -> None:
    """get_pandera_schema should fall back to SchemaProvider when registry misses."""
    table_schema = TableSchema(
        schema="test",
        name="synthetic",
        columns=[
            Column(name="repo", type="VARCHAR", nullable=False),
            Column(name="value", type="INTEGER"),
        ],
        primary_key=("repo",),
    )
    provider = MappingSchemaProvider({"test.synthetic": table_schema})

    schema = get_pandera_schema("test.synthetic", schema_provider=provider)
    if schema is None:
        pytest.fail("Expected schema_provider fallback to produce a Pandera schema")

    df = pd.DataFrame({"repo": ["r1"], "value": [1]})
    schema.validate(df, lazy=False)
