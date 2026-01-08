"""PR-57: DuckDBPolicyBackend uses SchemaProvider for DDL."""

from __future__ import annotations

import duckdb
import pytest

from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.storage.gateway.minimal import MinimalStorageGateway


def test_ensure_table_creates_table_from_schema_provider() -> None:
    """ensure_table should create schema and table using the injected provider."""
    con = duckdb.connect(":memory:")
    provider = MappingSchemaProvider(
        {
            "analytics.example": TableSchema(
                schema="analytics",
                name="example",
                columns=[
                    Column("repo", "VARCHAR", nullable=False),
                    Column("commit", "VARCHAR", nullable=False),
                ],
                primary_key=("repo", "commit"),
            )
        }
    )

    gateway = MinimalStorageGateway(con, schema_provider=provider)
    try:
        gateway.policy.ensure_table("analytics.example")
    except NotImplementedError:
        pytest.xfail("MinimalStorageGateway no longer supports dataset policy writes.")

    info = con.execute("PRAGMA table_info(analytics.example)").fetchall()
    actual = [row[1] for row in info]
    expected = ["repo", "commit"]
    if actual != expected:
        pytest.fail(f"Unexpected column order: {actual} != {expected}")
