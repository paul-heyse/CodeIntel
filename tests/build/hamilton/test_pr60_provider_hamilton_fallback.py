"""PR-60: HamiltonSchemaProvider fallback behavior."""

from __future__ import annotations

import pytest

from codeintel.build.schemas.inference_service import HamiltonSchemaProvider
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider


def test_pr60_provider_hamilton_prefers_inferred_for_inferable_keys() -> None:
    """Inferable table keys should use the inferer; others should use declared schemas."""
    declared_native = TableSchema(
        schema="analytics",
        name="native_table",
        columns=[Column(name="a", type="INTEGER")],
    )
    declared_wrapper = TableSchema(
        schema="analytics",
        name="wrapper_table",
        columns=[Column(name="x", type="VARCHAR")],
    )
    declared = MappingSchemaProvider(
        {
            "analytics.native_table": declared_native,
            "analytics.wrapper_table": declared_wrapper,
        }
    )

    inferred_native = TableSchema(
        schema="analytics",
        name="native_table",
        columns=[Column(name="a", type="INTEGER"), Column(name="b", type="VARCHAR")],
    )

    def inferer(table_key: str) -> TableSchema:
        if table_key != "analytics.native_table":
            msg = f"Unexpected inference request: {table_key}"
            raise KeyError(msg)
        return inferred_native

    provider = HamiltonSchemaProvider(
        declared=declared,
        inferer=inferer,
        inferable_table_keys=frozenset({"analytics.native_table"}),
    )

    native_schema = provider.require_table_schema("analytics.native_table")
    if native_schema != inferred_native:
        pytest.fail("Expected inferred schema for inferable table key")

    wrapper_schema = provider.require_table_schema("analytics.wrapper_table")
    if wrapper_schema != declared_wrapper:
        pytest.fail("Expected declared schema for non-inferable table key")
