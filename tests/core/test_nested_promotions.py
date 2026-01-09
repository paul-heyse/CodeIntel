"""Tests for nested schema promotion guardrails."""

from __future__ import annotations

import pyarrow as pa
import pytest

from codeintel.core.columnar.finalize_ops import FinalizeSpec, finalize_table
from codeintel.core.columnar.nested_ops import deep_cast_table_to_contract
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.core.schemas.service import (
    SchemaService,
    clear_schema_service,
    get_schema_service,
    set_schema_service,
)


def _install_schema_service() -> SchemaService | None:
    try:
        return get_schema_service()
    except RuntimeError:
        return None


def _reset_schema_service(previous: SchemaService | None) -> None:
    if previous is None:
        clear_schema_service()
        return
    set_schema_service(previous)


def test_deep_cast_rejects_disallowed_list_promotion() -> None:
    table = pa.table({"items": [[1, 2], [3]]})
    contract_schema = pa.schema([pa.field("items", pa.list_(pa.string()))])

    with pytest.raises(ValueError, match="Disallowed promotion"):
        deep_cast_table_to_contract(table, contract_schema)


def test_finalize_emits_nested_cast_failed() -> None:
    table_schema = TableSchema(
        schema="core",
        name="nested_cast_guardrail",
        columns=[Column("items", "LIST(VARCHAR)", nullable=True)],
    )
    provider = MappingSchemaProvider({table_schema.table_key: table_schema})
    previous = _install_schema_service()
    set_schema_service(SchemaService(table_provider=provider))
    try:
        table = pa.table({"items": [[1, 2], [3]]})
        result = finalize_table(
            table,
            spec=FinalizeSpec(
                table_key=table_schema.table_key,
                mode="tolerant",
                emit_artifacts=True,
            ),
        )
        codes = result.errors.column("error_code").to_pylist()
        assert codes == ["NESTED_CAST_FAILED", "NESTED_CAST_FAILED"]
    finally:
        _reset_schema_service(previous)
