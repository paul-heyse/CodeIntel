"""Tests for observed-first schema resolution."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pytest

from codeintel.core.columnar.ipc import schema_to_ipc_payload
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.core.schemas.resolution import SchemaResolutionSource, resolve_table_schema
from codeintel.storage.tracking.schema_catalog_models import SchemaObservationRecord
from tests._helpers.assertions.expectation_assertions import expect_equal


@dataclass(frozen=True)
class _StubObservationProvider:
    observation: SchemaObservationRecord

    def load_latest_schema_observation(
        self,
        *,
        table_key: str,
    ) -> SchemaObservationRecord | None:
        if table_key == self.observation.table_key:
            return self.observation
        return None


def test_resolve_table_schema_prefers_observation() -> None:
    """Observed schemas should override fallback schema providers."""
    table_key = "analytics.demo"
    fallback_schema = TableSchema(
        schema="analytics",
        name="demo",
        columns=[
            Column(name="id", type="BIGINT", nullable=False),
            Column(name="name", type="VARCHAR", nullable=True),
        ],
    )
    observed_schema = pa.schema([("id", pa.int64())])
    observation = SchemaObservationRecord(
        table_key=table_key,
        schema_digest="digest",
        schema_hash="hash",
        arrow_schema_ipc_b64=schema_to_ipc_payload(observed_schema),
    )
    provider = MappingSchemaProvider({table_key: fallback_schema})
    result = resolve_table_schema(
        table_key,
        observation_provider=_StubObservationProvider(observation),
        schema_provider=provider,
    )
    expect_equal(result.source, expected=SchemaResolutionSource.OBSERVED)
    if result.table_schema is None:
        pytest.fail("Expected observed table schema")
    expect_equal(result.table_schema.column_names(), expected=("id",))
