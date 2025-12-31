"""Tests for observed-first JSON schema generation."""

from __future__ import annotations

from dataclasses import dataclass

import pyarrow as pa
import pytest

from codeintel.core.columnar.ipc import schema_to_ipc_payload
from codeintel.core.schemas.primitives import Column, TableSchema
from codeintel.core.schemas.provider import MappingSchemaProvider
from codeintel.core.schemas.resolution import ResolvedArrowSchemaProvider, ResolvedSchemaProvider
from codeintel.core.schemas.service import SchemaService
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


def test_get_json_schema_prefers_observation() -> None:
    """JSON Schema generation should reflect observed schemas when available."""
    table_key = "core.modules"
    fallback_schema = TableSchema(
        schema="core",
        name="modules",
        columns=[
            Column(name="module", type="VARCHAR", nullable=False),
            Column(name="path", type="VARCHAR", nullable=False),
        ],
    )
    observed_schema = pa.schema([("module", pa.string())])
    observation = SchemaObservationRecord(
        table_key=table_key,
        schema_digest="digest",
        schema_hash="hash",
        arrow_schema_ipc_b64=schema_to_ipc_payload(observed_schema),
    )
    provider = MappingSchemaProvider({table_key: fallback_schema})
    resolved_provider = ResolvedSchemaProvider(
        observation_provider=_StubObservationProvider(observation),
        fallback_provider=provider,
    )
    arrow_provider = ResolvedArrowSchemaProvider(
        observation_provider=_StubObservationProvider(observation),
        fallback_provider=provider,
    )
    service = SchemaService(
        table_provider=resolved_provider,
        arrow_provider=arrow_provider,
    )
    schema = service.get_json_schema(table_key)
    if schema is None:
        pytest.fail("Expected JSON Schema to be generated")
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        pytest.fail("Expected JSON Schema properties")
    expect_equal(set(properties), expected={"module"})
