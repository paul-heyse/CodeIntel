"""Tests for observed-first export validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pyarrow as pa
import pyarrow.parquet as pq

from codeintel.build.exports.validation import validate_export_files
from codeintel.core.columnar.ipc import schema_to_ipc_payload
from codeintel.storage.tracking.schema_catalog_models import SchemaObservationRecord
from tests._helpers.assertions.expectation_assertions import expect_equal

if TYPE_CHECKING:
    from codeintel.storage.gateway.protocol import StorageGateway


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


@dataclass(frozen=True)
class _StubGateway:
    schemas: _StubObservationProvider


def test_validate_export_files_prefers_observation(tmp_path: Path) -> None:
    """Parquet validation should use observed schema when available."""
    table_key = "test.table"
    observed_schema = pa.schema([("id", pa.int64())])
    observation = SchemaObservationRecord(
        table_key=table_key,
        schema_digest="digest",
        schema_hash="hash",
        arrow_schema_ipc_b64=schema_to_ipc_payload(observed_schema),
    )
    gateway = cast(
        "StorageGateway",
        _StubGateway(_StubObservationProvider(observation)),
    )

    path = tmp_path / "observed.parquet"
    table = pa.Table.from_pydict({"id": [1, 2, 3]})
    pq.write_table(table, path)

    result = validate_export_files(table_key, [path], gateway=gateway)
    expect_equal(result, expected=0)
