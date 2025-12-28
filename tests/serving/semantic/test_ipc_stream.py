"""IPC streaming tests for the semantic query kernel."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pyarrow as pa
import pytest

from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import SemanticQueryRequest
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

if TYPE_CHECKING:
    from pathlib import Path


def _decode_metadata(metadata: dict[bytes, bytes]) -> dict[str, object]:
    return {
        key.decode("utf-8"): json.loads(value.decode("utf-8")) for key, value in metadata.items()
    }


@pytest.mark.anyio
async def test_query_ipc_stream_includes_metadata_and_rows(tmp_path: Path) -> None:
    """Arrow IPC stream includes schema metadata and expected rows."""
    snapshot = ServingSnapshotFactory(tmp_path).demo_snapshot(row_count=3)

    manager = ServingDBManager(
        pointer_path=snapshot.pointer_path,
        pool_cfg=PoolConfig(size=1),
        poll_interval_s=0.01,
        hot_swap=False,
    )
    await manager.start()
    try:
        kernel = SemanticQueryKernel(
            db=manager,
            settings=ServingSettings(
                serve_dir=snapshot.serve_dir,
                hot_swap=False,
                pool_size=1,
                poll_interval_s=0.01,
                result_engine="polars",
                schema_enforcement="strict",
            ),
        )

        request = SemanticQueryRequest(
            view_id="demo.view",
            filters=[],
            order_by=["id"],
            limit=2,
            offset=0,
        )
        stream = kernel.query_ipc_stream(request)
        data = b"".join(stream)
        expect_true(bool(data), message="Expected IPC stream payload")

        reader = pa.ipc.open_stream(pa.BufferReader(data))
        metadata = _decode_metadata(reader.schema.metadata or {})
        expect_equal(metadata.get("codeintel.table_key"), expected="docs.v_demo")
        expect_true(
            bool(metadata.get("codeintel.snapshot_id")),
            message="Expected snapshot_id metadata",
        )
        expect_true(
            bool(metadata.get("codeintel.query_hash")),
            message="Expected query_hash metadata",
        )
        if "codeintel.schema_hash" in metadata:
            expect_true(
                bool(metadata.get("codeintel.schema_hash")),
                message="Expected schema_hash metadata",
            )

        row_count = sum(batch.num_rows for batch in reader)
        expect_equal(row_count, expected=2)
    finally:
        await manager.stop()
