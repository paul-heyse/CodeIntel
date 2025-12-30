"""Scan metrics tests for semantic queries."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.core.iceberg.catalog import IcebergCatalogProvider
from codeintel.serving.db.manager import ServingDBManager
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.semantic.iceberg_scans import resolve_iceberg_ref
from codeintel.serving.semantic.kernel import SemanticQueryKernel
from codeintel.serving.semantic.models import SemanticQueryRequest
from codeintel.serving.settings import ServingSettings
from codeintel.storage.gateway.pool import PoolConfig
from codeintel.storage.iceberg.stats import iceberg_stats_for_table
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_is_not_none,
)
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.anyio
async def test_query_scan_metrics_from_iceberg(tmp_path: Path) -> None:
    """Query scan metrics should reflect Iceberg stats."""
    snapshot = ServingSnapshotFactory(tmp_path).demo_snapshot(row_count=4)
    pointer = ServingSnapshotPointer.load(snapshot.pointer_path)
    provider = IcebergCatalogProvider(snapshot.iceberg_settings)
    table = provider.load_table("docs.v_demo")
    ref = resolve_iceberg_ref(pointer=pointer, settings=snapshot.iceberg_settings)
    snapshot_id = None
    if ref:
        tagged = table.snapshot_by_name(ref)
        if tagged is not None:
            snapshot_id = tagged.snapshot_id
    stats = iceberg_stats_for_table(table, snapshot_id=snapshot_id)

    expected_row_count = stats.get("total_records")
    expected_file_count = stats.get("data_file_count")
    expected_total_bytes = stats.get("total_bytes")

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
                dataset_scan_metrics_enabled=True,
                iceberg=snapshot.iceberg_settings,
            ),
        )
        result = kernel.query(SemanticQueryRequest(view_id="demo.view", limit=2))
        metrics = expect_is_not_none(result.scan_metrics)
        expect_equal(metrics.row_count, expected_row_count)
        expect_equal(metrics.file_count, expected_file_count)
        expect_equal(metrics.total_bytes, expected_total_bytes)
        expect_equal(metrics.scan_source, "iceberg")
        expect_equal(metrics.pushdown_coverage, None)
    finally:
        await manager.stop()
