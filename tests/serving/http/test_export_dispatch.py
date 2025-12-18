"""Unit tests for HTTP export dispatch helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pytest

from codeintel.serving.http.export_dispatch import ExportMetricsContext, dispatch_semantic_export
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticExportRequest
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


@dataclass(frozen=True, slots=True)
class _FakeOps:
    rows_written: int

    def export_to_parquet(self, _request: SemanticExportRequest, *, output_path: Path) -> int:
        output_path.write_bytes(b"fake-parquet")
        return self.rows_written

    def export_to_arrow_ipc(self, _request: SemanticExportRequest, *, output_path: Path) -> int:
        output_path.write_bytes(b"fake-arrow")
        return self.rows_written

    def export_rows(self, _request: SemanticExportRequest) -> list[dict[str, object]]:
        return []


@pytest.mark.anyio
async def test_http_export_dispatch_returns_row_count_for_parquet() -> None:
    fake = _FakeOps(rows_written=7)
    payload = SemanticExportRequest(view_id="demo.view", format="parquet", limit=10)
    metrics = ExportMetricsContext(
        view_id="demo.view",
        correlation_id="corr-1",
        query_hash="q_123",
        schema_hash=None,
    )
    dispatched = await dispatch_semantic_export(
        cast("ServingOperations", fake),
        payload,
        metrics,
        headers={},
    )
    expect_equal(dispatched.metrics_row_count, 7)
    expect_true(hasattr(dispatched.response, "headers"))

