"""Unit tests for HTTP export dispatch helpers."""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pytest
from starlette.responses import StreamingResponse

from codeintel.serving.export.formats import mime_type_for_export_format
from codeintel.serving.http.export_dispatch import (
    ExportDispatchOptions,
    ExportMetricsContext,
    dispatch_semantic_export,
)
from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.semantic.models import SemanticExportRequest
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true


@dataclass(frozen=True, slots=True)
class _FakeOps:
    rows_written: int
    rows: Sequence[Mapping[str, object]] = field(default_factory=list)

    def export_rows(
        self,
        _request: SemanticExportRequest,
        *,
        cancel_check: object | None = None,
    ) -> Iterator[dict[str, object]]:
        _ = cancel_check
        return (dict(row) for row in self.rows)

    def export_to_parquet(
        self,
        _request: SemanticExportRequest,
        *,
        output_path: Path,
        cancel_check: object | None = None,
    ) -> int:
        _ = cancel_check
        output_path.write_bytes(b"fake-parquet")
        return self.rows_written

    def export_to_arrow_ipc(
        self,
        _request: SemanticExportRequest,
        *,
        output_path: Path,
        cancel_check: object | None = None,
    ) -> int:
        _ = cancel_check
        output_path.write_bytes(b"fake-arrow")
        return self.rows_written


@pytest.mark.anyio
async def test_http_export_dispatch_returns_row_count_for_parquet() -> None:
    """Ensure dispatch returns metrics when writing Parquet exports."""
    fake = _FakeOps(rows_written=7)
    payload = SemanticExportRequest(view_id="demo.view", format="parquet", limit=10)
    metrics = ExportMetricsContext(
        view_id="demo.view",
        correlation_id="corr-1",
        query_hash="q_123",
        schema_hash=None,
    )
    options = ExportDispatchOptions(headers={})
    dispatched = await dispatch_semantic_export(
        cast("ServingOperations", fake),
        payload,
        metrics,
        options=options,
    )
    expect_equal(dispatched.metrics_row_count, 7)
    expect_true(hasattr(dispatched.response, "headers"))


@pytest.mark.anyio
async def test_http_export_dispatch_streams_jsonl() -> None:
    """Ensure JSONL exports stream rows and omit metrics row counts."""
    fake = _FakeOps(
        rows_written=0,
        rows=[{"id": 1}, {"id": 2}],
    )
    payload = SemanticExportRequest(view_id="demo.view", format="jsonl", limit=10)
    metrics = ExportMetricsContext(
        view_id="demo.view",
        correlation_id="corr-2",
        query_hash="q_456",
        schema_hash=None,
    )
    options = ExportDispatchOptions(headers={})
    dispatched = await dispatch_semantic_export(
        cast("ServingOperations", fake),
        payload,
        metrics,
        options=options,
    )
    expect_equal(dispatched.metrics_row_count, None)
    expect_true(isinstance(dispatched.response, StreamingResponse))
    expect_equal(dispatched.response.media_type, mime_type_for_export_format("jsonl"))


@pytest.mark.anyio
async def test_http_export_dispatch_returns_json_rows() -> None:
    """Ensure JSON exports return rows in a JSON response payload."""
    rows = [{"id": 1}, {"id": 2}, {"id": 3}]
    fake = _FakeOps(rows_written=0, rows=rows)
    payload = SemanticExportRequest(view_id="demo.view", format="json", limit=10)
    metrics = ExportMetricsContext(
        view_id="demo.view",
        correlation_id="corr-3",
        query_hash="q_789",
        schema_hash=None,
    )
    options = ExportDispatchOptions(headers={})
    dispatched = await dispatch_semantic_export(
        cast("ServingOperations", fake),
        payload,
        metrics,
        options=options,
    )
    expect_equal(dispatched.metrics_row_count, len(rows))
    body = json.loads(bytes(dispatched.response.body).decode("utf-8"))
    expect_equal(body.get("count"), len(rows))
    expect_equal(body.get("rows"), rows)
