"""Tests for FastMCP background task execution (SEP-1686) on the serving surface."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import anyio
import pytest
from mcp import McpError

from tests._helpers.harnesses.serving_app import ServingAppHarness, ServingSettingsOverrides
from tests._helpers.mcp_payloads import extract_payload
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.serving.mcp.protocols import SemanticKernelProtocol as SemanticKernel
    from codeintel.serving.mcp.protocols import ServingDBManagerProtocol
    from codeintel.serving.meta.models import ServingKernelMetaResponse
    from codeintel.serving.runtime import ServingRuntime
    from codeintel.serving.search.models import SearchQueryRequest, SearchQueryResponse
    from codeintel.serving.semantic.models import (
        SemanticCatalogResponse,
        SemanticExplainResponse,
        SemanticExportRequest,
        SemanticQueryRequest,
        SemanticQueryResponse,
        SemanticViewDescriptionResponse,
    )


@dataclass(frozen=True, slots=True)
class _SlowExportKernel:
    inner: SemanticKernel
    delay_s: float

    @property
    def db(self) -> ServingDBManagerProtocol:
        return self.inner.db

    def catalog(self) -> SemanticCatalogResponse:
        return self.inner.catalog()

    def describe(self, view_id: str) -> SemanticViewDescriptionResponse:
        return self.inner.describe(view_id)

    def query(self, request: SemanticQueryRequest) -> SemanticQueryResponse:
        return self.inner.query(request)

    def explain(self, request: SemanticQueryRequest) -> SemanticExplainResponse:
        return self.inner.explain(request)

    def search(self, request: SearchQueryRequest) -> SearchQueryResponse:
        return self.inner.search(request)

    def meta(self) -> ServingKernelMetaResponse:
        return self.inner.meta()

    def export_rows(self, request: SemanticExportRequest) -> Iterator[dict[str, object]]:
        for index, row in enumerate(self.inner.export_rows(request)):
            if index > 0:
                time.sleep(self.delay_s)
            yield row

    def export_sql(self, request: SemanticExportRequest) -> str:
        return self.inner.export_sql(request)

    def export_fingerprint(self, request: SemanticExportRequest) -> tuple[str, str | None]:
        return self.inner.export_fingerprint(request)

    def export_to_parquet(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        return self.inner.export_to_parquet(request, output_path=output_path)

    def export_to_arrow_ipc(self, request: SemanticExportRequest, *, output_path: Path) -> int:
        return self.inner.export_to_arrow_ipc(request, output_path=output_path)

    def compile_query_sql(self, request: SemanticQueryRequest) -> str:
        return self.inner.compile_query_sql(request)


@pytest.mark.anyio
async def test_mcp_export_task_mode_completes(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Allow calling semantic_export as a background task and awaiting completion."""
    snapshot = serving_snapshot_factory.demo_snapshot(row_count=10)
    harness = ServingAppHarness.from_snapshot(snapshot)
    settings_overrides: ServingSettingsOverrides = {
        "hot_swap": False,
        "result_engine": "pandas",
        "schema_enforcement": "strict",
        "mcp_mask_errors": False,
        "mcp_export_enable_tasks": True,
    }
    async with harness.mcp_client(settings_overrides=settings_overrides) as client:
        task_or_result = await client.call_tool(
            "semantic_export",
            {"request": {"view_id": "demo.view", "export_format": "jsonl", "limit": 10}},
            task=True,
        )

        result_obj = task_or_result
        if hasattr(task_or_result, "result"):
            result_obj = await task_or_result.result()

        payload = extract_payload(result_obj)
        export_id = payload.get("export_id")
        if not isinstance(export_id, str) or not export_id:
            pytest.fail("Expected semantic_export task result to include export_id")


@pytest.mark.anyio
async def test_mcp_export_task_cancellation_cleans_up_artifacts(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Cancel a running export task and ensure partial artifacts are cleaned up."""
    snapshot = serving_snapshot_factory.demo_snapshot(row_count=2500)
    harness = ServingAppHarness.from_snapshot(snapshot)
    settings_overrides: ServingSettingsOverrides = {
        "hot_swap": False,
        "result_engine": "pandas",
        "schema_enforcement": "strict",
        "mcp_mask_errors": False,
        "mcp_export_enable_tasks": True,
    }

    def _slow_kernel(runtime: ServingRuntime) -> SemanticKernel:
        return _SlowExportKernel(inner=runtime.kernel, delay_s=0.001)

    async with harness.mcp_client(
        settings_overrides=settings_overrides,
        kernel_builder=_slow_kernel,
    ) as client:
        tool_task = await client.call_tool(
            "semantic_export",
            {"request": {"view_id": "demo.view", "export_format": "jsonl", "limit": 2500}},
            task=True,
        )
        if not hasattr(tool_task, "cancel"):
            pytest.fail("Expected task-capable client to return a ToolTask")

        await tool_task.cancel()

        exports_dir = snapshot.serve_dir / "exports"
        exports_dir.mkdir(parents=True, exist_ok=True)
        leftovers: list[str] = []
        for _ in range(200):
            leftovers = [
                path.name for path in exports_dir.iterdir() if not path.name.endswith(".cancelled")
            ]
            if not leftovers:
                break
            await anyio.sleep(0.02)
        else:
            pytest.fail(f"Expected export artifacts to be cleaned up, found: {leftovers}")

        with pytest.raises(McpError):
            await tool_task.result()
