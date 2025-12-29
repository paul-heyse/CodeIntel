"""Shared export dispatch helpers for serving transports."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from codeintel.serving.export.engine import (
    ExportDelivery,
    ExportPlan,
    build_export_plan,
    write_export_file,
)
from codeintel.serving.semantic.models import SemanticExportRequest

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.serving.operations.cancellation import CancelCheck
    from codeintel.serving.operations.ops import ServingOperations

T = TypeVar("T")
type DispatchResult[T] = T | Awaitable[T]


@dataclass(frozen=True, slots=True)
class ExportRowProvider:
    """Row provider for export dispatch handling."""

    ops: ServingOperations
    request: SemanticExportRequest
    cancel_check: CancelCheck | None = None

    def iter_rows(self) -> Iterator[dict[str, object]]:
        """Return an iterator over export rows."""
        return self.ops.export_rows(self.request, cancel_check=self.cancel_check)

    def collect_rows(self) -> list[dict[str, object]]:
        """Collect all export rows into memory."""
        return list(self.iter_rows())


@dataclass(frozen=True, slots=True)
class ExportDispatchHandlers[T]:
    """Handlers for each export delivery mode."""

    ndjson_stream: Callable[[ExportPlan, ExportRowProvider], DispatchResult[T]]
    json_rows: Callable[[ExportPlan, ExportRowProvider], DispatchResult[T]]
    binary_file: Callable[[ExportPlan, Callable[[Path], int]], DispatchResult[T]]


def dispatch_export[T](
    ops: ServingOperations,
    request: SemanticExportRequest,
    *,
    cancel_check: CancelCheck | None,
    handlers: ExportDispatchHandlers[T],
) -> DispatchResult[T]:
    """Dispatch an export request using the supplied handlers."""
    plan = build_export_plan(request)
    provider = ExportRowProvider(
        ops=ops,
        request=request,
        cancel_check=cancel_check,
    )
    if plan.delivery is ExportDelivery.ndjson_stream:
        return handlers.ndjson_stream(plan, provider)
    if plan.delivery is ExportDelivery.json_rows:
        return handlers.json_rows(plan, provider)
    if plan.delivery is ExportDelivery.binary_file:
        return handlers.binary_file(
            plan,
            lambda path: write_export_file(
                ops,
                request,
                output_path=path,
                cancel_check=cancel_check,
            ),
        )
    msg = f"Unsupported export format: {request.format}"
    raise ValueError(msg)


__all__ = ["ExportDispatchHandlers", "ExportRowProvider", "dispatch_export"]
