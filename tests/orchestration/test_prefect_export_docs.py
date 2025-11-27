"""Tests for Prefect export docs task validation."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.runner import ExportOptions
from codeintel.pipeline.orchestration.prefect_flow import ExportTaskHooks, t_export_docs
from codeintel.storage.gateway import StorageGateway


class _StubGateway:
    """Minimal gateway stub for export task."""

    def __init__(self) -> None:
        self.con = object()


def test_t_export_docs_invokes_validator_before_export(tmp_path: Path) -> None:
    """Prefect task validates registry before exporting datasets."""
    events: list[str] = []

    def validator(_gateway: StorageGateway) -> None:
        events.append("validator")

    def export_runner(
        *, gateway: StorageGateway, output_dir: Path, options: ExportOptions | None = None
    ) -> list[Path]:
        if options is None:
            pytest.fail("Expected options")
        options.validator(gateway)
        events.append(f"export:{output_dir}")
        return []

    def gateway_factory(_db_path: Path) -> StorageGateway:
        return _StubGateway()  # type: ignore[return-value]

    def create_views(_con: object) -> None:
        events.append("views")

    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)
    hooks = ExportTaskHooks(
        validator=validator,
        export_runner=export_runner,
        gateway_factory=gateway_factory,
        create_views=create_views,
    )
    export_options = ExportOptions(
        export=ExportCallOptions(validate_exports=True, schemas=["public"])
    )
    t_export_docs(
        db_path=tmp_path / "db.duckdb",
        document_output_dir=output_dir,
        options=export_options,
        hooks=hooks,
    )

    expected = ["views", "validator", f"export:{output_dir}"]
    if events != expected:
        pytest.fail(f"Unexpected event order: {events}")
