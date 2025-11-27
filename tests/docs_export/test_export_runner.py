"""Unit tests for validated export runner."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from codeintel.pipeline.export.export_jsonl import ExportCallOptions
from codeintel.pipeline.export.runner import ExportOptions, run_validated_exports
from codeintel.storage.gateway import StorageConfig, StorageGateway
from tests._helpers.fixtures import provision_docs_export_ready


class _StubGateway(Protocol):
    datasets: object
    config: StorageConfig
    con: object
    core: object
    graph: object
    docs: object
    analytics: object

    def close(self) -> None: ...
    def execute(self, sql: str, params: object | None = None) -> object: ...
    def table(self, name: str) -> object: ...


def test_run_validated_exports_invokes_validator_before_exports(tmp_path: Path) -> None:
    """
    Ensure the runner validates before invoking export functions.

    Raises
    ------
    AssertionError
        If the validator/export invocation order or outputs are unexpected.
    """
    calls: list[str] = []
    ctx = provision_docs_export_ready(tmp_path, db_path=tmp_path / "db.duckdb", file_backed=True)

    def validator(gateway: _StubGateway) -> None:
        _ = gateway.datasets
        calls.append("validator")

    def export_parquet_fn(
        gateway: StorageGateway,
        document_output_dir: Path,
        *,
        options: ExportCallOptions | None = None,
    ) -> None:
        _ = gateway
        _ = document_output_dir
        opts = options or ExportCallOptions()
        calls.append(f"parquet:{opts.datasets}")

    def export_jsonl_fn(
        gateway: StorageGateway,
        document_output_dir: Path,
        *,
        options: ExportCallOptions | None = None,
    ) -> list[Path]:
        _ = gateway
        _ = document_output_dir
        opts = options or ExportCallOptions()
        calls.append(f"jsonl:{opts.datasets}")
        return [tmp_path / "dummy.jsonl"]

    options = ExportOptions(
        export=ExportCallOptions(
            validate_exports=False,
            schemas=None,
            datasets=["a"],
        ),
        validator=cast("Callable[[StorageGateway], None]", validator),
        export_parquet_fn=export_parquet_fn,
        export_jsonl_fn=export_jsonl_fn,
    )

    try:
        result = run_validated_exports(
            gateway=cast("StorageGateway", ctx.gateway), output_dir=tmp_path, options=options
        )
    finally:
        ctx.close()

    expected_calls = ["validator", "parquet:['a']", "jsonl:['a']"]
    if calls != expected_calls:
        message = f"Unexpected call sequence: {calls}"
        raise AssertionError(message)
    expected_result = [tmp_path / "dummy.jsonl"]
    if result != expected_result:
        message = f"Unexpected export result: {result}"
        raise AssertionError(message)
