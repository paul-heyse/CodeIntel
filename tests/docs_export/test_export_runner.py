"""Unit tests for validated export runner."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol, cast

from codeintel.docs_export.runner import ExportOptions, run_validated_exports
from codeintel.storage.gateway import StorageConfig, StorageGateway


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

    class Gateway:
        def __init__(self) -> None:
            self.datasets = object()
            self.config = StorageConfig(db_path=Path(":memory:"))
            self.core = object()
            self.graph = object()
            self.docs = object()
            self.analytics = object()
            self.con = object()

        @staticmethod
        def close() -> None:
            return None

        @staticmethod
        def execute(_sql: str, _params: object | None = None) -> object:
            return object()

        @staticmethod
        def table(_name: str) -> object:
            return object()

    def validator(gateway: _StubGateway) -> None:
        _ = gateway.datasets
        calls.append("validator")

    def export_parquet_fn(*_: object, **kwargs: object) -> None:
        calls.append(f"parquet:{kwargs.get('datasets')}")

    def export_jsonl_fn(*_: object, **kwargs: object) -> list[Path]:
        calls.append(f"jsonl:{kwargs.get('datasets')}")
        return [tmp_path / "dummy.jsonl"]

    options = ExportOptions(
        validate_exports=False,
        schemas=None,
        datasets=["a"],
        validator=cast("Callable[[StorageGateway], None]", validator),
        export_parquet_fn=export_parquet_fn,
        export_jsonl_fn=export_jsonl_fn,
    )

    result = run_validated_exports(
        gateway=cast("StorageGateway", Gateway()), output_dir=tmp_path, options=options
    )

    expected_calls = ["validator", "parquet:['a']", "jsonl:['a']"]
    if calls != expected_calls:
        message = f"Unexpected call sequence: {calls}"
        raise AssertionError(message)
    expected_result = [tmp_path / "dummy.jsonl"]
    if result != expected_result:
        message = f"Unexpected export result: {result}"
        raise AssertionError(message)
