"""Orchestration helpers for validated Document Output exports."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from codeintel.pipeline.export.export_jsonl import export_all_jsonl
from codeintel.pipeline.export.export_parquet import export_all_parquet
from codeintel.serving.http.datasets import validate_dataset_registry
from codeintel.storage.gateway import StorageGateway


class Exporter(Protocol):
    """Protocol for export callables used by the runner."""

    def __call__(
        self,
        gateway: StorageGateway,
        document_output_dir: Path,
        *,
        validate_exports: bool,
        schemas: list[str] | None,
        datasets: list[str] | None,
    ) -> None:
        """Execute an export call."""
        ...


class JsonlExporter(Protocol):
    """Protocol for JSONL export callables that return written paths."""

    def __call__(
        self,
        gateway: StorageGateway,
        document_output_dir: Path,
        *,
        validate_exports: bool,
        schemas: list[str] | None,
        datasets: list[str] | None,
    ) -> list[Path]:
        """Execute an export call and return emitted file paths."""
        ...


class ExportRunner(Protocol):
    """Protocol for higher-level export runners invoked by CLI/Prefect."""

    def __call__(
        self,
        *,
        gateway: StorageGateway,
        output_dir: Path,
        options: ExportOptions | None = None,
    ) -> list[Path]:
        """Run exports and return emitted file paths."""
        ...


@dataclass
class ExportOptions:
    """Options controlling export validation and dataset selection."""

    validate_exports: bool = False
    schemas: list[str] | None = None
    datasets: list[str] | None = None
    validator: Callable[[StorageGateway], None] = validate_dataset_registry
    export_parquet_fn: Exporter = field(default=export_all_parquet)
    export_jsonl_fn: JsonlExporter = field(default=export_all_jsonl)


def run_validated_exports(
    *,
    gateway: StorageGateway,
    output_dir: Path,
    options: ExportOptions | None = None,
) -> list[Path]:
    """
    Validate registry and emit Parquet/JSONL exports.

    Parameters
    ----------
    gateway:
        StorageGateway providing datasets and connection metadata.
    output_dir:
        Document Output directory for emitted artifacts.
    options:
        ExportOptions controlling validation, schemas, and dataset selection.

    Returns
    -------
    list[Path]
        Paths written by the JSONL export (Parquet exports are written for side effects).
    """
    opts = options or ExportOptions()
    opts.validator(gateway)
    opts.export_parquet_fn(
        gateway,
        output_dir,
        validate_exports=opts.validate_exports,
        schemas=opts.schemas,
        datasets=opts.datasets,
    )
    return opts.export_jsonl_fn(
        gateway,
        output_dir,
        validate_exports=opts.validate_exports,
        schemas=opts.schemas,
        datasets=opts.datasets,
    )
