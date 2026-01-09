"""Shared dataset scan helpers for causal analysis scripts."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.core.columnar.conversion import reader_to_table
from codeintel.core.columnar.execution_context import ExecutionContext
from codeintel.core.columnar.normalization import normalize_table_for_compute
from codeintel.core.datasets.scanning import (
    ParquetScanOptions,
    scan_parquet_dataset_with_telemetry,
)

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pyarrow as pa


@dataclass(frozen=True, slots=True)
class TableScanResult:
    """Result of scanning a dataset table."""

    table: pa.Table
    used_fallback: bool
    primary_row_count: int
    fallback_row_count: int | None = None


@dataclass(frozen=True, slots=True)
class ScanConfig:
    """Shared scan configuration for dataset readers."""

    dataset_root: Path
    snapshot_id: str
    repo: str | None = None
    commit: str | None = None
    execution_ctx: ExecutionContext | None = None


def scan_table_with_fallback(
    config: ScanConfig,
    table_key: str,
    columns: tuple[str, ...],
) -> TableScanResult:
    """Scan a dataset table with optional repo/commit filtering and fallback.

    Returns
    -------
    TableScanResult
        Scan result with row counts and fallback metadata.

    Raises
    ------
    FileNotFoundError
        If the dataset snapshot directory is missing.
    """
    options = ParquetScanOptions(
        columns=columns,
        repo=config.repo,
        commit=config.commit,
        implicit_ordering=True,
        require_sequenced_output=True,
        metrics_enabled=True,
        execution_ctx=config.execution_ctx,
    )
    reader, telemetry = scan_parquet_dataset_with_telemetry(
        dataset_root=config.dataset_root,
        table_key=table_key,
        snapshot_id=config.snapshot_id,
        options=options,
    )
    if telemetry is not None:
        LOG.debug("Causal scan telemetry: %s", telemetry.to_mapping())
    if reader is None:
        msg = f"{table_key} snapshot not found for {config.snapshot_id}"
        raise FileNotFoundError(msg)
    table = normalize_table_for_compute(reader_to_table(reader))
    primary_rows = table.num_rows
    if primary_rows == 0 and (config.repo or config.commit):
        fallback_reader, fallback_telemetry = scan_parquet_dataset_with_telemetry(
            dataset_root=config.dataset_root,
            table_key=table_key,
            snapshot_id=config.snapshot_id,
            options=ParquetScanOptions(
                columns=columns,
                implicit_ordering=True,
                require_sequenced_output=True,
                metrics_enabled=True,
                execution_ctx=config.execution_ctx,
            ),
        )
        if fallback_telemetry is not None:
            LOG.debug("Causal scan fallback telemetry: %s", fallback_telemetry.to_mapping())
        if fallback_reader is not None:
            fallback = normalize_table_for_compute(reader_to_table(fallback_reader))
        else:
            fallback = None
        if fallback is not None and fallback.num_rows > 0:
            LOG.info(
                "Filtered scan returned 0 rows for %s; falling back to unfiltered scan.",
                table_key,
            )
            return TableScanResult(
                table=fallback,
                used_fallback=True,
                primary_row_count=primary_rows,
                fallback_row_count=fallback.num_rows,
            )
    return TableScanResult(
        table=table,
        used_fallback=False,
        primary_row_count=primary_rows,
        fallback_row_count=None,
    )


__all__ = ["ScanConfig", "TableScanResult", "scan_table_with_fallback"]
