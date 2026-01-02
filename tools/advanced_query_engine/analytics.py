"""Polars-based analytics helpers for persisted query results."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import pandera.polars as pa
import polars as pl
from pandera.typing.polars import DataFrame, Series

from tools.advanced_query_engine.contracts import QueryBudget


class MatchRecordModel(pa.DataFrameModel):
    """Pandera model for match record rows."""

    engine: Series[str]
    path: Series[str]
    start_byte: Series[int]
    end_byte: Series[int]
    start_line: Series[int] = pa.Field(nullable=True)
    end_line: Series[int] = pa.Field(nullable=True)
    rule_id: Series[str] = pa.Field(nullable=True)
    pattern_id: Series[str] = pa.Field(nullable=True)
    snippet: Series[str] = pa.Field(nullable=True)
    captures: Series[object] = pa.Field(nullable=True)

    class Config:
        """Pandera configuration for match record validation."""

        strict = True
        coerce = True


class WiringEdgeModel(pa.DataFrameModel):
    """Pandera model for wiring edge rows."""

    edge_id: Series[str]
    pack_id: Series[str]
    framework: Series[str] = pa.Field(nullable=True)
    entry_kind: Series[str]
    entry_key: Series[str]
    path: Series[str]
    start_byte: Series[int]
    end_byte: Series[int]
    rule_id: Series[str] = pa.Field(nullable=True)
    target_name: Series[str] = pa.Field(nullable=True)
    target_qname: Series[str] = pa.Field(nullable=True)
    evidence: Series[str] = pa.Field(nullable=True)
    captures: Series[object] = pa.Field(nullable=True)

    class Config:
        """Pandera configuration for wiring edge validation."""

        strict = True
        coerce = True


@dataclass(frozen=True)
class StreamResult:
    """Streamed batch results with budget metadata."""

    batches: list[pl.DataFrame]
    rows_seen: int
    budget_exhausted: bool


def scan_parquet(path: Path) -> pl.LazyFrame:
    """Return a lazy scan over a Parquet dataset.

    Parameters
    ----------
    path:
        Parquet dataset path.

    Returns
    -------
    pl.LazyFrame
        Lazy scan of the dataset.
    """
    return pl.scan_parquet(str(path))


def scan_match_records(path: Path) -> pl.LazyFrame:
    """Return a lazy scan for match records.

    Parameters
    ----------
    path:
        Parquet dataset path.

    Returns
    -------
    pl.LazyFrame
        Lazy scan for match records.
    """
    return scan_parquet(path)


def scan_wiring_edges(path: Path) -> pl.LazyFrame:
    """Return a lazy scan for wiring edges.

    Parameters
    ----------
    path:
        Parquet dataset path.

    Returns
    -------
    pl.LazyFrame
        Lazy scan for wiring edges.
    """
    return scan_parquet(path)


def stream_batches(
    lf: pl.LazyFrame,
    *,
    chunk_size: int,
    max_rows: int | None = None,
) -> StreamResult:
    """Collect lazy frames in batches with optional row budgets.

    Parameters
    ----------
    lf:
        Lazy frame to stream.
    chunk_size:
        Batch size for collection.
    max_rows:
        Optional maximum number of rows to emit.

    Returns
    -------
    StreamResult
        Streamed batches with budget metadata.
    """
    batches: list[pl.DataFrame] = []
    rows_seen = 0
    budget_exhausted = False
    for batch in lf.collect_batches(chunk_size=chunk_size):
        if max_rows is not None and rows_seen >= max_rows:
            budget_exhausted = True
            break
        current_batch = batch
        if max_rows is not None and rows_seen + batch.height > max_rows:
            current_batch = batch.head(max_rows - rows_seen)
            budget_exhausted = True
        rows_seen += current_batch.height
        batches.append(current_batch)
        if budget_exhausted:
            break
    return StreamResult(batches=batches, rows_seen=rows_seen, budget_exhausted=budget_exhausted)


def stream_with_budget(
    lf: pl.LazyFrame,
    *,
    budget: QueryBudget,
    chunk_size: int,
) -> StreamResult:
    """Stream batches with query-budget alignment.

    Parameters
    ----------
    lf:
        Lazy frame to stream.
    budget:
        Query budget controlling max rows.
    chunk_size:
        Batch size for collection.

    Returns
    -------
    StreamResult
        Streamed batches with budget metadata.
    """
    max_rows = budget.max_matches if budget.max_matches else None
    return stream_batches(lf, chunk_size=chunk_size, max_rows=max_rows)


def profile_query(lf: pl.LazyFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Execute and return result + profile frames.

    Parameters
    ----------
    lf:
        Lazy frame to profile.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame]
        Result dataframe and profile dataframe.
    """
    return lf.profile()


def validate_match_records(frame: pl.DataFrame) -> DataFrame[MatchRecordModel]:
    """Validate match records via Pandera.

    Parameters
    ----------
    frame:
        Frame of match records.

    Returns
    -------
    DataFrame[MatchRecordModel]
        Validated match records.
    """
    return MatchRecordModel.validate(frame)


def validate_wiring_edges(frame: pl.DataFrame) -> DataFrame[WiringEdgeModel]:
    """Validate wiring edges via Pandera.

    Parameters
    ----------
    frame:
        Frame of wiring edges.

    Returns
    -------
    DataFrame[WiringEdgeModel]
        Validated wiring edges.
    """
    return WiringEdgeModel.validate(frame)


def validate_batches(
    batches: Iterable[pl.DataFrame],
    *,
    model: type[pa.DataFrameModel],
) -> list[pl.DataFrame]:
    """Validate batches against a Pandera model.

    Parameters
    ----------
    batches:
        Iterable of frames to validate.
    model:
        Pandera model to validate against.

    Returns
    -------
    list[pl.DataFrame]
        Validated batches.
    """
    return [model.validate(batch) for batch in batches]


__all__ = [
    "MatchRecordModel",
    "StreamResult",
    "WiringEdgeModel",
    "profile_query",
    "scan_match_records",
    "scan_parquet",
    "scan_wiring_edges",
    "stream_batches",
    "stream_with_budget",
    "validate_batches",
    "validate_match_records",
    "validate_wiring_edges",
]
