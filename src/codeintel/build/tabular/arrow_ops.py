"""Arrow-first join and materialization helpers for build pipelines."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl
import pyarrow as pa

from codeintel.build.tabular.conversion import table_to_frame, tabular_to_arrow_reader
from codeintel.build.tabular.frames import JoinSpec, JoinValidation, join_validated
from codeintel.build.tabular.types import InferableTabularInput

_ARROW_JOIN_TYPES = {
    "left": "left outer",
    "inner": "inner",
    "right": "right outer",
    "full": "full outer",
    "outer": "full outer",
}


@dataclass(frozen=True, slots=True)
class ArrowJoinSpec:
    """Arrow join configuration for materialized joins."""

    on: Sequence[str] | None = None
    left_on: Sequence[str] | None = None
    right_on: Sequence[str] | None = None
    how: str = "left"
    validate: JoinValidation | None = None
    suffix: str = ""
    coalesce_keys: bool = True
    left_suffix: str | None = None
    right_suffix: str | None = None


def _arrow_spec_from_join_spec(spec: JoinSpec) -> ArrowJoinSpec:
    return ArrowJoinSpec(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
        how=spec.how,
        validate=spec.validate,
        suffix=spec.suffix,
    )


def _join_spec_from_arrow_spec(spec: ArrowJoinSpec) -> JoinSpec:
    """Build a JoinSpec fallback from an ArrowJoinSpec."""
    suffix = spec.suffix
    if not suffix and spec.right_suffix:
        suffix = spec.right_suffix
    return JoinSpec(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
        how=spec.how,
        validate=spec.validate,
        suffix=suffix,
    )


def _polars_join_fallback(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    spec: ArrowJoinSpec | JoinSpec,
) -> pl.DataFrame:
    join_spec = _join_spec_from_arrow_spec(spec) if isinstance(spec, ArrowJoinSpec) else spec
    left_lazy = left.lazy() if isinstance(left, pl.DataFrame) else left
    right_lazy = right.lazy() if isinstance(right, pl.DataFrame) else right
    return join_validated(left_lazy, right_lazy, spec=join_spec).collect()


def _resolve_join_keys(
    *,
    on: Sequence[str] | None,
    left_on: Sequence[str] | None,
    right_on: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[str, ...] | None]:
    if on:
        return tuple(on), tuple(right_on) if right_on else None
    if left_on:
        return tuple(left_on), tuple(right_on) if right_on else None
    if right_on:
        return tuple(right_on), tuple(right_on)
    return (), None


def _ensure_unique_keys(table: pa.Table, keys: Sequence[str], *, label: str) -> None:
    if not keys:
        return
    missing = [key for key in keys if key not in table.column_names]
    if missing:
        msg = f"Missing join keys on {label}: {', '.join(missing)}"
        raise ValueError(msg)
    count_source = keys[0]
    grouped = table.group_by(list(keys)).aggregate([(count_source, "count")])
    count_name = f"{count_source}_count"
    if grouped.num_rows == 0 or count_name not in grouped.column_names:
        return
    counts = grouped[count_name].to_pylist()
    if any(isinstance(value, int) and value > 1 for value in counts):
        msg = f"Join validation failed for {label}: keys not unique"
        raise ValueError(msg)


def _validate_join(
    left: pa.Table,
    right: pa.Table,
    *,
    left_keys: Sequence[str],
    right_keys: Sequence[str] | None,
    validate: JoinValidation | None,
) -> None:
    if validate is None or validate == "m:m":
        return
    right_key_values = right_keys if right_keys is not None else left_keys
    if validate in {"m:1", "1:1"}:
        _ensure_unique_keys(right, right_key_values, label="right")
    if validate in {"1:m", "1:1"}:
        _ensure_unique_keys(left, left_keys, label="left")
    if validate not in {"m:1", "1:m", "1:1", "m:m"}:
        msg = f"Unsupported join validation: {validate}"
        raise ValueError(msg)


def arrow_table_from_tabular(value: InferableTabularInput) -> pa.Table:
    """Convert a tabular input into a fully materialized Arrow Table.

    Returns
    -------
    pa.Table
        Materialized Arrow table.
    """
    reader = tabular_to_arrow_reader(value)
    return pa.Table.from_batches(list(reader), schema=reader.schema)


def arrow_table_from_lazyframe(frame: pl.LazyFrame) -> pa.Table:
    """Collect a LazyFrame into an Arrow Table.

    Returns
    -------
    pa.Table
        Materialized Arrow table.
    """
    return frame.collect().to_arrow()


def _arrow_table_from_frame(frame: pl.DataFrame | pl.LazyFrame) -> pa.Table:
    if isinstance(frame, pl.DataFrame):
        return frame.to_arrow()
    return arrow_table_from_lazyframe(frame)


def arrow_join_tables(
    left: pa.Table,
    right: pa.Table,
    *,
    spec: ArrowJoinSpec,
) -> pa.Table:
    """Join two Arrow tables using the provided keys.

    Parameters
    ----------
    left
        Left-hand table.
    right
        Right-hand table.
    spec
        Join configuration for Arrow joins.

    Raises
    ------
    ValueError
        If join keys are missing or if validation fails.

    Returns
    -------
    pa.Table
        Joined Arrow table.
    """
    keys, right_keys = _resolve_join_keys(
        on=spec.on,
        left_on=spec.left_on,
        right_on=spec.right_on,
    )
    if not keys:
        msg = "Arrow join requires join keys"
        raise ValueError(msg)
    right_suffix = spec.right_suffix
    if spec.suffix and right_suffix is None:
        right_suffix = spec.suffix
    _validate_join(
        left,
        right,
        left_keys=keys,
        right_keys=right_keys,
        validate=spec.validate,
    )
    join_type = _ARROW_JOIN_TYPES.get(spec.how, spec.how)
    return left.join(
        right,
        keys=tuple(keys),
        right_keys=tuple(right_keys) if right_keys is not None else None,
        join_type=join_type,
        left_suffix=spec.left_suffix,
        right_suffix=right_suffix,
        coalesce_keys=spec.coalesce_keys,
    )


def arrow_join_frames(
    left: pl.DataFrame | pl.LazyFrame,
    right: pl.DataFrame | pl.LazyFrame,
    *,
    spec: ArrowJoinSpec | JoinSpec,
) -> pl.DataFrame:
    """Collect, join in Arrow, and return a Polars DataFrame.

    Returns
    -------
    pl.DataFrame
        Joined Polars DataFrame.
    """
    resolved_spec = spec if isinstance(spec, ArrowJoinSpec) else _arrow_spec_from_join_spec(spec)
    left_table = _arrow_table_from_frame(left)
    right_table = _arrow_table_from_frame(right)
    try:
        joined = arrow_join_tables(
            left_table,
            right_table,
            spec=resolved_spec,
        )
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        return _polars_join_fallback(left, right, spec=resolved_spec)
    return table_to_frame(joined)


def arrow_join_lazyframes(
    left: pl.LazyFrame,
    right: pl.LazyFrame,
    *,
    spec: JoinSpec | ArrowJoinSpec | None = None,
) -> pl.LazyFrame:
    """Join two LazyFrames via Arrow, returning a LazyFrame.

    Returns
    -------
    pl.LazyFrame
        LazyFrame wrapping the Arrow join result.
    """
    if spec is None:
        resolved = _arrow_spec_from_join_spec(JoinSpec())
    elif isinstance(spec, ArrowJoinSpec):
        resolved = spec
    else:
        resolved = _arrow_spec_from_join_spec(spec)
    joined = arrow_join_frames(
        left,
        right,
        spec=resolved,
    )
    return joined.lazy()


__all__ = [
    "ArrowJoinSpec",
    "arrow_join_frames",
    "arrow_join_lazyframes",
    "arrow_join_tables",
    "arrow_table_from_lazyframe",
    "arrow_table_from_tabular",
]
