"""Deduplication helpers for Arrow tabular data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import pyarrow as pa

from codeintel.core.columnar.compute_helpers import array_from_compute, call_compute, sort_options
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.kernels import SortKey, hash_struct_ordinal, stable_sort_indices
from codeintel.core.schemas.service import get_schema_service

DedupeDeterminism = Literal["best_effort", "stable", "order_independent"]
DedupeTier = Literal["canonical", "throughput"]
DedupeStrategy = Literal["order_independent", "first"]
_HASH_ORDINAL_MODULUS = 2**31 - 1


@dataclass(frozen=True, slots=True)
class DedupeSpec:
    """Specification for dedupe behavior."""

    keys: Sequence[str] = ()
    prefer_columns: Sequence[str] = ()
    tie_breakers: Sequence[SortKey] = ()
    tier: DedupeTier = "canonical"
    strategy: DedupeStrategy = "order_independent"


@dataclass(frozen=True, slots=True)
class DedupeLegacy:
    """Legacy dedupe configuration parameters."""

    prefer_columns: Sequence[str] = ()
    determinism: DedupeDeterminism = "best_effort"
    tie_breaker_columns: Sequence[str] = ()


def _row_index_array(length: int) -> pa.Array | None:
    try:
        return pa.array(range(length), type=pa.int64())
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        return None


def _row_index_name(table: pa.Table, *, base: str) -> str:
    existing = set(table.column_names)
    name = base
    suffix = 1
    while name in existing:
        name = f"{base}_{suffix}"
        suffix += 1
    return name


def _sort_table_for_preference(table: pa.Table, prefer_columns: Sequence[str]) -> pa.Table:
    sort_keys = [(name, "descending") for name in prefer_columns]
    options = sort_options(sort_keys, null_placement="at_end")
    indices = call_compute("sort_indices", [table], options=options)
    if indices is None:
        return table
    return table.take(indices)


def _stable_sort_for_dedupe(
    table: pa.Table,
    *,
    sort_keys: Sequence[SortKey],
    hash_tiebreaker: bool,
) -> pa.Table:
    if table.num_rows <= 1 or not sort_keys:
        return table
    sort_table = table
    resolved_sort_keys: list[SortKey] = list(sort_keys)
    if hash_tiebreaker:
        try:
            ordinal = hash_struct_ordinal(
                table,
                columns=list(table.column_names),
                modulus=_HASH_ORDINAL_MODULUS,
            )
        except (RuntimeError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
            ordinal = None
        if ordinal is not None:
            temp_name = _row_index_name(table, base="__dedupe_ordinal")
            sort_table = table.append_column(temp_name, ordinal)
            resolved_sort_keys.append((temp_name, "ascending"))
    try:
        indices = stable_sort_indices(sort_table, sort_keys=resolved_sort_keys)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        msg = "Deterministic dedupe requires stable sorting; sort failed."
        raise RuntimeError(msg) from None
    return table.take(indices)


def _dedupe_sort_keys(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
    prefer_columns: Sequence[str],
    tie_breakers: Sequence[SortKey],
) -> list[SortKey]:
    available = set(table.column_names)
    keys: list[SortKey] = []
    used: set[str] = set()
    for name in key_columns:
        if name in available and name not in used:
            keys.append((name, "ascending"))
            used.add(name)
    for name in prefer_columns:
        if name in available and name not in used:
            keys.append((name, "descending"))
            used.add(name)
    for name, order in tie_breakers:
        if name in available and name not in used:
            keys.append((name, order))
            used.add(name)
    return keys


def _require_tie_breakers(
    table: pa.Table,
    *,
    tie_breakers: Sequence[SortKey],
) -> Sequence[SortKey]:
    if not tie_breakers:
        msg = "Deterministic dedupe requires tie_breaker_columns."
        raise ValueError(msg)
    missing = [name for name, _ in tie_breakers if name not in table.column_names]
    if missing:
        msg = f"Deterministic dedupe missing tie_breaker columns: {missing}"
        raise ValueError(msg)
    return tie_breakers


def _ascending_sort_keys(columns: Sequence[str]) -> list[SortKey]:
    return [cast("SortKey", (name, "ascending")) for name in columns]


def _require_key_columns(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
) -> Sequence[str]:
    if not key_columns:
        msg = "Deduplication requires at least one key column."
        raise ValueError(msg)
    missing = [name for name in key_columns if name not in table.column_names]
    if missing:
        msg = f"Deduplication missing key columns: {missing}"
        raise ValueError(msg)
    return key_columns


def _determinism_for_spec(spec: DedupeSpec) -> DedupeDeterminism:
    if spec.strategy == "order_independent":
        return "order_independent"
    if spec.tie_breakers:
        return "stable"
    return "best_effort"


def _dedupe_table_via_compute(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
) -> pa.Table | None:
    if table.num_rows == 0:
        return table
    row_index_name = _row_index_name(table, base="_row_index")
    row_index = _row_index_array(table.num_rows)
    if row_index is None:
        return None
    try:
        indexed = table.append_column(row_index_name, row_index)
        grouped = indexed.group_by(list(key_columns)).aggregate([(row_index_name, "min")])
        index_column = f"{row_index_name}_min"
        if index_column not in grouped.column_names:
            return None
        indices = grouped.column(index_column)
        mask = array_from_compute("is_in", [row_index, indices])
        if mask is None:
            return None
        return table.filter(mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return None


def dedupe_table_for_table(
    table_key: str,
    table: pa.Table,
    *,
    spec: DedupeSpec | None = None,
    legacy: DedupeLegacy | None = None,
) -> pa.Table:
    """Return a table with duplicate primary-key rows removed.

    Returns
    -------
    pa.Table
        Table with duplicate primary-key rows removed.
    """
    schema_service = get_schema_service()
    schema = schema_service.get_table_schema(table_key)
    if schema is None or not schema.primary_key:
        return table
    key_columns = list(schema.primary_key)
    if spec is not None:
        key_columns = list(spec.keys) if spec.keys else key_columns
        _require_key_columns(table, key_columns=key_columns)
        prefer = tuple(name for name in spec.prefer_columns if name in table.column_names)
        tie_breakers: tuple[SortKey, ...] = tuple(spec.tie_breakers)
        if spec.tier == "canonical" and spec.strategy == "first":
            tie_breakers = tuple(_require_tie_breakers(table, tie_breakers=tie_breakers))
        determinism = _determinism_for_spec(spec)
        if determinism != "best_effort":
            sort_keys = _dedupe_sort_keys(
                table,
                key_columns=key_columns,
                prefer_columns=prefer,
                tie_breakers=tie_breakers,
            )
            table = _stable_sort_for_dedupe(
                table,
                sort_keys=sort_keys,
                hash_tiebreaker=determinism == "order_independent",
            )
        elif prefer:
            table = _sort_table_for_preference(table, prefer)
        return _drop_duplicates(table, key_columns=key_columns)
    _require_key_columns(table, key_columns=key_columns)
    resolved_legacy = legacy or DedupeLegacy()
    if resolved_legacy.determinism != "best_effort":
        resolved_tie_breakers = _require_tie_breakers(
            table,
            tie_breakers=_ascending_sort_keys(resolved_legacy.tie_breaker_columns),
        )
        prefer = tuple(
            name for name in resolved_legacy.prefer_columns if name in table.column_names
        )
        sort_keys = _dedupe_sort_keys(
            table,
            key_columns=key_columns,
            prefer_columns=prefer,
            tie_breakers=resolved_tie_breakers,
        )
        table = _stable_sort_for_dedupe(
            table,
            sort_keys=sort_keys,
            hash_tiebreaker=resolved_legacy.determinism == "order_independent",
        )
    elif resolved_legacy.prefer_columns:
        prefer = [
            name for name in resolved_legacy.prefer_columns if name in set(table.column_names)
        ]
        if prefer:
            table = _sort_table_for_preference(table, prefer)
    return _drop_duplicates(table, key_columns=key_columns)


def _drop_duplicates(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
) -> pa.Table:
    try:
        return table.drop_duplicates(key_columns)
    except (AttributeError, pa.ArrowNotImplementedError, pa.ArrowTypeError):
        deduped = _dedupe_table_via_compute(table, key_columns=key_columns)
        if deduped is not None:
            return deduped
        seen: set[tuple[object, ...]] = set()
        rows: list[dict[str, object]] = []
        for row in iter_rows(table):
            key = tuple(row.get(col) for col in key_columns)
            if key in seen:
                continue
            seen.add(key)
            rows.append(row)
        if not rows:
            return pa.Table.from_batches([], schema=table.schema)
        return pa.Table.from_pylist(rows, schema=table.schema)


__all__ = [
    "DedupeDeterminism",
    "DedupeLegacy",
    "DedupeSpec",
    "DedupeStrategy",
    "DedupeTier",
    "dedupe_table_for_table",
]
