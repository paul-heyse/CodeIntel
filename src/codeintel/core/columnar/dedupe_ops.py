"""Deduplication helpers for Arrow tabular data."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Literal, cast

import pyarrow as pa

from codeintel.core import columnar
from codeintel.core.columnar.compute_helpers import (
    array_from_compute,
    call_compute,
    require_array,
    sort_options,
)
from codeintel.core.columnar.iter import iter_rows
from codeintel.core.columnar.kernels import (
    SortKey,
    hash_struct_ordinal,
    stable_sort_indices,
    stable_sort_table,
)
from codeintel.core.columnar.plan_ops import HashJoinSpec, Plan, materialize_plan
from codeintel.core.schemas.primitives import resolve_canonical_sort_keys
from codeintel.core.schemas.service import get_schema_service

if TYPE_CHECKING:
    from collections.abc import Sequence

DedupeDeterminism = Literal["best_effort", "stable", "order_independent"]
DedupeTier = Literal["canonical", "stable_set", "best_effort", "throughput"]
DedupeTierNormalized = Literal["canonical", "stable_set", "best_effort"]
DedupeStrategy = Literal[
    "order_independent",
    "first",
    "keep_best_by_score",
    "keep_arbitrary",
]
_HASH_ORDINAL_MODULUS = 2**31 - 1


@dataclass(frozen=True, slots=True)
class DedupeSpec:
    """Specification for dedupe behavior."""

    keys: Sequence[str] = ()
    prefer_columns: Sequence[str] = ()
    tie_breakers: Sequence[SortKey] = ()
    tier: DedupeTier = "stable_set"
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
        ordinal = _hash_ordinal_for_ties(table, columns=table.column_names)
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


def _hash_ordinal_for_ties(
    table: pa.Table,
    *,
    columns: Sequence[str],
) -> pa.Array | pa.ChunkedArray | None:
    available = [name for name in columns if name in table.column_names]
    if not available:
        return None
    try:
        safe_table = _join_safe_projection(table)
    except ValueError:
        return None
    safe_columns = [name for name in available if name in safe_table.column_names]
    if not safe_columns:
        return None
    try:
        return hash_struct_ordinal(
            safe_table,
            columns=safe_columns,
            modulus=_HASH_ORDINAL_MODULUS,
        )
    except (RuntimeError, ValueError, pa.ArrowInvalid, pa.ArrowTypeError, TypeError):
        return None


def _score_for_keep_best_by_score(
    table: pa.Table,
    *,
    sort_keys: Sequence[SortKey],
    hash_tiebreaker: bool,
) -> pa.Array | pa.ChunkedArray:
    if not sort_keys:
        msg = "keep_best_by_score requires ordering keys."
        raise ValueError(msg)
    sort_table = table
    resolved_sort_keys: list[SortKey] = list(sort_keys)
    if hash_tiebreaker:
        ordinal = _hash_ordinal_for_ties(
            table,
            columns=table.column_names,
        )
        if ordinal is not None:
            temp_name = _row_index_name(table, base="__dedupe_score_ordinal")
            sort_table = table.append_column(temp_name, ordinal)
            temp_key: SortKey = (temp_name, "ascending")
            resolved_sort_keys.append(temp_key)
    try:
        indices = stable_sort_indices(sort_table, sort_keys=resolved_sort_keys)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        msg = "Order-independent dedupe scoring requires stable sorting."
        raise RuntimeError(msg) from None
    rank = require_array(call_compute("inverse_permutation", [indices]), name="inverse_permutation")
    total = pa.scalar(sort_table.num_rows, type=pa.int64())
    return require_array(call_compute("subtract", [total, rank]), name="subtract")


def _join_safe_projection(
    table: pa.Table,
    *,
    allowed_columns: Sequence[str] = (),
) -> pa.Table:
    return columnar.join_safe_projection(table, allowed_columns=allowed_columns)


def _score_table_for_best_by_score(
    table: pa.Table,
    *,
    sort_keys: Sequence[SortKey],
    hash_tiebreaker: bool,
) -> tuple[pa.Table, str, str]:
    score = _score_for_keep_best_by_score(
        table,
        sort_keys=sort_keys,
        hash_tiebreaker=hash_tiebreaker,
    )
    row_id = _row_index_array(table.num_rows)
    if row_id is None:
        msg = "Order-independent dedupe requires a row index."
        raise RuntimeError(msg)
    score_name = _row_index_name(table, base="__dedupe_score")
    row_id_name = _row_index_name(table, base="__dedupe_row_id")
    scored = table.append_column(score_name, score).append_column(row_id_name, row_id)
    return scored, score_name, row_id_name


def _winner_indices_for_best_by_score(
    scored: pa.Table,
    *,
    key_columns: Sequence[str],
    score_name: str,
    row_id_name: str,
) -> pa.Array | pa.ChunkedArray:
    score_table = scored.select([*key_columns, score_name])
    try:
        winners = score_table.group_by(list(key_columns)).aggregate([(score_name, "max")])
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        msg = "Order-independent dedupe failed to compute winner scores."
        raise RuntimeError(msg) from None
    score_max_name = f"{score_name}_max"
    join_left = _join_safe_projection(
        scored,
        allowed_columns=(*key_columns, score_name, row_id_name),
    )
    join_right = _join_safe_projection(
        winners,
        allowed_columns=(*key_columns, score_max_name),
    )
    join_spec = HashJoinSpec(
        left_keys=[*key_columns, score_name],
        right_keys=[*key_columns, score_max_name],
        left_output=list(join_left.column_names),
        right_output=(),
    )
    selected_plan = Plan.table(join_left).hash_join(
        right=Plan.table(join_right),
        spec=join_spec,
    )
    selected = materialize_plan(selected_plan, use_threads=True)
    if row_id_name not in selected.column_names:
        msg = "Order-independent dedupe failed to retain row identifiers."
        raise RuntimeError(msg)
    return selected[row_id_name]


def _has_duplicate_keys(table: pa.Table, *, key_columns: Sequence[str]) -> bool:
    if table.num_rows <= 1:
        return False
    deduped = _drop_duplicates(table, key_columns=key_columns)
    return deduped.num_rows != table.num_rows


def _resolve_best_by_score_ties(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
    tie_breakers: Sequence[SortKey],
    require_tie_breakers: bool,
) -> pa.Table:
    if not _has_duplicate_keys(table, key_columns=key_columns):
        return table
    if require_tie_breakers and not tie_breakers:
        msg = "Deterministic dedupe requires tie_breaker_columns."
        raise ValueError(msg)
    sort_keys = _dedupe_sort_keys(
        table,
        key_columns=key_columns,
        prefer_columns=(),
        tie_breakers=tie_breakers,
    )
    ordered = _stable_sort_for_dedupe(
        table,
        sort_keys=sort_keys,
        hash_tiebreaker=True,
    )
    return _dedupe_keep_first(ordered, key_columns=key_columns)


def dedupe_keep_first_after_sort(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
    prefer_columns: Sequence[str] = (),
    tie_breakers: Sequence[SortKey] = (),
    require_tie_breakers: bool = False,
) -> pa.Table:
    """Deduplicate by keeping the first row after stable sorting.

    Parameters
    ----------
    table
        Table to deduplicate.
    key_columns
        Columns defining duplicate groups.
    prefer_columns
        Columns to use as descending tie-breakers.
    tie_breakers
        Explicit ordering keys for deterministic selection.
    require_tie_breakers
        Whether to enforce non-empty tie breakers.

    Returns
    -------
    pyarrow.Table
        Deduplicated table with the first row per key kept.
    """
    key_columns = _require_key_columns(table, key_columns=key_columns)
    if require_tie_breakers:
        tie_breakers = _require_tie_breakers(table, tie_breakers=tie_breakers)
    sort_keys = _dedupe_sort_keys(
        table,
        key_columns=key_columns,
        prefer_columns=prefer_columns,
        tie_breakers=tie_breakers,
    )
    ordered = stable_sort_table(table, sort_keys=sort_keys) if sort_keys else table
    return _dedupe_keep_first(ordered, key_columns=key_columns)


def stable_dedupe_with_ties(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
    order_by: Sequence[SortKey] = (),
    tie_breakers: Sequence[SortKey] = (),
    require_tie_breakers: bool = False,
) -> pa.Table:
    """Deduplicate by applying a stable sort with explicit tie handling.

    Parameters
    ----------
    table
        Table to deduplicate.
    key_columns
        Columns defining duplicate groups.
    order_by
        Ordering keys to apply before selecting the first row.
    tie_breakers
        Additional tie breakers applied after ``order_by``.
    require_tie_breakers
        Whether to enforce non-empty tie breakers for deterministic selection.

    Returns
    -------
    pyarrow.Table
        Deduplicated table with stable ordering applied.
    """
    sort_keys = (*order_by, *tie_breakers)
    return dedupe_keep_first_after_sort(
        table,
        key_columns=key_columns,
        tie_breakers=sort_keys,
        require_tie_breakers=require_tie_breakers,
    )


def _dedupe_keep_best_by_score(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
    prefer_columns: Sequence[str],
    tie_breakers: Sequence[SortKey],
    require_tie_breakers: bool,
) -> pa.Table:
    key_columns = _require_key_columns(table, key_columns=key_columns)
    if table.num_rows <= 1:
        return table
    if require_tie_breakers:
        tie_breakers = _require_tie_breakers(table, tie_breakers=tie_breakers)
    if not prefer_columns and not tie_breakers:
        msg = "keep_best_by_score requires prefer_columns or tie_breakers."
        raise ValueError(msg)
    sort_keys = _dedupe_sort_keys(
        table,
        key_columns=key_columns,
        prefer_columns=prefer_columns,
        tie_breakers=tie_breakers,
    )
    scored, score_name, row_id_name = _score_table_for_best_by_score(
        table,
        sort_keys=sort_keys,
        hash_tiebreaker=not tie_breakers,
    )
    indices = _winner_indices_for_best_by_score(
        scored,
        key_columns=key_columns,
        score_name=score_name,
        row_id_name=row_id_name,
    )
    deduped = scored.take(indices)
    if _has_duplicate_keys(deduped, key_columns=key_columns):
        deduped = _resolve_best_by_score_ties(
            deduped,
            key_columns=key_columns,
            tie_breakers=tie_breakers,
            require_tie_breakers=require_tie_breakers,
        )
    return deduped.drop([score_name, row_id_name])


def _dedupe_keep_first(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
) -> pa.Table:
    key_set = set(key_columns)
    non_keys = [name for name in table.column_names if name not in key_set]
    if not non_keys:
        deduped = _dedupe_table_via_compute(table, key_columns=key_columns)
        if deduped is not None:
            return deduped
        return _drop_duplicates(table, key_columns=key_columns)
    aggs = [(name, "first") for name in non_keys]
    try:
        grouped = table.group_by(list(key_columns), use_threads=False).aggregate(aggs)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError):
        return _dedupe_keep_first_python(table, key_columns=key_columns)
    return grouped.rename_columns([*key_columns, *non_keys])


def _dedupe_keep_first_python(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
) -> pa.Table:
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
    if spec.tier == "throughput":
        return "best_effort"
    if spec.strategy == "keep_arbitrary":
        return "best_effort"
    if spec.strategy == "order_independent":
        return "order_independent"
    if spec.tie_breakers or spec.prefer_columns:
        return "stable"
    return "best_effort"


def normalize_dedupe_tier(tier: DedupeTier | None) -> DedupeTierNormalized:
    """Return a normalized dedupe tier for policy enforcement.

    Returns
    -------
    DedupeTierNormalized
        Normalized tier value used by enforcement logic.
    """
    if tier is None:
        return "stable_set"
    if tier == "throughput":
        return "stable_set"
    return tier


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


def _dedupe_with_spec(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
    spec: DedupeSpec,
) -> pa.Table:
    resolved_keys = list(spec.keys) if spec.keys else list(key_columns)
    _require_key_columns(table, key_columns=resolved_keys)
    prefer = tuple(name for name in spec.prefer_columns if name in table.column_names)
    tie_breakers: tuple[SortKey, ...] = tuple(spec.tie_breakers)
    determinism = _determinism_for_spec(spec)
    resolved_tier = normalize_dedupe_tier(spec.tier)
    if spec.strategy == "keep_best_by_score":
        return _dedupe_keep_best_by_score(
            table,
            key_columns=resolved_keys,
            prefer_columns=prefer,
            tie_breakers=tie_breakers,
            require_tie_breakers=resolved_tier == "canonical",
        )
    if spec.strategy == "first":
        return dedupe_keep_first_after_sort(
            table,
            key_columns=resolved_keys,
            prefer_columns=prefer,
            tie_breakers=tie_breakers,
            require_tie_breakers=resolved_tier == "canonical",
        )
    if spec.strategy == "keep_arbitrary":
        return _drop_duplicates(table, key_columns=resolved_keys)
    if determinism != "best_effort":
        sort_keys = _dedupe_sort_keys(
            table,
            key_columns=resolved_keys,
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
    return _drop_duplicates(table, key_columns=resolved_keys)


def _resolve_canonical_tie_breakers(
    table_key: str,
    spec: DedupeSpec,
) -> DedupeSpec:
    if normalize_dedupe_tier(spec.tier) != "canonical":
        return spec
    if spec.tie_breakers:
        return spec
    schema = get_schema_service().get_table_schema(table_key)
    canonical_keys = resolve_canonical_sort_keys(schema)
    if not canonical_keys:
        return spec
    return replace(spec, tie_breakers=tuple(_ascending_sort_keys(canonical_keys)))


def _dedupe_with_legacy(
    table: pa.Table,
    *,
    key_columns: Sequence[str],
    legacy: DedupeLegacy,
) -> pa.Table:
    _require_key_columns(table, key_columns=key_columns)
    if legacy.determinism != "best_effort":
        resolved_tie_breakers = _require_tie_breakers(
            table,
            tie_breakers=_ascending_sort_keys(legacy.tie_breaker_columns),
        )
        prefer = tuple(name for name in legacy.prefer_columns if name in table.column_names)
        sort_keys = _dedupe_sort_keys(
            table,
            key_columns=key_columns,
            prefer_columns=prefer,
            tie_breakers=resolved_tie_breakers,
        )
        table = _stable_sort_for_dedupe(
            table,
            sort_keys=sort_keys,
            hash_tiebreaker=legacy.determinism == "order_independent",
        )
    elif legacy.prefer_columns:
        column_set = set(table.column_names)
        prefer = [name for name in legacy.prefer_columns if name in column_set]
        if prefer:
            table = _sort_table_for_preference(table, prefer)
    return _drop_duplicates(table, key_columns=key_columns)


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
        resolved_spec = _resolve_canonical_tie_breakers(table_key, spec)
        return _dedupe_with_spec(table, key_columns=key_columns, spec=resolved_spec)
    resolved_legacy = legacy or DedupeLegacy()
    return _dedupe_with_legacy(table, key_columns=key_columns, legacy=resolved_legacy)


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
    "DedupeTierNormalized",
    "dedupe_keep_first_after_sort",
    "dedupe_table_for_table",
    "normalize_dedupe_tier",
    "stable_dedupe_with_ties",
]
