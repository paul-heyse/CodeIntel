"""Arrow compute kernels for graph assembly."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, cast

import pyarrow as pa

from codeintel.core.columnar.compute_helpers import call_compute, require_array
from codeintel.core.columnar.kernels import (
    SortKey,
)
from codeintel.core.columnar.kernels import (
    hash_struct_ordinal as _hash_struct_ordinal,
)
from codeintel.core.columnar.kernels import (
    stable_sort_table as _stable_sort_table,
)
from codeintel.core.columnar.plan_kernels import ExplodeSpec
from codeintel.core.columnar.plan_kernels import (
    explode_edges_for_join as _explode_edges_for_join,
)
from codeintel.core.schemas.primitives import resolve_canonical_sort_keys
from codeintel.core.schemas.service import get_schema_service


@dataclass(frozen=True, slots=True)
class ExplodeEdgesResult:
    """Result of exploding list-based edges."""

    edges: pa.Table
    invalid_parents: pa.Table


@dataclass(frozen=True, slots=True)
class ExplodeEdgesSpec:
    """Configuration for exploding list-based edges."""

    src_col: str
    dst_list_col: str
    repeat_cols: Sequence[str] = ()
    src_name: str = "src_id"
    dst_name: str = "dst_id"
    aligned_list_cols: Sequence[str] = ()
    nulls_match: bool = True


def explode_edges(
    table: pa.Table,
    *,
    spec: ExplodeEdgesSpec,
) -> pa.Table:
    """Explode list-valued destination columns into edges.

    Returns
    -------
    pyarrow.Table
        Edge table with list elements flattened into destination rows.
    """
    result = _explode_edges_for_join(
        table,
        spec=ExplodeSpec(
            src_col=spec.src_col,
            dst_list_col=spec.dst_list_col,
            repeat_cols=spec.repeat_cols,
            null_list_policy="error",
            null_child_policy="drop",
            enforce_parent_valid=True,
        ),
    )
    return _rename_exploded_columns(result.good, spec)


def explode_edges_with_aligned_lists(
    table: pa.Table,
    *,
    spec: ExplodeEdgesSpec,
) -> ExplodeEdgesResult:
    """Explode list-valued edges with aligned list attributes.

    Returns
    -------
    ExplodeEdgesResult
        Exploded edge table and any rows with misaligned list lengths.
    """
    result = _explode_edges_for_join(
        table,
        spec=ExplodeSpec(
            src_col=spec.src_col,
            dst_list_col=spec.dst_list_col,
            repeat_cols=spec.repeat_cols,
            aligned_list_cols=spec.aligned_list_cols,
            null_list_policy="empty" if spec.nulls_match else "error",
            null_child_policy="drop",
            enforce_parent_valid=True,
        ),
    )
    invalid_parents = _invalid_parent_rows(table, result.errors)
    edges = _rename_exploded_columns(result.good, spec)
    return ExplodeEdgesResult(edges=edges, invalid_parents=invalid_parents)


def _rename_exploded_columns(table: pa.Table, spec: ExplodeEdgesSpec) -> pa.Table:
    rename: dict[str, str] = {}
    if spec.src_name != spec.src_col:
        rename[spec.src_col] = spec.src_name
    if spec.dst_name != spec.dst_list_col:
        rename[spec.dst_list_col] = spec.dst_name
    if not rename:
        return table
    return table.rename_columns([rename.get(name, name) for name in table.column_names])


def _invalid_parent_rows(table: pa.Table, errors: pa.Table) -> pa.Table:
    if errors.num_rows == 0 or "row_id" not in errors.column_names:
        return _empty_slice(table)
    row_ids = errors["row_id"]
    unique_ids = require_array(call_compute("unique", [row_ids]), name="unique")
    return table.take(unique_ids)


def _empty_slice(table: pa.Table) -> pa.Table:
    return table.slice(0, 0)


def hash_struct_ordinal(
    table: pa.Table,
    *,
    columns: Sequence[str],
    modulus: int,
) -> pa.Array | pa.ChunkedArray:
    """Return deterministic ordinals by hashing the provided columns.

    Returns
    -------
    pyarrow.Array | pyarrow.ChunkedArray
        Hash-based ordinals for each input row.
    """
    return _hash_struct_ordinal(table, columns=columns, modulus=modulus)


def stable_sort_table(
    table: pa.Table,
    *,
    sort_keys: Sequence[SortKey],
    null_placement: Literal["at_end", "at_start"] = "at_end",
) -> pa.Table:
    """Return a table sorted using stable Arrow sort indices.

    Returns
    -------
    pyarrow.Table
        Stably sorted table.
    """
    return _stable_sort_table(table, sort_keys=sort_keys, null_placement=null_placement)


def stable_sort_for_contract(
    table: pa.Table,
    *,
    table_key: str,
    provenance_keys: Sequence[str] = (),
    null_placement: Literal["at_end", "at_start"] = "at_end",
) -> pa.Table:
    """Return a table stably sorted by contract keys with optional provenance tie-breakers.

    Returns
    -------
    pyarrow.Table
        Table sorted using canonical or provenance-aware keys.
    """
    schema = get_schema_service().get_table_schema(table_key)
    canonical_keys = resolve_canonical_sort_keys(schema) or ()
    if not canonical_keys:
        return table
    order_keys = (*canonical_keys, *provenance_keys)
    sort_keys: list[SortKey] = [cast("SortKey", (key, "ascending")) for key in order_keys]
    return stable_sort_table(table, sort_keys=sort_keys, null_placement=null_placement)


__all__ = [
    "ExplodeEdgesResult",
    "ExplodeEdgesSpec",
    "explode_edges",
    "explode_edges_with_aligned_lists",
    "hash_struct_ordinal",
    "stable_sort_for_contract",
    "stable_sort_table",
]
