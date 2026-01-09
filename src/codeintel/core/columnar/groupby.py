"""Group-by aggregation helpers for Arrow tables."""

from __future__ import annotations

from collections.abc import Sequence

import pyarrow as pa

from codeintel.core.columnar.kernels import group_by_aggregate as _group_by_aggregate


def group_by_aggregate(
    table: pa.Table,
    *,
    keys: Sequence[str],
    aggregations: Sequence[tuple[str, str]],
) -> pa.Table:
    """Group by keys and aggregate columns.

    Parameters
    ----------
    table
        Arrow table to aggregate.
    keys
        Column names to group by.
    aggregations
        Sequence of (column, aggregation) tuples.

    Returns
    -------
    pyarrow.Table
        Aggregated Arrow table.
    """
    return _group_by_aggregate(table, keys=keys, aggregations=aggregations)


__all__ = [
    "group_by_aggregate",
]
