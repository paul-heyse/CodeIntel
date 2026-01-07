"""Snapshot-scoped row collection for tabular inputs."""

from __future__ import annotations

from collections.abc import Sequence

from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.arrow_ops import iter_rows
from codeintel.build.tabular.conversion import tabular_to_arrow_table, tabular_to_scoped_table
from codeintel.build.tabular.types import InferableTabularInput


def collect_scoped_rows(
    value: InferableTabularInput,
    columns: Sequence[str],
    *,
    scope: SnapshotScope,
    require_scope_columns: bool = True,
) -> list[dict[str, object]]:
    """Collect rows after applying strict snapshot filtering.

    Returns
    -------
    list[dict[str, object]]
        Snapshot-filtered rows for the requested columns.

    Raises
    ------
    ValueError
        If requested columns are missing in the input table.
    """
    table = tabular_to_arrow_table(value)
    missing = [name for name in columns if name not in table.column_names]
    if missing:
        msg = f"Missing columns for scoped rows: {missing}"
        raise ValueError(msg)
    scoped = tabular_to_scoped_table(
        table,
        columns=columns,
        scope=scope,
        require_scope_columns=require_scope_columns,
    )
    if scoped.num_rows == 0:
        return []
    return list(iter_rows(scoped))


__all__ = ["collect_scoped_rows"]
