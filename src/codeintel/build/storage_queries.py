"""Safe, centralized storage query helpers for build-time introspection.

The build system frequently needs to inspect persisted outputs (e.g., row counts) after a target
materializes. This module provides hardened helpers that avoid ad hoc SQL string building and
normalize snapshot filtering semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from codeintel.storage.ibis_types import and_predicates

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway.protocol import StorageGateway


def _coerce_scalar_int(raw: object) -> int:
    if isinstance(raw, pd.DataFrame):
        if raw.empty:
            return 0
        value = raw.iloc[0, 0]
        return int(str(value))
    if isinstance(raw, pd.Series):
        if raw.empty:
            return 0
        value = raw.iloc[0]
        return int(str(value))
    return int(str(raw))


def count_rows_for_snapshot(
    gateway: StorageGateway,
    *,
    table_key: str,
    snapshot: SnapshotRef,
) -> int:
    """Count rows for a table constrained to a specific repo+commit snapshot.

    Parameters
    ----------
    gateway
        Storage gateway used to construct the query.
    table_key
        Fully-qualified table key (e.g., "analytics.subsystems").
    snapshot
        Snapshot reference providing ``repo`` and ``commit`` filters.

    Returns
    -------
    int
        Row count for the table within the snapshot.

    Raises
    ------
    ValueError
        If the table is missing required snapshot columns.
    """
    table = gateway.ibis.table(table_key)
    try:
        filtered = table.filter(
            and_predicates(table.repo == snapshot.repo, table.commit == snapshot.commit)
        )
    except AttributeError as exc:
        msg = f"Table does not expose required snapshot columns: {table_key}"
        raise ValueError(msg) from exc

    return _coerce_scalar_int(filtered.count().execute())
