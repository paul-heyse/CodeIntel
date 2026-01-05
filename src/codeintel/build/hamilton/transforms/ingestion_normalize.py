"""Normalization utilities for ingestion Arrow outputs."""

from __future__ import annotations

import pyarrow as pa

from codeintel.build.tabular.arrow_ops import align_table_to_contract, dedupe_table_for_table
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput


def normalize_ingest_frame(
    frame: InferableTabularInput | None,
    *,
    table_key: str,
    add_missing: bool = True,
    keep_extras: bool | None = None,
) -> pa.Table | None:
    """Normalize ingestion frames for schema alignment and deduping.

    Parameters
    ----------
    frame
        Tabular input to normalize (None means skip).
    table_key
        Target table key for schema alignment.
    add_missing
        Whether to add missing schema columns as nulls.
    keep_extras
        Whether to keep extra columns not in the schema. When None, respect
        the extras policy encoded in the Arrow schema metadata.

    Returns
    -------
    pa.Table | None
        Normalized Arrow table or None if input is None.
    """
    if frame is None:
        return None
    table = tabular_to_arrow_table(frame)
    if table.num_rows == 0:
        return pa.Table.from_batches([], schema=table.schema)
    extras_policy = None
    if keep_extras is True:
        extras_policy = "retain"
    elif keep_extras is False:
        extras_policy = "drop"
    aligned = (
        align_table_to_contract(table_key, table, extras_policy=extras_policy)
        if add_missing or extras_policy is not None
        else table
    )
    return dedupe_table_for_table(table_key, aligned)


__all__ = ["normalize_ingest_frame"]
