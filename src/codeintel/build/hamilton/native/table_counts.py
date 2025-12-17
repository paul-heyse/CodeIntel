"""Normalize and validate native target table row count mappings."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def normalize_table_counts(
    expected_table_keys: tuple[str, ...],
    counts: Mapping[str, int] | None,
) -> dict[str, int]:
    """Normalize a row count mapping to exactly the expected table keys.

    Parameters
    ----------
    expected_table_keys
        Contract table keys the target is expected to produce.
    counts
        Observed row counts, keyed by table key.

    Returns
    -------
    dict[str, int]
        Mapping containing every expected key (missing keys set to 0).

    Raises
    ------
    ValueError
        If the observed mapping contains unexpected table keys.
    """
    row_counts = dict.fromkeys(expected_table_keys, 0)
    if counts is None:
        return row_counts

    for table_key, row_count in counts.items():
        if table_key not in row_counts:
            msg = (
                f"Unexpected table_counts key: {table_key}. "
                f"Expected one of: {sorted(expected_table_keys)}"
            )
            raise ValueError(msg)
        row_counts[table_key] = int(row_count)
    return row_counts


__all__ = [
    "normalize_table_counts",
]
