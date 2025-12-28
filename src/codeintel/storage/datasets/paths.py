"""Arrow dataset path helpers."""

from __future__ import annotations

from pathlib import Path

from codeintel.storage.helpers.table_key import parse_table_key


class SnapshotIdError(ValueError):
    """Raised when a snapshot identifier is invalid for path construction."""


def dataset_table_dir(dataset_root: Path, *, table_key: str) -> Path:
    """Return the root directory for a dataset table.

    Parameters
    ----------
    dataset_root
        Root directory where Arrow datasets are stored.
    table_key
        Fully qualified table key (schema.table).

    Returns
    -------
    Path
        Directory for the dataset table.
    """
    parsed = parse_table_key(table_key)
    return dataset_root / parsed.schema / parsed.name


def dataset_snapshot_dir(dataset_root: Path, *, table_key: str, snapshot_id: str) -> Path:
    """Return the directory for a dataset snapshot.

    Parameters
    ----------
    dataset_root
        Root directory where Arrow datasets are stored.
    table_key
        Fully qualified table key (schema.table).
    snapshot_id
        Snapshot identifier used in the directory name.

    Returns
    -------
    Path
        Directory for the dataset snapshot.

    Raises
    ------
    SnapshotIdError
        If snapshot_id is empty or contains path separators.
    """
    value = snapshot_id.strip()
    if not value:
        msg = "snapshot_id must be non-empty"
        raise SnapshotIdError(msg)
    if "/" in value or "\\" in value or value in {".", ".."}:
        msg = f"snapshot_id contains invalid characters: {snapshot_id!r}"
        raise SnapshotIdError(msg)
    return dataset_table_dir(dataset_root, table_key=table_key) / f"snapshot_id={snapshot_id}"


__all__ = [
    "SnapshotIdError",
    "dataset_snapshot_dir",
    "dataset_table_dir",
]
