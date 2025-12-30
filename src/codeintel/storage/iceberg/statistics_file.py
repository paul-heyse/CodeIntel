"""Iceberg statistics file persistence helpers."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from pyiceberg.table.puffin import MAGIC_BYTES, Footer
from pyiceberg.table.statistics import StatisticsFile

from codeintel.core.serialization.stable import stable_stringify

if TYPE_CHECKING:
    from collections.abc import Mapping

    from pyiceberg.table import Table

LOG = logging.getLogger(__name__)

_PUFFIN_FLAGS = (0).to_bytes(4, "little")
_STATS_PREFIX = "codeintel.stats."
_SNAPSHOT_PREFIX = "codeintel.snapshot."
_TABLE_KEY_KEY = "codeintel.table_key"
_STATS_JSON_KEY = "codeintel.stats_json"


def persist_iceberg_statistics(
    *,
    table: Table,
    table_key: str,
    stats: Mapping[str, object],
    snapshot_properties: Mapping[str, str],
) -> StatisticsFile | None:
    """Persist derived statistics as a Puffin stats file.

    Parameters
    ----------
    table
        Iceberg table to update with statistics metadata.
    table_key
        Fully qualified table key for the stats payload.
    stats
        Derived statistics payload for the snapshot.
    snapshot_properties
        Snapshot properties applied to the table write.

    Returns
    -------
    StatisticsFile | None
        Persisted statistics file metadata, or None when skipped.
    """
    snapshot_id = stats.get("snapshot_id")
    if not isinstance(snapshot_id, int):
        return None
    properties = _stats_properties(
        stats=stats,
        snapshot_properties=snapshot_properties,
        table_key=table_key,
    )
    puffin_bytes, footer_size = _puffin_bytes(properties)
    location = _statistics_location(table=table, snapshot_id=snapshot_id)
    try:
        output = table.io.new_output(location)
        with output.create(overwrite=False) as handle:
            handle.write(puffin_bytes)
        stats_file = StatisticsFile.model_validate(
            {
                "snapshot-id": snapshot_id,
                "statistics-path": location,
                "file-size-in-bytes": len(puffin_bytes),
                "file-footer-size-in-bytes": footer_size,
                "blob-metadata": [],
            }
        )
        table.update_statistics().set_statistics(stats_file).commit()
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        LOG.warning("Iceberg statistics persistence failed for %s: %s", table_key, exc)
        return None
    return stats_file


def _stats_properties(
    *,
    stats: Mapping[str, object],
    snapshot_properties: Mapping[str, str],
    table_key: str,
) -> dict[str, str]:
    properties: dict[str, str] = {}
    if table_key:
        properties[_TABLE_KEY_KEY] = table_key
    for key, value in snapshot_properties.items():
        if not key:
            continue
        properties[f"{_SNAPSHOT_PREFIX}{key}"] = value
    for key, value in stats.items():
        if value is None or not key:
            continue
        properties[f"{_STATS_PREFIX}{key}"] = stable_stringify(value)
    properties[_STATS_JSON_KEY] = stable_stringify(stats)
    return properties


def _statistics_location(*, table: Table, snapshot_id: int) -> str:
    location_provider = table.location_provider()
    file_name = f"statistics-{snapshot_id}-{uuid.uuid4()}.puffin"
    return location_provider.new_metadata_location(file_name)


def _puffin_bytes(properties: Mapping[str, str]) -> tuple[bytes, int]:
    footer = Footer(blobs=[], properties=dict(properties))
    footer_payload = footer.model_dump_json().encode("utf-8")
    footer_size = len(footer_payload)
    footer_size_bytes = footer_size.to_bytes(4, "little")
    payload = b"".join(
        [
            MAGIC_BYTES,
            _PUFFIN_FLAGS,
            footer_payload,
            footer_size_bytes,
            _PUFFIN_FLAGS,
            MAGIC_BYTES,
        ]
    )
    return payload, footer_size


__all__ = ["persist_iceberg_statistics"]
