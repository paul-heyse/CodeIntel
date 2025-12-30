"""Helpers for building Iceberg snapshot properties."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.serialization.stable import stable_stringify

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True, slots=True)
class SnapshotPropertyInputs:
    """Inputs used to populate snapshot properties."""

    table_key: str
    repo: str | None = None
    commit: str | None = None
    run_id: str | None = None
    target_name: str | None = None
    schema_hash: str | None = None
    producer_version: str | None = None
    write_settings: Mapping[str, object] | None = None


def snapshot_properties_for_write(inputs: SnapshotPropertyInputs) -> dict[str, str]:
    """Build stable snapshot properties for Iceberg writes.

    Returns
    -------
    dict[str, str]
        Snapshot property payload suitable for Iceberg writes.
    """
    properties = {
        "run_id": inputs.run_id,
        "commit": inputs.commit,
        "repo": inputs.repo,
        "table_key": inputs.table_key,
        "target_name": inputs.target_name,
        "schema_hash": inputs.schema_hash,
        "producer_version": inputs.producer_version,
    }
    payload = {key: stable_stringify(value) for key, value in properties.items() if value}
    payload.update(_write_settings_payload(inputs.write_settings))
    return payload


def _write_settings_payload(write_settings: Mapping[str, object] | None) -> dict[str, str]:
    if not write_settings:
        return {}
    payload: dict[str, str] = {}
    for key, value in write_settings.items():
        if value is None:
            continue
        payload[f"ci.write.{key}"] = stable_stringify(value)
    return payload


__all__ = ["SnapshotPropertyInputs", "snapshot_properties_for_write"]
