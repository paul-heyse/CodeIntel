"""Schema observation provider backed by build metadata bundles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import cast

from codeintel.core.schemas.resolution import SchemaObservationProvider
from codeintel.core.schemas.schema_catalog_models import (
    ColumnStatsPayload,
    DatasetStatsPayload,
    DerivedSettingsPayload,
    SchemaObservationRecord,
)


@dataclass
class BundleSchemaObservationProvider(SchemaObservationProvider):
    """Load schema observations from a build metadata bundle."""

    bundle_root: Path
    _cache: dict[str, SchemaObservationRecord] = field(default_factory=dict, init=False, repr=False)
    _loaded: bool = field(default=False, init=False, repr=False)

    def load_latest_schema_observation(
        self,
        *,
        table_key: str,
    ) -> SchemaObservationRecord | None:
        """Return the latest observation for a table key.

        Returns
        -------
        SchemaObservationRecord | None
            Latest observation for the table key, if available.
        """
        if not self._loaded:
            self._load_cache()
        return self._cache.get(table_key)

    def _load_cache(self) -> None:
        self._loaded = True
        path = self.bundle_root / "schema" / "schema_observations.jsonl"
        if not path.is_file():
            return
        latest: dict[str, tuple[datetime | None, int, SchemaObservationRecord]] = {}
        line_index = 0
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                line_index += 1
                try:
                    payload = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                record = _record_from_payload(payload)
                if record is None:
                    continue
                observed_at = record.observed_at
                existing = latest.get(record.table_key)
                if existing is None:
                    latest[record.table_key] = (observed_at, line_index, record)
                    continue
                prev_time, prev_index, _ = existing
                if _is_newer(observed_at, line_index, prev_time, prev_index):
                    latest[record.table_key] = (observed_at, line_index, record)
        self._cache = {key: record for key, (_, __, record) in latest.items()}


def _record_from_payload(payload: object) -> SchemaObservationRecord | None:
    if not isinstance(payload, Mapping):
        return None
    table_key = _optional_str(payload.get("table_key"))
    schema_digest = _optional_str(payload.get("schema_digest"))
    schema_hash = _optional_str(payload.get("schema_hash"))
    ipc_payload = _optional_str(payload.get("arrow_schema_ipc_b64"))
    if not (table_key and schema_digest and schema_hash and ipc_payload):
        return None
    return SchemaObservationRecord(
        table_key=table_key,
        schema_digest=schema_digest,
        schema_hash=schema_hash,
        arrow_schema_ipc_b64=ipc_payload,
        repo=_optional_str(payload.get("repo")),
        commit=_optional_str(payload.get("commit")),
        target_name=_optional_str(payload.get("target_name")),
        column_stats=_optional_column_stats(payload.get("column_stats")),
        dataset_stats=_optional_dataset_stats(payload.get("dataset_stats")),
        derived_settings=_optional_derived_settings(payload.get("derived_settings")),
        drift_summary=_optional_mapping(payload.get("drift_summary")),
        observed_at=_parse_datetime(payload.get("observed_at")),
        observation_id=_optional_str(payload.get("observation_id")),
    )


def _is_newer(
    observed_at: datetime | None,
    index: int,
    prev_time: datetime | None,
    prev_index: int,
) -> bool:
    if observed_at is None and prev_time is None:
        return index > prev_index
    if observed_at is None:
        return False
    if prev_time is None:
        return True
    if observed_at == prev_time:
        return index > prev_index
    return observed_at > prev_time


def _parse_datetime(value: object) -> datetime | None:
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _optional_str(value: object) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            return stripped
    return None


def _optional_mapping(value: object) -> dict[str, object] | None:
    if isinstance(value, Mapping):
        return {str(key): item for key, item in value.items()}
    return None


def _optional_column_stats(value: object) -> ColumnStatsPayload | None:
    mapping = _optional_mapping(value)
    if mapping is None:
        return None
    return cast("ColumnStatsPayload", mapping)


def _optional_dataset_stats(value: object) -> DatasetStatsPayload | None:
    mapping = _optional_mapping(value)
    if mapping is None:
        return None
    return cast("DatasetStatsPayload", mapping)


def _optional_derived_settings(value: object) -> DerivedSettingsPayload | None:
    mapping = _optional_mapping(value)
    if mapping is None:
        return None
    return cast("DerivedSettingsPayload", mapping)


__all__ = ["BundleSchemaObservationProvider"]
