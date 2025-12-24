"""Shard manifest utilities for incremental SCIP indexing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Mapping


MANIFEST_VERSION = 1


@dataclass(frozen=True)
class ScipShardRecord:
    """Metadata for a single per-module SCIP shard."""

    rel_path: str
    content_hash: str
    options_hash: str | None
    tool_version: str | None
    shard_path: str
    updated_at: datetime

    def to_dict(self) -> dict[str, object]:
        """Serialize the record to a JSON-safe mapping.

        Returns
        -------
        dict[str, object]
            JSON-serializable record payload.
        """
        return {
            "rel_path": self.rel_path,
            "content_hash": self.content_hash,
            "options_hash": self.options_hash,
            "tool_version": self.tool_version,
            "shard_path": self.shard_path,
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScipShardRecord:
        """Parse a shard record from a JSON mapping.

        Returns
        -------
        ScipShardRecord
            Parsed shard record instance.
        """
        updated_at_raw = payload.get("updated_at")
        updated_at = (
            datetime.fromisoformat(str(updated_at_raw))
            if isinstance(updated_at_raw, str) and updated_at_raw
            else datetime.now(tz=UTC)
        )
        return cls(
            rel_path=str(payload.get("rel_path", "")),
            content_hash=str(payload.get("content_hash", "")),
            options_hash=_coerce_optional_str(payload.get("options_hash")),
            tool_version=_coerce_optional_str(payload.get("tool_version")),
            shard_path=str(payload.get("shard_path", "")),
            updated_at=updated_at,
        )


@dataclass(frozen=True)
class ScipShardManifest:
    """Manifest of per-module SCIP shards."""

    records: dict[str, ScipShardRecord]
    generated_at: datetime
    version: int = MANIFEST_VERSION

    def to_dict(self) -> dict[str, object]:
        """Serialize the manifest to a JSON-safe mapping.

        Returns
        -------
        dict[str, object]
            JSON-serializable manifest payload.
        """
        return {
            "version": self.version,
            "generated_at": self.generated_at.isoformat(),
            "records": {key: record.to_dict() for key, record in self.records.items()},
        }

    @classmethod
    def empty(cls) -> ScipShardManifest:
        """Return an empty manifest.

        Returns
        -------
        ScipShardManifest
            Manifest with no shard records.
        """
        return cls(records={}, generated_at=datetime.now(tz=UTC))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ScipShardManifest:
        """Parse a manifest from JSON payload.

        Returns
        -------
        ScipShardManifest
            Manifest populated from the payload.
        """
        records_raw = payload.get("records", {})
        records: dict[str, ScipShardRecord] = {}
        if isinstance(records_raw, dict):
            for rel_path, record_payload in records_raw.items():
                if isinstance(record_payload, Mapping):
                    records[str(rel_path)] = ScipShardRecord.from_dict(record_payload)
        generated_at_raw = payload.get("generated_at")
        generated_at = (
            datetime.fromisoformat(str(generated_at_raw))
            if isinstance(generated_at_raw, str) and generated_at_raw
            else datetime.now(tz=UTC)
        )
        version = int(payload.get("version", MANIFEST_VERSION))
        return cls(records=records, generated_at=generated_at, version=version)


def manifest_path(scip_dir: Path) -> Path:
    """Return the default shard manifest path.

    Returns
    -------
    Path
        Path to the manifest JSON file.
    """
    return scip_dir / "shards" / "manifest.json"


def shard_path(scip_dir: Path, *, rel_path: str, content_hash: str) -> Path:
    """Return the canonical shard path for a module.

    Returns
    -------
    Path
        Canonical shard path for the module content hash.
    """
    prefix = content_hash[:2] if content_hash else "00"
    sanitized = rel_path.replace("/", "__").replace("\\", "__")
    filename = f"{content_hash}__{sanitized}.scip"
    return scip_dir / "shards" / prefix / filename


def load_manifest(path: Path) -> ScipShardManifest:
    """Load a shard manifest from disk.

    Returns
    -------
    ScipShardManifest
        Loaded manifest or an empty manifest when missing/invalid.
    """
    if not path.is_file():
        return ScipShardManifest.empty()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return ScipShardManifest.empty()
    return ScipShardManifest.from_dict(payload)


def write_manifest(path: Path, manifest: ScipShardManifest) -> None:
    """Write the shard manifest atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    payload = json.dumps(manifest.to_dict(), indent=2, sort_keys=True)
    tmp_path.write_text(payload, encoding="utf-8")
    tmp_path.replace(path)


def update_manifest(
    manifest: ScipShardManifest,
    *,
    updates: Mapping[str, ScipShardRecord],
    deleted: Mapping[str, object] | None = None,
) -> ScipShardManifest:
    """Return a new manifest updated with shard changes.

    Returns
    -------
    ScipShardManifest
        Updated manifest with applied shard changes.
    """
    records = dict(manifest.records)
    for rel_path, record in updates.items():
        records[rel_path] = record
    if deleted:
        for rel_path in deleted:
            records.pop(rel_path, None)
    return ScipShardManifest(records=records, generated_at=datetime.now(tz=UTC))


def _coerce_optional_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


__all__ = [
    "ScipShardManifest",
    "ScipShardRecord",
    "load_manifest",
    "manifest_path",
    "shard_path",
    "update_manifest",
    "write_manifest",
]
