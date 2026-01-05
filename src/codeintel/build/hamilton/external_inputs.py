"""External input allowlist helpers for seedless builds."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml

_ALLOWLIST_RELATIVE_PATH = Path("config/registry/external_inputs_allowlist.yaml")


@dataclass(frozen=True, slots=True)
class ExternalInputEntry:
    """Single external input allowlist entry."""

    table_key: str
    reason: str | None = None
    owner: str | None = None


@dataclass(frozen=True, slots=True)
class ExternalInputsAllowlist:
    """Parsed external input allowlist."""

    version: int
    entries: tuple[ExternalInputEntry, ...]

    def table_keys(self) -> frozenset[str]:
        """Return the set of allowlisted table keys.

        Returns
        -------
        frozenset[str]
            Allowlisted table keys.
        """
        return frozenset(entry.table_key for entry in self.entries)


def load_external_inputs_allowlist(*, repo_root: Path) -> ExternalInputsAllowlist:
    """Load the external input allowlist from the registry.

    Returns
    -------
    ExternalInputsAllowlist
        Parsed allowlist payload (empty when missing).
    """
    path = repo_root / _ALLOWLIST_RELATIVE_PATH
    if not path.is_file():
        return ExternalInputsAllowlist(version=1, entries=())
    raw_text = path.read_text(encoding="utf8")
    payload = yaml.safe_load(raw_text) or {}
    return _parse_allowlist_payload(payload, path=path)


def _parse_allowlist_payload(
    payload: object,
    *,
    path: Path,
) -> ExternalInputsAllowlist:
    if not isinstance(payload, Mapping):
        msg = f"External inputs allowlist must be a mapping: {path}"
        raise TypeError(msg)
    version = payload.get("version", 1)
    if not isinstance(version, int):
        msg = f"External inputs allowlist version must be an int: {path}"
        raise TypeError(msg)
    entries_raw = payload.get("external_inputs")
    entries = _parse_allowlist_entries(entries_raw, path=path)
    return ExternalInputsAllowlist(version=version, entries=tuple(entries))


def _parse_allowlist_entries(
    entries_raw: object,
    *,
    path: Path,
) -> list[ExternalInputEntry]:
    if entries_raw is None:
        return []
    if not isinstance(entries_raw, list):
        msg = f"external_inputs must be a list in {path}"
        raise TypeError(msg)
    return [_parse_allowlist_entry(entry, path=path) for entry in entries_raw]


def _parse_allowlist_entry(entry: object, *, path: Path) -> ExternalInputEntry:
    if not isinstance(entry, Mapping):
        msg = f"external_inputs entries must be mappings in {path}"
        raise TypeError(msg)
    table_key = entry.get("table_key")
    if not isinstance(table_key, str) or not table_key:
        msg = f"external_inputs entry missing table_key in {path}"
        raise ValueError(msg)
    reason = entry.get("reason")
    owner = entry.get("owner")
    if reason is not None and not isinstance(reason, str):
        msg = f"external_inputs reason must be a string in {path}"
        raise TypeError(msg)
    if owner is not None and not isinstance(owner, str):
        msg = f"external_inputs owner must be a string in {path}"
        raise TypeError(msg)
    return ExternalInputEntry(table_key=table_key, reason=reason, owner=owner)


__all__ = ["ExternalInputEntry", "ExternalInputsAllowlist", "load_external_inputs_allowlist"]
