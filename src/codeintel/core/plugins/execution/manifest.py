"""Manifest and caching utilities for plugin execution.

This module provides utilities for tracking plugin execution state,
computing content hashes for caching, and determining when plugins
can be skipped due to unchanged inputs.

These utilities are domain-agnostic and can be used by any plugin
executor that wants to implement manifest-based skip detection.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime

from codeintel.core.plugins.types.result import PluginExecutionRecord

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputHashPayload:
    """Inputs contributing to a plugin's content hash.

    A generic payload for computing content hashes that can be extended
    with domain-specific fields via the extra_fields parameter.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    plugin_name
        Name of the plugin.
    version_hash
        Plugin version hash.
    options_hash
        Hash of plugin options.
    extra_fields
        Additional domain-specific fields to include in the hash.

    Examples
    --------
    >>> payload = InputHashPayload(
    ...     repo="myrepo",
    ...     commit="abc123",
    ...     plugin_name="my_plugin",
    ...     version_hash="v1",
    ...     options_hash=None,
    ...     extra_fields={"scope_paths": ("src/",)},
    ... )
    >>> compute_input_hash(payload)
    '...'
    """

    repo: str
    commit: str
    plugin_name: str
    version_hash: str | None
    options_hash: str | None
    extra_fields: Mapping[str, object] = field(default_factory=dict)


def compute_input_hash(payload: InputHashPayload) -> str:
    """Compute a hash of the plugin's inputs for caching.

    Parameters
    ----------
    payload
        Payload containing all inputs to hash.

    Returns
    -------
    str
        SHA-256 hash of serialized inputs (first 16 chars).

    Examples
    --------
    >>> payload = InputHashPayload(
    ...     repo="myrepo",
    ...     commit="abc123",
    ...     plugin_name="my_plugin",
    ...     version_hash="v1",
    ...     options_hash=None,
    ... )
    >>> len(compute_input_hash(payload))
    16
    """
    data: dict[str, object] = {
        "repo": payload.repo,
        "commit": payload.commit,
        "plugin_name": payload.plugin_name,
        "version_hash": payload.version_hash,
        "options_hash": payload.options_hash,
    }
    # Include any extra fields
    data.update(payload.extra_fields)

    serialized = json.dumps(data, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def compute_options_hash(
    plugin_name: str,
    options: object | None,
) -> str | None:
    """Compute a hash of plugin options.

    Parameters
    ----------
    plugin_name
        Name of the plugin.
    options
        Options value to hash.

    Returns
    -------
    str | None
        SHA-256 hash of serialized options if present (first 16 chars).

    Examples
    --------
    >>> compute_options_hash("my_plugin", {"threshold": 0.5})
    '...'
    >>> compute_options_hash("my_plugin", None) is None
    True
    """
    if options is None:
        return None
    try:
        serialized = json.dumps(
            {"plugin_name": plugin_name, "options": options},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    except (TypeError, ValueError):
        log.warning("manifest.options_hash.serialize_failed plugin=%s", plugin_name)
        return None


@dataclass(frozen=True)
class ManifestState:
    """State used for manifest-based skip decisions.

    Contains all information needed to determine if a plugin's inputs
    have changed since the last execution.

    Attributes
    ----------
    plugin_name
        Name of the plugin.
    input_hash
        Input hash for comparison.
    options_hash
        Options hash for comparison.

    Examples
    --------
    >>> state = ManifestState(
    ...     plugin_name="my_plugin",
    ...     input_hash="abc123",
    ...     options_hash=None,
    ... )
    >>> is_unchanged(prior_manifest={"my_plugin": {"input_hash": "abc123"}}, state=state)
    True
    """

    plugin_name: str
    input_hash: str | None
    options_hash: str | None


def is_unchanged(
    prior_manifest: Mapping[str, Mapping[str, object]] | None,
    state: ManifestState,
) -> bool:
    """Check if plugin inputs have changed since the last run.

    Parameters
    ----------
    prior_manifest
        Manifest from prior execution keyed by plugin name.
    state
        Current manifest state.

    Returns
    -------
    bool
        True if inputs are unchanged and execution can be skipped.

    Examples
    --------
    >>> prior = {"my_plugin": {"input_hash": "abc", "options_hash": None}}
    >>> state = ManifestState("my_plugin", "abc", None)
    >>> is_unchanged(prior, state)
    True
    >>> state_changed = ManifestState("my_plugin", "xyz", None)
    >>> is_unchanged(prior, state_changed)
    False
    """
    if prior_manifest is None:
        return False
    prior = prior_manifest.get(state.plugin_name)
    if prior is None:
        return False
    prior_input_hash = prior.get("input_hash")
    prior_options_hash = prior.get("options_hash")
    if state.input_hash is None or prior_input_hash is None:
        return False
    if state.input_hash != prior_input_hash:
        return False
    return state.options_hash == prior_options_hash


def create_skip_record(
    *,
    plugin_name: str,
    reason: str,
    meta: dict[str, object] | None = None,
) -> PluginExecutionRecord:
    """Create an execution record for a skipped plugin.

    Parameters
    ----------
    plugin_name
        Name of the plugin.
    reason
        Reason for skipping.
    meta
        Additional metadata to include.

    Returns
    -------
    PluginExecutionRecord
        Record marked as skipped with the specified reason.

    Examples
    --------
    >>> record = create_skip_record(
    ...     plugin_name="my_plugin",
    ...     reason="unchanged",
    ...     meta={"input_hash": "abc"},
    ... )
    >>> record.status
    'skipped'
    """
    now = datetime.now(tz=UTC)
    record_meta: dict[str, object] = {"skipped_reason": reason}
    if meta:
        record_meta.update(meta)

    return PluginExecutionRecord(
        plugin_name=plugin_name,
        status="skipped",
        started_at=now,
        ended_at=now,
        duration_ms=0.0,
        attempts=0,
        partial=False,
        error=None,
        meta=record_meta,
    )


@dataclass
class PluginExecutionManifest:
    """Manifest tracking plugin execution history.

    A generic manifest that can be used by any plugin executor
    to track execution metadata for cache invalidation.

    Attributes
    ----------
    entries
        Map of plugin name to execution metadata.

    Examples
    --------
    >>> manifest = PluginExecutionManifest()
    >>> manifest.record(
    ...     plugin_name="my_plugin",
    ...     input_hash="abc123",
    ...     options_hash=None,
    ...     version_hash="v1",
    ...     row_counts={"table1": 100},
    ... )
    >>> manifest.entries["my_plugin"]["input_hash"]
    'abc123'
    """

    entries: dict[str, dict[str, object]] = field(default_factory=dict)

    def record_from_state(
        self,
        state: ManifestState,
        version_hash: str | None,
        row_counts: dict[str, int] | None,
    ) -> None:
        """Record execution metadata for a plugin from state.

        Parameters
        ----------
        state
            Manifest state with plugin name and hashes.
        version_hash
            Plugin version hash.
        row_counts
            Table row counts produced.
        """
        self.entries[state.plugin_name] = {
            "input_hash": state.input_hash,
            "options_hash": state.options_hash,
            "version_hash": version_hash,
            "row_counts": row_counts,
            "recorded_at": datetime.now(tz=UTC).isoformat(),
        }

    def record_entry(self, plugin_name: str, entry: dict[str, object]) -> None:
        """Record a pre-built manifest entry.

        Parameters
        ----------
        plugin_name
            Name of the plugin.
        entry
            Manifest entry dictionary.
        """
        self.entries[plugin_name] = entry

    def get(self, plugin_name: str) -> dict[str, object] | None:
        """Get manifest entry for a plugin.

        Parameters
        ----------
        plugin_name
            Name of the plugin.

        Returns
        -------
        dict[str, object] | None
            Manifest entry or None if not found.
        """
        return self.entries.get(plugin_name)

    def to_dict(self) -> dict[str, dict[str, object]]:
        """Return manifest entries as a dictionary.

        Returns
        -------
        dict[str, dict[str, object]]
            Manifest entries.
        """
        return dict(self.entries)

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Mapping[str, object]] | None,
    ) -> PluginExecutionManifest:
        """Create manifest from dictionary.

        Parameters
        ----------
        data
            Dictionary of manifest entries.

        Returns
        -------
        PluginExecutionManifest
            New manifest instance.
        """
        manifest = cls()
        if data:
            manifest.entries = {k: dict(v) for k, v in data.items()}
        return manifest


def build_manifest_entry(
    record: PluginExecutionRecord,
    *,
    input_hash: str | None = None,
    options_hash: str | None = None,
    version_hash: str | None = None,
) -> dict[str, object]:
    """Build a manifest entry from an execution record.

    Parameters
    ----------
    record
        Plugin execution record.
    input_hash
        Input hash to include.
    options_hash
        Options hash to include.
    version_hash
        Version hash to include.

    Returns
    -------
    dict[str, object]
        Manifest entry suitable for storage.
    """
    row_counts = (
        dict(record.result.row_counts)
        if record.result and record.result.row_counts
        else None
    )
    return {
        "input_hash": input_hash or record.meta.get("input_hash"),
        "options_hash": options_hash or record.meta.get("options_hash"),
        "version_hash": version_hash or record.meta.get("version_hash"),
        "row_counts": row_counts,
        "executed_at": record.ended_at.isoformat() if record.ended_at else None,
    }


__all__ = [
    "InputHashPayload",
    "ManifestState",
    "PluginExecutionManifest",
    "build_manifest_entry",
    "compute_input_hash",
    "compute_options_hash",
    "create_skip_record",
    "is_unchanged",
]
