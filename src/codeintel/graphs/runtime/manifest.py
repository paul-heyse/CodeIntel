"""Manifest and caching utilities for graph plugin execution.

This module provides utilities for tracking plugin execution state,
computing content hashes for caching, and determining when plugins
can be skipped due to unchanged inputs.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.core.plugins.types.result import PluginExecutionRecord

if TYPE_CHECKING:
    from codeintel.config.steps_graphs import GraphRunScope
    from codeintel.graphs.core.protocol import GraphPluginProtocol
    from codeintel.storage.gateway import StorageGateway


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputHashPayload:
    """Inputs contributing to a plugin's content hash.

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
    scope
        Execution scope.
    options_hash
        Hash of plugin options.
    """

    repo: str
    commit: str
    plugin_name: str
    version_hash: str | None
    scope: GraphRunScope
    options_hash: str | None


def compute_input_hash(payload: InputHashPayload) -> str:
    """Compute a hash of the plugin's inputs for caching.

    Parameters
    ----------
    payload
        Payload containing all inputs to hash.

    Returns
    -------
    str
        SHA-256 hash of serialized inputs.
    """
    serialized = json.dumps(
        {
            "repo": payload.repo,
            "commit": payload.commit,
            "plugin_name": payload.plugin_name,
            "version_hash": payload.version_hash,
            "scope_paths": payload.scope.paths,
            "scope_modules": payload.scope.modules,
            "options_hash": payload.options_hash,
        },
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


def compute_options_hash(
    plugin: GraphPluginProtocol,
    options: object | None,
) -> str | None:
    """Compute a hash of plugin options.

    Parameters
    ----------
    plugin
        Plugin whose options are being hashed.
    options
        Options value to hash.

    Returns
    -------
    str | None
        SHA-256 hash of serialized options if present.
    """
    if options is None:
        return None
    try:
        serialized = json.dumps(
            {"plugin_name": plugin.metadata.name, "options": options},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]
    except (TypeError, ValueError):
        log.warning("graph_manifest.options_hash.serialize_failed plugin=%s", plugin.metadata.name)
        return None


@dataclass(frozen=True)
class ManifestState:
    """State used for manifest-based skip decisions.

    Attributes
    ----------
    plugin_name
        Name of the plugin.
    row_count_tables
        Tables to query for row counts.
    gateway
        Storage gateway.
    repo
        Repository identifier.
    commit
        Commit SHA.
    input_hash
        Input hash for comparison.
    options_hash
        Options hash for comparison.
    """

    plugin_name: str
    row_count_tables: tuple[str, ...]
    gateway: StorageGateway
    repo: str
    commit: str
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


@dataclass
class RecordParams:
    """Parameters for constructing execution records.

    Attributes
    ----------
    severity
        Plugin severity level.
    timeout_ms
        Timeout in milliseconds.
    version_hash
        Plugin version hash.
    input_hash
        Input content hash.
    options_hash
        Options content hash.
    options
        Plugin options value.
    requires_isolation
        Whether isolation is required.
    isolation_kind
        Kind of isolation.
    policy_fail_fast
        Whether fail-fast is enabled.
    """

    severity: str
    timeout_ms: int | None
    version_hash: str | None
    input_hash: str | None
    options_hash: str | None
    options: object | None
    requires_isolation: bool = False
    isolation_kind: str | None = None
    policy_fail_fast: bool = True


def dry_run_record(
    *,
    plugin: GraphPluginProtocol,
    params: RecordParams,
) -> PluginExecutionRecord:
    """Create a record for a dry-run execution.

    Parameters
    ----------
    plugin
        Plugin being recorded.
    params
        Record parameters.

    Returns
    -------
    PluginExecutionRecord
        Record marked as skipped due to dry-run.
    """
    now = datetime.now(tz=UTC)
    return PluginExecutionRecord(
        plugin_name=plugin.metadata.name,
        status="skipped",
        started_at=now,
        ended_at=now,
        duration_ms=0.0,
        attempts=0,
        partial=False,
        error=None,
        meta={
            "skipped_reason": "dry_run",
            "severity": params.severity,
            "timeout_ms": params.timeout_ms,
            "version_hash": params.version_hash,
            "input_hash": params.input_hash,
            "options_hash": params.options_hash,
        },
    )


def skip_record(
    *,
    plugin: GraphPluginProtocol,
    params: RecordParams,
    reason: str,
) -> PluginExecutionRecord:
    """Create a record for a skipped execution.

    Parameters
    ----------
    plugin
        Plugin being recorded.
    params
        Record parameters.
    reason
        Reason for skipping.

    Returns
    -------
    PluginExecutionRecord
        Record marked as skipped with the specified reason.
    """
    now = datetime.now(tz=UTC)
    return PluginExecutionRecord(
        plugin_name=plugin.metadata.name,
        status="skipped",
        started_at=now,
        ended_at=now,
        duration_ms=0.0,
        attempts=0,
        partial=False,
        error=None,
        meta={
            "skipped_reason": reason,
            "severity": params.severity,
            "timeout_ms": params.timeout_ms,
            "version_hash": params.version_hash,
            "input_hash": params.input_hash,
            "options_hash": params.options_hash,
        },
    )


@dataclass
class GraphPluginManifest:
    """Manifest tracking plugin execution history.

    Attributes
    ----------
    entries
        Map of plugin name to execution metadata.
    """

    entries: dict[str, dict[str, object]] = field(default_factory=dict)

    def record(
        self,
        plugin_name: str,
        input_hash: str | None,
        options_hash: str | None,
        version_hash: str | None,
        row_counts: dict[str, int] | None,
    ) -> None:
        """Record execution metadata for a plugin.

        Parameters
        ----------
        plugin_name
            Name of the plugin.
        input_hash
            Computed input hash.
        options_hash
            Computed options hash.
        version_hash
            Plugin version hash.
        row_counts
            Table row counts produced.
        """
        self.entries[plugin_name] = {
            "input_hash": input_hash,
            "options_hash": options_hash,
            "version_hash": version_hash,
            "row_counts": row_counts,
            "recorded_at": datetime.now(tz=UTC).isoformat(),
        }

    def to_dict(self) -> dict[str, dict[str, object]]:
        """Return manifest entries as a dictionary.

        Returns
        -------
        dict[str, dict[str, object]]
            Manifest entries.
        """
        return dict(self.entries)


__all__ = [
    "GraphPluginManifest",
    "InputHashPayload",
    "ManifestState",
    "RecordParams",
    "compute_input_hash",
    "compute_options_hash",
    "dry_run_record",
    "is_unchanged",
    "skip_record",
]
