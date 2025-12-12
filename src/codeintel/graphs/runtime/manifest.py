"""Manifest and caching utilities for graph plugin execution.

This module provides utilities for tracking plugin execution state,
computing content hashes for caching, and determining when plugins
can be skipped due to unchanged inputs.

Re-exports common utilities from core and adds graph-specific extensions.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.core.plugins.execution.manifest import (
    ManifestState as CoreManifestState,
)
from codeintel.core.plugins.execution.manifest import (
    PluginExecutionManifest,
)
from codeintel.core.plugins.execution.manifest import (
    is_unchanged as core_is_unchanged,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.config.steps_graphs import GraphRunScope
    from codeintel.graphs.core.protocol import GraphPluginProtocol
    from codeintel.storage.gateway import StorageGateway


log = logging.getLogger(__name__)


@dataclass(frozen=True)
class InputHashPayload:
    """Inputs contributing to a plugin's content hash.

    Graph-specific version that includes scope from GraphRunScope.

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
        Graph execution scope.
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
        log.warning(
            "graph_manifest.options_hash.serialize_failed plugin=%s",
            plugin.metadata.name,
        )
        return None


@dataclass(frozen=True)
class ManifestState:
    """State used for manifest-based skip decisions.

    Graph-specific version with additional context fields.

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
    core_state = CoreManifestState(
        plugin_name=state.plugin_name,
        input_hash=state.input_hash,
        options_hash=state.options_hash,
    )
    return core_is_unchanged(prior_manifest, core_state)


class GraphPluginManifest(PluginExecutionManifest):
    """Manifest tracking plugin execution history.

    Extend core PluginExecutionManifest with graph-specific record() method
    for backward compatibility.

    Attributes
    ----------
    entries
        Map of plugin name to execution metadata.
    """

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


__all__ = [
    "GraphPluginManifest",
    "InputHashPayload",
    "ManifestState",
    "compute_input_hash",
    "compute_options_hash",
    "is_unchanged",
]
