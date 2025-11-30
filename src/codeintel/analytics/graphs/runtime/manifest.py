"""Manifest and unchanged-detection utilities for graph runtime."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from codeintel.analytics.graphs.contracts import (
    ContractChecker,
    PluginContractResult,
    run_contract_checkers,
)
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
)
from codeintel.analytics.graphs.runtime.model import GraphPluginRunRecord
from codeintel.config.steps_graphs import GraphRunScope
from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class InputHashPayload:
    """Payload required to compute an input hash for a plugin."""

    repo: str
    commit: str
    plugin_name: str
    version_hash: str | None
    scope: GraphRunScope | None
    options_hash: str | None


@dataclass(frozen=True)
class ManifestState:
    """State required to decide if a plugin should be skipped as unchanged."""

    plugin_name: str
    row_count_tables: Sequence[str]
    gateway: StorageGateway | None
    repo: str
    commit: str
    input_hash: str | None
    options_hash: str | None


@dataclass(frozen=True)
class RecordParams:
    """Reusable parameters for building GraphPluginRunRecord instances."""

    severity: Literal["fatal", "soft_fail", "skip_on_error"]
    timeout_ms: int | None
    version_hash: str | None
    input_hash: str | None
    options_hash: str | None
    options: object | None
    requires_isolation: bool
    isolation_kind: str | None
    policy_fail_fast: bool


def hash_json(payload: object) -> str:
    """
    Return a stable SHA256 hash for the given JSON-serializable payload.

    Returns
    -------
    str
        Hex digest of the serialized payload.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_options_hash(plugin: GraphMetricPlugin, options: object | None) -> str | None:
    """
    Compute a stable hash over plugin options, including the plugin name.

    Returns
    -------
    str | None
        Hex digest of the options payload or None when options are missing.
    """
    if options is None:
        return None
    return hash_json({"plugin": plugin.name, "options": options})


def compute_input_hash(
    payload: InputHashPayload,
) -> str:
    """
    Compute an input hash over repo, commit, scope, options, and plugin version.

    Returns
    -------
    str
        Hex digest representing the plugin inputs.
    """
    scope_payload: dict[str, object] | None = None
    if payload.scope is not None:
        scope_payload = {
            "paths": payload.scope.paths,
            "modules": payload.scope.modules,
            "time_window": (
                (
                    payload.scope.time_window[0].isoformat(),
                    payload.scope.time_window[1].isoformat(),
                )
                if payload.scope.time_window is not None
                else None
            ),
        }
    parts = {
        "repo": payload.repo,
        "commit": payload.commit,
        "plugin": payload.plugin_name,
        "version_hash": payload.version_hash or "0",
        "options_hash": payload.options_hash,
        "scope": scope_payload,
    }
    serialized = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def row_counts_equal(
    current: dict[str, int] | None,
    prior: object,
) -> bool:
    """
    Compare row-count dictionaries while tolerating missing or malformed entries.

    Returns
    -------
    bool
        True when row counts match or are unavailable; otherwise False.
    """
    if current is None:
        return True
    if not isinstance(prior, dict):
        return True
    for table, count in current.items():
        if prior.get(table) != count:
            return False
    for table, count in prior.items():
        if not isinstance(count, int):
            return False
        if table not in current or current[table] != count:
            return False
    return True


def current_row_counts(
    gateway: StorageGateway | None,
    tables: Sequence[str],
    *,
    repo: str,
    commit: str,
) -> dict[str, int] | None:
    """
    Compute row counts for the provided tables scoped to repo/commit.

    Returns
    -------
    dict[str, int] | None
        Mapping of table name to row count, or None when unavailable.
    """
    if gateway is None or not tables:
        return None
    counts: dict[str, int] = {}
    connection = getattr(gateway, "con", None)
    if connection is None:
        return None
    for table in tables:
        try:
            escaped_repo = repo.replace("'", "''")
            escaped_commit = commit.replace("'", "''")
            relation = connection.table(table).filter(
                f"repo = '{escaped_repo}' AND commit = '{escaped_commit}'"
            )
            count = relation.aggregate("count(*)").fetchone()[0]
            counts[table] = int(count)
        except Exception:  # noqa: BLE001 - defensive, should not block skip logic
            return None
    return counts


def is_unchanged(
    prior_manifest: Mapping[str, Mapping[str, object]] | None,
    state: ManifestState,
) -> bool:
    """
    Determine whether a plugin should be skipped due to unchanged inputs.

    Returns
    -------
    bool
        True when inputs match the prior manifest and row counts are stable.
    """
    if prior_manifest is None or state.input_hash is None:
        return False
    prior = prior_manifest.get(state.plugin_name)
    if prior is None or prior.get("status") != "succeeded":
        return False

    prior_input_hash = prior.get("input_hash")
    prior_options_hash = prior.get("options_hash")
    unchanged = prior_input_hash == state.input_hash
    if state.options_hash is not None:
        unchanged = unchanged and prior_options_hash in {state.options_hash, None}
    if not unchanged:
        return False

    if not state.row_count_tables:
        return True
    prior_rows = prior.get("row_counts")
    current_rows = current_row_counts(
        gateway=state.gateway,
        tables=state.row_count_tables,
        repo=state.repo,
        commit=state.commit,
    )
    if prior_rows is None or current_rows is None:
        return True
    return row_counts_equal(current_rows, prior_rows)


def dry_run_record(
    *,
    plugin: GraphMetricPlugin,
    params: RecordParams,
    run_id: str,
) -> GraphPluginRunRecord:
    """
    Create a record representing a dry-run skip.

    Returns
    -------
    GraphPluginRunRecord
        Record describing the skipped plugin.
    """
    now_ts = datetime.now(tz=UTC)
    return GraphPluginRunRecord(
        name=plugin.name,
        stage=plugin.stage,
        severity=params.severity,
        status="skipped",
        attempts=0,
        timeout_ms=params.timeout_ms,
        started_at=now_ts,
        ended_at=now_ts,
        duration_ms=0.0,
        partial=False,
        run_id=run_id,
        error=None,
        options=params.options,
        input_hash=params.input_hash,
        options_hash=params.options_hash,
        version_hash=params.version_hash,
        skipped_reason="dry_run",
        row_counts=None,
        requires_isolation=params.requires_isolation,
        isolation_kind=params.isolation_kind,
        policy_fail_fast=params.policy_fail_fast,
    )


def skip_record(
    *,
    plugin: GraphMetricPlugin,
    params: RecordParams,
    reason: str,
    run_id: str,
) -> GraphPluginRunRecord:
    """
    Create a record representing a skipped plugin.

    Returns
    -------
    GraphPluginRunRecord
        Record describing the skipped plugin.
    """
    now_ts = datetime.now(tz=UTC)
    return GraphPluginRunRecord(
        name=plugin.name,
        stage=plugin.stage,
        severity=params.severity,
        status="skipped",
        attempts=0,
        timeout_ms=params.timeout_ms,
        started_at=now_ts,
        ended_at=now_ts,
        duration_ms=0.0,
        partial=False,
        run_id=run_id,
        error=None,
        options=params.options,
        input_hash=params.input_hash,
        options_hash=params.options_hash,
        version_hash=params.version_hash,
        skipped_reason=reason,
        row_counts=None,
        contracts=(),
        requires_isolation=params.requires_isolation,
        isolation_kind=params.isolation_kind,
        policy_fail_fast=params.policy_fail_fast,
    )


def run_contracts(
    *,
    checkers: tuple[ContractChecker, ...],
    ctx: GraphMetricExecutionContext,
    status: Literal["succeeded", "failed", "skipped"],
) -> tuple[PluginContractResult, ...]:
    """
    Run contract checkers when a plugin succeeds.

    Returns
    -------
    tuple[PluginContractResult, ...]
        Contract results captured from the checkers.
    """
    if not checkers or status != "succeeded":
        return ()
    return run_contract_checkers(ctx=ctx, checkers=checkers)


def load_prior_manifest(path: Path | None) -> dict[str, dict[str, object]] | None:
    """
    Load the prior manifest and normalize records for unchanged detection.

    Returns
    -------
    dict[str, dict[str, object]] | None
        Mapping of plugin name to normalized manifest record.
    """
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    records = payload.get("records")
    if not isinstance(records, list):
        return None

    normalized: dict[str, dict[str, object]] = {}
    for record in records:
        if not isinstance(record, dict):
            continue
        name = record.get("name")
        if not isinstance(name, str):
            continue
        merged: dict[str, object] = dict(record)
        meta = record.get("meta")
        if isinstance(meta, dict):
            merged.update(meta)
        normalized[name] = merged
    return normalized


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    """Persist a manifest payload to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


__all__ = [
    "compute_input_hash",
    "compute_options_hash",
    "current_row_counts",
    "dry_run_record",
    "hash_json",
    "is_unchanged",
    "load_prior_manifest",
    "row_counts_equal",
    "run_contracts",
    "skip_record",
    "write_manifest",
]
