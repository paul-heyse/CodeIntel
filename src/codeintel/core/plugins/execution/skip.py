"""Plugin skip decision logic."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.core.plugins.types.result import PluginExecutionRecord

if TYPE_CHECKING:
    from codeintel.core.plugins.execution.manifest import ManifestStore
    from codeintel.core.plugins.execution.run_context import PluginRunContext, RunContextInputs

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SkipDecision:
    """Result of a skip decision check."""

    should_skip: bool
    reason: str
    prior_record: PluginExecutionRecord | None = None

    @classmethod
    def execute(cls, reason: str) -> SkipDecision:
        """Create a decision to execute.

        Parameters
        ----------
        reason
            Explanation for why execution should proceed.

        Returns
        -------
        SkipDecision
            Decision indicating execution should proceed.
        """
        return cls(should_skip=False, reason=reason)

    @classmethod
    def skip(cls, reason: str, prior_record: PluginExecutionRecord) -> SkipDecision:
        """Create a decision to skip.

        Parameters
        ----------
        reason
            Explanation for why skipping is safe.
        prior_record
            Prior execution record to reuse.

        Returns
        -------
        SkipDecision
            Decision indicating execution should be skipped.
        """
        return cls(should_skip=True, reason=reason, prior_record=prior_record)


def should_skip_plugin(
    run_context: PluginRunContext,
    manifest_store: ManifestStore,
    *,
    inputs: RunContextInputs,
    force: bool = False,
) -> SkipDecision:
    """Determine whether a plugin execution should be skipped.

    Parameters
    ----------
    run_context
        Prepared plugin run context with hashes.
    manifest_store
        Store used to look up prior execution records.
    inputs
        Repository, commit, and scope information for this run.
    force
        When True, always execute regardless of manifest state.

    Returns
    -------
    SkipDecision
        Decision indicating whether to skip or execute.
    """
    plugin_name = run_context.plugin_name
    current_input_hash = run_context.input_hash

    if force:
        log.debug("skip_check: %s force=True, executing", plugin_name)
        return SkipDecision.execute("force execution requested")

    prior_record = manifest_store.load_last_record(
        plugin_name=plugin_name,
        repo=inputs.repo,
        commit=inputs.commit,
        scope_id=inputs.scope_id,
        variant=inputs.variant,
    )

    if prior_record is None:
        log.debug("skip_check: %s no prior record, executing", plugin_name)
        return SkipDecision.execute("no prior execution record")

    if prior_record.status == "failed":
        log.debug("skip_check: %s prior failed, re-executing", plugin_name)
        return SkipDecision.execute("prior execution failed")

    prior_input_hash = prior_record.meta.get("input_hash")
    if prior_input_hash is None:
        log.debug("skip_check: %s prior has no input_hash, executing", plugin_name)
        return SkipDecision.execute("prior record missing input_hash")

    if prior_input_hash == current_input_hash:
        log.info(
            "skip_check: %s input_hash unchanged (%s), skipping",
            plugin_name,
            current_input_hash[:8],
        )
        return SkipDecision.skip("input_hash unchanged", prior_record)

    log.debug(
        "skip_check: %s input_hash changed (%s -> %s), executing",
        plugin_name,
        str(prior_input_hash)[:8],
        current_input_hash[:8],
    )
    return SkipDecision.execute(
        f"input_hash changed: {str(prior_input_hash)[:8]} -> {current_input_hash[:8]}"
    )


def create_skip_execution_record(
    run_context: PluginRunContext,
    prior_record: PluginExecutionRecord,
    *,
    inputs: RunContextInputs,
) -> PluginExecutionRecord:
    """Create an execution record for a skipped plugin.

    Parameters
    ----------
    run_context
        Current plugin run context.
    prior_record
        Prior record whose result is being reused.
    inputs
        Repository and scope inputs for the execution.

    Returns
    -------
    PluginExecutionRecord
        Record for the skipped execution.
    """
    now = datetime.now(tz=UTC)

    return PluginExecutionRecord(
        plugin_name=run_context.plugin_name,
        status="skipped",
        started_at=now,
        ended_at=now,
        duration_ms=0.0,
        result=prior_record.result,
        meta={
            "repo": inputs.repo,
            "commit": inputs.commit,
            "scope_id": inputs.scope_id,
            "variant": inputs.variant,
            "options_hash": run_context.options_hash,
            "input_hash": run_context.input_hash,
            "skip_reason": "input_hash_unchanged",
            "prior_input_hash": prior_record.meta.get("input_hash"),
        },
    )


__all__ = [
    "SkipDecision",
    "create_skip_execution_record",
    "should_skip_plugin",
]
