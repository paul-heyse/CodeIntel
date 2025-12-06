"""Graph plugin executor using unified execution infrastructure.

This module provides `GraphPluginExecutor` and `GraphExecutorContext`,
which use the common execution infrastructure (settings, manifest,
policy) for graph plugin execution.

Note: This executor does not extend `BasePluginExecutor` due to
protocol variance constraints. Instead, it follows the same patterns
and uses the same common types.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast

from codeintel.core.execution.errors import PLUGIN_CATCHABLE_ERRORS
from codeintel.core.execution.telemetry import get_runtime_telemetry
from codeintel.core.execution.timing import utc_now
from codeintel.core.plugins.execution.context import PluginScratch
from codeintel.core.plugins.execution.executor_context import BaseExecutorContext
from codeintel.core.plugins.execution.manifest import (
    ManifestState,
    is_unchanged,
)
from codeintel.core.plugins.execution.policy import BaseExecutionPolicy
from codeintel.core.plugins.types.report import BaseExecutionReport
from codeintel.core.plugins.types.result import PluginExecutionRecord
from codeintel.core.resources.registry import ResourceRegistry
from codeintel.graphs.core.context import GraphPluginExecutionContext
from codeintel.graphs.core.protocol import GraphPluginProtocol
from codeintel.graphs.resources.graphs import GraphResource
from codeintel.graphs.resources.storage import StorageResource

if TYPE_CHECKING:
    from codeintel.config.steps_graphs import GraphRunScope
    from codeintel.core.execution.telemetry import RuntimeTelemetry
    from codeintel.core.plugins.execution.settings import PluginExecutionSettings
    from codeintel.graphs.catalog import FunctionCatalogProvider
    from codeintel.graphs.engine import GraphEngine, NxGraphEngine

log = logging.getLogger(__name__)


# =============================================================================
# Graphs-Specific Executor Context (extends BaseExecutorContext)
# =============================================================================


@dataclass
class GraphExecutorContext(BaseExecutorContext):
    """Context for graph plugin execution.

    Extend BaseExecutorContext with graph-specific fields.

    Attributes
    ----------
    engine
        Graph engine.
    catalog_provider
        Function catalog provider.
    """

    engine: GraphEngine | None = None
    catalog_provider: FunctionCatalogProvider | None = None


def _status_counts(records: tuple[PluginExecutionRecord, ...]) -> dict[str, int]:
    """Summarize plugin run statuses.

    Parameters
    ----------
    records
        Plugin execution records.

    Returns
    -------
    dict[str, int]
        Counts keyed by success/failure/skipped.
    """
    return {
        "success": sum(1 for r in records if r.status == "succeeded"),
        "failure": sum(1 for r in records if r.status == "failed"),
        "skipped": sum(1 for r in records if r.status == "skipped"),
    }


@dataclass(frozen=True)
class GraphRunReport(BaseExecutionReport):
    """Report of a graph plugin execution run.

    Extend BaseExecutionReport with graph-specific fields like manifest.

    Attributes
    ----------
    repo
        Repository identifier.
    commit
        Commit SHA.
    manifest
        Final manifest state.
    """

    repo: str = ""
    commit: str = ""
    manifest: Mapping[str, Mapping[str, object]] = field(default_factory=dict)


class GraphPluginExecutor:
    """Graph plugin executor using common infrastructure.

    Use common types (settings, manifest, policy) for graph plugin
    execution with manifest tracking and scope-aware skip detection.

    Parameters
    ----------
    policy
        Execution policy.
    telemetry
        Runtime telemetry.
    prior_manifest
        Prior manifest for skip detection.
    scope
        Graph run scope for incremental execution.
    """

    def __init__(
        self,
        policy: BaseExecutionPolicy | None = None,
        telemetry: RuntimeTelemetry | None = None,
        prior_manifest: Mapping[str, Mapping[str, object]] | None = None,
        scope: GraphRunScope | None = None,
    ) -> None:
        """Initialize the graph executor.

        Parameters
        ----------
        policy
            Execution policy.
        telemetry
            Runtime telemetry.
        prior_manifest
            Prior manifest for skip detection.
        scope
            Graph run scope.
        """
        self._policy = policy or BaseExecutionPolicy()
        self._telemetry = telemetry or get_runtime_telemetry()
        self._prior_manifest = prior_manifest
        self._scope = scope
        self._manifest: dict[str, dict[str, object]] = {}

    @property
    def policy(self) -> BaseExecutionPolicy:
        """Return the execution policy."""
        return self._policy

    @property
    def telemetry(self) -> RuntimeTelemetry:
        """Return the telemetry instance."""
        return self._telemetry

    @property
    def manifest(self) -> dict[str, dict[str, object]]:
        """Return the execution manifest."""
        return self._manifest

    @property
    def prior_manifest(self) -> Mapping[str, Mapping[str, object]] | None:
        """Return the prior manifest."""
        return self._prior_manifest

    def execute(
        self,
        executor_ctx: GraphExecutorContext,
        plugins: tuple[GraphPluginProtocol, ...],
        *,
        scratch: PluginScratch | None = None,
        run_id: str | None = None,
        settings_by_plugin: Mapping[str, PluginExecutionSettings] | None = None,
    ) -> GraphRunReport:
        """Execute all plugins with retry and telemetry.

        Parameters
        ----------
        executor_ctx
            Executor-level context.
        plugins
            Ordered plugins to execute.
        scratch
            Optional shared scratch store.
        run_id
            Optional run identifier.
        settings_by_plugin
            Optional per-plugin execution settings.

        Returns
        -------
        GraphRunReport
            Execution report with all plugin results.
        """
        effective_run_id = run_id or executor_ctx.effective_run_id or "graph-run"
        started_at = utc_now()
        start_time = time.perf_counter()
        records: list[PluginExecutionRecord] = []
        shared_scratch = scratch or PluginScratch()
        fatal_error = False

        # Reset manifest for this run
        self._manifest = {}

        log.info(
            "graph_executor.run.start run_id=%s plugin_count=%d",
            effective_run_id,
            len(plugins),
        )

        try:
            for plugin in plugins:
                settings = self._get_settings(plugin, settings_by_plugin)

                # Check for skip conditions
                skip_reason = self._should_skip(plugin, settings)
                if skip_reason is not None:
                    record = self._create_skip_record(plugin, skip_reason, settings)
                    records.append(record)
                    continue

                # Build plugin context
                plugin_ctx = self._build_context(executor_ctx, plugin, shared_scratch)

                # Execute the plugin
                record = self._execute_plugin(plugin, plugin_ctx, settings)
                records.append(record)

                # Build manifest entry for successful plugins
                if record.status == "succeeded":
                    self._manifest[plugin.metadata.name] = self._build_manifest_entry(
                        record, settings
                    )

                # Check for fail-fast condition
                if record.status == "failed":
                    severity = (
                        settings.severity
                        if settings is not None
                        else self._policy.get_severity(plugin.metadata.name)
                    )
                    if severity == "fatal" and self._policy.fail_fast:
                        log.error(
                            "graph_executor.run.fatal_error plugin=%s",
                            plugin.metadata.name,
                        )
                        fatal_error = True
                        break
        finally:
            shared_scratch.cleanup()

        ended_at = utc_now()
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Record run-level telemetry
        self._telemetry.record_run_metrics(
            run_id=effective_run_id,
            success_count=sum(1 for r in records if r.status == "succeeded"),
            failure_count=sum(1 for r in records if r.status == "failed"),
            skip_count=sum(1 for r in records if r.status == "skipped"),
            duration_s=duration_ms / 1000,
        )

        log.info(
            "graph_executor.run.complete run_id=%s duration_ms=%.2f fatal_error=%s",
            effective_run_id,
            duration_ms,
            fatal_error,
        )

        return GraphRunReport(
            run_id=effective_run_id,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            records=tuple(records),
            fatal_error=fatal_error,
            repo=executor_ctx.snapshot.repo,
            commit=executor_ctx.snapshot.commit,
            manifest=self._manifest,
        )

    @staticmethod
    def _get_settings(
        plugin: GraphPluginProtocol,
        settings_by_plugin: Mapping[str, PluginExecutionSettings] | None,
    ) -> PluginExecutionSettings | None:
        """Get settings for a plugin.

        Parameters
        ----------
        plugin
            Plugin to get settings for.
        settings_by_plugin
            Settings map keyed by plugin name.

        Returns
        -------
        PluginExecutionSettings | None
            Settings if available.
        """
        if settings_by_plugin is None:
            return None
        return settings_by_plugin.get(plugin.metadata.name)

    def _should_skip(
        self,
        plugin: GraphPluginProtocol,
        settings: PluginExecutionSettings | None,
    ) -> str | None:
        """Check if plugin should be skipped.

        Parameters
        ----------
        plugin
            Plugin to check.
        settings
            Plugin settings with hashes.

        Returns
        -------
        str | None
            Skip reason if plugin should be skipped.
        """
        if self._policy.dry_run:
            return "dry_run"

        # Check manifest-based skip
        if (
            self._policy.skip_on_unchanged
            and self._prior_manifest is not None
            and settings is not None
        ):
            state = ManifestState(
                plugin_name=plugin.metadata.name,
                input_hash=settings.input_hash,
                options_hash=settings.options_hash,
            )
            if is_unchanged(self._prior_manifest, state):
                return "unchanged"

        return None

    def _build_context(
        self,
        executor_ctx: GraphExecutorContext,
        plugin: GraphPluginProtocol,
        scratch: PluginScratch,
    ) -> GraphPluginExecutionContext:
        """Build graph plugin execution context.

        Parameters
        ----------
        executor_ctx
            Executor-level context.
        plugin
            Plugin being executed.
        scratch
            Shared scratch store.

        Returns
        -------
        GraphPluginExecutionContext
            Graph-specific execution context.
        """
        resources = ResourceRegistry()
        resources.register(
            StorageResource,
            StorageResource(executor_ctx.gateway, executor_ctx.snapshot.repo_root),
        )
        if executor_ctx.engine is not None:
            resources.register(
                GraphResource,
                GraphResource(cast("NxGraphEngine", executor_ctx.engine)),
            )

        return GraphPluginExecutionContext(
            gateway=executor_ctx.gateway,
            snapshot=executor_ctx.snapshot,
            run_id=executor_ctx.effective_run_id,
            resources=resources,
            scratch=scratch,
            plugin_name=plugin.metadata.name,
            scope=self._scope,
            run_context=executor_ctx.run_context,
            _catalog_provider=executor_ctx.catalog_provider,
        )

    def _execute_plugin(
        self,
        plugin: GraphPluginProtocol,
        ctx: GraphPluginExecutionContext,
        settings: PluginExecutionSettings | None,
    ) -> PluginExecutionRecord:
        """Execute a single plugin.

        Parameters
        ----------
        plugin
            Plugin to execute.
        ctx
            Execution context.
        settings
            Plugin settings.

        Returns
        -------
        PluginExecutionRecord
            Execution record.
        """
        meta = plugin.metadata
        started_at = utc_now()
        start_time = time.perf_counter()

        # Build record metadata
        record_meta: dict[str, object] = {}
        if settings is not None:
            record_meta["input_hash"] = settings.input_hash
            record_meta["options_hash"] = settings.options_hash
            record_meta["version_hash"] = settings.version_hash

        # Start telemetry span
        span = self._telemetry.start_span(
            meta.name,
            ctx.run_id or "unknown",
            attributes={"stage": meta.stage, "kind": meta.kind},
        )

        log.info(
            "graph_executor.plugin.start name=%s stage=%s",
            meta.name,
            meta.stage,
        )

        # Execute plugin
        result = None
        error: str | None = None
        try:
            result = plugin.execute(ctx)
            if not result.success:
                error = result.error
        except PLUGIN_CATCHABLE_ERRORS as exc:
            error = repr(exc)
            log.warning(
                "graph_executor.plugin.error name=%s error=%s",
                meta.name,
                error,
            )

        ended_at = utc_now()
        duration_ms = round((time.perf_counter() - start_time) * 1000, 2)

        # Determine status
        severity = (
            settings.severity
            if settings is not None
            else self._policy.get_severity(meta.name)
        )

        if result is not None and result.success:
            status = "skipped" if result.skipped else "succeeded"
            rows_written = sum(result.row_counts.values()) if result.row_counts else 0
            self._telemetry.end_span(span, success=True, rows_written=rows_written)
        else:
            status = "skipped" if severity == "skip_on_error" else "failed"
            self._telemetry.end_span(span, success=False, error=error)

        log.info(
            "graph_executor.plugin.complete name=%s status=%s duration_ms=%.2f",
            meta.name,
            status,
            duration_ms,
        )

        return PluginExecutionRecord(
            plugin_name=meta.name,
            status=status,
            started_at=started_at,
            ended_at=ended_at,
            duration_ms=duration_ms,
            attempts=1,
            result=result,
            error=error,
            meta=record_meta,
        )

    @staticmethod
    def _create_skip_record(
        plugin: GraphPluginProtocol,
        reason: str,
        settings: PluginExecutionSettings | None,
    ) -> PluginExecutionRecord:
        """Create a skip record for a plugin.

        Parameters
        ----------
        plugin
            Plugin that was skipped.
        reason
            Reason for skipping.
        settings
            Plugin settings.

        Returns
        -------
        PluginExecutionRecord
            Skip record.
        """
        now = utc_now()
        record_meta: dict[str, object] = {"skipped_reason": reason}
        if settings is not None:
            record_meta["input_hash"] = settings.input_hash
            record_meta["options_hash"] = settings.options_hash
            record_meta["version_hash"] = settings.version_hash

        return PluginExecutionRecord(
            plugin_name=plugin.metadata.name,
            status="skipped",
            started_at=now,
            ended_at=now,
            duration_ms=0.0,
            attempts=0,
            error=reason,
            meta=record_meta,
        )

    @staticmethod
    def _build_manifest_entry(
        record: PluginExecutionRecord,
        settings: PluginExecutionSettings | None,
    ) -> dict[str, object]:
        """Build manifest entry for a successful plugin.

        Parameters
        ----------
        record
            Execution record.
        settings
            Plugin settings.

        Returns
        -------
        dict[str, object]
            Manifest entry.
        """
        row_counts = (
            dict(record.result.row_counts)
            if record.result and record.result.row_counts
            else None
        )

        return {
            "input_hash": settings.input_hash if settings else record.meta.get("input_hash"),
            "options_hash": settings.options_hash if settings else record.meta.get("options_hash"),
            "version_hash": settings.version_hash if settings else record.meta.get("version_hash"),
            "row_counts": row_counts,
            "executed_at": record.ended_at.isoformat() if record.ended_at else None,
        }


__all__ = [
    "GraphExecutorContext",
    "GraphPluginExecutor",
    "GraphRunReport",
]
