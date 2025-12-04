"""History timeseries plugin using the new protocol."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from codeintel.analytics.core.execution_context import PluginExecutionContext
from codeintel.analytics.core.plugin_protocol import (
    PluginInputSpec,
    PluginMetadata,
    PluginOutputSpec,
    PluginResourceHints,
    PluginResult,
    ValidationResult,
)
from codeintel.analytics.history import compute_history_timeseries_gateways
from codeintel.config.steps_analytics import HistoryTimeseriesStepConfig

if TYPE_CHECKING:
    from codeintel.storage.gateway import SnapshotGatewayResolver


@dataclass
class HistoryTimeseriesPlugin:
    """Plugin for aggregating history timeseries.

    Computes historical trends by:
    - Aggregating analytics across commits
    - Building time-based metrics
    - Tracking evolution patterns
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata."""
        return PluginMetadata(
            name="history.timeseries",
            description="Aggregate analytics across commits into history timeseries.",
            kind="analytics",
            stage="history",
            version="2.0.0",
            enabled_by_default=True,
            severity="fatal",
            inputs=(
                PluginInputSpec(
                    name="history_cfg",
                    type_ref="HistoryTimeseriesStepConfig",
                    required=True,
                    source="config",
                ),
            ),
            outputs=(
                PluginOutputSpec(
                    name="history_timeseries", tables=("analytics.history_timeseries",)
                ),
            ),
            provides=(
                "analytics.history_timeseries",
            ),
            requires=(
                "analytics.function_profile",
            ),
            depends_on=("profiles.build",),
            resource_hints=PluginResourceHints(
                max_runtime_ms=120_000,
                priority=80,
            ),
            tags=("history", "timeseries", "trends"),
        )

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate required inputs.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        ValidationResult
            Validation result.
        """
        _ = self.metadata
        errors: list[str] = []

        if not ctx.has_config(HistoryTimeseriesStepConfig):
            errors.append("HistoryTimeseriesStepConfig is required")

        if ctx.extra.get("history_snapshot_resolver") is None:
            errors.append("history_snapshot_resolver is required in ctx.extra")

        if errors:
            return ValidationResult.failure(tuple(errors))
        return ValidationResult.success()

    def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        PluginResult
            Execution result.
        """
        _ = self.metadata
        try:
            cfg = ctx.get_config(HistoryTimeseriesStepConfig)
        except ValueError as e:
            return PluginResult.fail(str(e))

        snapshot_resolver = ctx.extra.get("history_snapshot_resolver")
        if snapshot_resolver is None:
            return PluginResult.fail("history_snapshot_resolver is required")

        resolver = cast("SnapshotGatewayResolver", snapshot_resolver)

        try:
            compute_history_timeseries_gateways(
                ctx.gateway,
                cfg,
                resolver,
                runner=ctx.extra.get("tool_runner"),
            )
        except (RuntimeError, ValueError, OSError) as e:
            return PluginResult.fail(f"History timeseries computation failed: {e}")

        return PluginResult.ok()


__all__ = ["HistoryTimeseriesPlugin"]
