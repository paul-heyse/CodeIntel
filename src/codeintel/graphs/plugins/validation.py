"""Graph validation plugin.

This module provides graph validation as a graph plugin, wrapping the existing
validation functionality with the new plugin protocol.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from codeintel.graphs.core import (
    GraphExecutionContext,
    GraphPluginMetadata,
    GraphPluginProtocol,
    GraphPluginResult,
)
from codeintel.graphs.core.registry import register_graph_plugin
from codeintel.graphs.engine import GraphKind

log = logging.getLogger(__name__)


@dataclass
class GraphValidationPlugin:
    """Plugin that validates graph construction outputs.

    This plugin wraps the existing graph validation functionality,
    emitting warnings for common graph integrity issues.
    """

    @property
    def metadata(self) -> GraphPluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        GraphPluginMetadata
            Metadata describing the graph validation plugin.
        """
        return GraphPluginMetadata(
            name="graph_validation",
            description="Validate graph construction outputs and emit warnings for issues.",
            kind="validation",
            stage="validation",
            enabled_by_default=True,
            depends_on=(
                "goid_builder",
                "callgraph_builder",
                "import_graph_builder",
            ),
            provides=("validation_report",),
            requires=("goids", "call_graph", "import_graph"),
            produces_tables=("analytics.graph_validation",),
            produces_graphs=(),
            requires_graphs=(GraphKind.CALL_GRAPH, GraphKind.IMPORT_GRAPH),
            supports_incremental=False,
            isolation_kind="none",
            row_count_tables=("analytics.graph_validation",),
        )

    def execute(self, ctx: GraphExecutionContext) -> GraphPluginResult:
        """Execute graph validation.

        Parameters
        ----------
        ctx
            Graph plugin execution context.

        Returns
        -------
        GraphPluginResult
            Result of the validation operation.
        """
        log.info(
            "graph_validation.start repo=%s commit=%s",
            ctx.repo,
            ctx.commit,
        )

        try:
            # Import validation module
            from codeintel.analytics.graph_runtime import (  # noqa: PLC0415
                GraphRuntimeOptions,
                resolve_graph_runtime,
            )
            from codeintel.config.primitives import GraphBackendConfig  # noqa: PLC0415
            from codeintel.graphs.validation import (  # noqa: PLC0415
                GraphValidationOptions,
                run_graph_validations,
            )

            # Build minimal runtime
            runtime_options = GraphRuntimeOptions(
                snapshot=ctx.snapshot,
                backend=GraphBackendConfig(),
            )
            runtime = resolve_graph_runtime(
                ctx.gateway,
                ctx.snapshot,
                runtime_options,
            )

            validation_options = GraphValidationOptions(
                hard_fail=False,
            )

            run_graph_validations(
                ctx.gateway,
                snapshot=ctx.snapshot,
                catalog_provider=ctx.catalog_provider,
                runtime=runtime,
                options=validation_options,
            )

            # Query finding counts
            finding_count = self._query_finding_count(ctx)

            log.info(
                "graph_validation.complete repo=%s commit=%s findings=%d",
                ctx.repo,
                ctx.commit,
                finding_count,
            )

            return GraphPluginResult.ok(
                row_counts={"analytics.graph_validation": finding_count}
            )

        except Exception as exc:
            log.exception(
                "graph_validation.failed repo=%s commit=%s",
                ctx.repo,
                ctx.commit,
            )
            return GraphPluginResult.fail(str(exc), error_kind="validation_error")

    @staticmethod
    def _query_finding_count(ctx: GraphExecutionContext) -> int:
        """Query count of validation findings.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        int
            Count of validation findings.
        """
        try:
            result = ctx.gateway.con.execute(
                """
                SELECT COUNT(*)
                FROM analytics.graph_validation
                WHERE repo = ? AND commit = ?
                """,
                [ctx.repo, ctx.commit],
            ).fetchone()
            return int(result[0]) if result else 0
        except Exception:  # noqa: BLE001
            return 0


# Singleton instance
_GRAPH_VALIDATION_PLUGIN: GraphValidationPlugin | None = None


def get_graph_validation_plugin() -> GraphPluginProtocol:
    """Return the graph validation plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The singleton plugin instance.
    """
    global _GRAPH_VALIDATION_PLUGIN  # noqa: PLW0603
    if _GRAPH_VALIDATION_PLUGIN is None:
        _GRAPH_VALIDATION_PLUGIN = GraphValidationPlugin()
    return _GRAPH_VALIDATION_PLUGIN


def _register_plugin() -> None:
    """Register the graph validation plugin with the global registry."""
    import contextlib  # noqa: PLC0415

    with contextlib.suppress(ValueError):
        register_graph_plugin(get_graph_validation_plugin())


# Auto-register on import
_register_plugin()


__all__ = [
    "GraphValidationPlugin",
    "get_graph_validation_plugin",
]
