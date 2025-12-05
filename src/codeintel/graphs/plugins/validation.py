"""Graph validation plugin.

This module provides graph validation as a graph plugin, wrapping the existing
validation functionality with the new plugin protocol.

Architecture Notes
------------------
This plugin imports from analytics.graph_runtime for GraphRuntime resolution.
This is an intentional delegation - the graphs package orchestrates validation
but delegates runtime construction to analytics (Option B architecture).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass

from codeintel.analytics.runtime import GraphRuntimeOptions, resolve_graph_runtime
from codeintel.config.primitives import GraphBackendConfig
from codeintel.core.plugins.types.result import PluginResult
from codeintel.core.singleton import SingletonHolder
from codeintel.graphs.core import (
    GraphPluginExecutionContext,
    GraphPluginMetadata,
    GraphPluginProtocol,
    create_graph_metadata,
)
from codeintel.graphs.core.registry import register_graph_plugin
from codeintel.graphs.engine import GraphKind
from codeintel.graphs.validation import GraphValidationOptions, run_graph_validations
from codeintel.storage.gateway import (
    DuckDBCatalogException,
    DuckDBDatabaseError,
    DuckDBProgrammingError,
)

log = logging.getLogger(__name__)


class _GraphValidationPluginHolder(SingletonHolder["GraphValidationPlugin"]):
    """Singleton holder for GraphValidationPlugin.

    Uses the thread-safe SingletonHolder pattern from core.
    """


@dataclass
class GraphValidationPlugin:
    """Plugin that validates graph construction outputs.

    This plugin wraps the existing graph validation functionality,
    emitting warnings for common graph integrity issues.

    Use `instance()` to get the global plugin instance.
    """

    @classmethod
    def instance(cls) -> GraphValidationPlugin:
        """Return the singleton plugin instance.

        Returns
        -------
        GraphValidationPlugin
            The global plugin instance.
        """
        return _GraphValidationPluginHolder.get(cls)

    @classmethod
    def reset_instance(cls) -> None:
        """Reset the singleton instance for testing."""
        _GraphValidationPluginHolder.reset()

    @property
    def metadata(self) -> GraphPluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        GraphPluginMetadata
            Metadata describing the graph validation plugin.
        """
        return create_graph_metadata(
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
            produces_graph_kinds=(),
            requires_graph_kinds=(GraphKind.CALL_GRAPH, GraphKind.IMPORT_GRAPH),
            supports_incremental=False,
            isolation_kind="none",
            row_count_tables=("analytics.graph_validation",),
        )

    def execute(self, ctx: GraphPluginExecutionContext) -> PluginResult:
        """Execute graph validation.

        Parameters
        ----------
        ctx
            Graph plugin execution context.

        Returns
        -------
        PluginResult
            Result of the validation operation.
        """
        log.info(
            "graph_validation.start repo=%s commit=%s",
            ctx.repo,
            ctx.commit,
        )

        try:
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

            return PluginResult.ok(row_counts={"analytics.graph_validation": finding_count})

        except (
            RuntimeError,
            ValueError,
            TypeError,
            LookupError,
            OSError,
            DuckDBDatabaseError,
            DuckDBCatalogException,
            DuckDBProgrammingError,
        ) as exc:
            log.exception(
                "graph_validation.failed repo=%s commit=%s",
                ctx.repo,
                ctx.commit,
            )
            return PluginResult.fail(str(exc), error_kind="validation_error")

    @staticmethod
    def _query_finding_count(ctx: GraphPluginExecutionContext) -> int:
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
        except (DuckDBCatalogException, DuckDBProgrammingError, DuckDBDatabaseError):
            # Table may not exist yet or query may fail - return 0
            return 0


def get_graph_validation_plugin() -> GraphPluginProtocol:
    """Return the graph validation plugin instance.

    Returns
    -------
    GraphPluginProtocol
        The singleton plugin instance.
    """
    return GraphValidationPlugin.instance()


def _register_plugin() -> None:
    """Register the graph validation plugin with the global registry."""
    with contextlib.suppress(ValueError):
        register_graph_plugin(get_graph_validation_plugin())


# Auto-register on import
_register_plugin()


__all__ = [
    "GraphValidationPlugin",
    "get_graph_validation_plugin",
]
