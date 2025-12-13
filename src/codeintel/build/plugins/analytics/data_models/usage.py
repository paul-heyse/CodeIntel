"""Data model usage plugin.

This plugin classifies per-function data model read/write usage patterns.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.analytics.compute.data_models.usage import compute_data_model_usage
from codeintel.analytics.parsing.ast_cache import FunctionAstLoadRequest, load_function_asts
from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugins.analytics._metadata import to_plugin_metadata
from codeintel.config.steps_analytics import DataModelUsageStepConfig
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata

log = logging.getLogger(__name__)


DATA_MODEL_USAGE_METADATA = CorePluginMetadata(
    name="analytics.data_model_usage",
    version="3.0.0",
    description="Classify per-function data model read/write usage patterns.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="data_model_usage",
    provides=("analytics.data_model_usage",),
    requires=(
        "analytics.data_model_fields",
        "analytics.data_model_relationships",
        "core.modules",
    ),
    produces_tables=("analytics.data_model_usage",),
    consumes_tables=(
        "analytics.data_model_fields",
        "analytics.data_model_relationships",
        "core.modules",
    ),
)


class DataModelUsagePlugin(TargetPlugin):
    """Classify per-function data model read/write usage patterns.

    Analyzes per-function:
    - Read/write patterns
    - Model field access
    - Data flow through models

    Outputs
    -------
    - analytics.data_model_usage: Data model usage patterns
    """

    plugin_name: ClassVar[str] = "data_model_usage"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = (
        "Classify per-function data model read/write usage patterns."
    )
    _core_metadata: ClassVar[CorePluginMetadata] = DATA_MODEL_USAGE_METADATA

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata."""
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return canonical metadata."""
        return self._core_metadata

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        TargetResult
            Execution result.
        """
        _ = self

        cfg = DataModelUsageStepConfig(
            snapshot=ctx.snapshot,
        )

        catalog_provider = ctx.resources.catalog
        if catalog_provider is None:
            log.info("data_model_usage: No catalog available, skipping")
            return TargetResult.succeeded(row_counts={"analytics.data_model_usage": 0})

        module_rows = ctx.gateway.execute(
            "SELECT path, module FROM core.modules WHERE repo = ? AND commit = ?",
            [ctx.repo, ctx.commit],
        ).fetchall()

        if not module_rows:
            log.info("data_model_usage: No modules found, skipping")
            return TargetResult.succeeded(row_counts={"analytics.data_model_usage": 0})

        module_map = {row[0]: row[1] for row in module_rows}

        catalog = catalog_provider.catalog()
        function_spans = catalog.function_spans
        if not function_spans:
            log.info("data_model_usage: No functions in catalog, skipping")
            return TargetResult.succeeded(row_counts={"analytics.data_model_usage": 0})

        request = FunctionAstLoadRequest(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.snapshot.repo_root,
            catalog_provider=catalog_provider,
        )
        ast_by_goid, missing_goids = load_function_asts(ctx.gateway, request)

        if not ast_by_goid:
            log.info("data_model_usage: No ASTs loaded, skipping")
            return TargetResult.succeeded(row_counts={"analytics.data_model_usage": 0})

        try:
            compute_data_model_usage(
                ctx.gateway,
                cfg,
                module_map=module_map,
                ast_by_goid=ast_by_goid,
                missing_goids=missing_goids,
            )
        except (RuntimeError, ValueError, OSError) as e:
            return TargetResult.failed(f"Data model usage computation failed: {e}")

        return TargetResult.succeeded()


__all__ = ["DATA_MODEL_USAGE_METADATA", "DataModelUsagePlugin"]
