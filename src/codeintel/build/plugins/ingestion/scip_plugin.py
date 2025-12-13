"""SCIP ingest plugin."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.errors import ToolNotAvailableError
from codeintel.build.plugin import MetadataPlugin
from codeintel.build.plugins._helpers import compute_row_counts
from codeintel.build.plugins.ingestion.helpers import get_module_paths, paths_to_modules
from codeintel.build.plugins.ingestion.scip_options import ScipIngestOptions
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.ingestion.adapters import BuildToolAdapter, DuckDBStorageAdapter
from codeintel.ingestion.compute.scip_ingest import ScipIngestConfig, ScipIngestStep

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext

log = logging.getLogger(__name__)


SCIP_INGEST_METADATA = CorePluginMetadata(
    name="ingest.scip_python",
    version="3.0.0",
    description="Run scip-python and persist symbols and GOID crosswalk.",
    domain=PluginDomain.INGEST,
    kind="builder",
    stage="goid",
    provides=(
        "core.scip_symbols",
        "core.goid_crosswalk",
    ),
    requires=("core.modules",),
    produces_tables=(
        "core.scip_symbols",
        "core.goid_crosswalk",
    ),
    consumes_tables=("core.modules",),
    supports_incremental=False,
    scope_aware=True,
    options_model=ScipIngestOptions,
    resource_hints={
        "requires_tools": ["scip-python"],
    },
)


def _filter_paths(paths: list[str], scope_paths: list[str] | None) -> list[str]:
    """Filter module paths by scope.

    Returns
    -------
    list[str]
        Filtered paths respecting scope prefixes.
    """
    if not scope_paths:
        return paths
    prefixes = tuple(scope_paths)
    return [path for path in paths if path.startswith(prefixes)]


class ScipIngestPlugin(MetadataPlugin):
    """Run scip-python and persist symbols and GOID crosswalk.

    This plugin executes the SCIP-Python indexer to generate semantic
    code intelligence data, including symbol information and global
    identifier crosswalk.

    Outputs
    -------
    - index.scip: SCIP index file
    - core.scip_symbols: Symbol table
    - core.goid_crosswalk: GOID crosswalk
    """

    _core_metadata: ClassVar[CorePluginMetadata] = SCIP_INGEST_METADATA

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute SCIP indexing.

        Parameters
        ----------
        ctx
            Execution context with resources and parameters.

        Returns
        -------
        TargetResult
            Success result with row counts.

        Raises
        ------
        ToolNotAvailableError
            When the scip-python tool is not available.
        """
        _ = self

        if ctx.resources.scip_indexer is None:
            raise ToolNotAvailableError(target=self.plugin_name, tool="scip-python")

        opts = self.resolve_options(
            ScipIngestOptions,
            dynamic_overrides={"scip_output_dir": ctx.scip_dir},
        )

        paths = _filter_paths(get_module_paths(ctx), opts.scope_paths)
        modules = paths_to_modules(paths, ctx.repo_root)

        storage = DuckDBStorageAdapter(ctx.gateway)
        tool = BuildToolAdapter(scip_indexer=ctx.resources.scip_indexer)

        scip_dir = opts.scip_output_dir or ctx.scip_dir
        config = ScipIngestConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            output_scip=scip_dir / "index.scip",
            output_json=scip_dir / "index.json",
        )

        step = ScipIngestStep(storage=storage, tools=tool)
        result = await step.execute_async(modules, config)

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            return TargetResult.failed(f"SCIP ingest failed: {errors}")

        row_counts = compute_row_counts(ctx)
        return TargetResult.succeeded(
            row_counts=row_counts,
            artifacts_written=["index.scip", "index.json"],
        )


__all__ = [
    "SCIP_INGEST_METADATA",
    "ScipIngestPlugin",
    "get_module_paths",
    "paths_to_modules",
]
