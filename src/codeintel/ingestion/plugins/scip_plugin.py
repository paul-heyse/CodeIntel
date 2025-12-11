"""SCIP ingest plugin."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar, SupportsInt, cast

from codeintel.build.errors import ToolNotAvailableError
from codeintel.build.plugin import TargetPlugin
from codeintel.build.result import TargetResult
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata
from codeintel.ingestion.adapters import BuildToolAdapter, DuckDBStorageAdapter
from codeintel.ingestion.compute.scip_ingest import ScipIngestConfig, ScipIngestStep
from codeintel.ingestion.plugins.helpers import get_module_paths, paths_to_modules
from codeintel.ingestion.plugins.scip_options import ScipIngestOptions
from codeintel.storage.ibis_types import filter_by, ibis_bool

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.execution.options import PluginOptionsResolver
    from codeintel.core.plugins.types.protocol import PluginKind, PluginStage

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


def _to_plugin_metadata(core: CorePluginMetadata) -> PluginMetadata:
    """Convert CorePluginMetadata to PluginMetadata for protocol compliance.

    Returns
    -------
    PluginMetadata
        Protocol-compatible metadata instance.
    """
    return PluginMetadata(
        name=core.name,
        version=core.version,
        description=core.description,
        kind=cast("PluginKind", core.kind),
        stage=cast("PluginStage", core.stage or "goid"),
        provides=core.provides,
        requires=core.requires,
        produces_tables=core.produces_tables,
    )


def _compute_row_counts(ctx: TargetExecutionContext) -> dict[str, int]:
    """Compute row counts for output tables.

    Returns
    -------
    dict[str, int]
        Row counts per table.
    """
    row_counts: dict[str, int] = {}
    for table_key in ctx.contract.table_keys:
        try:
            table = ctx.gateway.ibis.table(table_key)
            count_expr = filter_by(
                table,
                ibis_bool(table.repo == ctx.repo),
                ibis_bool(table.commit == ctx.commit),
            ).count()
            row_counts[table_key] = int(cast("SupportsInt", count_expr.execute()))
        except (RuntimeError, OSError):
            row_counts[table_key] = 0
    return row_counts


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


class ScipIngestPlugin(TargetPlugin):
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

    plugin_name: ClassVar[str] = "scip_ingest"
    plugin_version: ClassVar[str] = "3.0.0"
    plugin_description: ClassVar[str] = "Run scip-python and persist symbols and GOID crosswalk."
    _core_metadata: ClassVar[CorePluginMetadata] = SCIP_INGEST_METADATA

    def __init__(self, *, options_resolver: PluginOptionsResolver | None = None) -> None:
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        PluginMetadata
            Protocol-compatible metadata.
        """
        return _to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return full core metadata.

        Returns
        -------
        CorePluginMetadata
            Canonical metadata definition.
        """
        return self._core_metadata

    def resolve_options(
        self,
        *,
        dynamic_overrides: Mapping[str, Any] | None = None,
    ) -> ScipIngestOptions:
        """Resolve typed options from configuration.

        Returns
        -------
        ScipIngestOptions
            Resolved options instance.
        """
        if self._options_resolver is None:
            if dynamic_overrides:
                return ScipIngestOptions(**dynamic_overrides)
            return ScipIngestOptions()

        return self._options_resolver.get_options(
            self._core_metadata,
            ScipIngestOptions,
            dynamic_overrides=dynamic_overrides,
        )

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
        _ = self  # Protocol method requires instance

        # Check tool availability
        if ctx.resources.scip_indexer is None:
            raise ToolNotAvailableError(target=self.plugin_name, tool="scip-python")

        opts = self.resolve_options(
            dynamic_overrides={"scip_output_dir": ctx.scip_dir},
        )

        # Get module paths and convert to ModuleRecord
        paths = _filter_paths(get_module_paths(ctx), opts.scope_paths)
        modules = paths_to_modules(paths, ctx.repo_root)

        # Create adapters using build protocols
        storage = DuckDBStorageAdapter(ctx.gateway)
        tool = BuildToolAdapter(scip_indexer=ctx.resources.scip_indexer)

        # Create config
        scip_dir = opts.scip_output_dir or ctx.scip_dir
        config = ScipIngestConfig(
            repo=ctx.repo,
            commit=ctx.commit,
            repo_root=ctx.repo_root,
            output_scip=scip_dir / "index.scip",
            output_json=scip_dir / "index.json",
        )

        # Execute step
        step = ScipIngestStep(storage=storage, tools=tool)
        result = await step.execute_async(modules, config)

        if not result.success:
            errors = "; ".join(result.errors) if result.errors else "Unknown error"
            return TargetResult.failed(f"SCIP ingest failed: {errors}")

        # Compute row counts
        row_counts = _compute_row_counts(ctx)
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
