"""Type coverage analytics plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import TargetPlugin
from codeintel.build.plugins._metadata import to_plugin_metadata
from codeintel.build.plugins.analytics.types.options import TypeCoverageOptions
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain
from codeintel.core.plugins.types.protocol import PluginMetadata

if TYPE_CHECKING:
    from collections.abc import Mapping

    from codeintel.build.context import TargetExecutionContext
    from codeintel.core.plugins.execution.options import PluginOptionsResolver


TYPE_COVERAGE_METADATA = CorePluginMetadata(
    name="analytics.type_coverage",
    version="2.0.0",
    description="Compute type annotation coverage metrics.",
    domain=PluginDomain.ANALYTICS,
    kind="metric",
    stage="function",
    provides=("analytics.type_coverage",),
    requires=("core.goids", "analytics.function_types"),
    produces_tables=("analytics.type_coverage",),
    consumes_tables=("core.goids", "analytics.function_types"),
    scope_aware=True,
    options_model=TypeCoverageOptions,
    resource_hints={"max_memory_mb": 512},
)


class TypeCoveragePlugin(TargetPlugin):
    """Compute type annotation coverage metrics."""

    plugin_name: ClassVar[str] = "type_coverage"
    plugin_version: ClassVar[str] = "2.0.0"
    plugin_description: ClassVar[str] = "Compute type annotation coverage metrics."
    _core_metadata: ClassVar[CorePluginMetadata] = TYPE_COVERAGE_METADATA

    def __init__(self, *, options_resolver: PluginOptionsResolver | None = None) -> None:
        self._options_resolver = options_resolver

    @property
    def metadata(self) -> PluginMetadata:
        """Return protocol-compatible metadata.

        Returns
        -------
        PluginMetadata
            Protocol metadata facade.
        """
        return to_plugin_metadata(self._core_metadata)

    @property
    def core_metadata(self) -> CorePluginMetadata:
        """Return core metadata.

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
    ) -> TypeCoverageOptions:
        """Resolve options from config and runtime overrides.

        Returns
        -------
        TypeCoverageOptions
            Resolved options instance.
        """
        if self._options_resolver is None:
            if dynamic_overrides:
                return TypeCoverageOptions(**dynamic_overrides)
            return TypeCoverageOptions()

        return self._options_resolver.get_options(
            self._core_metadata,
            TypeCoverageOptions,
            dynamic_overrides=dynamic_overrides,
        )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute type coverage analysis.

        Returns
        -------
        TargetResult
            Success result placeholder until computation is wired.
        """
        _ = self
        _ = ctx
        return TargetResult.succeeded()


__all__ = [
    "TYPE_COVERAGE_METADATA",
    "TypeCoveragePlugin",
]
