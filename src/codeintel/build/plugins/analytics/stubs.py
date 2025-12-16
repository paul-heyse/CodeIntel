"""Stub classes for migrated analytics plugins.

These stub classes are provided for backward compatibility with test infrastructure
that may reference the original plugin classes. The actual implementations have been
migrated to native Hamilton modules in Phase 4.

Do NOT use these classes in new code. They are deprecated and will be removed in
a future version.

Migration targets:
- FunctionMetricsPlugin -> codeintel.build.hamilton.native.analytics.function_metrics
- FunctionAstFeaturesPlugin -> codeintel.build.hamilton.native.analytics.ast_features
- FunctionEffectsPlugin -> codeintel.build.hamilton.native.analytics.function_effects
- FunctionContractsPlugin -> codeintel.build.hamilton.native.analytics.function_contracts
- CoverageTestEdgesPlugin -> codeintel.build.hamilton.native.analytics.coverage_test_edges
- TestProfilePlugin -> codeintel.build.hamilton.native.analytics.test_profile
- BehavioralCoveragePlugin -> codeintel.build.hamilton.native.analytics.behavioral_coverage
- SemanticRolesPlugin -> codeintel.build.hamilton.native.analytics.semantic_roles
- SubsystemGraphMetricsPlugin -> codeintel.build.hamilton.native.analytics.subsystem_graph_metrics
- SubsystemAgreementPlugin -> codeintel.build.hamilton.native.analytics.subsystem_agreement
- ConfigDataFlowPlugin -> codeintel.build.hamilton.native.analytics.config_data_flow
- ProfilesPlugin -> codeintel.build.hamilton.native.analytics.profiles
- SymbolGraphMetricsPlugin -> codeintel.build.hamilton.native.analytics.symbol_graph_metrics
- HotspotsPlugin -> codeintel.build.hamilton.native.analytics.hotspots
- RiskFactorsPlugin -> codeintel.build.hamilton.native.analytics.risk_factors
- SubsystemsPlugin -> codeintel.build.hamilton.native.analytics.subsystems
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, ClassVar

from codeintel.build.context import TargetResult
from codeintel.build.plugin import MetadataPlugin
from codeintel.core.plugins.types.metadata import CorePluginMetadata, PluginDomain

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext


def _deprecated_warning(cls_name: str, native_module: str) -> None:
    """Emit a deprecation warning for stub plugin usage.

    Parameters
    ----------
    cls_name
        Name of the deprecated plugin class.
    native_module
        Path to the native Hamilton module replacement.
    """
    warnings.warn(
        f"{cls_name} is deprecated. Use native Hamilton module: {native_module}",
        DeprecationWarning,
        stacklevel=3,
    )


class FunctionMetricsPlugin(MetadataPlugin):
    """Stub for migrated FunctionMetricsPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.function_metrics
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.function_metrics",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="function",
        provides=("analytics.function_metrics", "analytics.function_types"),
        requires=("core.goids",),
        produces_tables=("analytics.function_metrics", "analytics.function_types"),
        consumes_tables=("core.goids",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "FunctionMetricsPlugin",
            "codeintel.build.hamilton.native.analytics.function_metrics",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class FunctionAstFeaturesPlugin(MetadataPlugin):
    """Stub for migrated FunctionAstFeaturesPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.ast_features
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.function_ast_features",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="function",
        provides=("analytics.function_ast_features",),
        requires=("core.goids",),
        produces_tables=("analytics.function_ast_features",),
        consumes_tables=("core.goids",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "FunctionAstFeaturesPlugin",
            "codeintel.build.hamilton.native.analytics.ast_features",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class FunctionEffectsPlugin(MetadataPlugin):
    """Stub for migrated FunctionEffectsPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.function_effects
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.function_effects",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="function",
        provides=("analytics.function_effects", "analytics.function_effect_types"),
        requires=("core.goids",),
        produces_tables=("analytics.function_effects", "analytics.function_effect_types"),
        consumes_tables=("core.goids",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "FunctionEffectsPlugin",
            "codeintel.build.hamilton.native.analytics.function_effects",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class FunctionContractsPlugin(MetadataPlugin):
    """Stub for migrated FunctionContractsPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.function_contracts
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.function_contracts",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="function",
        provides=("analytics.function_contracts",),
        requires=("core.goids",),
        produces_tables=("analytics.function_contracts",),
        consumes_tables=("core.goids",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "FunctionContractsPlugin",
            "codeintel.build.hamilton.native.analytics.function_contracts",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class CoverageTestEdgesPlugin(MetadataPlugin):
    """Stub for migrated CoverageTestEdgesPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.coverage_test_edges
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.coverage_test_edges",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="coverage",
        provides=("analytics.coverage_test_edges",),
        requires=("core.goids",),
        produces_tables=("analytics.coverage_test_edges",),
        consumes_tables=("core.goids",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "CoverageTestEdgesPlugin",
            "codeintel.build.hamilton.native.analytics.coverage_test_edges",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class TestProfilePlugin(MetadataPlugin):
    """Stub for migrated TestProfilePlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.test_profile
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.test_profile",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="test",
        provides=("analytics.test_profile",),
        requires=("core.goids",),
        produces_tables=("analytics.test_profile",),
        consumes_tables=("core.goids",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "TestProfilePlugin",
            "codeintel.build.hamilton.native.analytics.test_profile",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class BehavioralCoveragePlugin(MetadataPlugin):
    """Stub for migrated BehavioralCoveragePlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.behavioral_coverage
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.behavioral_coverage",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="coverage",
        provides=("analytics.behavioral_coverage",),
        requires=("core.goids",),
        produces_tables=("analytics.behavioral_coverage",),
        consumes_tables=("core.goids",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "BehavioralCoveragePlugin",
            "codeintel.build.hamilton.native.analytics.behavioral_coverage",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class SemanticRolesPlugin(MetadataPlugin):
    """Stub for migrated SemanticRolesPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.semantic_roles
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.semantic_roles",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="semantic",
        provides=("analytics.semantic_roles",),
        requires=("core.modules",),
        produces_tables=("analytics.semantic_roles",),
        consumes_tables=("core.modules",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "SemanticRolesPlugin",
            "codeintel.build.hamilton.native.analytics.semantic_roles",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class SubsystemGraphMetricsPlugin(MetadataPlugin):
    """Stub for migrated SubsystemGraphMetricsPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.subsystem_graph_metrics
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.subsystem_graph_metrics",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="subsystem",
        provides=("analytics.subsystem_graph_metrics",),
        requires=("analytics.subsystems",),
        produces_tables=("analytics.subsystem_graph_metrics",),
        consumes_tables=("analytics.subsystems",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "SubsystemGraphMetricsPlugin",
            "codeintel.build.hamilton.native.analytics.subsystem_graph_metrics",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class SubsystemAgreementPlugin(MetadataPlugin):
    """Stub for migrated SubsystemAgreementPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.subsystem_agreement
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.subsystem_agreement",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="subsystem",
        provides=("analytics.subsystem_agreement",),
        requires=("analytics.subsystems",),
        produces_tables=("analytics.subsystem_agreement",),
        consumes_tables=("analytics.subsystems",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "SubsystemAgreementPlugin",
            "codeintel.build.hamilton.native.analytics.subsystem_agreement",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class ConfigDataFlowPlugin(MetadataPlugin):
    """Stub for migrated ConfigDataFlowPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.config_data_flow
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.config_data_flow",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="config",
        provides=("analytics.config_data_flow",),
        requires=("graph.call_graph_edges",),
        produces_tables=("analytics.config_data_flow",),
        consumes_tables=("graph.call_graph_edges",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "ConfigDataFlowPlugin",
            "codeintel.build.hamilton.native.analytics.config_data_flow",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class ProfilesPlugin(MetadataPlugin):
    """Stub for migrated ProfilesPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.profiles
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.profiles",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="profiles",
        provides=("analytics.function_profile", "analytics.file_profile", "analytics.module_profile"),
        requires=("graph.call_graph_edges",),
        produces_tables=("analytics.function_profile", "analytics.file_profile", "analytics.module_profile"),
        consumes_tables=("graph.call_graph_edges",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "ProfilesPlugin",
            "codeintel.build.hamilton.native.analytics.profiles",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class SymbolGraphMetricsPlugin(MetadataPlugin):
    """Stub for migrated SymbolGraphMetricsPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.symbol_graph_metrics
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.symbol_graph_metrics",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="symbol",
        provides=("analytics.symbol_graph_metrics_functions", "analytics.symbol_graph_metrics_modules"),
        requires=("graph.symbol_use_edges",),
        produces_tables=("analytics.symbol_graph_metrics_functions", "analytics.symbol_graph_metrics_modules"),
        consumes_tables=("graph.symbol_use_edges",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "SymbolGraphMetricsPlugin",
            "codeintel.build.hamilton.native.analytics.symbol_graph_metrics",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class HotspotsPlugin(MetadataPlugin):
    """Stub for migrated HotspotsPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.hotspots
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.hotspots",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="hotspots",
        provides=("analytics.hotspots",),
        requires=("core.modules",),
        produces_tables=("analytics.hotspots",),
        consumes_tables=("core.modules",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "HotspotsPlugin",
            "codeintel.build.hamilton.native.analytics.hotspots",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class RiskFactorsPlugin(MetadataPlugin):
    """Stub for migrated RiskFactorsPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.risk_factors
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.risk_factors",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="risk",
        provides=("analytics.goid_risk_factors",),
        requires=("analytics.function_metrics",),
        produces_tables=("analytics.goid_risk_factors",),
        consumes_tables=("analytics.function_metrics",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "RiskFactorsPlugin",
            "codeintel.build.hamilton.native.analytics.risk_factors",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


class SubsystemsPlugin(MetadataPlugin):
    """Stub for migrated SubsystemsPlugin.

    This plugin has been migrated to a native Hamilton module.
    See: codeintel.build.hamilton.native.analytics.subsystems
    """

    _core_metadata: ClassVar[CorePluginMetadata] = CorePluginMetadata(
        name="analytics.subsystems",
        version="4.0.0-stub",
        description="[STUB] Migrated to native Hamilton module",
        domain=PluginDomain.ANALYTICS,
        kind="metric",
        stage="subsystem",
        provides=("analytics.subsystems",),
        requires=("core.modules",),
        produces_tables=("analytics.subsystems",),
        consumes_tables=("core.modules",),
    )

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute stub - raises deprecation warning."""
        _ = (self, ctx)
        _deprecated_warning(
            "SubsystemsPlugin",
            "codeintel.build.hamilton.native.analytics.subsystems",
        )
        return TargetResult.failed("Plugin deprecated - use native Hamilton module")


__all__ = [
    "BehavioralCoveragePlugin",
    "ConfigDataFlowPlugin",
    "CoverageTestEdgesPlugin",
    "FunctionAstFeaturesPlugin",
    "FunctionContractsPlugin",
    "FunctionEffectsPlugin",
    "FunctionMetricsPlugin",
    "HotspotsPlugin",
    "ProfilesPlugin",
    "RiskFactorsPlugin",
    "SemanticRolesPlugin",
    "SubsystemAgreementPlugin",
    "SubsystemGraphMetricsPlugin",
    "SubsystemsPlugin",
    "SymbolGraphMetricsPlugin",
    "TestProfilePlugin",
]
