"""Compatibility exports for schema-generated row types.

This module provides stable, legacy-friendly names (e.g., ``FunctionMetricsRow``)
that resolve to schema-generated row models in ``codeintel.core.schemas.generated_rows``.

Notes
-----
Prefer importing the schema-prefixed types directly from
``codeintel.core.schemas.generated_rows`` in new code.
"""

from __future__ import annotations

from codeintel.core.data_models.rows import (
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    GoidCrosswalkRow,
    GoidRow,
    ImportEdgeRow,
    ImportModuleRow,
    SymbolUseRow,
)
from codeintel.core.schemas.generated_rows.analytics import (
    AnalyticsBehavioralCoverageRow,
    AnalyticsConfigValuesRow,
    AnalyticsCoverageLinesRow,
    AnalyticsFileProfileRow,
    AnalyticsFunctionAstFeaturesRow,
    AnalyticsFunctionContractsRow,
    AnalyticsFunctionEffectsRow,
    AnalyticsFunctionMetricsRow,
    AnalyticsFunctionProfileRow,
    AnalyticsFunctionTypesRow,
    AnalyticsFunctionValidationRow,
    AnalyticsGraphMetricsFunctionsExtRow,
    AnalyticsGraphMetricsFunctionsRow,
    AnalyticsGraphMetricsModulesExtRow,
    AnalyticsGraphMetricsModulesRow,
    AnalyticsGraphValidationRow,
    AnalyticsHotspotsRow,
    AnalyticsModuleProfileRow,
    AnalyticsStaticDiagnosticsRow,
    AnalyticsSubsystemCoverageCacheRow,
    AnalyticsSubsystemProfileCacheRow,
    AnalyticsTestCatalogRow,
    AnalyticsTestCoverageEdgesRow,
    AnalyticsTestProfileRow,
    AnalyticsTypednessRow,
)
from codeintel.core.schemas.generated_rows.core import CoreDocstringsRow
from codeintel.core.schemas.generated_rows.graph import (
    GraphCallGraphEdgesRow,
    GraphCallGraphNodesRow,
)

BehavioralCoverageRowModel = AnalyticsBehavioralCoverageRow
CallGraphEdgeRow = GraphCallGraphEdgesRow
CallGraphNodeRow = GraphCallGraphNodesRow
ConfigValueRow = AnalyticsConfigValuesRow
CoverageLineRow = AnalyticsCoverageLinesRow
DocstringRow = CoreDocstringsRow
FileProfileRowModel = AnalyticsFileProfileRow
FunctionAstFeaturesRow = AnalyticsFunctionAstFeaturesRow
FunctionContractsRow = AnalyticsFunctionContractsRow
FunctionEffectsRow = AnalyticsFunctionEffectsRow
FunctionMetricsRow = AnalyticsFunctionMetricsRow
FunctionProfileRowModel = AnalyticsFunctionProfileRow
FunctionTypesRow = AnalyticsFunctionTypesRow
FunctionValidationRow = AnalyticsFunctionValidationRow
GraphMetricsFunctionsExtRow = AnalyticsGraphMetricsFunctionsExtRow
GraphMetricsFunctionsRow = AnalyticsGraphMetricsFunctionsRow
GraphMetricsModulesExtRow = AnalyticsGraphMetricsModulesExtRow
GraphMetricsModulesRow = AnalyticsGraphMetricsModulesRow
GraphValidationRow = AnalyticsGraphValidationRow
HotspotRow = AnalyticsHotspotsRow
ModuleProfileRowModel = AnalyticsModuleProfileRow
ProfileRowModel = AnalyticsTestProfileRow
StaticDiagnosticRow = AnalyticsStaticDiagnosticsRow
SubsystemCoverageCacheRow = AnalyticsSubsystemCoverageCacheRow
SubsystemProfileCacheRow = AnalyticsSubsystemProfileCacheRow
TestCatalogRowModel = AnalyticsTestCatalogRow
TestCoverageEdgeRow = AnalyticsTestCoverageEdgesRow
TypednessRow = AnalyticsTypednessRow

__all__ = [
    "BehavioralCoverageRowModel",
    "CFGBlockRow",
    "CFGEdgeRow",
    "CallGraphEdgeRow",
    "CallGraphNodeRow",
    "ConfigValueRow",
    "CoverageLineRow",
    "DFGEdgeRow",
    "DocstringRow",
    "FileProfileRowModel",
    "FunctionAstFeaturesRow",
    "FunctionContractsRow",
    "FunctionEffectsRow",
    "FunctionMetricsRow",
    "FunctionProfileRowModel",
    "FunctionTypesRow",
    "FunctionValidationRow",
    "GoidCrosswalkRow",
    "GoidRow",
    "GraphMetricsFunctionsExtRow",
    "GraphMetricsFunctionsRow",
    "GraphMetricsModulesExtRow",
    "GraphMetricsModulesRow",
    "GraphValidationRow",
    "HotspotRow",
    "ImportEdgeRow",
    "ImportModuleRow",
    "ModuleProfileRowModel",
    "ProfileRowModel",
    "StaticDiagnosticRow",
    "SubsystemCoverageCacheRow",
    "SubsystemProfileCacheRow",
    "SymbolUseRow",
    "TestCatalogRowModel",
    "TestCoverageEdgeRow",
    "TypednessRow",
]
