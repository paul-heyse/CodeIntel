"""Analytics-focused pipeline steps.

This package contains all analytics-related pipeline steps.

Steps are organized into categories:

- Function analytics (metrics, effects, contracts, history)
- Coverage and testing (coverage edges, test profiles)
- Graph metrics (hotspots, graph metrics)
- Data models and config (data models, config data flow)
- Profiles and organization (profiles, subsystems, risk factors)
"""

from __future__ import annotations

from codeintel.pipeline.steps.analytics.steps import (
    ANALYTICS_STEPS,
    BehavioralCoverageStep,
    ConfigDataFlowStep,
    CoverageAnalyticsStep,
    DataModelUsageStep,
    DataModelsStep,
    EntryPointsStep,
    ExternalDependenciesStep,
    FunctionAnalyticsStep,
    FunctionContractsStep,
    FunctionEffectsStep,
    FunctionHistoryStep,
    GraphMetricsStep,
    HistoryTimeseriesStep,
    HotspotsStep,
    ProfilesStep,
    RiskFactorsStep,
    SemanticRolesStep,
    SubsystemsStep,
    TestCoverageEdgesStep,
    TestProfileStep,
)

__all__ = [
    "ANALYTICS_STEPS",
    "BehavioralCoverageStep",
    "ConfigDataFlowStep",
    "CoverageAnalyticsStep",
    "DataModelUsageStep",
    "DataModelsStep",
    "EntryPointsStep",
    "ExternalDependenciesStep",
    "FunctionAnalyticsStep",
    "FunctionContractsStep",
    "FunctionEffectsStep",
    "FunctionHistoryStep",
    "GraphMetricsStep",
    "HistoryTimeseriesStep",
    "HotspotsStep",
    "ProfilesStep",
    "RiskFactorsStep",
    "SemanticRolesStep",
    "SubsystemsStep",
    "TestCoverageEdgesStep",
    "TestProfileStep",
]
