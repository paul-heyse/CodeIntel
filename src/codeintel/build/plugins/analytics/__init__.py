"""Analytics plugins using the TargetPlugin protocol.

.. deprecated::
    All analytics plugins have been migrated to native Hamilton modules.
    Use the native modules in ``codeintel.build.hamilton.native.analytics`` instead.

For the actual implementations, see:
``codeintel.build.hamilton.native.analytics``

Migrated plugins:
- BehavioralCoveragePlugin -> behavioral_coverage.py
- ConfigDataFlowPlugin -> config_data_flow.py
- CoverageTestEdgesPlugin -> coverage_test_edges.py
- FunctionAstFeaturesPlugin -> ast_features.py
- FunctionContractsPlugin -> function_contracts.py
- FunctionEffectsPlugin -> function_effects.py
- FunctionMetricsPlugin -> function_metrics.py
- HotspotsPlugin -> hotspots.py
- ProfilesPlugin -> profiles.py
- RiskFactorsPlugin -> risk_factors.py
- SemanticRolesPlugin -> semantic_roles.py
- SubsystemAgreementPlugin -> subsystem_agreement.py
- SubsystemGraphMetricsPlugin -> subsystem_graph_metrics.py
- SubsystemsPlugin -> subsystems.py
- SymbolGraphMetricsPlugin -> symbol_graph_metrics.py
- TestProfilePlugin -> test_profile.py
"""

from __future__ import annotations

__all__: list[str] = []
