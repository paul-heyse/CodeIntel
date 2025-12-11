"""Central registry of all plugin metadata."""

from __future__ import annotations

from functools import cache

from codeintel.analytics.plugins.cfg_dfg.metrics import CFG_DFG_METRICS_METADATA
from codeintel.analytics.plugins.config_data_flow.compute import CONFIG_DATA_FLOW_METADATA
from codeintel.analytics.plugins.coverage.functions import FUNCTION_COVERAGE_METADATA
from codeintel.analytics.plugins.coverage.test_edges import TEST_COVERAGE_EDGES_METADATA
from codeintel.analytics.plugins.data_models.build import DATA_MODELS_METADATA
from codeintel.analytics.plugins.data_models.usage import DATA_MODEL_USAGE_METADATA
from codeintel.analytics.plugins.dependencies.external import EXTERNAL_DEPS_METADATA
from codeintel.analytics.plugins.entrypoints.build import ENTRYPOINTS_METADATA
from codeintel.analytics.plugins.functions.ast_features import FUNCTION_AST_FEATURES_METADATA
from codeintel.analytics.plugins.functions.contracts import FUNCTION_CONTRACTS_METADATA
from codeintel.analytics.plugins.functions.effects import FUNCTION_EFFECTS_METADATA
from codeintel.analytics.plugins.functions.history import FUNCTION_HISTORY_METADATA
from codeintel.analytics.plugins.functions.metrics import FUNCTION_METRICS_METADATA
from codeintel.analytics.plugins.history.timeseries import HISTORY_TIMESERIES_METADATA
from codeintel.analytics.plugins.hotspots.build import HOTSPOTS_METADATA
from codeintel.analytics.plugins.profiles.build import PROFILES_METADATA
from codeintel.analytics.plugins.risk.factors import RISK_FACTORS_METADATA
from codeintel.analytics.plugins.semantic_roles.compute import SEMANTIC_ROLES_METADATA
from codeintel.analytics.plugins.subsystem_metrics.agreement import (
    SUBSYSTEM_AGREEMENT_METADATA,
)
from codeintel.analytics.plugins.subsystem_metrics.graph_metrics import (
    SUBSYSTEM_GRAPH_METRICS_METADATA,
)
from codeintel.analytics.plugins.subsystems.build import SUBSYSTEMS_METADATA
from codeintel.analytics.plugins.symbol_graph_metrics.compute import (
    SYMBOL_GRAPH_METRICS_METADATA,
)
from codeintel.analytics.plugins.tests.behavioral_coverage import (
    BEHAVIORAL_COVERAGE_METADATA,
)
from codeintel.analytics.plugins.tests.graph_metrics import TEST_GRAPH_METRICS_METADATA
from codeintel.analytics.plugins.tests.profile import TEST_PROFILE_METADATA
from codeintel.analytics.plugins.types.coverage import TYPE_COVERAGE_METADATA
from codeintel.core.plugins.registry.capability_index import (
    PluginRegistryIndex,
    build_registry_index,
)
from codeintel.core.plugins.types.metadata import CorePluginMetadata
from codeintel.graphs.plugins.builders.callgraph import CALLGRAPH_METADATA
from codeintel.graphs.plugins.builders.cfg_dfg import CFG_DFG_METADATA as GRAPH_CFG_DFG_METADATA
from codeintel.graphs.plugins.builders.goid import GOID_BUILDER_METADATA
from codeintel.graphs.plugins.builders.import_graph import IMPORT_GRAPH_METADATA
from codeintel.graphs.plugins.builders.symbol_uses import SYMBOL_USES_METADATA
from codeintel.ingestion.plugins.ast_extract import AST_EXTRACT_METADATA
from codeintel.ingestion.plugins.config_plugin import CONFIG_INGEST_METADATA
from codeintel.ingestion.plugins.coverage_plugin import COVERAGE_INGEST_METADATA
from codeintel.ingestion.plugins.cst_extract import CST_EXTRACT_METADATA
from codeintel.ingestion.plugins.docstrings_plugin import DOCSTRINGS_METADATA
from codeintel.ingestion.plugins.modules_plugin import MODULE_INGEST_METADATA
from codeintel.ingestion.plugins.repo_scan import REPO_SCAN_METADATA
from codeintel.ingestion.plugins.scip_plugin import SCIP_INGEST_METADATA
from codeintel.ingestion.plugins.tests_plugin import TESTS_INGEST_METADATA
from codeintel.ingestion.plugins.typing_plugin import TYPING_INGEST_METADATA

ALL_PLUGIN_METADATA: tuple[CorePluginMetadata, ...] = (
    # Analytics
    BEHAVIORAL_COVERAGE_METADATA,
    CFG_DFG_METRICS_METADATA,
    CONFIG_DATA_FLOW_METADATA,
    DATA_MODELS_METADATA,
    DATA_MODEL_USAGE_METADATA,
    ENTRYPOINTS_METADATA,
    EXTERNAL_DEPS_METADATA,
    FUNCTION_AST_FEATURES_METADATA,
    FUNCTION_CONTRACTS_METADATA,
    FUNCTION_COVERAGE_METADATA,
    FUNCTION_EFFECTS_METADATA,
    FUNCTION_HISTORY_METADATA,
    FUNCTION_METRICS_METADATA,
    HISTORY_TIMESERIES_METADATA,
    HOTSPOTS_METADATA,
    PROFILES_METADATA,
    RISK_FACTORS_METADATA,
    SEMANTIC_ROLES_METADATA,
    SUBSYSTEMS_METADATA,
    SUBSYSTEM_AGREEMENT_METADATA,
    SUBSYSTEM_GRAPH_METRICS_METADATA,
    SYMBOL_GRAPH_METRICS_METADATA,
    TEST_COVERAGE_EDGES_METADATA,
    TEST_GRAPH_METRICS_METADATA,
    TEST_PROFILE_METADATA,
    TYPE_COVERAGE_METADATA,
    # Graphs
    CALLGRAPH_METADATA,
    GRAPH_CFG_DFG_METADATA,
    GOID_BUILDER_METADATA,
    IMPORT_GRAPH_METADATA,
    SYMBOL_USES_METADATA,
    # Ingestion
    REPO_SCAN_METADATA,
    MODULE_INGEST_METADATA,
    AST_EXTRACT_METADATA,
    CST_EXTRACT_METADATA,
    DOCSTRINGS_METADATA,
    CONFIG_INGEST_METADATA,
    COVERAGE_INGEST_METADATA,
    TESTS_INGEST_METADATA,
    TYPING_INGEST_METADATA,
    SCIP_INGEST_METADATA,
)

@cache
def get_global_registry_index() -> PluginRegistryIndex:
    """Return the global plugin registry index.

    Returns
    -------
    PluginRegistryIndex
        Registry index built from all plugin metadata.
    """
    return build_registry_index(ALL_PLUGIN_METADATA)


def get_provider_lookup() -> dict[str, str]:
    """Return capability → provider name lookup.

    Returns
    -------
    dict[str, str]
        Mapping of capability name to provider plugin name.
    """
    return get_global_registry_index().provider_lookup()


__all__ = [
    "ALL_PLUGIN_METADATA",
    "get_global_registry_index",
    "get_provider_lookup",
]
