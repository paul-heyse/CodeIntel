"""Validation check implementations.

This package contains all the specific validation checks for graph integrity,
structure analysis, and anomaly detection.

Submodules
----------
database
    Database integrity checks (missing GOIDs, orphan modules).
structure
    Graph structure checks (SCCs, hubs, bridges, connectivity).
anomaly
    Community and subsystem-level anomaly detection.

Check Classes
-------------
All check classes implement CheckProtocol from core/validation and can be
registered with ValidationRunner for unified validation orchestration.
"""

from codeintel.graphs.validation.checks.anomaly import (
    ALL_ANOMALY_CHECKS,
    SubsystemDisagreementCheck,
    SymbolCommunityCheck,
    subsystem_disagreement_findings,
    symbol_community_findings,
)
from codeintel.graphs.validation.checks.database import (
    ALL_DATABASE_CHECKS,
    CallsiteSpanMismatchCheck,
    MissingFunctionGoidsCheck,
    OrphanModulesCheck,
    warn_callsite_span_mismatches,
    warn_missing_function_goids,
    warn_orphan_modules,
)
from codeintel.graphs.validation.checks.structure import (
    ALL_STRUCTURE_CHECKS,
    CallGraphStructureCheck,
    ConfigKeyCheck,
    ImportBridgeCheck,
    ImportCycleCheck,
    ImportGraphStructureCheck,
    ImportHubCheck,
    ImportUpwardCheck,
    SymbolGraphCheck,
    call_graph_findings,
    config_key_findings,
    import_bridge_findings,
    import_cycle_findings,
    import_graph_findings,
    import_hub_findings,
    import_upward_findings,
    symbol_graph_findings,
    warn_graph_structure,
)

__all__ = [
    # Check class tuples
    "ALL_ANOMALY_CHECKS",
    "ALL_DATABASE_CHECKS",
    "ALL_STRUCTURE_CHECKS",
    # Structure check classes
    "CallGraphStructureCheck",
    # Database check classes
    "CallsiteSpanMismatchCheck",
    "ConfigKeyCheck",
    "ImportBridgeCheck",
    "ImportCycleCheck",
    "ImportGraphStructureCheck",
    "ImportHubCheck",
    "ImportUpwardCheck",
    "MissingFunctionGoidsCheck",
    "OrphanModulesCheck",
    # Anomaly check classes
    "SubsystemDisagreementCheck",
    "SymbolCommunityCheck",
    "SymbolGraphCheck",
    # Backward-compatible functions (structure)
    "call_graph_findings",
    "config_key_findings",
    "import_bridge_findings",
    "import_cycle_findings",
    "import_graph_findings",
    "import_hub_findings",
    "import_upward_findings",
    # Backward-compatible functions (anomaly)
    "subsystem_disagreement_findings",
    "symbol_community_findings",
    "symbol_graph_findings",
    # Backward-compatible functions (database)
    "warn_callsite_span_mismatches",
    "warn_graph_structure",
    "warn_missing_function_goids",
    "warn_orphan_modules",
]
