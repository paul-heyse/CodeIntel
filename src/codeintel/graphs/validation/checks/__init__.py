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
)
from codeintel.graphs.validation.checks.database import (
    ALL_DATABASE_CHECKS,
    CallsiteSpanMismatchCheck,
    MissingFunctionGoidsCheck,
    OrphanModulesCheck,
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
]
