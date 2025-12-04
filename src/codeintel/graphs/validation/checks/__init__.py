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
"""

from codeintel.graphs.validation.checks.anomaly import (
    subsystem_disagreement_findings,
    symbol_community_findings,
)
from codeintel.graphs.validation.checks.database import (
    warn_callsite_span_mismatches,
    warn_missing_function_goids,
    warn_orphan_modules,
)
from codeintel.graphs.validation.checks.structure import (
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
    "call_graph_findings",
    "config_key_findings",
    "import_bridge_findings",
    "import_cycle_findings",
    "import_graph_findings",
    "import_hub_findings",
    "import_upward_findings",
    "subsystem_disagreement_findings",
    "symbol_community_findings",
    "symbol_graph_findings",
    "warn_callsite_span_mismatches",
    "warn_graph_structure",
    "warn_missing_function_goids",
    "warn_orphan_modules",
]
