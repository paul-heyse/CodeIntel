"""Graph validation framework.

This package provides comprehensive validation for graph construction outputs,
including integrity checks, structural analysis, and anomaly detection.

Key Components
--------------
- runner: Main orchestration for running validations
- checks: Individual validation check implementations
- findings: Finding types, persistence, and severity handling
- context: GraphValidationContext for CheckProtocol-based checks
- base: GraphCheckBase for implementing CheckProtocol

Example
-------
```python
from codeintel.build.graphs.validation import (
    GraphValidationContext,
    GraphValidationOptions,
    create_validation_runner,
    run_graph_validations_with_runner,
)


report = run_graph_validations_with_runner(gateway, snapshot=snapshot, runtime=runtime)
print(f"Found {report.error_count} errors")
```
"""

from __future__ import annotations

from codeintel.build.graphs.validation.base import GraphCheckBase
from codeintel.build.graphs.validation.checks import (
    ALL_ANOMALY_CHECKS,
    ALL_DATABASE_CHECKS,
    ALL_STRUCTURE_CHECKS,
    CallGraphStructureCheck,
    CallsiteSpanMismatchCheck,
    ConfigKeyCheck,
    ImportBridgeCheck,
    ImportCycleCheck,
    ImportGraphStructureCheck,
    ImportHubCheck,
    ImportUpwardCheck,
    MissingFunctionGoidsCheck,
    OrphanModulesCheck,
    SubsystemDisagreementCheck,
    SymbolCommunityCheck,
    SymbolGraphCheck,
)
from codeintel.build.graphs.validation.context import GraphValidationContext
from codeintel.build.graphs.validation.findings import (
    CALL_SCC_MIN,
    CONFIG_KEY_MIN_THRESHOLD,
    HUB_DEGREE_RATIO,
    HUB_MIN_DEGREE_FLOOR,
    SAMPLE_LIMIT,
    SYMBOL_COMMUNITY_MIN,
    GraphValidationOptions,
    apply_severity_overrides,
    cap_findings,
    has_error_findings,
    hub_threshold,
    persist_findings,
    resolve_validation_options,
)
from codeintel.build.graphs.validation.runner import (
    ALL_GRAPH_CHECKS,
    create_validation_runner,
    log_db_snapshot,
    resolve_validation_runtime,
    run_graph_validations_with_runner,
    warn_graph_structure,
)

__all__ = [
    # Check class tuples
    "ALL_ANOMALY_CHECKS",
    "ALL_DATABASE_CHECKS",
    "ALL_GRAPH_CHECKS",
    "ALL_STRUCTURE_CHECKS",
    # Constants
    "CALL_SCC_MIN",
    "CONFIG_KEY_MIN_THRESHOLD",
    "HUB_DEGREE_RATIO",
    "HUB_MIN_DEGREE_FLOOR",
    "SAMPLE_LIMIT",
    "SYMBOL_COMMUNITY_MIN",
    # Structure check classes
    "CallGraphStructureCheck",
    # Database check classes
    "CallsiteSpanMismatchCheck",
    "ConfigKeyCheck",
    # Base classes
    "GraphCheckBase",
    "GraphValidationContext",
    "GraphValidationOptions",
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
    # Finding utilities
    "apply_severity_overrides",
    "cap_findings",
    # Runner functions
    "create_validation_runner",
    "has_error_findings",
    "hub_threshold",
    "log_db_snapshot",
    "persist_findings",
    "resolve_validation_options",
    "resolve_validation_runtime",
    "run_graph_validations_with_runner",
    "warn_graph_structure",
]
