"""Graph validation framework.

This package provides comprehensive validation for graph construction outputs,
including integrity checks, structural analysis, and anomaly detection.

Key Components
--------------
- runner: Main orchestration for running validations
- checks: Individual validation check implementations
- findings: Finding types, persistence, and severity handling

Example
-------
```python
from codeintel.graphs.validation import (
    GraphValidationOptions,
    run_graph_validations,
    warn_graph_structure,
)


run_graph_validations(gateway, snapshot=snapshot, runtime=runtime)


findings = warn_graph_structure(engine, repo, commit)
```
"""

from __future__ import annotations

from codeintel.graphs.validation.checks import (
    call_graph_findings,
    config_key_findings,
    import_bridge_findings,
    import_cycle_findings,
    import_graph_findings,
    import_hub_findings,
    import_upward_findings,
    subsystem_disagreement_findings,
    symbol_community_findings,
    symbol_graph_findings,
    warn_callsite_span_mismatches,
    warn_graph_structure,
    warn_missing_function_goids,
    warn_orphan_modules,
)
from codeintel.graphs.validation.findings import (
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
from codeintel.graphs.validation.runner import (
    log_db_snapshot,
    resolve_validation_runtime,
    run_graph_validations,
)

__all__ = [
    "CALL_SCC_MIN",
    "CONFIG_KEY_MIN_THRESHOLD",
    "HUB_DEGREE_RATIO",
    "HUB_MIN_DEGREE_FLOOR",
    "SAMPLE_LIMIT",
    "SYMBOL_COMMUNITY_MIN",
    "GraphValidationOptions",
    "apply_severity_overrides",
    "call_graph_findings",
    "cap_findings",
    "config_key_findings",
    "has_error_findings",
    "hub_threshold",
    "import_bridge_findings",
    "import_cycle_findings",
    "import_graph_findings",
    "import_hub_findings",
    "import_upward_findings",
    "log_db_snapshot",
    "persist_findings",
    "resolve_validation_options",
    "resolve_validation_runtime",
    "run_graph_validations",
    "subsystem_disagreement_findings",
    "symbol_community_findings",
    "symbol_graph_findings",
    "warn_callsite_span_mismatches",
    "warn_graph_structure",
    "warn_missing_function_goids",
    "warn_orphan_modules",
]
