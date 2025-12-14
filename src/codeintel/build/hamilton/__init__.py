"""Hamilton integration for the CodeIntel build system.

This package provides Hamilton-based execution orchestration as an alternative
to the legacy BuildExecutor. Hamilton owns dependency ordering and observability
while existing plugins retain their computation and write logic.

Phase 0 Implementation
----------------------
- Wraps existing target plugins as Hamilton nodes
- Reuses existing manifest/hashing infrastructure
- Provides skip-if-unchanged caching via manifest checks
- Explicit node definitions for the risk_factors chain

Phase 1 Implementation (Full Production Features)
-------------------------------------------------
- HamiltonNodeMode: Support for "phase0" and "generated" node modes
- HamiltonRuntime: Extended with target↔node mappings
- Closure execution: Full dependency closure computed and executed
- Upstream failure gating: Downstream skipped if upstream fails
- Force targets: --force flag bypasses skip checks
- Run tracking: Builds tracked in build.runs table
- DatasetRef: Type-safe dataset references populated on success
- Dataset nodes: d__* nodes generated for all contract tables
- Observability: DAG export and visualization via CLI

Observability Architecture
--------------------------
This package uses Hamilton's native lifecycle adapter pattern for telemetry:

- ``NodeTelemetryHook``: Records per-node execution timing via Hamilton's
  ``pre_node_execute`` / ``post_node_execute`` hooks. Telemetry is persisted
  to the ``build.run_nodes`` table for profiling and debugging.

- ``ContractEnforcementHook``: Activates contract validation per-node using
  the same lifecycle adapter pattern.

- DAG observability functions (``get_dag_info``, ``export_dag_json``, etc.)
  use Hamilton's native Driver introspection APIs.

Future Enhancement: HamiltonTracker Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Hamilton provides an official ``HamiltonTracker`` adapter for integration with
the Hamilton UI (https://hamilton.dagworks.io/). This would enable:

- Visual DAG exploration in a web UI
- Execution history and lineage tracking
- Data quality monitoring dashboards

To integrate, add ``sf-hamilton[ui]`` to dependencies and configure::

    from hamilton_sdk import adapters
    tracker = adapters.HamiltonTracker(
        project_id=<project_id>,
        username=<username>,
        dag_name="codeintel-build",
    )
    # Pass tracker in execute_kwargs["adapters"]

This is not currently implemented but is a natural extension of the existing
lifecycle adapter pattern.

Example
-------
>>> from codeintel.build.hamilton import HamiltonBuildExecutor, BuildEnv
>>> executor = HamiltonBuildExecutor(profile="default", mode="generated")
>>> result = executor.run(env=env, targets=["risk_factors"])
>>> print(f"Computed: {result.computed_targets}")
>>> print(f"Skipped: {result.skipped_targets}")
"""

from __future__ import annotations

from codeintel.build.hamilton.compat import (
    LEGACY_PHASE0,
    build_driver_compat,
    list_available_nodes_compat,
)
from codeintel.build.hamilton.contracts import (
    get_pandera_schema,
    validate_dataframe,
    validate_dataset_ref,
    with_contract,
)
from codeintel.build.hamilton.driver_factory import (
    HamiltonNodeMode,
    HamiltonRuntime,
    build_driver,
    list_available_nodes,
    target_to_node_name,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.executor import HamiltonBuildExecutor, HamiltonBuildResult
from codeintel.build.hamilton.io import DatasetRef, IbisIOConfig, refs_from_target_result
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.metadata_bridge import CanonicalPluginMeta
from codeintel.build.hamilton.naming import dataset_node, target_node, to_node_name
from codeintel.build.hamilton.observability import (
    export_dag_json,
    export_execution_json,
    get_dag_info,
    list_execution_order,
    list_execution_targets,
)
from codeintel.build.hamilton.planner import (
    HamiltonBuildPlan,
    PlanEntry,
    StalenessExplanation,
    compute_plan,
    explain_plan,
)

__all__ = [
    "LEGACY_PHASE0",
    "BuildEnv",
    "CanonicalPluginMeta",
    "DatasetRef",
    "HamiltonBuildExecutor",
    "HamiltonBuildPlan",
    "HamiltonBuildResult",
    "HamiltonNodeMode",
    "HamiltonRuntime",
    "IbisIOConfig",
    "PlanEntry",
    "StalenessExplanation",
    "TargetRunRecord",
    "build_driver",
    "build_driver_compat",
    "compute_plan",
    "dataset_node",
    "explain_plan",
    "export_dag_json",
    "export_execution_json",
    "get_dag_info",
    "get_pandera_schema",
    "list_available_nodes",
    "list_available_nodes_compat",
    "list_execution_order",
    "list_execution_targets",
    "refs_from_target_result",
    "target_node",
    "target_to_node_name",
    "to_node_name",
    "validate_dataframe",
    "validate_dataset_ref",
    "with_contract",
]
