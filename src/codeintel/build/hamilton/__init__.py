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

Example
-------
>>> from codeintel.build.hamilton import HamiltonBuildExecutor, BuildEnv
>>> executor = HamiltonBuildExecutor(profile="default", mode="generated")
>>> result = executor.run(env=env, targets=["risk_factors"])
>>> print(f"Computed: {result.computed_targets}")
>>> print(f"Skipped: {result.skipped_targets}")
"""

from __future__ import annotations

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

__all__ = [
    "BuildEnv",
    "CanonicalPluginMeta",
    "DatasetRef",
    "HamiltonBuildExecutor",
    "HamiltonBuildResult",
    "HamiltonNodeMode",
    "HamiltonRuntime",
    "IbisIOConfig",
    "TargetRunRecord",
    "build_driver",
    "dataset_node",
    "export_dag_json",
    "export_execution_json",
    "get_dag_info",
    "get_pandera_schema",
    "list_available_nodes",
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
