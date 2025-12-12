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

Example
-------
>>> from codeintel.build.hamilton import HamiltonBuildExecutor, BuildEnv
>>> executor = HamiltonBuildExecutor(profile="default")
>>> result = executor.run(env=env, targets=["risk_factors"])
"""

from __future__ import annotations

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.executor import HamiltonBuildExecutor, HamiltonBuildResult
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.metadata_bridge import CanonicalPluginMeta
from codeintel.build.hamilton.naming import dataset_node, target_node, to_node_name

__all__ = [
    "BuildEnv",
    "CanonicalPluginMeta",
    "HamiltonBuildExecutor",
    "HamiltonBuildResult",
    "TargetRunRecord",
    "dataset_node",
    "target_node",
    "to_node_name",
]
