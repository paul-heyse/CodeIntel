"""Build system for computing minimal execution plans.

This package provides the target graph and state validation
infrastructure for the CodeIntel build system.

Key concepts:

- **OutputTarget**: A discrete output that can be requested and validated
- **TargetGraph**: Complete dependency graph of all output targets
- **OutputManifest**: Record of a target's computation with input/output hashes
- **BuildRunRecord**: Record of a build system run for observability

Use ``get_target_graph()`` to access the singleton target graph instance.
Use ``compute_input_hash()`` to compute content-addressable hashes for targets.
"""

from __future__ import annotations

from codeintel.core.build.hashing import compute_input_hash, compute_options_hash
from codeintel.core.build.manifest import BuildRunRecord, OutputManifest
from codeintel.core.build.registry import build_target_graph, get_target_graph
from codeintel.core.build.targets import OutputTarget, TargetGraph, TargetModule

__all__ = [
    "BuildRunRecord",
    "OutputManifest",
    "OutputTarget",
    "TargetGraph",
    "TargetModule",
    "build_target_graph",
    "compute_input_hash",
    "compute_options_hash",
    "get_target_graph",
]
