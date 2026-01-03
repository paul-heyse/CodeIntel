"""External dependency analytics plugins package.

For Hamilton native execution, use the orchestration helpers:
- ``compute_dependency_calls_pure`` loads patterns/aliases and returns rows
- ``compute_external_dependencies_pure`` loads patterns and returns rows

Pure compute functions live in ``codeintel.build.analytics.compute.dependencies.compute``.

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.dependencies``
"""

from codeintel.build.analytics.compute.dependencies.compute import (
    DependencyAggregate,
    DependencyContext,
    load_config_key_map,
)
from codeintel.build.analytics.compute.dependencies.detection import DependencyCall
from codeintel.build.analytics.dependencies.compute import (
    DependencyCallsResult,
    ExternalDependenciesResult,
    compute_dependency_calls_pure,
    compute_external_dependencies_pure,
)

__all__ = [
    "DependencyAggregate",
    "DependencyCall",
    "DependencyCallsResult",
    "DependencyContext",
    "ExternalDependenciesResult",
    "compute_dependency_calls_pure",
    "compute_external_dependencies_pure",
    "load_config_key_map",
]
