"""External dependency analytics plugins package.

For Hamilton native execution, use the pure compute functions:
- ``compute_dependency_calls_pure`` returns ``DependencyCallsResult`` without writing
- ``compute_external_dependencies_pure`` returns ``ExternalDependenciesResult`` without writing

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.dependencies``
"""

from codeintel.build.analytics.dependencies.compute import (
    DependencyCallsResult,
    ExternalDependenciesResult,
    compute_dependency_calls_pure,
    compute_external_dependencies_pure,
)
from codeintel.build.analytics.dependencies.core import (
    DependencyAggregate,
    DependencyCall,
    DependencyContext,
    load_config_key_map,
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
