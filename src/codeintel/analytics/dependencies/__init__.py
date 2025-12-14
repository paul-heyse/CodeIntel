"""External dependency analytics plugins package.

For Hamilton native execution, use the pure compute functions:
- `compute_dependency_calls_pure` returns `DependencyCallsResult` without writing
- `compute_external_dependencies_pure` returns `ExternalDependenciesResult` without writing
"""

from codeintel.analytics.dependencies.compute import (
    DependencyCallsResult,
    ExternalDependenciesResult,
    compute_dependency_calls_pure,
    compute_external_dependencies_pure,
)
from codeintel.analytics.dependencies.core import (
    DependencyAggregate,
    DependencyCall,
    DependencyContext,
    build_external_dependencies,
    build_external_dependency_calls,
    load_config_key_map,
)

__all__ = [
    "DependencyAggregate",
    "DependencyCall",
    "DependencyCallsResult",
    "DependencyContext",
    "ExternalDependenciesResult",
    "build_external_dependencies",
    "build_external_dependency_calls",
    "compute_dependency_calls_pure",
    "compute_external_dependencies_pure",
    "load_config_key_map",
]
