"""External dependency analytics plugins package."""

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
    "DependencyContext",
    "build_external_dependencies",
    "build_external_dependency_calls",
    "load_config_key_map",
]
