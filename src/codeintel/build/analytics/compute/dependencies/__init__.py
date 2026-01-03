"""Pure computation functions for dependency analysis.

This module provides side-effect-free functions for:
- Detecting dependency calls in AST nodes
- Classifying dependency modes and severity
- Computing risk scores

All functions are pure - they do not perform I/O operations.
"""

from __future__ import annotations

from codeintel.build.analytics.compute.dependencies.classification import (
    SEVERITY_SCORES,
    DependencyModePattern,
    LibraryPattern,
    classify_modes,
    risk_level,
    risk_score,
    severity_score,
)
from codeintel.build.analytics.compute.dependencies.compute import (
    DependencyAggregate,
    DependencyCallsResult,
    DependencyContext,
    ExternalDependenciesInputs,
    ExternalDependenciesResult,
    ExternalDependencyInputs,
    compute_dependency_calls_pure,
    compute_external_dependencies_pure,
    load_config_key_map,
)
from codeintel.build.analytics.compute.dependencies.detection import (
    DependencyCall,
    DependencyCallVisitor,
    build_alias_map,
    build_alias_maps,
    build_alias_maps_from_sources,
    group_calls_by_library,
)

__all__ = [
    "SEVERITY_SCORES",
    "DependencyAggregate",
    "DependencyCall",
    "DependencyCallVisitor",
    "DependencyCallsResult",
    "DependencyContext",
    "DependencyModePattern",
    "ExternalDependenciesInputs",
    "ExternalDependenciesResult",
    "ExternalDependencyInputs",
    "LibraryPattern",
    "build_alias_map",
    "build_alias_maps",
    "build_alias_maps_from_sources",
    "classify_modes",
    "compute_dependency_calls_pure",
    "compute_external_dependencies_pure",
    "group_calls_by_library",
    "load_config_key_map",
    "risk_level",
    "risk_score",
    "severity_score",
]
