"""Pure computation functions for dependency analysis.

This module provides side-effect-free functions for:
- Detecting dependency calls in AST nodes
- Classifying dependency modes and severity
- Computing risk scores

All functions are pure - they do not perform I/O operations.
"""

from __future__ import annotations

from codeintel.analytics.compute.dependencies.classification import (
    SEVERITY_SCORES,
    LibraryPattern,
    DependencyModePattern,
    classify_modes,
    risk_level,
    risk_score,
    severity_score,
)
from codeintel.analytics.compute.dependencies.detection import (
    DependencyCall,
    DependencyCallVisitor,
    build_alias_map,
    build_alias_maps,
    group_calls_by_library,
)

__all__ = [
    "DependencyCall",
    "DependencyCallVisitor",
    "DependencyModePattern",
    "LibraryPattern",
    "SEVERITY_SCORES",
    "build_alias_map",
    "build_alias_maps",
    "classify_modes",
    "group_calls_by_library",
    "risk_level",
    "risk_score",
    "severity_score",
]

