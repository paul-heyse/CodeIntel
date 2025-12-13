"""Pure computation layer for analytics.

This package contains pure functions with no I/O or side effects.
All computation modules operate solely on in-memory data structures,
making them easily testable and parallelizable.

Subpackages
-----------
functions
    Function-level analysis (complexity, typedness, signatures).
graphs
    Graph-theoretic algorithms (centrality, components).
profiles
    Profile aggregation logic.
dependencies
    Dependency detection and classification.
subsystems
    Subsystem clustering and classification.
semantic_roles
    Semantic role classification for functions and modules.
row_builders
    Row builder functions for constructing typed rows from computed metrics.
"""

from __future__ import annotations

__all__: list[str] = []
