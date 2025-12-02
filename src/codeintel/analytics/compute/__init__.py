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
risk
    Risk scoring algorithms.
profiles
    Profile aggregation logic.
"""

from __future__ import annotations

__all__: list[str] = []
