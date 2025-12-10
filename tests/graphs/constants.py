"""Shared graph test constants for parametrized sweeps."""

from __future__ import annotations

from typing import Final

# Common sizes for complete graph sweeps in community/statistics suites.
COMPLETE_GRAPH_SIZES: Final[tuple[int, ...]] = (5, 10, 15, 20)
# Smaller complete graph sizes used for edge-count assertions.
SMALL_COMPLETE_GRAPH_SIZES: Final[tuple[int, ...]] = (3, 4, 5)

# Cycle sizes used across SCC and condensation tests.
CYCLE_SCC_SIZES: Final[tuple[int, ...]] = (3, 5, 10)

# Star spoke counts used across reachability/degree sweeps.
STAR_SPOKE_SWEEP: Final[tuple[int, ...]] = (1, 3, 5, 10)

# Cycle sizes used for cycle/condensation limit tests.
CYCLE_SIZE_SWEEP: Final[tuple[int, ...]] = (3, 4, 5)

# Tree depth/branching shapes for structural sweeps.
TREE_SHAPES: Final[tuple[tuple[int, int], ...]] = ((2, 2), (3, 2), (3, 3))
