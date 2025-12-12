"""Shared graph test constants for parametrized sweeps."""

from __future__ import annotations

from typing import Final

COMPLETE_GRAPH_SIZES: Final[tuple[int, ...]] = (5, 10, 15, 20)

SMALL_COMPLETE_GRAPH_SIZES: Final[tuple[int, ...]] = (3, 4, 5)


CYCLE_SCC_SIZES: Final[tuple[int, ...]] = (3, 5, 10)


STAR_SPOKE_SWEEP: Final[tuple[int, ...]] = (1, 3, 5, 10)


CYCLE_SIZE_SWEEP: Final[tuple[int, ...]] = (2, 3, 4, 5, 10)


TREE_SHAPES: Final[tuple[tuple[int, int], ...]] = ((2, 2), (3, 2), (3, 3))
