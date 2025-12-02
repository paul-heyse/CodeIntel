"""Type-safe wrappers for NetworkX operations.

This module provides typed wrapper functions for NetworkX operations that
have incomplete or incorrect type stubs. These wrappers isolate type quirks
to a single module and provide type-safe interfaces for the rest of the codebase.

The NetworkX stubs incorrectly type `in_degree()`, `out_degree()`, and `degree()`
as returning `int` rather than `DegreeView`. We use `cast()` to specify the
correct iterable type, which is documented and intentional.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypeVar, cast

import networkx as nx

if TYPE_CHECKING:
    from collections.abc import Hashable

# Type variable for node types (typically int for GOIDs)
NodeT = TypeVar("NodeT", bound="Hashable")

# NetworkX DegreeView is Iterable[tuple[node, degree]] but stubs type it as int
DegreeViewT = Iterable[tuple[int, int]]


def get_in_degrees(graph: nx.DiGraph) -> list[tuple[int, int]]:
    """Extract in-degree tuples from a directed graph.

    Parameters
    ----------
    graph
        A directed graph to analyze.

    Returns
    -------
    list[tuple[int, int]]
        List of (node, in_degree) tuples for all nodes in the graph.

    Example
    -------
    >>> g = nx.DiGraph([(1, 2), (1, 3), (2, 3)])
    >>> get_in_degrees(g)
    [(1, 0), (2, 1), (3, 2)]
    """
    # NetworkX stubs incorrectly type this as int; cast to actual iterable type
    degrees = cast("DegreeViewT", graph.in_degree())
    return [(node, degree) for node, degree in degrees]


def get_out_degrees(graph: nx.DiGraph) -> list[tuple[int, int]]:
    """Extract out-degree tuples from a directed graph.

    Parameters
    ----------
    graph
        A directed graph to analyze.

    Returns
    -------
    list[tuple[int, int]]
        List of (node, out_degree) tuples for all nodes in the graph.

    Example
    -------
    >>> g = nx.DiGraph([(1, 2), (1, 3), (2, 3)])
    >>> get_out_degrees(g)
    [(1, 2), (2, 1), (3, 0)]
    """
    # NetworkX stubs incorrectly type this as int; cast to actual iterable type
    degrees = cast("DegreeViewT", graph.out_degree())
    return [(node, degree) for node, degree in degrees]


def get_degrees(graph: nx.Graph) -> list[tuple[int, int]]:
    """Extract degree tuples from an undirected graph.

    Parameters
    ----------
    graph
        An undirected graph to analyze.

    Returns
    -------
    list[tuple[int, int]]
        List of (node, degree) tuples for all nodes in the graph.

    Example
    -------
    >>> g = nx.Graph([(1, 2), (1, 3), (2, 3)])
    >>> get_degrees(g)
    [(1, 2), (2, 2), (3, 2)]
    """
    # NetworkX stubs have issues with degree(); use attribute access which is also DegreeView
    degrees = cast("DegreeViewT", graph.degree)
    return [(node, degree) for node, degree in degrees]


def get_in_degree_values(graph: nx.DiGraph) -> list[int]:
    """Extract just the in-degree values from a directed graph.

    Parameters
    ----------
    graph
        A directed graph to analyze.

    Returns
    -------
    list[int]
        List of in-degree values for all nodes (in node iteration order).
    """
    # NetworkX stubs incorrectly type this as int; cast to actual iterable type
    degrees = cast("DegreeViewT", graph.in_degree())
    return [degree for _, degree in degrees]


def get_out_degree_values(graph: nx.DiGraph) -> list[int]:
    """Extract just the out-degree values from a directed graph.

    Parameters
    ----------
    graph
        A directed graph to analyze.

    Returns
    -------
    list[int]
        List of out-degree values for all nodes (in node iteration order).
    """
    # NetworkX stubs incorrectly type this as int; cast to actual iterable type
    degrees = cast("DegreeViewT", graph.out_degree())
    return [degree for _, degree in degrees]


def get_degree_values(graph: nx.Graph) -> list[int]:
    """Extract just the degree values from an undirected graph.

    Parameters
    ----------
    graph
        An undirected graph to analyze.

    Returns
    -------
    list[int]
        List of degree values for all nodes (in node iteration order).
    """
    # NetworkX stubs have issues with degree(); use attribute access which is also DegreeView
    degrees = cast("DegreeViewT", graph.degree)
    return [degree for _, degree in degrees]


__all__ = [
    "get_degree_values",
    "get_degrees",
    "get_in_degree_values",
    "get_in_degrees",
    "get_out_degree_values",
    "get_out_degrees",
]
