"""Stable naming conventions for Hamilton nodes.

This module provides functions to convert logical identifiers (like table keys
and target names) into valid Python identifiers suitable for Hamilton nodes.

Hamilton requires node names to be valid Python identifiers, but logical IDs
often contain dots, dashes, and slashes. This module standardizes the conversion
so that metadata remains stable while Hamilton nodes are valid.

Examples
--------
>>> to_node_name("analytics.function_metrics", prefix="t")
't__analytics__function_metrics'

>>> target_node("risk_factors")
't__risk_factors'

>>> dataset_node("graph.call_graph_edges")
'd__graph__call_graph_edges'
"""

from __future__ import annotations

import re


def to_node_name(logical_name: str, *, prefix: str) -> str:
    """Convert a logical identifier to a valid Hamilton node name.

    Transform stable logical IDs like ``analytics.function_metrics`` or
    ``graph/call_graph_edges`` into Hamilton-compatible Python identifiers
    like ``t__analytics__function_metrics``.

    Parameters
    ----------
    logical_name
        The logical identifier to convert (e.g., table key or target name).
    prefix
        Single-character prefix to distinguish node types (e.g., "t" for
        targets, "d" for datasets).

    Returns
    -------
    str
        A valid Python identifier suitable for use as a Hamilton node name.

    Examples
    --------
    >>> to_node_name("analytics.function_metrics", prefix="t")
    't__analytics__function_metrics'

    >>> to_node_name("graph-call-edges", prefix="d")
    'd__graph_call_edges'

    >>> to_node_name("some/path/name", prefix="p")
    'p__some__path__name'
    """
    cleaned = logical_name.strip()
    # Replace common separators with double underscores
    cleaned = cleaned.replace(".", "__")
    cleaned = cleaned.replace("-", "_")
    cleaned = cleaned.replace("/", "__")
    # Remove any remaining non-identifier characters
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", cleaned)
    # Collapse multiple underscores
    cleaned = re.sub(r"_+", "_", cleaned)
    # Remove leading/trailing underscores from the cleaned part
    cleaned = cleaned.strip("_")
    return f"{prefix}__{cleaned}"


def target_node(target_name: str) -> str:
    """Convert a target name to a Hamilton node identifier.

    Targets use the "t" prefix to distinguish them from dataset nodes.

    Parameters
    ----------
    target_name
        The target name (e.g., "risk_factors", "call_graph").

    Returns
    -------
    str
        Hamilton node name with "t__" prefix.

    Examples
    --------
    >>> target_node("risk_factors")
    't__risk_factors'

    >>> target_node("function_metrics")
    't__function_metrics'
    """
    return to_node_name(target_name, prefix="t")


def dataset_node(dataset_key: str) -> str:
    """Convert a dataset key to a Hamilton node identifier.

    Datasets use the "d" prefix to distinguish them from target nodes.
    Dataset keys typically follow the pattern "schema.table_name".

    Parameters
    ----------
    dataset_key
        The dataset key (e.g., "graph.call_graph_edges").

    Returns
    -------
    str
        Hamilton node name with "d__" prefix.

    Examples
    --------
    >>> dataset_node("graph.call_graph_edges")
    'd__graph__call_graph_edges'

    >>> dataset_node("analytics.function_metrics")
    'd__analytics__function_metrics'
    """
    return to_node_name(dataset_key, prefix="d")


def node_to_target(node_name: str) -> str | None:
    """Extract the original target name from a Hamilton node identifier.

    Reverses the transformation performed by ``target_node()``. Returns None
    if the node name doesn't follow the target naming convention.

    Parameters
    ----------
    node_name
        The Hamilton node name to convert back.

    Returns
    -------
    str | None
        The original target name, or None if not a valid target node.

    Examples
    --------
    >>> node_to_target("t__risk_factors")
    'risk_factors'

    >>> node_to_target("d__some_dataset")  # Not a target node
    """
    if not node_name.startswith("t__"):
        return None
    return node_name[3:]  # Remove "t__" prefix


__all__ = [
    "dataset_node",
    "node_to_target",
    "target_node",
    "to_node_name",
]
