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

    cleaned = cleaned.replace("-", "_")

    cleaned = cleaned.replace(".", "__")
    cleaned = cleaned.replace("/", "__")

    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", cleaned)

    cleaned = re.sub(r"_{3,}", "__", cleaned)

    cleaned = cleaned.strip("_")
    return f"{prefix}__{cleaned}"


def sanitize_pipeline_component(component: str, *, default: str = "pipeline") -> str:
    """Sanitize a pipeline name component into a valid identifier fragment.

    Parameters
    ----------
    component
        Pipeline namespace or step component to sanitize.
    default
        Fallback value when the component is empty after sanitization.

    Returns
    -------
    str
        Sanitized component suitable for use in a node name.
    """
    cleaned = component.strip()
    cleaned = cleaned.replace(".", "_")
    cleaned = cleaned.replace("/", "_")
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "_", cleaned)
    cleaned = re.sub(r"_+", "_", cleaned)
    cleaned = cleaned.strip("_")
    if not cleaned:
        cleaned = default
    if cleaned[0].isdigit():
        cleaned = f"p_{cleaned}"
    return cleaned


def pipeline_node_name(namespace: str, step_name: str) -> str:
    """Build a stable node name for a pipeline step.

    Parameters
    ----------
    namespace
        Pipeline namespace string.
    step_name
        Step function name.

    Returns
    -------
    str
        Sanitized node name combining namespace and step name.
    """
    namespace_part = sanitize_pipeline_component(namespace)
    step_part = sanitize_pipeline_component(step_name.lstrip("_"))
    return f"{namespace_part}__{step_part}"


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


def compute_node(target_name: str) -> str:
    """Return the canonical compute node name for a target.

    Native targets commonly expose a pure compute node named
    ``t__<target>__compute`` that returns a DuckDB relation.

    Parameters
    ----------
    target_name
        Target name (e.g., "risk_factors").

    Returns
    -------
    str
        Hamilton node name for the compute node.
    """
    return f"{target_node(target_name)}__compute"


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


def artifact_node(artifact_name: str) -> str:
    """Convert an artifact name to a Hamilton node identifier.

    Artifacts use the "a" prefix to distinguish them from target and dataset nodes.

    Parameters
    ----------
    artifact_name
        The artifact name (e.g., "faiss_index", "model_weights").

    Returns
    -------
    str
        Hamilton node name with "a__" prefix.

    Examples
    --------
    >>> artifact_node("faiss_index")
    'a__faiss_index'

    >>> artifact_node("model_weights")
    'a__model_weights'
    """
    return to_node_name(artifact_name, prefix="a")


def path_node(artifact_name: str) -> str:
    """Convert an artifact name to a Path support node identifier.

    Path nodes use the "p" prefix and return resolved filesystem paths.

    Parameters
    ----------
    artifact_name
        The artifact name (e.g., "faiss_index", "model_weights").

    Returns
    -------
    str
        Hamilton node name with "p__" prefix.
    """
    return to_node_name(artifact_name, prefix="p")


def query_node(table_key: str) -> str:
    """Convert a table key to a query loader node identifier.

    Query nodes use the "q" prefix and return DuckDB relations.

    Parameters
    ----------
    table_key
        The table key (e.g., "analytics.function_metrics").

    Returns
    -------
    str
        Hamilton node name with "q__" prefix.

    Examples
    --------
    >>> query_node("analytics.function_metrics")
    'q__analytics__function_metrics'
    """
    return to_node_name(table_key, prefix="q")


def materialize_node(table_key: str) -> str:
    """Convert a table key to a DuckDB materialization node identifier.

    Materialization nodes use the "m" prefix and represent an explicit I/O
    boundary that persists an upstream compute node output to DuckDB.

    Parameters
    ----------
    table_key
        The table key (e.g., "analytics.function_metrics").

    Returns
    -------
    str
        Hamilton node name with "m__" prefix.

    Examples
    --------
    >>> materialize_node("analytics.function_metrics")
    'm__analytics__function_metrics'
    """
    return to_node_name(table_key, prefix="m")


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

    >>> node_to_target("d__some_dataset")
    """
    if not node_name.startswith("t__"):
        return None
    return node_name[3:]


__all__ = [
    "artifact_node",
    "compute_node",
    "dataset_node",
    "materialize_node",
    "node_to_target",
    "pipeline_node_name",
    "query_node",
    "sanitize_pipeline_component",
    "target_node",
    "to_node_name",
]
