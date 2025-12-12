"""Dataset extraction nodes for Hamilton lineage.

This module generates nodes that expose individual datasets from target
execution results, enabling fine-grained lineage tracking in the DAG.

Design Principles
-----------------
1. Each dataset node extracts a DatasetRef from a parent target's TargetRunRecord.
2. Dataset nodes use the "d__" prefix for clear visual distinction.
3. These nodes establish lineage edges from targets to their output datasets.
"""

from __future__ import annotations

from hamilton.function_modifiers import tag

from codeintel.build.hamilton.io.dataset_ref import DatasetRef, refs_from_target_result
from codeintel.build.hamilton.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.naming import dataset_node

__all__ = [
    "DATASET_NODES",
    "d__analytics__function_metrics",
    "d__analytics__risk_factors",
    "d__graph__call_graph_edges",
    "d__graph__call_graph_nodes",
    "extract_datasets_from_record",
]


def extract_datasets_from_record(
    record: TargetRunRecord,
    table_keys: tuple[str, ...],
) -> dict[str, DatasetRef]:
    """Extract DatasetRef instances from a TargetRunRecord.

    Parameters
    ----------
    record
        Execution record from a target node.
    table_keys
        Table keys to extract as DatasetRef instances.

    Returns
    -------
    dict[str, DatasetRef]
        Mapping of dataset node names to DatasetRef instances.

    Examples
    --------
    >>> refs = extract_datasets_from_record(
    ...     record, ("analytics.function_metrics",)
    ... )
    >>> refs["d__analytics__function_metrics"]
    DatasetRef(table_key='analytics.function_metrics', ...)
    """
    refs = refs_from_target_result(
        target_name=record.target,
        table_keys=table_keys,
        row_counts=dict(record.row_counts),
    )
    # Map to dataset node names
    return {dataset_node(key): ref for key, ref in refs.items()}


# =============================================================================
# Dataset Extraction Nodes
# =============================================================================
# These nodes expose individual datasets from Phase 0 targets.
# Each returns a single DatasetRef for Hamilton lineage tracking.


@tag(domain="graphs", dataset="call_graph_edges")
def d__graph__call_graph_edges(
    t__call_graph: TargetRunRecord,
) -> DatasetRef:
    """Extract call_graph_edges dataset from call_graph target.

    Parameters
    ----------
    t__call_graph
        Execution record from the call_graph target.

    Returns
    -------
    DatasetRef
        Reference to the graph.call_graph_edges table.
    """
    table_key = "graph.call_graph_edges"
    refs = refs_from_target_result(
        target_name=t__call_graph.target,
        table_keys=(table_key,),
        row_counts=dict(t__call_graph.row_counts),
    )
    return refs[table_key]


@tag(domain="graphs", dataset="call_graph_nodes")
def d__graph__call_graph_nodes(
    t__call_graph: TargetRunRecord,
) -> DatasetRef:
    """Extract call_graph_nodes dataset from call_graph target.

    Parameters
    ----------
    t__call_graph
        Execution record from the call_graph target.

    Returns
    -------
    DatasetRef
        Reference to the graph.call_graph_nodes table.
    """
    table_key = "graph.call_graph_nodes"
    refs = refs_from_target_result(
        target_name=t__call_graph.target,
        table_keys=(table_key,),
        row_counts=dict(t__call_graph.row_counts),
    )
    return refs[table_key]


@tag(domain="analytics", dataset="function_metrics")
def d__analytics__function_metrics(
    t__function_metrics: TargetRunRecord,
) -> DatasetRef:
    """Extract function_metrics dataset from function_metrics target.

    Parameters
    ----------
    t__function_metrics
        Execution record from the function_metrics target.

    Returns
    -------
    DatasetRef
        Reference to the analytics.function_metrics table.
    """
    table_key = "analytics.function_metrics"
    refs = refs_from_target_result(
        target_name=t__function_metrics.target,
        table_keys=(table_key,),
        row_counts=dict(t__function_metrics.row_counts),
    )
    return refs[table_key]


@tag(domain="analytics", dataset="risk_factors")
def d__analytics__risk_factors(
    t__risk_factors: TargetRunRecord,
) -> DatasetRef:
    """Extract risk_factors dataset from risk_factors target.

    Parameters
    ----------
    t__risk_factors
        Execution record from the risk_factors target.

    Returns
    -------
    DatasetRef
        Reference to the analytics.risk_factors table.
    """
    table_key = "analytics.risk_factors"
    refs = refs_from_target_result(
        target_name=t__risk_factors.target,
        table_keys=(table_key,),
        row_counts=dict(t__risk_factors.row_counts),
    )
    return refs[table_key]


# Registry of dataset nodes for discovery
DATASET_NODES = (
    d__graph__call_graph_edges,
    d__graph__call_graph_nodes,
    d__analytics__function_metrics,
    d__analytics__risk_factors,
)
