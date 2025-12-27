"""Tests for the contract-driven dataset/dataflow graph."""

from __future__ import annotations

import pytest

from codeintel.config.datasets.composites import get_composite_schemas
from codeintel.storage.contracts.dataflow import alias_docs_views, build_contract_dataflow_graph
from codeintel.storage.contracts.provider import iter_contracts, iter_contracts_by_table_key
from codeintel.storage.metadata import bootstrap_metadata_datasets
from codeintel.storage.metadata.meta_catalog import meta_table_ref
from tests._helpers.gateway import GatewayFactory


def _require(*, condition: bool, message: str) -> None:
    if not condition:
        pytest.fail(message)


def test_contract_dataflow_includes_all_datasets() -> None:
    """Every DatasetContract must appear as a DataflowNode."""
    nodes, _ = build_contract_dataflow_graph()
    node_ids = {node.id for node in nodes}

    for contract in iter_contracts():
        _require(
            condition=contract.table_key in node_ids,
            message=f"DatasetContract {contract.name} missing node for {contract.table_key}",
        )


def test_composite_edges_align_with_composite_schemas() -> None:
    """COMPOSITE_SCHEMAS must be fully represented in the dataflow graph."""
    _, edges = build_contract_dataflow_graph()
    builds_edges = {(edge.src, edge.dst) for edge in edges if edge.edge_type == "builds"}

    contracts_by_key = dict(iter_contracts_by_table_key())
    for table_key, composite in get_composite_schemas().items():
        target = contracts_by_key.get(table_key)
        if target is None:
            pytest.fail(f"CompositeSchema target {table_key} missing DatasetContract")
        dst_id = table_key
        for src_table_key in composite.composed_of:
            _require(
                condition=(src_table_key, dst_id) in builds_edges,
                message=f"Missing builds edge {src_table_key} -> {dst_id} in composite graph",
            )


def test_metadata_dataflow_tables_populated() -> None:
    """bootstrap_metadata_datasets must populate metadata.dataset_dataflow_* tables."""
    gateway = GatewayFactory().without_validation().open()
    try:
        bootstrap_metadata_datasets(gateway.con, include_views=True)
        nodes_ref = meta_table_ref("metadata.dataset_dataflow_nodes")
        edges_ref = meta_table_ref("metadata.dataset_dataflow_edges")
        node_row = gateway.con.execute(
            f"SELECT COUNT(*) FROM {nodes_ref}"
        ).fetchone()
        edge_row = gateway.con.execute(
            f"SELECT COUNT(*) FROM {edges_ref}"
        ).fetchone()
        if node_row is None or edge_row is None:
            pytest.fail("Failed to count dataflow tables")
        node_count = int(node_row[0])
        edge_count = int(edge_row[0])

        _require(condition=node_count > 0, message="Expected at least one dataflow node")
        _require(condition=edge_count > 0, message="Expected at least one dataflow edge")
    finally:
        gateway.close()


def test_alias_docs_views_have_nodes() -> None:
    """Alias docs views should produce DataflowNode entries."""
    nodes, _ = build_contract_dataflow_graph()
    node_ids = {node.id for node in nodes}
    mapping = alias_docs_views()
    missing = [view for view in mapping if view not in node_ids]
    _require(
        condition=not missing,
        message=f"Missing alias docs view nodes: {', '.join(missing)}",
    )
