"""Tests for PR-17: Support nodes compiled into the driver graph.

This module verifies that the driver graph includes support nodes:
1. Dataset nodes (d__*)
2. Loader nodes (q__*)
3. Artifact nodes (a__*)
"""

from __future__ import annotations

import pytest

from codeintel.runtime.runtime_bundle import RuntimeBundle


def test_assets_module_has_dataset_nodes(hamilton_runtime: RuntimeBundle) -> None:
    """Verify driver graph contains dataset nodes."""
    node_names = set(hamilton_runtime.dr.graph.nodes)

    # Should have at least one dataset node
    dataset_nodes = [name for name in node_names if name.startswith("d__")]
    if not dataset_nodes:
        pytest.fail("Driver graph should contain dataset nodes (d__*)")


def test_assets_module_has_loader_nodes(hamilton_runtime: RuntimeBundle) -> None:
    """Verify driver graph contains query loader nodes."""
    node_names = set(hamilton_runtime.dr.graph.nodes)

    # Should have query nodes (q__*)
    query_nodes = [name for name in node_names if name.startswith("q__")]
    if not query_nodes:
        pytest.fail("Driver graph should contain query nodes (q__*)")


def test_assets_module_has_artifact_nodes(hamilton_runtime: RuntimeBundle) -> None:
    """Verify driver graph contains artifact nodes for SCIP/exports."""
    node_names = set(hamilton_runtime.dr.graph.nodes)

    # Should have artifact nodes (a__*)
    artifact_nodes = [name for name in node_names if name.startswith("a__")]
    if not artifact_nodes:
        pytest.fail("Driver graph should contain artifact nodes (a__*)")


def test_assets_module_all_node_types_independent(hamilton_runtime: RuntimeBundle) -> None:
    """Verify support nodes are compiled alongside targets."""
    node_names = set(hamilton_runtime.dr.graph.nodes)

    # Should have all asset types
    has_datasets = any(name.startswith("d__") for name in node_names)
    has_queries = any(name.startswith("q__") for name in node_names)
    has_artifacts = any(name.startswith("a__") for name in node_names)

    if not has_datasets:
        pytest.fail("Should have dataset nodes")
    if not has_queries:
        pytest.fail("Should have query nodes")
    if not has_artifacts:
        pytest.fail("Should have artifact nodes")
