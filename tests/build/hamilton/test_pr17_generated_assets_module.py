"""Tests for PR-17: Support nodes compiled into the driver graph.

This module verifies that the driver graph includes support nodes:
1. Dataset nodes (d__*)
2. Loader nodes (q__*, df__*)
3. Artifact nodes (a__*)
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver


def test_assets_module_has_dataset_nodes() -> None:
    """Verify driver graph contains dataset nodes."""
    runtime = build_driver()
    node_names = set(runtime.dr.graph.nodes)

    # Should have at least one dataset node
    dataset_nodes = [name for name in node_names if name.startswith("d__")]
    if not dataset_nodes:
        pytest.fail("Driver graph should contain dataset nodes (d__*)")


def test_assets_module_has_loader_nodes() -> None:
    """Verify driver graph contains query and dataframe loader nodes."""
    runtime = build_driver()
    node_names = set(runtime.dr.graph.nodes)

    # Should have query nodes (q__*)
    query_nodes = [name for name in node_names if name.startswith("q__")]
    if not query_nodes:
        pytest.fail("Driver graph should contain query nodes (q__*)")

    # Should have dataframe nodes (df__*)
    dataframe_nodes = [name for name in node_names if name.startswith("df__")]
    if not dataframe_nodes:
        pytest.fail("Driver graph should contain dataframe nodes (df__*)")


def test_assets_module_has_artifact_nodes() -> None:
    """Verify driver graph contains artifact nodes for SCIP/exports."""
    runtime = build_driver()
    node_names = set(runtime.dr.graph.nodes)

    # Should have artifact nodes (a__*)
    artifact_nodes = [name for name in node_names if name.startswith("a__")]
    if not artifact_nodes:
        pytest.fail("Driver graph should contain artifact nodes (a__*)")


def test_assets_module_all_node_types_independent() -> None:
    """Verify support nodes are compiled alongside targets."""
    runtime = build_driver()
    node_names = set(runtime.dr.graph.nodes)

    # Should have all asset types
    has_datasets = any(name.startswith("d__") for name in node_names)
    has_queries = any(name.startswith("q__") for name in node_names)
    has_dataframes = any(name.startswith("df__") for name in node_names)
    has_artifacts = any(name.startswith("a__") for name in node_names)

    if not has_datasets:
        pytest.fail("Should have dataset nodes")
    if not has_queries:
        pytest.fail("Should have query nodes")
    if not has_dataframes:
        pytest.fail("Should have dataframe nodes")
    if not has_artifacts:
        pytest.fail("Should have artifact nodes")
