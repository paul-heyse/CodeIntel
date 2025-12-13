"""Tests for PR-17: Assets module generation (no target nodes).

This module verifies that when include_target_nodes=False:
1. Assets module contains dataset nodes (d__*)
2. Assets module contains loader nodes (q__*, df__*)
3. Assets module contains artifact nodes (a__*)
4. Assets module does NOT contain target nodes (t__*)
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.nodes.node_factory import (
    GenerationOptions,
    build_target_module,
    clear_generated_module_cache,
)


def test_assets_module_has_dataset_nodes() -> None:
    """Verify assets module contains dataset nodes."""
    clear_generated_module_cache()
    options = GenerationOptions(
        include_target_nodes=False,
        include_dataset_nodes=True,
        include_loader_nodes=True,
        include_artifact_nodes=True,
    )
    module = build_target_module(options=options)

    # Should have at least one dataset node
    dataset_nodes = [name for name in dir(module) if name.startswith("d__")]
    if not dataset_nodes:
        pytest.fail("Assets module should contain dataset nodes (d__*)")

    # Verify DATASET_TO_NODE mapping exists
    if not hasattr(module, "DATASET_TO_NODE"):
        pytest.fail("Assets module should define DATASET_TO_NODE")
    if len(module.DATASET_TO_NODE) == 0:
        pytest.fail("DATASET_TO_NODE should map at least one dataset")


def test_assets_module_has_loader_nodes() -> None:
    """Verify assets module contains query and dataframe loader nodes."""
    clear_generated_module_cache()
    options = GenerationOptions(
        include_target_nodes=False,
        include_dataset_nodes=True,
        include_loader_nodes=True,
        include_artifact_nodes=True,
    )
    module = build_target_module(options=options)

    # Should have query nodes (q__*)
    query_nodes = [name for name in dir(module) if name.startswith("q__")]
    if not query_nodes:
        pytest.fail("Assets module should contain query nodes (q__*)")

    # Should have dataframe nodes (df__*)
    dataframe_nodes = [name for name in dir(module) if name.startswith("df__")]
    if not dataframe_nodes:
        pytest.fail("Assets module should contain dataframe nodes (df__*)")

    # Verify mappings exist
    if not hasattr(module, "QUERY_TO_NODE"):
        pytest.fail("Assets module should define QUERY_TO_NODE")
    if not hasattr(module, "DATAFRAME_TO_NODE"):
        pytest.fail("Assets module should define DATAFRAME_TO_NODE")
    if len(module.QUERY_TO_NODE) == 0:
        pytest.fail("QUERY_TO_NODE should map at least one query")
    if len(module.DATAFRAME_TO_NODE) == 0:
        pytest.fail("DATAFRAME_TO_NODE should map at least one dataframe")


def test_assets_module_has_artifact_nodes() -> None:
    """Verify assets module contains artifact nodes for SCIP/exports."""
    clear_generated_module_cache()
    options = GenerationOptions(
        include_target_nodes=False,
        include_dataset_nodes=True,
        include_loader_nodes=True,
        include_artifact_nodes=True,
    )
    module = build_target_module(options=options)

    # Should have artifact nodes (a__*)
    artifact_nodes = [name for name in dir(module) if name.startswith("a__")]
    if not artifact_nodes:
        pytest.fail("Assets module should contain artifact nodes (a__*)")

    # Verify ARTIFACT_TO_NODE mapping exists
    if not hasattr(module, "ARTIFACT_TO_NODE"):
        pytest.fail("Assets module should define ARTIFACT_TO_NODE")
    if len(module.ARTIFACT_TO_NODE) == 0:
        pytest.fail("ARTIFACT_TO_NODE should map at least one artifact")


def test_assets_module_no_target_nodes() -> None:
    """Verify assets module does NOT contain target nodes."""
    clear_generated_module_cache()
    options = GenerationOptions(
        include_target_nodes=False,
        include_dataset_nodes=True,
        include_loader_nodes=True,
        include_artifact_nodes=True,
    )
    module = build_target_module(options=options)

    # Should NOT have target nodes (t__*)
    target_nodes = [name for name in dir(module) if name.startswith("t__")]
    if target_nodes:
        pytest.fail(f"Assets module should NOT contain target nodes, found: {target_nodes}")

    # TARGET_TO_NODE should be empty
    if not hasattr(module, "TARGET_TO_NODE"):
        pytest.fail("Assets module should define TARGET_TO_NODE")
    if len(module.TARGET_TO_NODE) != 0:
        keys = list(module.TARGET_TO_NODE.keys())
        pytest.fail(f"TARGET_TO_NODE should be empty in assets module, found: {keys}")


def test_assets_module_all_node_types_independent() -> None:
    """Verify assets module generation works with all node types enabled."""
    clear_generated_module_cache()
    options = GenerationOptions(
        include_target_nodes=False,
        include_dataset_nodes=True,
        include_loader_nodes=True,
        include_artifact_nodes=True,
    )
    module = build_target_module(options=options)

    # Should have all asset types
    has_datasets = any(name.startswith("d__") for name in dir(module))
    has_queries = any(name.startswith("q__") for name in dir(module))
    has_dataframes = any(name.startswith("df__") for name in dir(module))
    has_artifacts = any(name.startswith("a__") for name in dir(module))

    if not has_datasets:
        pytest.fail("Should have dataset nodes")
    if not has_queries:
        pytest.fail("Should have query nodes")
    if not has_dataframes:
        pytest.fail("Should have dataframe nodes")
    if not has_artifacts:
        pytest.fail("Should have artifact nodes")
