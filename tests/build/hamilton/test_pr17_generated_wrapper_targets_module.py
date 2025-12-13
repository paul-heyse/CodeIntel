"""Tests for PR-17: Wrapper targets module generation (target nodes only).

This module verifies that when configured for wrapper targets:
1. Module contains target nodes (t__*)
2. Module does NOT contain dataset nodes (d__*)
3. Module does NOT contain loader nodes (q__*, df__*)
4. Module does NOT contain artifact nodes (a__*)
5. Module respects exclude_targets for native targets
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.nodes.node_factory import (
    GenerationOptions,
    build_target_module,
    clear_generated_module_cache,
)


def test_wrapper_module_has_target_nodes() -> None:
    """Verify wrapper module contains target nodes."""
    clear_generated_module_cache()
    options = GenerationOptions(
        include_target_nodes=True,
        include_dataset_nodes=False,
        include_loader_nodes=False,
        include_artifact_nodes=False,
    )
    module = build_target_module(options=options)

    # Should have target nodes (t__*)
    target_nodes = [name for name in dir(module) if name.startswith("t__")]
    if not target_nodes:
        pytest.fail("Wrapper module should contain target nodes (t__*)")

    # Verify TARGET_TO_NODE mapping exists and is non-empty
    if not hasattr(module, "TARGET_TO_NODE"):
        pytest.fail("Wrapper module should define TARGET_TO_NODE")
    if len(module.TARGET_TO_NODE) == 0:
        pytest.fail("TARGET_TO_NODE should map at least one target")


def test_wrapper_module_no_asset_nodes() -> None:
    """Verify wrapper module does NOT contain asset nodes."""
    clear_generated_module_cache()
    options = GenerationOptions(
        include_target_nodes=True,
        include_dataset_nodes=False,
        include_loader_nodes=False,
        include_artifact_nodes=False,
    )
    module = build_target_module(options=options)

    # Should NOT have dataset nodes (d__*)
    dataset_nodes = [name for name in dir(module) if name.startswith("d__")]
    if dataset_nodes:
        pytest.fail(f"Wrapper module should NOT contain dataset nodes, found: {dataset_nodes}")

    # Should NOT have query nodes (q__*)
    query_nodes = [name for name in dir(module) if name.startswith("q__")]
    if query_nodes:
        pytest.fail(f"Wrapper module should NOT contain query nodes, found: {query_nodes}")

    # Should NOT have dataframe nodes (df__*)
    dataframe_nodes = [name for name in dir(module) if name.startswith("df__")]
    if dataframe_nodes:
        pytest.fail(f"Wrapper module should NOT contain dataframe nodes, found: {dataframe_nodes}")

    # Should NOT have artifact nodes (a__*)
    artifact_nodes = [name for name in dir(module) if name.startswith("a__")]
    if artifact_nodes:
        pytest.fail(f"Wrapper module should NOT contain artifact nodes, found: {artifact_nodes}")


def test_wrapper_module_respects_exclude_targets() -> None:
    """Verify wrapper module excludes native targets when specified."""
    clear_generated_module_cache()

    # Exclude risk_factors as if it were a native target
    options = GenerationOptions(
        include_target_nodes=True,
        include_dataset_nodes=False,
        include_loader_nodes=False,
        include_artifact_nodes=False,
        exclude_targets={"risk_factors"},
    )
    module = build_target_module(options=options)

    # Should NOT have t__risk_factors
    if hasattr(module, "t__risk_factors"):
        pytest.fail("Excluded target should not have node in wrapper module")

    # TARGET_TO_NODE should not contain risk_factors
    if "risk_factors" in module.TARGET_TO_NODE:
        pytest.fail("Excluded target should not be in TARGET_TO_NODE mapping")


def test_wrapper_module_respects_include_targets() -> None:
    """Verify wrapper module includes only specified targets when given."""
    clear_generated_module_cache()

    # Include only modules and ast targets
    options = GenerationOptions(
        include_target_nodes=True,
        include_dataset_nodes=False,
        include_loader_nodes=False,
        include_artifact_nodes=False,
        include_targets={"modules", "ast"},
    )
    module = build_target_module(options=options)

    # Should have t__modules and t__ast
    if not hasattr(module, "t__modules"):
        pytest.fail("Should have included target modules")
    if not hasattr(module, "t__ast"):
        pytest.fail("Should have included target ast")

    # Should NOT have other targets like t__risk_factors
    if hasattr(module, "t__risk_factors"):
        pytest.fail("Non-included target should not have node")

    # TARGET_TO_NODE should only contain specified targets
    if "modules" not in module.TARGET_TO_NODE:
        pytest.fail("modules should be in TARGET_TO_NODE")
    if "ast" not in module.TARGET_TO_NODE:
        pytest.fail("ast should be in TARGET_TO_NODE")
    if "risk_factors" in module.TARGET_TO_NODE:
        pytest.fail("risk_factors should not be in TARGET_TO_NODE")


def test_wrapper_module_all_targets_by_default() -> None:
    """Verify wrapper module includes all targets when no filters specified."""
    clear_generated_module_cache()
    options = GenerationOptions(
        include_target_nodes=True,
        include_dataset_nodes=False,
        include_loader_nodes=False,
        include_artifact_nodes=False,
    )
    module = build_target_module(options=options)

    # Should have many target nodes (at least 20 from the registry)
    target_count = len(module.TARGET_TO_NODE)
    min_expected_targets = 20
    if target_count < min_expected_targets:
        pytest.fail(
            "Wrapper module should contain many targets "
            f"(at least {min_expected_targets}), got {target_count}"
        )
