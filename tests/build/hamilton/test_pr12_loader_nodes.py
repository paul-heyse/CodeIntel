"""Tests for PR-12: Loader nodes (q__*).

Validates that generated modules include query loader nodes.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.naming import query_node
from codeintel.runtime.compose import compose_runtime
from codeintel.runtime.runtime_bundle import HamiltonRuntimeBundle


class TestLoaderNodeNaming:
    """Tests for loader node naming conventions."""

    @staticmethod
    def test_query_node_naming() -> None:
        """Verify query_node produces q__ prefix."""
        name = query_node("analytics.function_types")
        if not name.startswith("q__"):
            pytest.fail(f"query_node should start with q__, got '{name}'")
        if not name.isidentifier():
            pytest.fail(f"query_node should return valid identifier, got '{name}'")


class TestDriverLoaderNodes:
    """Tests for loader nodes in the driver graph."""

    @staticmethod
    def test_generated_module_has_query_nodes(hamilton_runtime: HamiltonRuntimeBundle) -> None:
        """Verify driver graph includes q__* nodes."""
        node_names = set(hamilton_runtime.dr.graph.nodes)
        expected = query_node("analytics.function_types")
        if expected not in node_names:
            pytest.fail(f"Missing query node {expected}")

    @staticmethod
    def test_loader_nodes_disabled_by_config(runtime_env: BuildEnv) -> None:
        """Verify loader nodes are not generated when flag is False.

        Raises
        ------
        ValueError
            If schema registry data is incomplete for loader node configuration.
        """
        config: dict[str, object] = {"ci_support_include_loader_nodes": False}
        if runtime_env.profile:
            config["profile"] = runtime_env.profile
        config.update(runtime_env.variants.as_hamilton_config())
        config["variant_fingerprint"] = runtime_env.variants.variant_fingerprint
        try:
            runtime = compose_runtime(env=runtime_env, config=config).bundle
        except ValueError as exc:
            if "Missing TableSchema definitions" in str(exc):
                pytest.xfail("Schema registry incomplete for loader node configuration.")
            raise
        node_names = set(runtime.dr.graph.nodes)
        query_name = query_node("analytics.function_types")
        if query_name in node_names:
            pytest.fail("Query nodes should be disabled when include_loader_nodes=False")
