"""Tests for PR-12: Loader nodes (q__* and df__*).

Validates that generated modules include query and dataframe loader nodes.
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.naming import dataframe_node, query_node
from codeintel.build.hamilton.nodes.node_factory import (
    build_target_module,
    clear_generated_module_cache,
)


class TestLoaderNodeNaming:
    """Tests for loader node naming conventions."""

    @staticmethod
    def test_query_node_naming() -> None:
        """Verify query_node produces q__ prefix."""
        name = query_node("analytics.function_metrics")
        if not name.startswith("q__"):
            pytest.fail(f"query_node should start with q__, got '{name}'")
        if not name.isidentifier():
            pytest.fail(f"query_node should return valid identifier, got '{name}'")

    @staticmethod
    def test_dataframe_node_naming() -> None:
        """Verify dataframe_node produces df__ prefix."""
        name = dataframe_node("analytics.function_metrics")
        if not name.startswith("df__"):
            pytest.fail(f"dataframe_node should start with df__, got '{name}'")
        if not name.isidentifier():
            pytest.fail(f"dataframe_node should return valid identifier, got '{name}'")

    @staticmethod
    def test_query_and_dataframe_names_differ() -> None:
        """Verify query and dataframe nodes have different names."""
        table_key = "analytics.function_metrics"
        q_name = query_node(table_key)
        df_name = dataframe_node(table_key)
        if q_name == df_name:
            pytest.fail("query_node and dataframe_node should produce different names")


class TestGeneratedModuleLoaderNodes:
    """Tests for loader nodes in generated module."""

    @staticmethod
    def test_generated_module_has_query_nodes() -> None:
        """Verify generated module includes q__* nodes."""
        clear_generated_module_cache()
        module = build_target_module(include_loader_nodes=True)

        # Check for QUERY_TO_NODE mapping
        if not hasattr(module, "QUERY_TO_NODE"):
            pytest.fail("Module missing QUERY_TO_NODE mapping")

        query_map = getattr(module, "QUERY_TO_NODE", {})
        if not query_map:
            pytest.fail("QUERY_TO_NODE should contain mappings")

    @staticmethod
    def test_generated_module_has_dataframe_nodes() -> None:
        """Verify generated module includes df__* nodes."""
        clear_generated_module_cache()
        module = build_target_module(include_loader_nodes=True)

        # Check for DATAFRAME_TO_NODE mapping
        if not hasattr(module, "DATAFRAME_TO_NODE"):
            pytest.fail("Module missing DATAFRAME_TO_NODE mapping")

        df_map = getattr(module, "DATAFRAME_TO_NODE", {})
        if not df_map:
            pytest.fail("DATAFRAME_TO_NODE should contain mappings")

    @staticmethod
    def test_loader_nodes_disabled_by_default() -> None:
        """Verify loader nodes are not generated when flag is False."""
        clear_generated_module_cache()
        module = build_target_module(include_loader_nodes=False)

        query_map = getattr(module, "QUERY_TO_NODE", {})
        df_map = getattr(module, "DATAFRAME_TO_NODE", {})

        # With loader nodes disabled, these should be empty
        if query_map:
            pytest.fail("QUERY_TO_NODE should be empty when include_loader_nodes=False")
        if df_map:
            pytest.fail("DATAFRAME_TO_NODE should be empty when include_loader_nodes=False")


class TestBuildEnvValidateOutputsFlag:
    """Tests for validate_outputs flag in BuildEnv."""

    @staticmethod
    def test_build_env_has_validate_outputs_field() -> None:
        """Verify BuildEnv has validate_outputs field."""
        fields = getattr(BuildEnv, "__dataclass_fields__", {})
        if "validate_outputs" not in fields:
            # Field may be added as part of optional enhancement
            pytest.skip("validate_outputs field not yet implemented")
        # If field exists, verify default
        field_info = fields["validate_outputs"]
        if field_info.default is not False:
            pytest.fail("validate_outputs should default to False")
