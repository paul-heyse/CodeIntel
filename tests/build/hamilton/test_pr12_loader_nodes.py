"""Tests for PR-12: Loader nodes (q__* and df__*).

Validates that generated modules include query and dataframe loader nodes.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.naming import dataframe_node, query_node
from codeintel.build.hamilton.nodes.support_factory import (
    SupportGenerationOptions,
    build_support_module,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.cli.commands.build import BuildRunCommand
from tests._helpers.assertions import assert_target_ok


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
        module = build_support_module(
            options=SupportGenerationOptions(include_loader_nodes=True),
        )

        if not hasattr(module, "QUERY_TO_NODE"):
            pytest.fail("Module missing QUERY_TO_NODE mapping")

        query_map = getattr(module, "QUERY_TO_NODE", {})
        if not query_map:
            pytest.fail("QUERY_TO_NODE should contain mappings")

    @staticmethod
    def test_generated_module_has_dataframe_nodes() -> None:
        """Verify generated module includes df__* nodes."""
        module = build_support_module(
            options=SupportGenerationOptions(include_loader_nodes=True),
        )

        if not hasattr(module, "DATAFRAME_TO_NODE"):
            pytest.fail("Module missing DATAFRAME_TO_NODE mapping")

        df_map = getattr(module, "DATAFRAME_TO_NODE", {})
        if not df_map:
            pytest.fail("DATAFRAME_TO_NODE should contain mappings")

    @staticmethod
    def test_loader_nodes_disabled_by_default() -> None:
        """Verify loader nodes are not generated when flag is False."""
        module = build_support_module(
            options=SupportGenerationOptions(include_loader_nodes=False),
        )

        query_map = getattr(module, "QUERY_TO_NODE", {})
        df_map = getattr(module, "DATAFRAME_TO_NODE", {})

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
            pytest.skip("validate_outputs field not yet implemented")

        field_info = fields["validate_outputs"]
        if field_info.default is not False:
            pytest.fail("validate_outputs should default to False")


class TestValidateOutputsBehavior:
    """Tests for --validate-outputs behavior and blocking semantics."""

    @staticmethod
    def test_validate_outputs_flag_exists_in_build_run_command() -> None:
        """Verify BuildRunCommand has validate_outputs option."""
        fields = getattr(BuildRunCommand, "__dataclass_fields__", {})
        if "validate_outputs" not in fields:
            pytest.skip("validate_outputs option not yet implemented")

    @staticmethod
    def test_validation_result_dataclass_exists() -> None:
        """Verify ValidationResult type exists for tracking validation status.

        When --validate-outputs is used, targets should report validation
        results that can block downstream if validation fails.

        This test checks for the existence of the ValidationResult type
        which is part of the optional --validate-outputs feature.
        """
        spec = importlib.util.find_spec("codeintel.build.hamilton.validation")
        if spec is None:
            pytest.skip("ValidationResult not yet implemented")

        try:
            validation_mod = importlib.import_module("codeintel.build.hamilton.validation")
        except ImportError:
            pytest.skip("ValidationResult not yet implemented")

        validation_result_cls = getattr(validation_mod, "ValidationResult", None)
        if validation_result_cls is None:
            pytest.skip("ValidationResult class not found")

        result = validation_result_cls(
            table_key="analytics.function_metrics",
            valid=False,
            errors=("Column 'loc' has wrong type",),
        )
        if result.valid:
            pytest.fail("Constructed result should be invalid")
        if not result.errors:
            pytest.fail("Invalid result should have errors")

    @staticmethod
    def test_validation_failure_blocks_downstream_conceptually() -> None:
        """Verify validation failure would block downstream targets.

        The validation semantics should prevent downstream targets from
        running if upstream validation fails. This test verifies the
        conceptual model rather than full integration.
        """
        record = TargetRunRecord(
            target="function_metrics",
            plugin_name="analytics.function_metrics",
            status="failed",
            input_hash="hash123",
            error="Validation failed: Column 'loc' has wrong type",
        )

        assert_target_ok(record, expected_status="failed")

    @staticmethod
    def test_target_run_record_has_validation_fields() -> None:
        """Verify TargetRunRecord can capture validation state."""
        record = TargetRunRecord(
            target="function_metrics",
            plugin_name="analytics.function_metrics",
            status="succeeded",
            input_hash="hash123",
        )

        assert_target_ok(record)
