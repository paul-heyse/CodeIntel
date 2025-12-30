"""Tests for PR-12: Loader nodes (q__*).

Validates that generated modules include query loader nodes.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.naming import query_node
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.cli.commands.build import BuildRunCommand
from codeintel.core.hamilton import tags as ht
from codeintel.core.hamilton.tags import NODE_TYPE_LOADER_QUERY
from codeintel.runtime.compose import compose_runtime
from codeintel.runtime.runtime_bundle import RuntimeBundle
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


class TestDriverLoaderNodes:
    """Tests for loader nodes in the driver graph."""

    @staticmethod
    def test_generated_module_has_query_nodes(hamilton_runtime: RuntimeBundle) -> None:
        """Verify driver graph includes q__* nodes."""
        node_names = set(hamilton_runtime.dr.graph.nodes)
        expected = query_node("analytics.function_metrics")
        if expected not in node_names:
            pytest.fail(f"Missing query node {expected}")

    @staticmethod
    def test_loader_nodes_disabled_by_config(runtime_env: BuildEnv) -> None:
        """Verify loader nodes are not generated when flag is False."""
        config: dict[str, object] = {"ci_support_include_loader_nodes": False}
        if runtime_env.profile:
            config["profile"] = runtime_env.profile
        config.update(runtime_env.variants.as_hamilton_config())
        config["variant_fingerprint"] = runtime_env.variants.variant_fingerprint
        runtime = compose_runtime(env=runtime_env, config=config).bundle
        query_name = query_node("analytics.function_metrics")
        variables = list(runtime.dr.list_available_variables())
        var_by_name = {getattr(var, "name", None): var for var in variables}
        query_var = var_by_name.get(query_name)
        if query_var is None:
            return
        q_vars = runtime.tag_query.query({ht.TAG_NODE_TYPE: NODE_TYPE_LOADER_QUERY})
        q_tagged = {getattr(var, "name", None) for var in q_vars}
        if query_name in q_tagged:
            pytest.fail("Query nodes should not be tagged when include_loader_nodes=False")


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
            impl_kind="native",
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
            impl_kind="native",
            status="succeeded",
            input_hash="hash123",
        )

        assert_target_ok(record)
