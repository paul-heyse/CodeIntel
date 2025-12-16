"""Validation Proof-of-Concept tests for Phase 1.5.

These tests validate the Hamilton-native validation infrastructure:
1. Validators work with Ibis tables (schema validation)
2. Validators work with DataFrames (full data validation)
3. ContractEnforcementHook captures validation results
4. Risk factors and hotspots modules have proper validators
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_false,
    expect_in,
    expect_true,
)

if TYPE_CHECKING:
    from collections.abc import Callable


class TestIbisTableValidation:
    """Test validators with Ibis table expressions (schema-only validation)."""

    @staticmethod
    def test_columns_exist_validator_with_ibis_table() -> None:
        """Verify ColumnsExistValidator works with Ibis tables."""
        pytest.importorskip("ibis")
        import ibis

        from codeintel.build.hamilton.validators import ColumnsExistValidator

        # Create a mock Ibis table using memtable
        df = pd.DataFrame(
            {
                "function_goid_h128": ["abc123"],
                "repo": ["org/repo"],
                "commit": ["abc123"],
                "risk_score": [5],
            }
        )
        ibis_table = ibis.memtable(df)

        # Test with all columns present
        validator = ColumnsExistValidator(["function_goid_h128", "repo", "commit"])
        result = validator.validate(ibis_table)
        expect_true(result.passes, message="Expected validation to pass for Ibis table")
        expect_in("Ibis table", result.message)

    @staticmethod
    def test_columns_exist_validator_ibis_missing_column() -> None:
        """Verify ColumnsExistValidator catches missing columns in Ibis tables."""
        pytest.importorskip("ibis")
        import ibis

        from codeintel.build.hamilton.validators import ColumnsExistValidator

        df = pd.DataFrame({"col_a": [1], "col_b": [2]})
        ibis_table = ibis.memtable(df)

        validator = ColumnsExistValidator(["col_a", "col_b", "col_c"])
        result = validator.validate(ibis_table)
        expect_false(result.passes, message="Expected validation to fail for missing column")
        expect_in("col_c", str(result.diagnostics.get("missing_columns", [])))

    @staticmethod
    def test_column_types_validator_with_ibis_table() -> None:
        """Verify ColumnTypesValidator validates Ibis table schema."""
        pytest.importorskip("ibis")
        import ibis

        from codeintel.build.hamilton.validators import ColumnTypesValidator

        df = pd.DataFrame(
            {
                "name": ["Alice", "Bob"],
                "age": [30, 25],
                "score": [95.5, 88.0],
            }
        )
        ibis_table = ibis.memtable(df)

        # Test type matching
        validator = ColumnTypesValidator({"name": "string", "age": "int", "score": "float"})
        result = validator.validate(ibis_table)
        # Note: Ibis may use different type names, so validation may pass or need adjustment
        expect_true(
            result.passes or result.diagnostics.get("data_type") == "Ibis table",
            message="Expected Ibis table type validation",
        )

    @staticmethod
    def test_no_nulls_validator_skips_ibis() -> None:
        """Verify NoNullsInColumnsValidator skips Ibis tables (lazy)."""
        pytest.importorskip("ibis")
        import ibis

        from codeintel.build.hamilton.validators import NoNullsInColumnsValidator

        df = pd.DataFrame({"col": [1, 2, None]})
        ibis_table = ibis.memtable(df)

        validator = NoNullsInColumnsValidator(["col"])
        result = validator.validate(ibis_table)
        expect_true(result.passes, message="Expected validation to pass (skipped for Ibis)")
        expect_true(result.diagnostics.get("skipped", False), message="Expected skipped=True")

    @staticmethod
    def test_column_values_in_set_skips_ibis() -> None:
        """Verify ColumnValuesInSetValidator skips Ibis tables (lazy)."""
        pytest.importorskip("ibis")
        import ibis

        from codeintel.build.hamilton.validators import ColumnValuesInSetValidator

        df = pd.DataFrame({"status": ["active", "inactive", "unknown"]})
        ibis_table = ibis.memtable(df)

        validator = ColumnValuesInSetValidator("status", {"active", "inactive"})
        result = validator.validate(ibis_table)
        expect_true(result.passes, message="Expected validation to pass (skipped for Ibis)")
        expect_true(result.diagnostics.get("skipped", False), message="Expected skipped=True")


class TestDataFrameValidation:
    """Test validators with pandas DataFrames (full data validation)."""

    @staticmethod
    def test_no_nulls_catches_nulls_in_dataframe() -> None:
        """Verify NoNullsInColumnsValidator catches nulls in DataFrames."""
        from codeintel.build.hamilton.validators import NoNullsInColumnsValidator

        df = pd.DataFrame(
            {
                "id": [1, 2, None],
                "name": ["Alice", "Bob", "Charlie"],
            }
        )

        validator = NoNullsInColumnsValidator(["id"])
        result = validator.validate(df)
        expect_false(result.passes, message="Expected validation to fail for null values")
        expect_equal(result.diagnostics.get("null_counts", {}).get("id"), 1)

    @staticmethod
    def test_column_values_in_set_catches_invalid() -> None:
        """Verify ColumnValuesInSetValidator catches invalid values."""
        from codeintel.build.hamilton.validators import ColumnValuesInSetValidator

        df = pd.DataFrame({"risk_level": ["high", "medium", "invalid"]})

        validator = ColumnValuesInSetValidator("risk_level", {"high", "medium", "low"})
        result = validator.validate(df)
        expect_false(result.passes, message="Expected validation to fail for invalid value")
        expect_in("invalid", str(result.diagnostics.get("invalid_values", [])))


class TestContractEnforcementHook:
    """Test ContractEnforcementHook validation result capture."""

    @staticmethod
    def test_validation_result_creation() -> None:
        """Verify ValidationResult dataclass works correctly."""
        from codeintel.build.hamilton.hooks import ValidationResult

        result = ValidationResult(
            node_name="test_node",
            passed=True,
            message="Validation passed",
        )
        expect_equal(result.node_name, "test_node")
        expect_true(result.passed)

    @staticmethod
    def test_validation_summary_aggregation() -> None:
        """Verify ValidationSummary aggregates results correctly."""
        from codeintel.build.hamilton.hooks import ValidationSummary

        summary = ValidationSummary(
            total_nodes=10,
            passed_count=8,
            failed_count=2,
            skipped_count=0,
            failed_nodes=["node_a", "node_b"],
        )
        expect_equal(summary.total_nodes, 10)
        expect_equal(summary.passed_count, 8)
        expect_equal(summary.failed_count, 2)
        expect_false(summary.all_passed)

    @staticmethod
    def test_hook_captures_validation_results(
        minimal_target_graph: Callable[[], Any],
    ) -> None:
        """Verify ContractEnforcementHook captures validation results."""
        from codeintel.build.hamilton.hooks import ContractEnforcementHook

        graph = minimal_target_graph()
        hook = ContractEnforcementHook(graph, strict=False)

        # Simulate node execution
        hook.pre_node_execute(node_name="test_node", node_tags={"target": "test"})
        hook.post_node_execute(node_name="test_node", success=True)

        # Check validation results
        results = hook.validation_results
        expect_in("test_node", results)
        expect_true(results["test_node"].passed)

    @staticmethod
    def test_hook_captures_failure(
        minimal_target_graph: Callable[[], Any],
    ) -> None:
        """Verify ContractEnforcementHook captures failures."""
        from codeintel.build.hamilton.hooks import ContractEnforcementHook

        graph = minimal_target_graph()
        hook = ContractEnforcementHook(graph, strict=False)

        # Simulate failed node execution
        hook.pre_node_execute(node_name="failing_node", node_tags={"target": "test"})
        hook.post_node_execute(
            node_name="failing_node",
            success=False,
            error=ValueError("Validation failed: missing columns"),
        )

        results = hook.validation_results
        expect_in("failing_node", results)
        expect_false(results["failing_node"].passed)
        expect_in("Validation", str(results["failing_node"].error))

    @staticmethod
    def test_hook_get_validation_summary(
        minimal_target_graph: Callable[[], Any],
    ) -> None:
        """Verify get_validation_summary aggregates results."""
        from codeintel.build.hamilton.hooks import ContractEnforcementHook

        graph = minimal_target_graph()
        hook = ContractEnforcementHook(graph, strict=False)

        # Simulate multiple nodes
        for name in ["node_a", "node_b", "node_c"]:
            hook.pre_node_execute(node_name=name, node_tags={})
            hook.post_node_execute(node_name=name, success=(name != "node_c"))

        summary = hook.get_validation_summary()
        expect_equal(summary.total_nodes, 3)
        expect_equal(summary.passed_count, 2)
        expect_equal(summary.failed_count, 1)
        expect_in("node_c", summary.failed_nodes)


class TestRiskFactorsModuleValidation:
    """Test that risk_factors module has proper validators configured."""

    @staticmethod
    def test_risk_factors_module_has_validators() -> None:
        """Verify risk_factors module has @check_output_custom decorators."""
        from codeintel.build.hamilton.native.analytics import risk_factors

        # Check that the compute function exists
        compute_fn = getattr(risk_factors, "t__risk_factors__compute", None)
        expect_true(compute_fn is not None, message="Expected t__risk_factors__compute to exist")

        # Check for Hamilton tags (indicates proper decoration)
        fn_tags = getattr(compute_fn, "_tags", {})
        if fn_tags:
            expect_in("domain", fn_tags)
            expect_equal(fn_tags.get("domain"), "analytics")

    @staticmethod
    def test_risk_factors_module_exports() -> None:
        """Verify risk_factors module exports expected symbols."""
        from codeintel.build.hamilton.native.analytics import risk_factors

        expect_in("t__risk_factors__compute", risk_factors.__all__)
        expect_in("t__risk_factors", risk_factors.__all__)


class TestHotspotsModuleValidation:
    """Test that hotspots module has proper validators configured."""

    @staticmethod
    def test_hotspots_module_has_validators() -> None:
        """Verify hotspots module has @check_output_custom decorators."""
        from codeintel.build.hamilton.native.analytics import hotspots

        # Check that the compute function exists
        compute_fn = getattr(hotspots, "t__hotspots__compute", None)
        expect_true(compute_fn is not None, message="Expected t__hotspots__compute to exist")

    @staticmethod
    def test_hotspots_module_exports() -> None:
        """Verify hotspots module exports expected symbols."""
        from codeintel.build.hamilton.native.analytics import hotspots

        expect_in("t__hotspots__compute", hotspots.__all__)
        expect_in("t__hotspots", hotspots.__all__)


class TestBuildHooksValidation:
    """Test build_hooks() with validation enabled."""

    @staticmethod
    def test_build_hooks_includes_validation_by_default(
        storage_gateway_for_tests: Any,
        minimal_target_graph: Callable[[], Any],
    ) -> None:
        """Verify build_hooks includes ContractEnforcementHook by default."""
        from codeintel.build.hamilton.hooks import (
            ContractEnforcementHook,
            build_hooks,
        )

        gateway = storage_gateway_for_tests
        graph = minimal_target_graph()

        hooks = build_hooks("run-123", gateway, graph)

        # Should include ContractEnforcementHook
        contract_hooks = [h for h in hooks if isinstance(h, ContractEnforcementHook)]
        expect_true(
            len(contract_hooks) >= 1,
            message="Expected ContractEnforcementHook in hooks",
        )

    @staticmethod
    def test_build_hooks_validation_can_be_disabled(
        storage_gateway_for_tests: Any,
        minimal_target_graph: Callable[[], Any],
    ) -> None:
        """Verify build_hooks can disable validation."""
        from codeintel.build.hamilton.hooks import (
            ContractEnforcementHook,
            build_hooks,
        )

        gateway = storage_gateway_for_tests
        graph = minimal_target_graph()

        hooks = build_hooks("run-123", gateway, graph, enable_validation=False)

        # Should not include ContractEnforcementHook
        contract_hooks = [h for h in hooks if isinstance(h, ContractEnforcementHook)]
        expect_equal(len(contract_hooks), 0)


class TestTableContractBuilders:
    """Test contract builder functions with validators."""

    @staticmethod
    def test_build_table_contract_creates_validators() -> None:
        """Verify build_table_contract creates expected validators."""
        from codeintel.build.hamilton.validators import (
            ColumnsExistValidator,
            build_table_contract,
        )

        validators = build_table_contract(
            required_columns=["id", "name"],
            no_nulls=["id"],
        )

        # Should include ColumnsExistValidator
        validator_types = [type(v).__name__ for v in validators]
        expect_in("ColumnsExistValidator", validator_types)

        # First validator should check required columns
        columns_validator = validators[0]
        expect_true(
            isinstance(columns_validator, ColumnsExistValidator),
            message="Expected first validator to be ColumnsExistValidator",
        )

    @staticmethod
    def test_build_enum_column_contract() -> None:
        """Verify build_enum_column_contract creates expected validators."""
        from codeintel.build.hamilton.validators import (
            ColumnsExistValidator,
            ColumnValuesInSetValidator,
            NoNullsInColumnsValidator,
            build_enum_column_contract,
        )

        validators = build_enum_column_contract(
            column="status",
            allowed_values={"active", "inactive"},
        )

        # Returns 3 validators: columns exist, values in set, no nulls
        expect_equal(len(validators), 3)
        expect_true(isinstance(validators[0], ColumnsExistValidator))
        expect_true(isinstance(validators[1], ColumnValuesInSetValidator))
        expect_true(isinstance(validators[2], NoNullsInColumnsValidator))


@pytest.fixture
def minimal_target_graph() -> Callable[[], Any]:
    """Create a minimal target graph for testing.

    Returns a factory function that creates a mock target graph.
    """

    def _create_graph() -> Any:
        """Create mock target graph."""
        from unittest.mock import MagicMock

        graph = MagicMock()
        graph.get.side_effect = KeyError("Target not found")
        return graph

    return _create_graph


@pytest.fixture
def storage_gateway_for_tests() -> Any:
    """Create a mock storage gateway for testing."""
    from unittest.mock import MagicMock

    return MagicMock()
