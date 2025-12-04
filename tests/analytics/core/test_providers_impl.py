"""Tests for codeintel.analytics.core.providers_impl module.

Testing Charter Compliance:
- Uses real DuckDB via TestContext fixtures
- No monkeypatching or test-only code paths
- Tests production code paths for analytics support providers
"""

from __future__ import annotations

from codeintel.analytics.core.contracts import OutputContractSpec
from codeintel.analytics.core.providers_impl import DefaultAnalyticsSupportProvider
from tests._helpers import CORE_PACK, METRICS_PACK, TestContext


class TestDefaultAnalyticsSupportProvider:
    """Tests for DefaultAnalyticsSupportProvider class."""

    @staticmethod
    def test_initialization() -> None:
        """Verify provider initializes successfully."""
        provider = DefaultAnalyticsSupportProvider()
        # Provider should be usable - verify via public interface
        assert provider is not None
        assert callable(provider.compute_row_counts)
        assert callable(provider.validate_contracts)


class TestComputeRowCounts:
    """Tests for compute_row_counts method."""

    @staticmethod
    def test_returns_empty_for_empty_tables(test_ctx: TestContext) -> None:
        """Verify empty tables tuple returns empty dict."""
        provider = DefaultAnalyticsSupportProvider()
        snapshot = test_ctx.to_snapshot_ref()

        result = provider.compute_row_counts(test_ctx.gateway, snapshot, ())

        assert result == {}

    @staticmethod
    def test_counts_seeded_tables(metrics_ctx: TestContext) -> None:
        """Verify row counts for seeded tables."""
        provider = DefaultAnalyticsSupportProvider()
        snapshot = metrics_ctx.to_snapshot_ref()

        tables = ("analytics.function_metrics", "analytics.goid_risk_factors")
        result = provider.compute_row_counts(metrics_ctx.gateway, snapshot, tables)

        # METRICS_PACK seeds these tables with data
        assert "analytics.function_metrics" in result
        assert "analytics.goid_risk_factors" in result
        # Both should have positive counts from seeds
        assert result["analytics.function_metrics"] > 0
        assert result["analytics.goid_risk_factors"] > 0

    @staticmethod
    def test_returns_zero_for_empty_table(test_ctx: TestContext) -> None:
        """Verify zero count for empty tables."""
        test_ctx.require(CORE_PACK)  # Only core data, no metrics
        provider = DefaultAnalyticsSupportProvider()
        snapshot = test_ctx.to_snapshot_ref()

        tables = ("analytics.function_metrics",)
        result = provider.compute_row_counts(test_ctx.gateway, snapshot, tables)

        # Table exists but is empty
        assert result.get("analytics.function_metrics", 0) == 0

    @staticmethod
    def test_handles_multiple_tables(core_ctx: TestContext) -> None:
        """Verify counting multiple tables at once."""
        provider = DefaultAnalyticsSupportProvider()
        snapshot = core_ctx.to_snapshot_ref()

        tables = ("core.modules", "core.goids")
        result = provider.compute_row_counts(core_ctx.gateway, snapshot, tables)

        # CORE_PACK seeds these tables
        assert "core.modules" in result
        assert "core.goids" in result
        assert result["core.modules"] > 0
        assert result["core.goids"] > 0


class TestValidateContracts:
    """Tests for validate_contracts method."""

    @staticmethod
    def test_returns_valid_for_empty_contracts(test_ctx: TestContext) -> None:
        """Verify empty contracts returns valid."""
        provider = DefaultAnalyticsSupportProvider()
        snapshot = test_ctx.to_snapshot_ref()

        valid, errors = provider.validate_contracts(test_ctx.gateway, (), snapshot)

        assert valid is True
        assert errors == ()

    @staticmethod
    def test_returns_valid_for_non_contract_objects(test_ctx: TestContext) -> None:
        """Verify non-OutputContractSpec objects are ignored."""
        provider = DefaultAnalyticsSupportProvider()
        snapshot = test_ctx.to_snapshot_ref()

        # Pass objects that aren't OutputContractSpec
        contracts: tuple[object, ...] = ("not_a_contract", 123, {"key": "value"})
        valid, errors = provider.validate_contracts(test_ctx.gateway, contracts, snapshot)

        assert valid is True
        assert errors == ()

    @staticmethod
    def test_validates_contract_with_data(metrics_ctx: TestContext) -> None:
        """Verify contract validation works with real data."""
        provider = DefaultAnalyticsSupportProvider()
        snapshot = metrics_ctx.to_snapshot_ref()

        # Create a contract that should pass (table has data from METRICS_PACK)
        contract = OutputContractSpec(
            table="analytics.function_metrics",
            description="Function metrics must exist",
            min_rows=1,
        )
        contracts: tuple[object, ...] = (contract,)

        valid, errors = provider.validate_contracts(metrics_ctx.gateway, contracts, snapshot)

        assert valid is True
        assert len(errors) == 0

    @staticmethod
    def test_validates_failing_contract(test_ctx: TestContext) -> None:
        """Verify contract validation detects failures."""
        test_ctx.require(CORE_PACK)  # Only core data
        provider = DefaultAnalyticsSupportProvider()
        snapshot = test_ctx.to_snapshot_ref()

        # Create a contract that should fail (requires data that doesn't exist)
        contract = OutputContractSpec(
            table="analytics.function_metrics",
            description="Function metrics must have 100 rows",
            min_rows=100,  # This will fail since table is empty
        )
        contracts: tuple[object, ...] = (contract,)

        valid, errors = provider.validate_contracts(test_ctx.gateway, contracts, snapshot)

        assert valid is False
        assert len(errors) > 0

    @staticmethod
    def test_mixed_contracts_filters_correctly(test_ctx: TestContext) -> None:
        """Verify mixed contract types are filtered correctly."""
        test_ctx.require(METRICS_PACK)
        provider = DefaultAnalyticsSupportProvider()
        snapshot = test_ctx.to_snapshot_ref()

        # Mix of valid contracts and non-contracts
        valid_contract = OutputContractSpec(
            table="analytics.function_metrics",
            description="Valid contract",
            min_rows=1,
        )
        contracts: tuple[object, ...] = (
            "string",
            valid_contract,
            123,
        )

        valid, validation_errors = provider.validate_contracts(
            test_ctx.gateway, contracts, snapshot
        )

        # Should only validate the actual contract, which should pass
        assert valid is True
        assert len(validation_errors) == 0
