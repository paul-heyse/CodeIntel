"""Tests for core providers module.

This module tests:
- AnalyticsSupportProvider protocol
- get_support_provider function
- set_support_provider function
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.analytics.core.providers import (
    AnalyticsSupportProvider,
    get_support_provider,
    set_support_provider,
)
from codeintel.analytics.core.providers_impl import DefaultAnalyticsSupportProvider

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


class MockSupportProvider:
    """Mock support provider for testing.

    Implements the AnalyticsSupportProvider protocol with no-op methods.
    """

    call_count: int = 0

    def compute_row_counts(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        tables: tuple[str, ...],
    ) -> dict[str, int]:
        """Return mock row counts.

        Returns
        -------
        dict[str, int]
            Empty row counts for each table.
        """
        # Use parameters to avoid ARG002
        _ = gateway
        _ = snapshot
        self.call_count += 1
        return dict.fromkeys(tables, 0)

    def validate_contracts(
        self,
        gateway: StorageGateway,
        contracts: tuple[object, ...],
        snapshot: SnapshotRef,
    ) -> tuple[bool, tuple[str, ...]]:
        """Return mock validation result.

        Returns
        -------
        tuple[bool, tuple[str, ...]]
            Always returns (True, ()).
        """
        # Use parameters to avoid ARG002
        _ = gateway
        _ = contracts
        _ = snapshot
        self.call_count += 1
        return (True, ())


class TestAnalyticsSupportProviderProtocol:
    """Tests for AnalyticsSupportProvider protocol."""

    @staticmethod
    def test_mock_provider_has_required_methods() -> None:
        """Verify mock provider has required protocol methods."""
        provider = MockSupportProvider()
        assert callable(provider.compute_row_counts)
        assert callable(provider.validate_contracts)

    @staticmethod
    def test_default_provider_has_required_methods() -> None:
        """Verify default provider has required protocol methods."""
        provider = DefaultAnalyticsSupportProvider()
        assert callable(provider.compute_row_counts)
        assert callable(provider.validate_contracts)


class TestGetSupportProvider:
    """Tests for get_support_provider function."""

    @staticmethod
    def test_returns_provider_with_required_methods() -> None:
        """Verify get_support_provider returns a provider with protocol methods."""
        provider = get_support_provider()
        assert callable(provider.compute_row_counts)
        assert callable(provider.validate_contracts)

    @staticmethod
    def test_returns_same_provider_on_subsequent_calls() -> None:
        """Verify same provider is returned on subsequent calls."""
        provider1 = get_support_provider()
        provider2 = get_support_provider()
        assert provider1 is provider2


class TestSetSupportProvider:
    """Tests for set_support_provider function."""

    @staticmethod
    def test_set_and_get_provider() -> None:
        """Verify set provider is returned by get."""
        mock_provider = MockSupportProvider()
        set_support_provider(cast("AnalyticsSupportProvider", mock_provider))

        provider = get_support_provider()
        assert provider is mock_provider

    @staticmethod
    def test_overwrites_previous_provider() -> None:
        """Verify previous provider is overwritten."""
        provider1 = MockSupportProvider()
        provider2 = MockSupportProvider()

        set_support_provider(cast("AnalyticsSupportProvider", provider1))
        set_support_provider(cast("AnalyticsSupportProvider", provider2))

        result = get_support_provider()
        assert result is provider2
        assert result is not provider1
