"""Support provider interfaces for analytics plugin runtime."""

from __future__ import annotations

from typing import Protocol

from codeintel.analytics.core.providers_impl import DefaultAnalyticsSupportProvider
from codeintel.config.primitives import SnapshotRef
from codeintel.storage.gateway import StorageGateway


class AnalyticsSupportProvider(Protocol):
    """Thin interface for analytics runtime helpers."""

    def compute_row_counts(
        self,
        gateway: StorageGateway,
        snapshot: SnapshotRef,
        tables: tuple[str, ...],
    ) -> dict[str, int]:
        """Return row counts for the given tables scoped to the snapshot."""
        ...

    def validate_contracts(
        self,
        gateway: StorageGateway,
        contracts: tuple[object, ...],
        snapshot: SnapshotRef,
    ) -> tuple[bool, tuple[str, ...]]:
        """Validate output contracts and return (valid, errors)."""
        ...


class _SupportProviderState:
    """Mutable holder for the analytics support provider."""

    provider: AnalyticsSupportProvider | None = None


_support_provider_state = _SupportProviderState()


def set_support_provider(provider: AnalyticsSupportProvider) -> None:
    """Override the global analytics support provider (for testing/experiments)."""
    _support_provider_state.provider = provider


def get_support_provider() -> AnalyticsSupportProvider:
    """Return the active analytics support provider.

    Returns
    -------
    AnalyticsSupportProvider
        Active support provider instance.
    """
    provider = _support_provider_state.provider
    if provider is None:
        provider = DefaultAnalyticsSupportProvider()
        _support_provider_state.provider = provider
    return provider


__all__ = [
    "AnalyticsSupportProvider",
    "get_support_provider",
    "set_support_provider",
]
