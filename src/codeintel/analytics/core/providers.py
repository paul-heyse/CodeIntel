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


_support_provider: AnalyticsSupportProvider | None = None


def set_support_provider(provider: AnalyticsSupportProvider) -> None:
    """Override the global analytics support provider (for testing/experiments)."""
    global _support_provider
    _support_provider = provider


def get_support_provider() -> AnalyticsSupportProvider:
    """Return the active analytics support provider."""
    global _support_provider
    if _support_provider is None:
        _support_provider = DefaultAnalyticsSupportProvider()
    return _support_provider


__all__ = [
    "AnalyticsSupportProvider",
    "get_support_provider",
    "set_support_provider",
]
