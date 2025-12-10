"""Factory helpers for analytics plugin harnesses.

These wrappers provide pre-seeded plugin harnesses for common packs so
tests can reuse consistent contexts without reapplying seeds in every
module. Prefer these over bespoke harness setup in plugin tests.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from tests._helpers.context import SeedPack
from tests._helpers.harnesses import (
    AnalyticsPluginHarness,
    coverage_plugin_harness,
    data_models_plugin_harness,
    entrypoints_plugin_harness,
    graph_plugin_harness,
    plugin_harness_with_packs,
)
from tests._helpers.seeds import CORE_PACK


class PluginHarnessFactory:
    """Factory for building analytics plugin harnesses with common seed packs."""

    def __init__(self, tmp_path: Path) -> None:
        self._tmp_path = tmp_path

    @contextmanager
    def with_packs(self, *packs: SeedPack) -> Iterator[AnalyticsPluginHarness]:
        """Create a harness with arbitrary packs applied (defaults to CORE)."""
        packs_to_apply = packs or (CORE_PACK,)
        with plugin_harness_with_packs(self._tmp_path, *packs_to_apply) as harness:
            yield harness

    @contextmanager
    def core(self) -> Iterator[AnalyticsPluginHarness]:
        """Harness seeded with CORE_PACK."""
        with self.with_packs(CORE_PACK) as harness:
            yield harness

    @contextmanager
    def coverage(self) -> Iterator[AnalyticsPluginHarness]:
        """Harness seeded with core + coverage packs."""
        with coverage_plugin_harness(self._tmp_path) as harness:
            yield harness

    @contextmanager
    def graphs(self) -> Iterator[AnalyticsPluginHarness]:
        """Harness seeded with core + graph packs."""
        with graph_plugin_harness(self._tmp_path) as harness:
            yield harness

    @contextmanager
    def data_models(self) -> Iterator[AnalyticsPluginHarness]:
        """Harness seeded with core + data models pack."""
        with data_models_plugin_harness(self._tmp_path) as harness:
            yield harness

    @contextmanager
    def entrypoints(self) -> Iterator[AnalyticsPluginHarness]:
        """Harness seeded with core + entrypoints pack."""
        with entrypoints_plugin_harness(self._tmp_path) as harness:
            yield harness


__all__ = [
    "PluginHarnessFactory",
]
