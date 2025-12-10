"""Factory helpers for analytics plugin harnesses.

These wrappers provide pre-seeded plugin harnesses for common packs so
tests can reuse consistent contexts without reapplying seeds in every
module. Prefer these over bespoke harness setup in plugin tests.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from tests._helpers.catalogs import ensure_catalog_with_goids
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
        """Create a harness with arbitrary packs applied (defaults to CORE).

        Yields
        ------
        AnalyticsPluginHarness
            Harness configured with the requested seed packs.
        """
        packs_to_apply = packs or (CORE_PACK,)
        with plugin_harness_with_packs(self._tmp_path, *packs_to_apply) as harness:
            yield harness

    @contextmanager
    def core(self) -> Iterator[AnalyticsPluginHarness]:
        """Harness seeded with CORE_PACK.

        Yields
        ------
        AnalyticsPluginHarness
            Harness preloaded with the core seed pack.
        """
        with self.with_packs(CORE_PACK) as harness:
            yield harness

    @contextmanager
    def coverage(self) -> Iterator[AnalyticsPluginHarness]:
        """Harness seeded with core + coverage packs.

        Yields
        ------
        AnalyticsPluginHarness
            Harness preloaded with coverage-related packs.
        """
        with coverage_plugin_harness(self._tmp_path) as harness:
            yield harness

    @contextmanager
    def graphs(self) -> Iterator[AnalyticsPluginHarness]:
        """Harness seeded with core + graph packs.

        Yields
        ------
        AnalyticsPluginHarness
            Harness preloaded with graph packs.
        """
        with graph_plugin_harness(self._tmp_path) as harness:
            yield harness

    @contextmanager
    def data_models(self) -> Iterator[AnalyticsPluginHarness]:
        """Harness seeded with core + data models pack.

        Yields
        ------
        AnalyticsPluginHarness
            Harness preloaded with data model pack.
        """
        with data_models_plugin_harness(self._tmp_path) as harness:
            yield harness

    @contextmanager
    def entrypoints(self) -> Iterator[AnalyticsPluginHarness]:
        """Harness seeded with core + entrypoints pack.

        Yields
        ------
        AnalyticsPluginHarness
            Harness preloaded with entrypoint pack.
        """
        with entrypoints_plugin_harness(self._tmp_path) as harness:
            yield harness

    @contextmanager
    def with_catalog(
        self,
        catalog: object | Callable[[AnalyticsPluginHarness], object],
        *packs: SeedPack,
    ) -> Iterator[tuple[AnalyticsPluginHarness, object]]:
        """Create a harness and seed GOIDs for the provided catalog or builder.

        Yields
        ------
        tuple[AnalyticsPluginHarness, object]
            Harness with seeded catalog alongside the catalog object.
        """
        packs_to_apply = packs or (CORE_PACK,)
        with plugin_harness_with_packs(self._tmp_path, *packs_to_apply) as harness:
            catalog_obj = catalog(harness) if callable(catalog) else catalog
            ensure_catalog_with_goids(harness.ctx, catalog_obj)
            yield harness, catalog_obj


__all__ = [
    "PluginHarnessFactory",
]
