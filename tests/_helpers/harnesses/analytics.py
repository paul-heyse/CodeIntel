"""Analytics plugin harness helpers.

These helpers provide pre-seeded harnesses to reduce boilerplate in analytics tests.
Prefer using the context managers here instead of manually constructing TestContext.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers import TestScenario
from tests._helpers.seeds import CORE_PACK, DATA_MODELS_PACK, ENTRYPOINTS_PACK, GRAPH_PACK

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from tests._helpers.context import SeedPack, TestContext


@dataclass
class AnalyticsPluginHarness:
    """Lightweight harness providing seeded TestContext for analytics tests."""

    ctx: TestContext

    def close(self) -> None:
        """Close the underlying TestContext."""
        self.ctx.close()


def _apply_packs(ctx: TestContext, packs: tuple[SeedPack, ...]) -> None:
    if packs:
        ctx.require(*packs)


@contextmanager
def plugin_harness_with_packs(tmp_path: Path, *packs: SeedPack) -> Iterator[AnalyticsPluginHarness]:
    """Create a plugin harness with the provided seed packs applied.

    Yields
    ------
    AnalyticsPluginHarness
        Harness with the requested seed packs applied.
    """
    scenario = TestScenario().with_seeds(*packs)
    ctx = scenario.build(tmp_path)
    harness = AnalyticsPluginHarness(ctx=ctx)
    try:
        yield harness
    finally:
        harness.close()


@contextmanager
def graph_plugin_harness(tmp_path: Path) -> Iterator[AnalyticsPluginHarness]:
    """Plugin harness seeded with core + graph data.

    Yields
    ------
    AnalyticsPluginHarness
        Harness seeded with graph packs.
    """
    with plugin_harness_with_packs(tmp_path, CORE_PACK, GRAPH_PACK) as harness:
        yield harness


@contextmanager
def entrypoints_plugin_harness(tmp_path: Path) -> Iterator[AnalyticsPluginHarness]:
    """Plugin harness seeded with entrypoints data and canonical sources.

    Yields
    ------
    AnalyticsPluginHarness
        Harness seeded with entrypoint packs.
    """
    with plugin_harness_with_packs(tmp_path, CORE_PACK, ENTRYPOINTS_PACK) as harness:
        yield harness


@contextmanager
def data_models_plugin_harness(tmp_path: Path) -> Iterator[AnalyticsPluginHarness]:
    """Plugin harness seeded with data models pack.

    Yields
    ------
    AnalyticsPluginHarness
        Harness seeded with data model packs.
    """
    with plugin_harness_with_packs(tmp_path, CORE_PACK, DATA_MODELS_PACK) as harness:
        yield harness


__all__ = [
    "AnalyticsPluginHarness",
    "data_models_plugin_harness",
    "entrypoints_plugin_harness",
    "graph_plugin_harness",
    "plugin_harness_with_packs",
]
