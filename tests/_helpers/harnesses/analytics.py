"""Analytics plugin harness helpers.

These helpers provide pre-seeded plugin harnesses to reduce boilerplate
in plugin tests. Prefer using the context managers here instead of
manually constructing TestContext + ExecutionContextBuilder in each test.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

from tests._helpers.context import create_test_context
from tests._helpers.fakes.contexts import ExecutionContextBuilder
from tests._helpers.seeds import (
    CORE_PACK,
    COVERAGE_LINES_PACK,
    COVERAGE_PACK,
    DATA_MODELS_PACK,
    ENTRYPOINTS_PACK,
    GRAPH_PACK,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.build.context import TargetResult
    from codeintel.build.parameters import TargetParameters
    from tests._helpers.context import SeedPack, TestContext
    from tests._helpers.fakes.contexts import (
        PluginLike,
        TargetResourceOverrides,
    )


@dataclass
class AnalyticsPluginHarness:
    """Lightweight harness for executing analytics plugins."""

    ctx: TestContext

    def close(self) -> None:
        """Close the underlying TestContext."""
        self.ctx.close()

    def execute_plugin(
        self,
        plugin: PluginLike,
        *,
        parameters: TargetParameters | None = None,
        resources: TargetResourceOverrides | None = None,
    ) -> TargetResult:
        """Execute a plugin using the harness's context.

        Returns
        -------
        TargetResult
            Result produced by the executed plugin.
        """
        builder = ExecutionContextBuilder(
            gateway=self.ctx.gateway,
            snapshot=self.ctx.snapshot,
            paths=self.ctx.build_paths,
        )
        return builder.execute_plugin(plugin, parameters=parameters, resources=resources)


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
    ctx = create_test_context(tmp_path)
    _apply_packs(ctx, packs)
    harness = AnalyticsPluginHarness(ctx=ctx)
    try:
        yield harness
    finally:
        harness.close()


@contextmanager
def coverage_plugin_harness(tmp_path: Path) -> Iterator[AnalyticsPluginHarness]:
    """Plugin harness seeded with core + coverage (catalog/edges/lines).

    Yields
    ------
    AnalyticsPluginHarness
        Harness seeded with coverage-related packs.
    """
    with plugin_harness_with_packs(
        tmp_path,
        CORE_PACK,
        COVERAGE_PACK,
        COVERAGE_LINES_PACK,
    ) as harness:
        yield harness


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
    "coverage_plugin_harness",
    "data_models_plugin_harness",
    "entrypoints_plugin_harness",
    "graph_plugin_harness",
    "plugin_harness_with_packs",
]
