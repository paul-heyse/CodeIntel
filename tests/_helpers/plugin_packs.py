"""Reusable plugin packs for analytics/serving graph plugin tests."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Literal, cast

from codeintel.analytics.graphs.contracts import ContractChecker, PluginContractResult
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
    register_graph_metric_plugin,
    unregister_graph_metric_plugin,
)


@dataclass
class GraphPluginPackSettings:
    """Tunables for the default graph plugin pack."""

    slow_sleep_ms: int = 50
    flaky_failures: int = 1
    base_stage: Literal[
        "core",
        "cfg",
        "dfg",
        "test",
        "symbol",
        "subsystem",
        "config",
        "stats",
    ] = "core"
    success_row_counts: dict[str, int] = field(
        default_factory=lambda: {"analytics.pack.success": 1}
    )


@dataclass
class GraphPluginPackCounters:
    """Execution counters surfaced to tests."""

    success_calls: int = 0
    soft_fail_calls: int = 0
    fatal_calls: int = 0
    slow_calls: int = 0
    flaky_calls: int = 0


@dataclass
class GraphPluginPack:
    """Bundle of realistic plugins plus lifecycle helpers."""

    success: GraphMetricPlugin
    soft_fail: GraphMetricPlugin
    fatal_fail: GraphMetricPlugin
    slow: GraphMetricPlugin
    flaky: GraphMetricPlugin
    counters: GraphPluginPackCounters
    _registered: set[str] = field(default_factory=set)

    def register(self, *plugins: GraphMetricPlugin) -> None:
        """Register selected plugins against the global registry."""
        for plugin in plugins:
            register_graph_metric_plugin(plugin)
            self._registered.add(plugin.name)

    def register_all(self) -> None:
        """Register all pack plugins."""
        self.register(
            self.success,
            self.soft_fail,
            self.fatal_fail,
            self.slow,
            self.flaky,
        )

    def unregister_all(self) -> None:
        """Unregister any plugins registered by this pack."""
        for name in list(self._registered):
            unregister_graph_metric_plugin(name)
            self._registered.discard(name)

    @staticmethod
    def names(*plugins: GraphMetricPlugin) -> tuple[str, ...]:
        """
        Return plugin names for convenience when invoking runtimes.

        Returns
        -------
        tuple[str, ...]
            Names of the provided plugins in order.
        """
        return tuple(plugin.name for plugin in plugins)


def build_graph_plugin_pack(settings: GraphPluginPackSettings | None = None) -> GraphPluginPack:
    """
    Create a reusable pack of plugins covering success/failure/slow paths.

    Returns
    -------
    GraphPluginPack
        Pack containing pre-wired plugins and counters.
    """
    cfg = settings or GraphPluginPackSettings()
    counters = GraphPluginPackCounters()

    def _success(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        counters.success_calls += 1
        return GraphPluginResult(row_counts=dict(cfg.success_row_counts))

    def _soft_fail(_ctx: GraphMetricExecutionContext) -> None:
        counters.soft_fail_calls += 1
        message = "soft_fail_triggered"
        raise RuntimeError(message)

    def _fatal(_ctx: GraphMetricExecutionContext) -> None:
        counters.fatal_calls += 1
        message = "fatal_triggered"
        raise RuntimeError(message)

    flaky_threshold = cfg.flaky_failures

    def _flaky(_ctx: GraphMetricExecutionContext) -> GraphPluginResult:
        counters.flaky_calls += 1
        if counters.flaky_calls <= flaky_threshold:
            message = "transient"
            raise RuntimeError(message)
        return GraphPluginResult(row_counts={"analytics.pack.flaky": counters.flaky_calls})

    def _slow(_ctx: GraphMetricExecutionContext) -> None:
        counters.slow_calls += 1
        time.sleep(cfg.slow_sleep_ms / 1000)

    def soft_contract(_ctx: GraphMetricExecutionContext) -> PluginContractResult:
        return PluginContractResult(
            name="soft_contract",
            status="soft_failed",
            message="soft_contract_failure",
        )

    return GraphPluginPack(
        success=GraphMetricPlugin(
            name="pack_success",
            description="succeeds",
            stage=cfg.base_stage,
            enabled_by_default=False,
            run=_success,
        ),
        soft_fail=GraphMetricPlugin(
            name="pack_soft_fail",
            description="soft fail",
            stage=cfg.base_stage,
            enabled_by_default=False,
            run=_soft_fail,
            severity="soft_fail",
        ),
        fatal_fail=GraphMetricPlugin(
            name="pack_fatal_fail",
            description="fatal fail",
            stage=cfg.base_stage,
            enabled_by_default=False,
            run=_fatal,
            severity="fatal",
            contract_checkers=(cast("ContractChecker", soft_contract),),
        ),
        slow=GraphMetricPlugin(
            name="pack_slow",
            description="slow plugin",
            stage=cfg.base_stage,
            enabled_by_default=False,
            run=_slow,
            severity="soft_fail",
        ),
        flaky=GraphMetricPlugin(
            name="pack_flaky",
            description="flaky transient failure",
            stage=cfg.base_stage,
            enabled_by_default=False,
            run=_flaky,
        ),
        counters=counters,
    )


__all__ = [
    "GraphPluginPack",
    "GraphPluginPackCounters",
    "GraphPluginPackSettings",
    "build_graph_plugin_pack",
]
