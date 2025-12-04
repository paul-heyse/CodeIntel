"""Fake graph plugin implementations for testing.

This module provides fake implementations of graph plugins for tests
that need deterministic plugin behavior.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

from codeintel.graphs.core import (
    GraphExecutionContext,
    GraphPluginMetadata,
    GraphPluginProtocol,
    GraphPluginResult,
    register_graph_plugin,
)
from codeintel.graphs.core.registry import unregister_graph_plugin

# Type alias for plugin execute functions
ExecuteFn = Callable[[GraphExecutionContext], GraphPluginResult]


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
class TestGraphPlugin:
    """Test plugin implementing GraphPluginProtocol."""

    _metadata: GraphPluginMetadata
    _execute_fn: ExecuteFn

    @property
    def metadata(self) -> GraphPluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        GraphPluginMetadata
            Metadata for this plugin.
        """
        return self._metadata

    def execute(self, ctx: GraphExecutionContext) -> GraphPluginResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context.

        Returns
        -------
        GraphPluginResult
            Result of execution.
        """
        return self._execute_fn(ctx)


@dataclass
class GraphPluginPack:
    """Bundle of realistic plugins plus lifecycle helpers."""

    success: GraphPluginProtocol
    soft_fail: GraphPluginProtocol
    fatal_fail: GraphPluginProtocol
    slow: GraphPluginProtocol
    flaky: GraphPluginProtocol
    counters: GraphPluginPackCounters
    _registered: set[str] = field(default_factory=set)

    def register(self, *plugins: GraphPluginProtocol) -> None:
        """Register selected plugins against the global registry."""
        for plugin in plugins:
            try:
                register_graph_plugin(plugin)
                self._registered.add(plugin.metadata.name)
            except ValueError:
                # Already registered
                pass

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
            with contextlib.suppress(KeyError):
                unregister_graph_plugin(name)
            self._registered.discard(name)

    @staticmethod
    def names(*plugins: GraphPluginProtocol) -> tuple[str, ...]:
        """Return plugin names for convenience when invoking runtimes.

        Returns
        -------
        tuple[str, ...]
            Names of the provided plugins in order.
        """
        return tuple(plugin.metadata.name for plugin in plugins)


def build_graph_plugin_pack(settings: GraphPluginPackSettings | None = None) -> GraphPluginPack:
    """Create a reusable pack of plugins covering success/failure/slow paths.

    Parameters
    ----------
    settings
        Optional pack configuration.

    Returns
    -------
    GraphPluginPack
        Pack containing pre-wired plugins and counters.
    """
    cfg = settings or GraphPluginPackSettings()
    counters = GraphPluginPackCounters()

    def _success(_ctx: GraphExecutionContext) -> GraphPluginResult:
        counters.success_calls += 1
        return GraphPluginResult.ok(row_counts=dict(cfg.success_row_counts))

    def _soft_fail(_ctx: GraphExecutionContext) -> GraphPluginResult:
        counters.soft_fail_calls += 1
        message = "soft_fail_triggered"
        raise RuntimeError(message)

    def _fatal(_ctx: GraphExecutionContext) -> GraphPluginResult:
        counters.fatal_calls += 1
        message = "fatal_triggered"
        raise RuntimeError(message)

    flaky_threshold = cfg.flaky_failures

    def _flaky(_ctx: GraphExecutionContext) -> GraphPluginResult:
        counters.flaky_calls += 1
        if counters.flaky_calls <= flaky_threshold:
            message = "transient"
            raise RuntimeError(message)
        return GraphPluginResult.ok(row_counts={"analytics.pack.flaky": counters.flaky_calls})

    def _slow(_ctx: GraphExecutionContext) -> GraphPluginResult:
        counters.slow_calls += 1
        time.sleep(cfg.slow_sleep_ms / 1000)
        return GraphPluginResult.ok()

    return GraphPluginPack(
        success=TestGraphPlugin(
            _metadata=GraphPluginMetadata(
                name="pack_success",
                description="succeeds",
                kind="metric",
                stage=cfg.base_stage,
                enabled_by_default=False,
            ),
            _execute_fn=_success,
        ),
        soft_fail=TestGraphPlugin(
            _metadata=GraphPluginMetadata(
                name="pack_soft_fail",
                description="soft fail",
                kind="metric",
                stage=cfg.base_stage,
                enabled_by_default=False,
                severity="soft_fail",
            ),
            _execute_fn=_soft_fail,
        ),
        fatal_fail=TestGraphPlugin(
            _metadata=GraphPluginMetadata(
                name="pack_fatal_fail",
                description="fatal fail",
                kind="metric",
                stage=cfg.base_stage,
                enabled_by_default=False,
                severity="fatal",
            ),
            _execute_fn=_fatal,
        ),
        slow=TestGraphPlugin(
            _metadata=GraphPluginMetadata(
                name="pack_slow",
                description="slow plugin",
                kind="metric",
                stage=cfg.base_stage,
                enabled_by_default=False,
                severity="soft_fail",
            ),
            _execute_fn=_slow,
        ),
        flaky=TestGraphPlugin(
            _metadata=GraphPluginMetadata(
                name="pack_flaky",
                description="flaky transient failure",
                kind="metric",
                stage=cfg.base_stage,
                enabled_by_default=False,
            ),
            _execute_fn=_flaky,
        ),
        counters=counters,
    )


__all__ = [
    "ExecuteFn",
    "GraphPluginPack",
    "GraphPluginPackCounters",
    "GraphPluginPackSettings",
    "TestGraphPlugin",
    "build_graph_plugin_pack",
]
