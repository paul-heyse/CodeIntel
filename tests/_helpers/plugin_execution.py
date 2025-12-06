"""Test utilities for executing TargetPlugin instances.

This module provides infrastructure for executing async TargetPlugin.execute()
methods in tests, allowing tests to verify plugin behavior without going
through the full build executor.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.context import (
    ContextResources,
    TargetExecutionContext,
    TargetResult,
)
from codeintel.build.contracts import EMPTY_CONTRACT
from codeintel.build.parameters import EMPTY_PARAMETERS
from codeintel.build.plugin import TargetPlugin
from codeintel.build.targets import OutputTarget
from codeintel.config.primitives import BuildPaths, SnapshotRef

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


@dataclass
class PluginTestContext:
    """Bundle of resources for plugin testing.

    Attributes
    ----------
    gateway
        Storage gateway for database access.
    snapshot
        Repository snapshot reference.
    paths
        Build paths for directory resolution.
    repo_root
        Repository root directory (convenience accessor).
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    resources: ContextResources = field(default_factory=ContextResources)

    @property
    def repo_root(self) -> Path:
        """Return the repository root.

        Returns
        -------
        Path
            Repository root directory.
        """
        return self.snapshot.repo_root


def make_test_output_target(plugin: TargetPlugin) -> OutputTarget:
    """Create a minimal OutputTarget for testing a plugin.

    Parameters
    ----------
    plugin
        Plugin instance to create target for.

    Returns
    -------
    OutputTarget
        Minimal target suitable for test execution.
    """
    return OutputTarget(
        name=plugin.plugin_name,
        module="analytics",  # Default module for testing
        plugin=plugin.plugin_name,
        contract=EMPTY_CONTRACT,
        dependencies=(),
        description=plugin.plugin_description,
    )


def build_execution_context(
    plugin: TargetPlugin,
    test_ctx: PluginTestContext,
) -> TargetExecutionContext:
    """Build a TargetExecutionContext for testing.

    Parameters
    ----------
    plugin
        Plugin to create context for.
    test_ctx
        Plugin test context with gateway, snapshot, and paths.

    Returns
    -------
    TargetExecutionContext
        Context ready for plugin execution.
    """
    target = make_test_output_target(plugin)
    resources = ContextResources(
        gateway=test_ctx.gateway,
        providers=test_ctx.resources.providers,
        modules=test_ctx.resources.modules,
        change_tracker=test_ctx.resources.change_tracker,
        graph_runtime=test_ctx.resources.graph_runtime,
        catalog=test_ctx.resources.catalog,
    )
    return TargetExecutionContext(
        target=target,
        snapshot=test_ctx.snapshot,
        paths=test_ctx.paths,
        resources=resources,
        parameters=EMPTY_PARAMETERS,
    )


def execute_target_plugin(
    plugin: TargetPlugin,
    test_ctx: PluginTestContext,
) -> TargetResult:
    """Execute a TargetPlugin synchronously for testing.

    This function wraps the async plugin.execute() method and runs it
    using asyncio.run() for use in synchronous test code.

    Parameters
    ----------
    plugin
        Plugin instance to execute.
    test_ctx
        Plugin test context with gateway, snapshot, and paths.

    Returns
    -------
    TargetResult
        Result of plugin execution.
    """
    ctx = build_execution_context(plugin, test_ctx)
    return asyncio.run(plugin.execute(ctx))


async def execute_target_plugin_async(
    plugin: TargetPlugin,
    test_ctx: PluginTestContext,
) -> TargetResult:
    """Execute a TargetPlugin asynchronously.

    Use this in async test functions or when running multiple plugins.

    Parameters
    ----------
    plugin
        Plugin instance to execute.
    test_ctx
        Plugin test context with gateway, snapshot, and paths.

    Returns
    -------
    TargetResult
        Result of plugin execution.
    """
    ctx = build_execution_context(plugin, test_ctx)
    return await plugin.execute(ctx)


__all__ = [
    "PluginTestContext",
    "build_execution_context",
    "execute_target_plugin",
    "execute_target_plugin_async",
    "make_test_output_target",
]
