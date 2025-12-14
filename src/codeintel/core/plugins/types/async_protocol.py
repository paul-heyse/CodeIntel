"""Async plugin protocol for asynchronous plugin execution.

This module extends the core plugin protocol to support asynchronous
execution patterns, primarily used for tool plugins that invoke external
CLI tools or services.

Architecture
------------
The async protocol mirrors the sync PluginProtocol but uses async/await
for the execute method, enabling non-blocking execution of external tools.

Integration
-----------
Tool plugins from ingestion can implement this protocol while sharing
the same PluginMetadata structure used by sync plugins.

Example
-------
```python
from codeintel.core.plugins.types import AsyncPluginProtocol, PluginMetadata


class MyToolPlugin:
    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            name="my.tool",
            description="My tool plugin",
            kind="tool",
            stage="pipeline_ingestion",
        )

    async def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        # Run external tool
        result = await run_tool(ctx.repo_root)
        return PluginResult.ok(artifacts={"output": result})
```
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.core.plugins.execution.context import PluginExecutionContext
    from codeintel.core.plugins.types.protocol import PluginMetadata, ValidationResult
    from codeintel.core.plugins.types.result import PluginResult


@runtime_checkable
class AsyncPluginProtocol(Protocol):
    """Protocol for async plugin execution.

    This protocol extends the plugin concept to support asynchronous
    execution, primarily for plugins that invoke external tools or services.

    Implementations should:
    - Return PluginMetadata with kind="tool" for tool plugins
    - Use async/await for external I/O operations
    - Handle timeouts and cancellation appropriately

    Notes
    -----
    The async protocol shares PluginMetadata with sync plugins, enabling
    unified metadata handling across all plugin types.
    """

    @property
    def metadata(self) -> PluginMetadata:
        """Return plugin metadata.

        Returns
        -------
        PluginMetadata
            Static metadata describing the plugin.
        """
        ...

    async def execute(self, ctx: PluginExecutionContext) -> PluginResult:
        """Execute the plugin asynchronously.

        Parameters
        ----------
        ctx
            Execution context providing access to storage, config, and runtime.

        Returns
        -------
        PluginResult
            Result of the plugin execution.

        Notes
        -----
        Implementations should:
        - Use async operations for external tool invocation
        - Handle timeouts gracefully
        - Capture and report errors in the result
        """
        ...

    def validate_inputs(self, ctx: PluginExecutionContext) -> ValidationResult:
        """Validate that required inputs are available.

        Parameters
        ----------
        ctx
            Execution context to validate against.

        Returns
        -------
        ValidationResult
            Validation outcome with any errors or warnings.

        Notes
        -----
        This is a synchronous method since validation typically
        doesn't require external I/O.
        """
        ...


@runtime_checkable
class AsyncPluginWithCleanup(AsyncPluginProtocol, Protocol):
    """Extended async plugin protocol with cleanup support.

    This protocol adds cleanup capabilities for plugins that need
    to release resources after execution, such as temporary files
    or connections.
    """

    async def cleanup(self, ctx: PluginExecutionContext) -> None:
        """Clean up resources after execution.

        Parameters
        ----------
        ctx
            Execution context providing access to cleanup utilities.

        Notes
        -----
        This method is called after execute() completes, regardless
        of success or failure. Implementations should handle errors
        gracefully and not raise exceptions.
        """
        ...


__all__ = [
    "AsyncPluginProtocol",
    "AsyncPluginWithCleanup",
]
