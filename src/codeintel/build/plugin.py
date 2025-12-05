"""Target plugin protocol and base class.

This module defines the unified TargetPlugin protocol that all plugins
implement. Plugins are pure executors - all metadata about what they
produce and what they depend on lives in the OutputTarget definition.

Example
-------
>>> class MyPlugin(TargetPlugin):
...     plugin_name: ClassVar[str] = "my_plugin"
...     plugin_version: ClassVar[str] = "1.0.0"
...
...     async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
...         rows = self._compute_rows(ctx)
...         ctx.write_table("core.my_table", rows)
...         return TargetResult.succeeded(row_counts={"core.my_table": len(rows)})
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext, TargetResult

__all__ = [
    "TargetPlugin",
    "TargetPluginProtocol",
]


@runtime_checkable
class TargetPluginProtocol(Protocol):
    """Protocol for target plugins.

    This is the minimal interface that all plugins must satisfy.
    Plugins receive everything they need via TargetExecutionContext
    and return a TargetResult.

    Class Variables
    ---------------
    plugin_name
        Unique identifier for the plugin (e.g., "ast_extract").
    plugin_version
        Semantic version string (e.g., "1.0.0").
    plugin_description
        Human-readable description of what the plugin does.
    """

    plugin_name: ClassVar[str]
    plugin_version: ClassVar[str]
    plugin_description: ClassVar[str]

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin with the given context.

        Parameters
        ----------
        ctx
            Execution context with resources, parameters, and write methods.

        Returns
        -------
        TargetResult
            Success or failure result with row counts and artifacts.
        """
        ...


class TargetPlugin(ABC):
    """Base class for target plugins.

    Provides the abstract interface for all plugins in the build system.
    Subclasses must define class variables and implement execute().

    Class Variables
    ---------------
    plugin_name
        Unique identifier for the plugin.
    plugin_version
        Semantic version for change tracking.
    plugin_description
        Human-readable description.

    Example
    -------
    >>> class RepoScanPlugin(TargetPlugin):
    ...     plugin_name = "repo_scan"
    ...     plugin_version = "3.0.0"
    ...     plugin_description = "Scan repository for Python modules."
    ...
    ...     async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
    ...         modules = self._scan_modules(ctx.repo_root)
    ...         ctx.write_table("core.modules", modules)
    ...         return TargetResult.succeeded(row_counts={"core.modules": len(modules)})
    """

    plugin_name: ClassVar[str] = ""
    plugin_version: ClassVar[str] = "1.0.0"
    plugin_description: ClassVar[str] = ""

    @abstractmethod
    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin.

        Parameters
        ----------
        ctx
            Execution context with everything the plugin needs.

        Returns
        -------
        TargetResult
            Result indicating success/failure with row counts.
        """
        ...

    def validate_context(self, ctx: TargetExecutionContext) -> list[str]:
        """Validate that the context has everything needed.

        Override this method to add plugin-specific validation.
        The default implementation returns an empty list (no errors).

        Parameters
        ----------
        ctx
            Execution context to validate.

        Returns
        -------
        list[str]
            List of validation error messages. Empty if valid.
        """
        # Base implementation performs no validation; subclasses may override
        _ = (self, ctx)  # Protocol method signature - both used in subclasses
        return []
