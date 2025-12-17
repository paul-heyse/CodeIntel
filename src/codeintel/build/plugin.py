"""Legacy build plugin protocol.

The build system has moved to Hamilton-native execution, but some test harnesses and
legacy utilities still model a "plugin" as an object with a name, description, and an
async ``execute()`` method.

This module provides the minimal protocol needed by those call sites while keeping
the production build execution path Hamilton-first.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from codeintel.build.context import TargetExecutionContext, TargetResult

__all__ = [
    "TargetPlugin",
]


@runtime_checkable
class TargetPlugin(Protocol):
    """Protocol for legacy build plugins.

    Notes
    -----
    New build features should prefer Hamilton-native targets (see
    ``codeintel.build.hamilton``). This protocol exists primarily for
    backwards-compatible test utilities.
    """

    @property
    def plugin_name(self) -> str:
        """Return the plugin name used as a target identifier."""
        ...

    @property
    def plugin_description(self) -> str:
        """Return a short human-readable description."""
        ...

    async def execute(self, ctx: TargetExecutionContext) -> TargetResult:
        """Execute the plugin with the provided context.

        Parameters
        ----------
        ctx
            Build-layer execution context for the plugin.

        Returns
        -------
        TargetResult
            Result describing success/failure for the target run.
        """
        ...
