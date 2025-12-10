"""External compatibility shims.

This module provides compatibility for external code that depends on
legacy CLI patterns. Internal code MUST NOT use this module.

All exports are deprecated and will be removed in a future version.

Migration Guide
---------------
- ``command_context`` → Use ``@cli_command`` decorator
- ``get_operation_registry`` → Use ``get_registry`` from ``execution.registry``
- ``EnhancedHandlerContext`` → Use ``HandlerContext`` from ``handlers.context``
- ``build_handler_context`` → Use ``handler_context_manager`` from ``handlers.context``

Version History
---------------
- Added: v2.0 (Phase 7)
- Planned removal: v3.0
"""

from __future__ import annotations

import warnings


def __getattr__(name: str) -> object:
    """Lazy attribute access with deprecation warnings.

    Parameters
    ----------
    name
        Attribute name to access.

    Returns
    -------
    object
        The requested attribute.

    Raises
    ------
    AttributeError
        If the attribute is not found.
    """
    if name == "command_context":
        warnings.warn(
            "command_context is deprecated. Use @cli_command decorator instead. "
            "See: docs/migration/cli-commands.md",
            DeprecationWarning,
            stacklevel=2,
        )
        from codeintel.cli.commands.context import (  # noqa: PLC0415
            command_context as _cmd_ctx,
        )

        return _cmd_ctx

    if name == "CommandContextError":
        warnings.warn(
            "CommandContextError is deprecated. Use standard exceptions instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from codeintel.cli.commands.context import (  # noqa: PLC0415
            CommandContextError,
        )

        return CommandContextError

    if name == "get_operation_registry":
        warnings.warn(
            "get_operation_registry is deprecated. Use get_registry from "
            "codeintel.cli.execution.registry instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from codeintel.cli.execution.registry import get_registry  # noqa: PLC0415

        return get_registry

    if name == "EnhancedHandlerContext":
        warnings.warn(
            "EnhancedHandlerContext is deprecated. Use HandlerContext from "
            "codeintel.cli.handlers.context instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from codeintel.cli.handlers.context import HandlerContext  # noqa: PLC0415

        return HandlerContext

    if name == "build_handler_context":
        warnings.warn(
            "build_handler_context is deprecated. Use handler_context_manager from "
            "codeintel.cli.handlers.context instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from codeintel.cli.handlers.context import (  # noqa: PLC0415
            handler_context_manager,
        )

        return handler_context_manager

    if name == "LegacyHandlerContext":
        warnings.warn(
            "LegacyHandlerContext is deprecated. Use HandlerContext from "
            "codeintel.cli.handlers.context instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        from codeintel.cli.handlers.context import HandlerContext  # noqa: PLC0415

        return HandlerContext

    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


# Note: All exports are via __getattr__ for lazy loading with deprecation warnings
# Available exports (all deprecated):
# - command_context
# - CommandContextError
# - get_operation_registry
# - EnhancedHandlerContext
# - build_handler_context
# - LegacyHandlerContext
