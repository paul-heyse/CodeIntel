"""Temporary feature flags for CLI migration.

WARNING: This module is temporary scaffolding for the CLI architecture
migration. It will be DELETED in Phase 6 of the migration.

Do not add permanent feature flags here. This module exists solely to
enable gradual rollout of new code paths during the migration.
"""

from __future__ import annotations

import logging
import os

LOG = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Migration Feature Flags
# -----------------------------------------------------------------------------
# These flags control which code paths are used during the migration.
# Set via environment variables for testing/rollout.
# -----------------------------------------------------------------------------

# Phase 1: Use new unified HandlerContext
# When True, new context implementation is active
# When False (default), existing EnhancedHandlerContext is used
USE_NEW_HANDLER_CONTEXT: bool = os.environ.get("CODEINTEL_CLI_NEW_CONTEXT", "0") == "1"

# Phase 2: Use UnifiedRenderer everywhere
# When True, executor uses UnifiedRenderer from service.py
# When False (default), executor uses renderers from renderers.py
USE_UNIFIED_RENDERER: bool = os.environ.get("CODEINTEL_CLI_UNIFIED_RENDERER", "0") == "1"

# Phase 5: Use @cli_command decorator
# When True, new decorator-based commands are active
# When False (default), traditional __call__ commands are used
USE_CLI_COMMAND_DECORATOR: bool = os.environ.get("CODEINTEL_CLI_DECORATOR", "0") == "1"


def log_migration_flags() -> None:
    """Log current migration flag states.

    Call at CLI startup to log which migration features are enabled.
    Useful for debugging and rollout verification.
    """
    LOG.debug(
        "Migration flags: context=%s, renderer=%s, decorator=%s",
        USE_NEW_HANDLER_CONTEXT,
        USE_UNIFIED_RENDERER,
        USE_CLI_COMMAND_DECORATOR,
    )


def is_any_migration_flag_enabled() -> bool:
    """Check if any migration flag is enabled.

    Returns
    -------
    bool
        True if any migration feature flag is enabled.
    """
    return USE_NEW_HANDLER_CONTEXT or USE_UNIFIED_RENDERER or USE_CLI_COMMAND_DECORATOR


__all__ = [
    "USE_CLI_COMMAND_DECORATOR",
    "USE_NEW_HANDLER_CONTEXT",
    "USE_UNIFIED_RENDERER",
    "is_any_migration_flag_enabled",
    "log_migration_flags",
]
