"""MCP auto-pipeline wrapper for tool invocations.

This module provides wrapper functionality to automatically run
pipeline prerequisites before MCP tool execution when enabled.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from codeintel.serving.auto_pipeline import ensure_prereqs_for_mcp

if TYPE_CHECKING:
    from codeintel.config.serving_models import ServingConfig
    from codeintel.serving.mcp.backend import QueryBackend

LOG = logging.getLogger(__name__)


def wrap_tool_with_prereqs[T](
    tool_fn: Callable[..., T],
    op_id: str,
    config: ServingConfig,
    backend: QueryBackend,
) -> Callable[..., T]:
    """Wrap a tool function to run prerequisites first.

    When auto-pipeline is enabled, this wrapper will ensure that the
    necessary pipeline stages have been run before executing the tool.

    Parameters
    ----------
    tool_fn
        The original tool function to wrap.
    op_id
        Operation identifier for prerequisite check.
    config
        Serving configuration with repo/commit info.
    backend
        Query backend with gateway access.

    Returns
    -------
    Callable[..., T]
        Wrapped function that runs prereqs before the tool.
    """

    @functools.wraps(tool_fn)
    def _wrapped(**kwargs: object) -> T:
        LOG.debug("auto_pipeline check for op=%s", op_id)
        ensure_prereqs_for_mcp(op_id=op_id, config=config, backend=backend)
        return tool_fn(**kwargs)

    return _wrapped


__all__ = ["wrap_tool_with_prereqs"]
