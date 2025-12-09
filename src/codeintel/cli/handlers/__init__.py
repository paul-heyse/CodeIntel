"""Unified CLI handlers package.

This package provides:
1. Base utilities (logging, context) in `handlers.base`
2. Common utilities (gateway, project) in `handlers.common`
3. Domain-specific handlers in `handlers.<domain>`

Examples
--------
>>> from codeintel.cli.handlers import setup_logging, build_handler_context
>>> setup_logging(verbosity=1)
>>> ctx = build_handler_context("build.run", {"target": "all"})
"""

from __future__ import annotations

from codeintel.cli.handlers.base import (
    HandlerContext,
    build_handler_context,
    get_handler_logger,
    open_handler_gateway,
    setup_logging,
)

__all__ = [
    "HandlerContext",
    "build_handler_context",
    "get_handler_logger",
    "open_handler_gateway",
    "setup_logging",
]
