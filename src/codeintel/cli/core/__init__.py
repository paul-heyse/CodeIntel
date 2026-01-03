"""Core CLI abstractions and types.

This package provides fundamental abstractions used across the CLI:

- ``CliResult``: Generic result wrapper for CLI operations
- ``Command``: Base class for type-safe commands (new pattern)
- Result type dataclasses for all handlers
- Value parsing utilities for consistent type coercion
"""

from __future__ import annotations

from codeintel.cli.core.command import Command
from codeintel.cli.core.parsing import (
    is_truthy_string,
    parse_bool,
    parse_bool_or_none,
    parse_cli_value,
)
from codeintel.cli.core.results import CliResult, ResultBase

__all__ = [
    "CliResult",
    "Command",
    "ResultBase",
    "is_truthy_string",
    "parse_bool",
    "parse_bool_or_none",
    "parse_cli_value",
]
