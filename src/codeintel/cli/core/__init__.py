"""Core CLI abstractions and types.

This package provides fundamental abstractions used across the CLI:

- ``CliResult``: Generic result wrapper for CLI operations
- ``OutputEnvelope``: I/O utilities for stdin/stdout handling
- Result type dataclasses for all handlers
- Common option definitions
"""

from __future__ import annotations

from codeintel.cli.options.common import CommonOptions
from codeintel.cli.core.output import (
    OutputEnvelope,
    iter_stdin_records,
    merge_stdin_with_args,
    read_stdin_records,
)
from codeintel.cli.core.results import CliResult, TextRenderer

__all__ = [
    "CliResult",
    "CommonOptions",
    "OutputEnvelope",
    "TextRenderer",
    "iter_stdin_records",
    "merge_stdin_with_args",
    "read_stdin_records",
]
