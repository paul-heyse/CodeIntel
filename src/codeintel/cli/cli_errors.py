"""Compatibility shim for cli_errors module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.errors`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.cli_errors import ProblemDetail, ValidationError

    # New (preferred):
    from codeintel.cli.errors import ProblemDetail, ValidationError
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.cli_errors' is deprecated. "
    "Use 'codeintel.cli.errors' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.errors import (
    CLI_EXIT_SUCCESS,
    CLI_EXIT_USAGE,
    CLI_EXIT_VALIDATION,
    CliError,
    DocsValidationError,
    ErrorType,
    OutputFormat,
    ProblemDetail,
    StorageConnectionError,
    StorageError,
    StorageQueryError,
    StorageSchemaError,
    UnknownCommandCliError,
    UnknownOptionCliError,
    ValidationError,
    handle_cli_error,
    run_handler,
    run_structured_handler,
    runtime_required,
)

__all__ = [
    "CLI_EXIT_SUCCESS",
    "CLI_EXIT_USAGE",
    "CLI_EXIT_VALIDATION",
    "CliError",
    "DocsValidationError",
    "ErrorType",
    "OutputFormat",
    "ProblemDetail",
    "StorageConnectionError",
    "StorageError",
    "StorageQueryError",
    "StorageSchemaError",
    "UnknownCommandCliError",
    "UnknownOptionCliError",
    "ValidationError",
    "handle_cli_error",
    "run_handler",
    "run_structured_handler",
    "runtime_required",
]
