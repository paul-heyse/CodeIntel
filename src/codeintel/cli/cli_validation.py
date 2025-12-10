"""Compatibility shim for cli_validation module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.introspection`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.cli_validation import ValidationSchema, StringValidator

    # New (preferred):
    from codeintel.cli.introspection import ValidationSchema, StringValidator
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.cli_validation' is deprecated. "
    "Use 'codeintel.cli.introspection' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.introspection.validation import (
    IntValidator,
    StringValidator,
    ValidationError,
    ValidationResult,
    ValidationSchema,
    Validator,
)

__all__ = [
    "IntValidator",
    "StringValidator",
    "ValidationError",
    "ValidationResult",
    "ValidationSchema",
    "Validator",
]
