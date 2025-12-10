"""Compatibility shim for operation_registry module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.introspection`` instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.operation_registry import get_operation_registry

    # New (preferred):
    from codeintel.cli.introspection import get_operation_registry
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.operation_registry' is deprecated. "
    "Use 'codeintel.cli.introspection' instead. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.introspection.registry import (
    OperationRegistry,
    get_operation_registry,
    register_operation,
)

__all__ = [
    "OperationRegistry",
    "get_operation_registry",
    "register_operation",
]
