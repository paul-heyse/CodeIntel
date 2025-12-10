"""Compatibility shim for introspection module.

.. deprecated::
    This module is deprecated. Import from ``codeintel.cli.introspection`` package instead.
    This shim will be removed in a future version.

Example migration::

    # Old (deprecated):
    from codeintel.cli.introspection import get_operation_info, search_operations

    # New (preferred):
    from codeintel.cli.introspection import get_operation_info, search_operations
    # (import from the package, not this file)
"""

from __future__ import annotations

import warnings

warnings.warn(
    "Importing from 'codeintel.cli.introspection' (module) is deprecated. "
    "The introspection package now provides these exports directly. "
    "This compatibility shim will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2,
)

# Re-export everything from the canonical location
from codeintel.cli.introspection.discovery import (
    OperationInfo,
    get_operation_info,
    get_operation_schema,
    list_all_operations,
    list_operations_by_category,
    search_operations,
)

__all__ = [
    "OperationInfo",
    "get_operation_info",
    "get_operation_schema",
    "list_all_operations",
    "list_operations_by_category",
    "search_operations",
]
