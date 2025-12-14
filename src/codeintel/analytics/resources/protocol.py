"""Protocol and base classes for resource providers.

This module re-exports unified resource provider types from codeintel.core.resources,
providing a consistent interface for analytics resource management.

.. deprecated:: 1.0
    Import directly from ``codeintel.core.resources`` instead.
    This module will be removed in a future version.

Examples
--------
Instead of:

>>> from codeintel.analytics.resources.protocol import LazyResource

Use:

>>> from codeintel.core.resources import LazyResource
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from codeintel.core.resources import (
    LazyResource,
    ResourceError,
    ResourceNotFoundError,
    ResourceNotLoadedError,
    ResourceProvider,
    ResourceProviderBase,
    ResourceRegistry,
)

if not TYPE_CHECKING:
    warnings.warn(
        "Importing from codeintel.analytics.resources.protocol is deprecated. "
        "Import from codeintel.core.resources instead.",
        DeprecationWarning,
        stacklevel=2,
    )

__all__ = [
    "LazyResource",
    "ResourceError",
    "ResourceNotFoundError",
    "ResourceNotLoadedError",
    "ResourceProvider",
    "ResourceProviderBase",
    "ResourceRegistry",
]
