"""Snapshot key and caching utilities.

This module provides types for snapshot-scoped caching,
where data is keyed by (repo, commit) pairs.

.. deprecated:: 1.0
    Import directly from ``codeintel.core.cache`` instead.
    This module will be removed in a future version.

Examples
--------
Instead of:

>>> from codeintel.core.data.snapshot import SnapshotKey

Use:

>>> from codeintel.core.cache import SnapshotKey
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

from codeintel.core.cache import SnapshotKey, SnapshotScopedCache

if not TYPE_CHECKING:
    warnings.warn(
        "Importing from codeintel.core.data.snapshot is deprecated. "
        "Import from codeintel.core.cache instead.",
        DeprecationWarning,
        stacklevel=2,
    )

__all__ = [
    "SnapshotKey",
    "SnapshotScopedCache",
]
