"""Unified data loading types for CodeIntel.

This module provides the canonical data loading protocol and utilities
for snapshot-scoped data access with caching.

Examples
--------
>>> from codeintel.core.data import DataLoaderProtocol, SnapshotKey
>>> key = SnapshotKey(repo="org/repo", commit="abc123")
>>> key.as_tuple()
('org/repo', 'abc123')
"""

from __future__ import annotations

from codeintel.core.data.loader import BaseDataLoader
from codeintel.core.data.protocol import DataLoaderProtocol
from codeintel.core.data.snapshot import SnapshotKey, SnapshotScopedCache

__all__ = [
    "BaseDataLoader",
    "DataLoaderProtocol",
    "SnapshotKey",
    "SnapshotScopedCache",
]
