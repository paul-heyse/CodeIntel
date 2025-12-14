"""Test environment types for graph tests.

This module provides environment types for graph testing scenarios.
These types wrap gateway, snapshot, and optional runtime into a single
environment object for convenient test parameterization.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.config.primitives import SnapshotRef
    from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class GraphTestEnv:
    """Test environment for graph integration tests.

    Parameters
    ----------
    gateway
        Storage gateway with schema applied.
    snapshot
        Snapshot reference for test data.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
