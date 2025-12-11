"""Protocols for CLI dependency injection.

This package defines protocols used by the CLI services.
The main execution context is CommandContext from codeintel.cli.context.

Public API
----------
- ``StorageAccess``: Protocol for storage operations
- ``JobManagerProtocol``: Protocol for job management
- ``ServingAccess``: Protocol for serving layer
"""

from __future__ import annotations

from codeintel.cli.deps.protocols import JobManagerProtocol, ServingAccess, StorageAccess

__all__ = [
    "JobManagerProtocol",
    "ServingAccess",
    "StorageAccess",
]
