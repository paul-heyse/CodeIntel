"""Transport-agnostic serving operations layer."""

from codeintel.serving.operations.ops import ServingOperations
from codeintel.serving.operations.protocols import (
    ServingDBManagerProtocol,
    ServingKernelProtocol,
    ServingSnapshotPointerProtocol,
)

__all__ = [
    "ServingDBManagerProtocol",
    "ServingKernelProtocol",
    "ServingOperations",
    "ServingSnapshotPointerProtocol",
]
