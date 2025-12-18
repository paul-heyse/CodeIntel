"""MCP protocol definitions.

Re-exports the transport-agnostic serving protocols from
``codeintel.serving.operations.protocols``.
"""

from codeintel.serving.operations.protocols import (
    ServingDBManagerProtocol,
    ServingSnapshotPointerProtocol,
)
from codeintel.serving.operations.protocols import (
    ServingKernelProtocol as SemanticKernelProtocol,
)

__all__ = [
    "SemanticKernelProtocol",
    "ServingDBManagerProtocol",
    "ServingSnapshotPointerProtocol",
]
