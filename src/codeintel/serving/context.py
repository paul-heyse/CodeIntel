"""Shared serving context for HTTP and MCP runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.serving.db.manager import ServingDBManager
    from codeintel.serving.operations.ops import ServingOperations
    from codeintel.serving.semantic.kernel import SemanticQueryKernel
    from codeintel.serving.settings import ServingSettings


@dataclass(frozen=True, slots=True)
class ServingContext:
    """Constructed serving dependencies shared across transports.

    Parameters
    ----------
    settings
        Serving settings used to configure the application.
    db_manager
        Snapshot pointer + connection manager.
    kernel
        Semantic query kernel used by HTTP/MCP surfaces.
    ops
        Transport-agnostic operations facade.
    """

    settings: ServingSettings
    db_manager: ServingDBManager
    kernel: SemanticQueryKernel
    ops: ServingOperations


__all__ = ["ServingContext"]
