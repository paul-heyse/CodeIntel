"""Typed FastAPI state for the serving application.

FastAPI exposes a dynamic ``app.state`` container. This module provides a typed
container we attach under ``app.state.serving`` so dependencies can remain
type-safe and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.serving.db.manager import ServingDBManager
    from codeintel.serving.operations.ops import ServingOperations
    from codeintel.serving.semantic.kernel import SemanticQueryKernel
    from codeintel.serving.settings import ServingSettings


@dataclass(frozen=True, slots=True)
class ServingState:
    """Application-scoped serving state.

    Parameters
    ----------
    settings
        Serving settings used to configure the application.
    db
        Snapshot pointer + connection manager.
    kernel
        Semantic query kernel used by HTTP/MCP surfaces.
    ops
        Transport-agnostic operations facade.
    """

    settings: ServingSettings
    db: ServingDBManager
    kernel: SemanticQueryKernel
    ops: ServingOperations


__all__ = ["ServingState"]
