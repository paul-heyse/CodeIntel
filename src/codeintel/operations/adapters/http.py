"""HTTP adapter for operations.

Generates FastAPI routes from the operation registry,
enabling HTTP access to all registered operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from codeintel.operations.registry import OperationRegistry, get_default_registry

if TYPE_CHECKING:
    from pathlib import Path

    from fastapi import FastAPI

    from codeintel.operations.base import OperationSpec


LOG = logging.getLogger(__name__)


@dataclass
class HttpAdapter:
    """Adapt operations to FastAPI HTTP routes.

    The adapter iterates over the operation registry and generates
    HTTP endpoints for each operation.

    Parameters
    ----------
    app
        FastAPI application.
    registry
        Operation registry (defaults to global).
    prefix
        URL prefix for all routes (e.g., "/api/v1").
    """

    app: FastAPI
    registry: OperationRegistry = field(default_factory=get_default_registry)
    prefix: str = "/api/v1"
    _db_path: Path | None = field(default=None, repr=False)

    def register_all(self) -> None:
        """Register all operations as HTTP routes.

        Creates endpoints for each operation, using POST for write
        operations and GET for read-only operations.
        """
        for spec in self.registry.list_operations(include_hidden=False):
            self._register_operation(spec)

    def _register_operation(self, spec: OperationSpec) -> None:
        """Register a single operation as an HTTP endpoint.

        Parameters
        ----------
        spec
            Operation specification.
        """
        _ = self  # Instance method for adapter pattern
        # Placeholder - full implementation will create routes
        LOG.debug("Would register HTTP route: %s", spec.operation_id)


def register_operations_with_fastapi(app: FastAPI, *, prefix: str = "/api/v1") -> HttpAdapter:
    """Register all operations with a FastAPI app.

    Parameters
    ----------
    app
        FastAPI application.
    prefix
        URL prefix for all routes.

    Returns
    -------
    HttpAdapter
        The configured adapter.
    """
    adapter = HttpAdapter(app, prefix=prefix)
    adapter.register_all()
    return adapter


__all__ = [
    "HttpAdapter",
    "register_operations_with_fastapi",
]
