"""Typed serving backend fakes aligned with production interfaces.

These helpers replace ad-hoc ``SimpleNamespace`` backends with protocol-aligned
fakes that expose the same attributes (gateway, limits, service) and provide
deterministic close semantics for tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving.backend.pagination import BackendLimits
from tests._helpers.gateway import GatewayFactory

if TYPE_CHECKING:
    from collections.abc import Callable

    from codeintel.storage.gateway import StorageGateway


@dataclass
class ServingBackend[ServiceT]:
    """Minimal backend wrapper carrying gateway, limits, and service."""

    gateway: StorageGateway
    service: ServiceT
    limits: BackendLimits

    def close(self) -> None:
        """Close the underlying gateway."""
        self.gateway.close()


@dataclass
class BackendHandle[ServiceT]:
    """Container matching backend factory expectations in serving/MCP tests."""

    backend: ServingBackend[ServiceT]
    close: Callable[[], None]


def build_serving_backend[ServiceT](
    *,
    factory: GatewayFactory | None = None,
    service: ServiceT,
    limits: BackendLimits | None = None,
) -> BackendHandle[ServiceT]:
    """
    Build a typed backend handle with a fresh StorageGateway.

    Parameters
    ----------
    factory
        Optional GatewayFactory configured for the test; defaults to in-memory.
    service
        Service instance to expose on the backend.
    limits
        Optional pagination limits; defaults to BackendLimits().

    Returns
    -------
    BackendHandle
        Handle exposing ``backend`` and ``close()`` consistent with production factories.
    """
    gateway_factory = factory or GatewayFactory()
    gateway = gateway_factory.open()
    backend = ServingBackend(
        gateway=gateway,
        service=service,
        limits=limits or BackendLimits(),
    )
    return BackendHandle(backend=backend, close=backend.close)


__all__ = ["BackendHandle", "ServingBackend", "build_serving_backend"]
