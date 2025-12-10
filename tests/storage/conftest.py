"""Storage-specific fixtures."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import GatewayFactory


@pytest.fixture
def macro_gateway(fresh_gateway: StorageGateway) -> StorageGateway:
    """
    Provide a gateway with ingest macros ensured.

    Returns
    -------
    StorageGateway
        Macro-ready gateway instance.
    """
    return fresh_gateway


@pytest.fixture
def schema_gateway() -> Iterator[StorageGateway]:
    """
    Provide a schema-ready in-memory gateway with validation and views.

    Yields
    ------
    StorageGateway
        Gateway configured with schemas, views, and validation.
    """
    gateway = GatewayFactory().with_views().with_validation().strict().open()
    try:
        yield gateway
    finally:
        gateway.close()


@pytest.fixture
def relaxed_schema_gateway() -> Iterator[StorageGateway]:
    """
    Provide a relaxed schema gateway (validation off, strict_schema disabled).

    Yields
    ------
    StorageGateway
        Gateway configured without strict schema enforcement.
    """
    gateway = GatewayFactory().with_views().without_validation().relaxed().open()
    try:
        yield gateway
    finally:
        gateway.close()
