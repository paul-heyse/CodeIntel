"""Storage-specific fixtures."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.storage.gateway import StorageGateway
from tests._helpers import docs_views_ready_gateway
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


@pytest.fixture
def docs_views_gateway(tmp_path: Path) -> Iterator[StorageGateway]:
    """
    Provide a gateway provisioned with docs view seeds for profiling/coverage tests.

    Yields
    ------
    StorageGateway
        Gateway configured with docs export seeds and coverage-ready views.
    """
    ctx = docs_views_ready_gateway(tmp_path / "docs_views")
    try:
        yield ctx.gateway
    finally:
        ctx.close()
