"""Pytest fixtures shared across ingestion tests."""

from __future__ import annotations

import pytest

from tests._helpers.gateway import GatewayFactory


@pytest.fixture
def ingestion_gateway():
    """Provide a fresh gateway with schema, views, and macros applied."""
    gateway = GatewayFactory().with_macros().open()
    try:
        yield gateway
    finally:
        gateway.close()


__all__ = ["ingestion_gateway"]
