"""Pytest configuration for the CodeIntel test suite."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.storage.gateway import StorageGateway, open_memory_gateway
from tests._helpers.fixtures import ProvisionedGateway, provision_ingested_repo


@pytest.fixture
def fresh_gateway() -> Iterator[StorageGateway]:
    """Provide an in-memory gateway with schema and views applied.

    Yields
    ------
    StorageGateway
        Gateway configured with schemas/views; caller must not close.
    """
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    try:
        yield gateway
    finally:
        gateway.close()


@pytest.fixture
def provisioned_repo(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a repo-root and ingest baseline data via production entry points.

    Yields
    ------
    ProvisionedGateway
        Gateway plus repo root populated with baseline ingestion data.
    """
    ctx = provision_ingested_repo(tmp_path / "repo")
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def codeintel_env() -> Iterator[None]:
    """Snapshot CODEINTEL_* environment variables and restore after the test."""
    prior = {key: value for key, value in os.environ.items() if key.startswith("CODEINTEL_")}
    try:
        yield
    finally:
        for key in list(os.environ.keys()):
            if key.startswith("CODEINTEL_") and key not in prior:
                os.environ.pop(key, None)
        for key, value in prior.items():
            os.environ[key] = value
