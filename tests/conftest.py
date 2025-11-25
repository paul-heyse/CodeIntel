"""Pytest configuration for the CodeIntel test suite."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.storage.gateway import StorageGateway
from tests._helpers.architecture import open_seeded_architecture_gateway
from tests._helpers.fixtures import (
    GatewayOptions,
    ProvisionedGateway,
    ProvisioningConfig,
    provision_docs_export_ready,
    provision_graph_ready_repo,
    provision_ingested_repo,
    provisioned_gateway,
)


@pytest.fixture
def fresh_gateway(tmp_path: Path) -> Iterator[StorageGateway]:
    """Provide an in-memory gateway with schema and views applied.

    Yields
    ------
    StorageGateway
        Gateway configured with schemas/views; caller must not close.
    """
    with provisioned_gateway(
        tmp_path / "fresh",
        config=ProvisioningConfig(run_ingestion=False),
    ) as ctx:
        yield ctx.gateway


@pytest.fixture
def provisioned_repo(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a repo-root and ingest baseline data via production entry points.

    Yields
    ------
    ProvisionedGateway
        Gateway plus repo root populated with baseline ingestion data.
    """
    with provision_ingested_repo(tmp_path / "repo") as ctx:
        yield ctx


@pytest.fixture
def graph_ready_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a repo with graph metrics seeds for graph-centric tests.

    Yields
    ------
    ProvisionedGateway
        Gateway plus repo context seeded with graph metrics data.
    """
    with provision_graph_ready_repo(tmp_path / "repo") as ctx:
        yield ctx


@pytest.fixture
def docs_export_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a gateway ready for docs export scenarios.

    Yields
    ------
    ProvisionedGateway
        Gateway populated with docs export seeds.
    """
    ctx = provision_docs_export_ready(tmp_path, repo="demo/repo", commit="deadbeef")
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def ingestion_only_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Provision a gateway without ingestion for custom seeding.

    Yields
    ------
    ProvisionedGateway
        Gateway prepared with schemas but without ingestion.
    """
    with provisioned_gateway(
        tmp_path / "repo",
        config=ProvisioningConfig(run_ingestion=False),
    ) as ctx:
        yield ctx


@pytest.fixture
def loose_gateway(tmp_path: Path) -> Iterator[ProvisionedGateway]:
    """Opt-out gateway for tests that intentionally drift schemas.

    Yields
    ------
    ProvisionedGateway
        Gateway configured without strict schema enforcement.
    """
    with provisioned_gateway(
        tmp_path / "repo",
        config=ProvisioningConfig(
            run_ingestion=False, gateway_options=GatewayOptions(strict_schema=False)
        ),
    ) as ctx:
        yield ctx


@pytest.fixture
def architecture_gateway(tmp_path: Path) -> Iterator[StorageGateway]:
    """Provide a gateway seeded with architecture data (subsystems, call/import graphs).

    Yields
    ------
    StorageGateway
        Gateway configured with architecture dataset seeds.
    """
    gateway = open_seeded_architecture_gateway(
        repo="demo/repo",
        commit="deadbeef",
        db_path=tmp_path / "arch.duckdb",
        strict_schema=True,
    )
    try:
        yield gateway
    finally:
        gateway.close()


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
