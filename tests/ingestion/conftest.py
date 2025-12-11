"""Pytest fixtures shared across ingestion tests."""

from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace

import pytest

from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import GatewayFactory
from tests._helpers.ingestion import (
    build_ingestion_context_bundle,
    build_scan_profile,
    closing_gateway,
    create_scan_step,
)
from tests._helpers.orchestration.tooling import tooling_outputs_session


@pytest.fixture
def ingestion_gateway_factory() -> GatewayFactory:
    """Provide a gateway factory with schema/views applied (no macros).

    Returns
    -------
    GatewayFactory
        Factory preconfigured for ingestion tests.
    """
    return GatewayFactory()


@pytest.fixture
def ingestion_gateway() -> Generator[StorageGateway]:
    """Provide a fresh gateway with schema and views applied (no macros).

    Yields
    ------
    StorageGateway
        Gateway instance opened for tests.
    """
    gateway = GatewayFactory().open()
    with closing_gateway(gateway):
        yield gateway


@pytest.fixture
def ingestion_ctx_bundle(
    tmp_path_factory: pytest.TempPathFactory,
) -> Generator[SimpleNamespace]:
    """Provision a reusable ingestion context with repo, gateway, and scan wiring.

    Yields
    ------
    SimpleNamespace
        Namespace containing repo_root, gateway, profile, scan_step, storage,
        discovery, change_detection, ctx, module_paths, and tools.
    """
    tmp_root = tmp_path_factory.mktemp("ingestion")
    bundle = build_ingestion_context_bundle(tmp_root)
    profile = build_scan_profile(bundle.repo_root)
    scan_root = tmp_path_factory.mktemp("scan")
    scan_step, _, _ = create_scan_step(bundle.gateway, bundle.repo_root, scan_root)
    ctx_ns = SimpleNamespace(
        repo_root=bundle.repo_root,
        gateway=bundle.gateway,
        profile=profile,
        scan_step=scan_step,
        storage=bundle.storage,
        discovery=bundle.discovery,
        change_detection=bundle.change_detection,
        ctx=bundle.ctx,
        module_paths=bundle.module_paths,
        tools=bundle.tools,
    )
    with closing_gateway(bundle.gateway):
        yield ctx_ns


__all__ = [
    "ingestion_ctx_bundle",
    "ingestion_gateway",
    "ingestion_gateway_factory",
    "tooling_outputs_session",
]
