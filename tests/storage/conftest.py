"""Storage-specific fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from tests._helpers import docs_views_ready_gateway
from tests._helpers.gateway import GatewayFactory
from tests._helpers.run_tracking import RunTrackingHarness, make_tracking

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.storage.gateway import StorageGateway


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
    Provide a gateway provisioned with docs view seeds for profiling tests.

    Yields
    ------
    StorageGateway
        Gateway configured with docs export seeds and view materializations.
    """
    ctx = docs_views_ready_gateway(tmp_path / "docs_views")
    try:
        yield ctx.gateway
    finally:
        ctx.close()


@pytest.fixture
def run_tracking_harness(tmp_path: Path) -> Iterator[RunTrackingHarness]:
    """
    Provide a macro-ready gateway and tracking accessor for run tracking tests.

    Yields
    ------
    RunTrackingHarness
        Harness bundling gateway, tracking accessor, and repo root.
    """
    repo_root = tmp_path / "run_repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    gateway = GatewayFactory().with_views().open()
    try:
        tracking = make_tracking(gateway.con)
        yield RunTrackingHarness(gateway=gateway, tracking=tracking, repo_root=repo_root)
    finally:
        gateway.close()
