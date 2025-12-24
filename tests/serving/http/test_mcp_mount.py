"""Tests for MCP mount path contract."""

from __future__ import annotations

from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
)
from tests._helpers.harnesses.serving_app import ServingAppHarness
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory


def test_mcp_health_at_mcp_path(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """Verify MCP health endpoint is at /mcp/health, not /mcp/mcp/health."""
    snapshot = serving_snapshot_factory.demo_snapshot()
    harness = ServingAppHarness.from_snapshot(snapshot)
    with harness.http_client(mount_mcp=True) as client:
        # /mcp/health should respond (200 or 503 if no snapshot)
        response = client.get("/mcp/health")
        expect_in(response.status_code, {200, 503})

        # /mcp/mcp/health should NOT exist (404)
        response_double = client.get("/mcp/mcp/health")
        expect_equal(response_double.status_code, 404)


def test_mcp_ready_at_mcp_path(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """Verify MCP ready endpoint is at /mcp/ready."""
    snapshot = serving_snapshot_factory.demo_snapshot()
    harness = ServingAppHarness.from_snapshot(snapshot)
    with harness.http_client(mount_mcp=True) as client:
        # /mcp/ready should respond
        response = client.get("/mcp/ready")
        expect_in(response.status_code, {200, 503})

        # /mcp/mcp/ready should NOT exist
        response_double = client.get("/mcp/mcp/ready")
        expect_equal(response_double.status_code, 404)


def test_mcp_not_mounted_when_disabled(
    serving_snapshot_factory: ServingSnapshotFactory,
) -> None:
    """Verify MCP routes are absent when mount_mcp=False."""
    snapshot = serving_snapshot_factory.demo_snapshot()
    harness = ServingAppHarness.from_snapshot(snapshot)
    with harness.http_client(mount_mcp=False) as client:
        # Both paths should 404 when MCP is not mounted
        response = client.get("/mcp/health")
        expect_equal(response.status_code, 404)
