"""Integration tests for the combined FastAPI + MCP serving app."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from starlette.routing import Mount

from tests._helpers.assertions import assert_http_success
from tests._helpers.assertions.expectation_assertions import expect_in
from tests._helpers.harnesses.serving_app import ServingAppHarness
from tests._helpers.serving_snapshot_factory import ServingSnapshotFactory

if TYPE_CHECKING:
    from fastapi import FastAPI


def test_fastapi_app_mounts_mcp(serving_snapshot_factory: ServingSnapshotFactory) -> None:
    """create_serving_app mounts the MCP app when enabled."""
    snapshot = serving_snapshot_factory.demo_snapshot()
    harness = ServingAppHarness.from_snapshot(snapshot)
    with harness.http_client(mount_mcp=True) as client:
        app = cast("FastAPI", client.app)
        mount_paths = {route.path for route in app.routes if isinstance(route, Mount)}
        expect_in("/mcp", mount_paths)
        _ = assert_http_success(client, "/v1/semantic/views")
