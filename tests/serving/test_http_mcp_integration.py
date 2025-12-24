"""Integration tests for the combined FastAPI + MCP serving app."""

from __future__ import annotations

from fastapi.testclient import TestClient
from starlette.routing import Mount

from codeintel.serving.http.app import create_serving_app
from codeintel.serving.settings import ServingSettings
from tests._helpers.assertions import assert_http_success, assert_target_ok
from tests._helpers.assertions.expectation_assertions import expect_in
from tests._helpers.harnesses.serving_harness import ServingTargetHarness


def _publish_serving_snapshot(serving_target_harness: ServingTargetHarness) -> None:
    records = serving_target_harness.run_targets()
    assert_target_ok(records["serving_artifacts"])
    serving_target_harness.publish_snapshot(run_id="run-1")


def test_fastapi_app_mounts_mcp(serving_target_harness: ServingTargetHarness) -> None:
    """create_serving_app mounts the MCP app when enabled."""
    _publish_serving_snapshot(serving_target_harness)
    serve_dir = serving_target_harness.harness.ctx.build_paths.build_dir / "serving"

    settings = ServingSettings(serve_dir=serve_dir, pool_size=1, poll_interval_s=0.01)
    app = create_serving_app(settings=settings, mount_mcp=True)

    mount_paths = {route.path for route in app.routes if isinstance(route, Mount)}
    expect_in("/mcp", mount_paths)

    with TestClient(app) as client:
        _ = assert_http_success(client, "/v1/semantic/views")
