"""Tests for MCP mount path contract."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from fastapi.testclient import TestClient

from codeintel.config.primitives import BuildPaths
from codeintel.serving.db.pointer import ServingSnapshotPointer
from codeintel.serving.http.app import create_serving_app
from codeintel.serving.settings import ServingSettings
from tests._helpers.assertions.expectation_assertions import (
    expect_equal,
    expect_in,
)
from tests._helpers.gateway import GatewayFactory
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts

if TYPE_CHECKING:
    from pathlib import Path


def _write_pointer(tmp_path: Path, *, repo: str, commit: str) -> None:
    db_path = tmp_path / "codeintel.duckdb"
    gateway = GatewayFactory().file_backed(db_path).open()
    gateway.close()

    artifacts = HarnessArtifacts(
        repo_root=tmp_path,
        paths=BuildPaths.from_explicit(build_dir=tmp_path),
    )
    semantic_registry_path = artifacts.write_semantic_registry(
        path=tmp_path / "semantic_registry.json"
    )
    schema_manifest_path = artifacts.write_schema_manifest(path=tmp_path / "schema_manifest.json")
    buildspec_path = artifacts.write_buildspec(path=tmp_path / "buildspec.json")

    pointer = ServingSnapshotPointer(
        db_path=db_path,
        semantic_registry_path=semantic_registry_path,
        schema_manifest_path=schema_manifest_path,
        buildspec_path=buildspec_path,
        repo=repo,
        commit=commit,
        run_id="run-1",
        published_at=datetime.now(tz=UTC),
        semantic_layer_version="v1",
    )
    (tmp_path / "current.json").write_text(pointer.to_json(), encoding="utf-8")


def test_mcp_health_at_mcp_path(tmp_path: Path) -> None:
    """Verify MCP health endpoint is at /mcp/health, not /mcp/mcp/health."""
    _write_pointer(tmp_path, repo="test", commit="abc123")
    settings = ServingSettings(serve_dir=tmp_path)
    app = create_serving_app(settings, mount_mcp=True)
    client = TestClient(app, raise_server_exceptions=False)

    # /mcp/health should respond (200 or 503 if no snapshot)
    response = client.get("/mcp/health")
    expect_in(response.status_code, {200, 503})

    # /mcp/mcp/health should NOT exist (404)
    response_double = client.get("/mcp/mcp/health")
    expect_equal(response_double.status_code, 404)


def test_mcp_ready_at_mcp_path(tmp_path: Path) -> None:
    """Verify MCP ready endpoint is at /mcp/ready."""
    _write_pointer(tmp_path, repo="test", commit="def456")
    settings = ServingSettings(serve_dir=tmp_path)
    app = create_serving_app(settings, mount_mcp=True)
    client = TestClient(app, raise_server_exceptions=False)

    # /mcp/ready should respond
    response = client.get("/mcp/ready")
    expect_in(response.status_code, {200, 503})

    # /mcp/mcp/ready should NOT exist
    response_double = client.get("/mcp/mcp/ready")
    expect_equal(response_double.status_code, 404)


def test_mcp_not_mounted_when_disabled(tmp_path: Path) -> None:
    """Verify MCP routes are absent when mount_mcp=False."""
    settings = ServingSettings(serve_dir=tmp_path)
    app = create_serving_app(settings, mount_mcp=False)
    client = TestClient(app, raise_server_exceptions=False)

    # Both paths should 404 when MCP is not mounted
    response = client.get("/mcp/health")
    expect_equal(response.status_code, 404)
