"""Tests for the debug pipeline prerequisites endpoint.

These tests verify that the `/meta/debug/pipeline/prereqs` endpoint
correctly returns prerequisite checking information for operations.
"""

from __future__ import annotations

from http import HTTPStatus

import pytest
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import BackendResource, create_app
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import build_duckdb_query_service


def _create_test_client(
    gateway: StorageGateway,
    repo: str = "demo/repo",
    commit: str = "deadbeef",
) -> TestClient:
    """Create a test client with the given gateway.

    Parameters
    ----------
    gateway
        Storage gateway with test data.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    TestClient
        FastAPI test client configured for the gateway.
    """
    limits = BackendLimits(default_limit=10, max_rows_per_call=100)
    query = build_duckdb_query_service(gateway, repo=repo, commit=commit, limits=limits)
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(gateway.datasets.mapping),
    )
    backend = DuckDBBackend(
        gateway=gateway,
        repo=repo,
        commit=commit,
        limits=limits,
        observability=None,
        service=service,
    )

    def load_config() -> ServingConfig:
        return ServingConfig(
            mode="local_db",
            repo=repo,
            commit=commit,
            api_base_url="http://test",
        )

    def backend_factory(_: ServingConfig, **__: object) -> BackendResource:
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(
        config_loader=load_config,
        backend_factory=backend_factory,
        gateway=gateway,
    )
    return TestClient(app)


# -----------------------------------------------------------------------------
# Debug Endpoint Tests
# -----------------------------------------------------------------------------


def test_debug_prereqs_endpoint_returns_structured_response(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify debug endpoint returns properly structured response.

    Parameters
    ----------
    architecture_gateway
        Gateway seeded with architecture data.
    """
    with _create_test_client(architecture_gateway) as client:
        response = client.get(
            "/meta/debug/pipeline/prereqs",
            params={"op_id": "function.summary"},
        )

    assert response.status_code == HTTPStatus.OK
    payload = response.json()

    # Check required fields exist
    assert "op_id" in payload
    assert payload["op_id"] == "function.summary"
    assert "repo" in payload
    assert "commit" in payload
    assert "required_datasets" in payload
    assert "expanded_datasets" in payload
    assert "dataset_statuses" in payload
    assert "runs_considered" in payload
    assert "data_satisfied" in payload
    assert "run_satisfied" in payload
    assert "overall_satisfied" in payload


def test_debug_prereqs_returns_404_for_unknown_operation(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify debug endpoint returns 404 for unknown operations.

    Parameters
    ----------
    architecture_gateway
        Gateway seeded with architecture data.
    """
    with _create_test_client(architecture_gateway) as client:
        response = client.get(
            "/meta/debug/pipeline/prereqs",
            params={"op_id": "nonexistent.operation"},
        )

    assert response.status_code == HTTPStatus.NOT_FOUND
    payload = response.json()
    assert "Unknown operation" in payload.get("detail", "")


def test_debug_prereqs_shows_dataset_statuses(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify debug endpoint includes dataset status information.

    Parameters
    ----------
    architecture_gateway
        Gateway seeded with architecture data.
    """
    with _create_test_client(architecture_gateway) as client:
        response = client.get(
            "/meta/debug/pipeline/prereqs",
            params={"op_id": "function.summary"},
        )

    assert response.status_code == HTTPStatus.OK
    payload = response.json()

    dataset_statuses = payload.get("dataset_statuses", [])
    # Should be a list (may be empty if no required datasets)
    assert isinstance(dataset_statuses, list)

    # If there are statuses, verify structure
    for status in dataset_statuses:
        assert "table_key" in status
        assert "name" in status
        assert "has_rows" in status
        assert "checked" in status


def test_debug_prereqs_uses_config_defaults(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify debug endpoint uses config repo/commit when not provided.

    Parameters
    ----------
    architecture_gateway
        Gateway seeded with architecture data.
    """
    with _create_test_client(architecture_gateway, repo="test/repo", commit="abc123") as client:
        response = client.get(
            "/meta/debug/pipeline/prereqs",
            params={"op_id": "function.summary"},
        )

    assert response.status_code == HTTPStatus.OK
    payload = response.json()

    # Should use config defaults
    assert payload["repo"] == "test/repo"
    assert payload["commit"] == "abc123"


def test_debug_prereqs_accepts_custom_repo_commit(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify debug endpoint accepts custom repo/commit params.

    Parameters
    ----------
    architecture_gateway
        Gateway seeded with architecture data.
    """
    with _create_test_client(architecture_gateway) as client:
        response = client.get(
            "/meta/debug/pipeline/prereqs",
            params={
                "op_id": "function.summary",
                "repo": "custom/repo",
                "commit": "custom123",
            },
        )

    assert response.status_code == HTTPStatus.OK
    payload = response.json()

    # Should use custom values
    assert payload["repo"] == "custom/repo"
    assert payload["commit"] == "custom123"


def test_debug_prereqs_returns_boolean_satisfaction_flags(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify debug endpoint returns boolean satisfaction flags.

    Parameters
    ----------
    architecture_gateway
        Gateway seeded with architecture data.
    """
    with _create_test_client(architecture_gateway) as client:
        response = client.get(
            "/meta/debug/pipeline/prereqs",
            params={"op_id": "function.summary"},
        )

    assert response.status_code == HTTPStatus.OK
    payload = response.json()

    # Satisfaction flags should be booleans
    assert isinstance(payload["data_satisfied"], bool)
    assert isinstance(payload["run_satisfied"], bool)
    assert isinstance(payload["overall_satisfied"], bool)


def test_debug_prereqs_runs_considered_is_list(
    architecture_gateway: StorageGateway,
) -> None:
    """Verify runs_considered is always a list.

    Parameters
    ----------
    architecture_gateway
        Gateway seeded with architecture data.
    """
    with _create_test_client(architecture_gateway) as client:
        response = client.get(
            "/meta/debug/pipeline/prereqs",
            params={"op_id": "function.summary"},
        )

    assert response.status_code == HTTPStatus.OK
    payload = response.json()

    runs_considered = payload.get("runs_considered", [])
    assert isinstance(runs_considered, list)


@pytest.mark.parametrize(
    "op_id",
    [
        "function.summary",
        "file.summary",
        "architecture.module",
        "graph.call_neighbors",
    ],
)
def test_debug_prereqs_works_for_various_operations(
    architecture_gateway: StorageGateway,
    op_id: str,
) -> None:
    """Verify debug endpoint works for various operation types.

    Parameters
    ----------
    architecture_gateway
        Gateway seeded with architecture data.
    op_id
        Operation identifier to test.
    """
    with _create_test_client(architecture_gateway) as client:
        response = client.get(
            "/meta/debug/pipeline/prereqs",
            params={"op_id": op_id},
        )

    # Should return 200 for known operations
    assert response.status_code == HTTPStatus.OK
    payload = response.json()
    assert payload["op_id"] == op_id
