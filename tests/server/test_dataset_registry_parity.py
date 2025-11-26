"""Ensure dataset registry and limits wiring are consistent across backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.http.datasets import build_dataset_registry
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.factory import DatasetRegistryOptions, build_service_from_config
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.serving.services.wiring import build_backend_resource
from codeintel.storage.gateway import StorageGateway
from tests._helpers.builders import RepoMapRow, insert_repo_map


def _seed_repo_identity(gateway: StorageGateway, repo: str, commit: str) -> None:
    insert_repo_map(
        gateway,
        [
            RepoMapRow(
                repo=repo,
                commit=commit,
                modules={},
                overlays={},
            )
        ],
    )


def test_dataset_registry_parity_across_factory_defaults(fresh_gateway: StorageGateway) -> None:
    """Local factory should expose the canonical dataset registry by default."""
    base_registry = build_dataset_registry()
    cfg = ServingConfig(
        mode="local_db",
        repo="r",
        commit="c",
        repo_root=Path.cwd(),
        db_path=Path(":memory:"),
        read_only=True,
        default_limit=50,
        max_rows_per_call=500,
    )
    gateway = fresh_gateway
    _seed_repo_identity(gateway, cfg.repo, cfg.commit)
    svc = build_service_from_config(
        cfg,
        gateway=gateway,
        registry=DatasetRegistryOptions(tables=base_registry, validate=False),
    )
    if not isinstance(svc, LocalQueryService):
        pytest.fail("Expected LocalQueryService")
    if svc.dataset_tables != gateway.datasets.mapping:
        pytest.fail("Dataset registry mismatch")


def test_backend_resource_limits_and_registry_align(fresh_gateway: StorageGateway) -> None:
    """BackendResource should carry registry and limits derived from config."""
    base_registry = build_dataset_registry()
    cfg = ServingConfig(
        mode="local_db",
        repo="r",
        commit="c",
        repo_root=Path.cwd(),
        db_path=Path(":memory:"),
        read_only=True,
        default_limit=50,
        max_rows_per_call=500,
    )
    gateway = fresh_gateway
    _seed_repo_identity(gateway, cfg.repo, cfg.commit)
    resource = build_backend_resource(
        cfg,
        gateway=gateway,
        registry=DatasetRegistryOptions(tables=base_registry, validate=False),
    )
    backend = resource.backend
    if not isinstance(backend, DuckDBBackend):
        pytest.fail("Expected DuckDBBackend for local_db mode")
    limits = backend.limits
    if limits.default_limit != cfg.default_limit:
        pytest.fail("default_limit mismatch")
    if limits.max_rows_per_call != cfg.max_rows_per_call:
        pytest.fail("max_rows_per_call mismatch")
    if backend.gateway.datasets.mapping != gateway.datasets.mapping:
        pytest.fail("backend registry mismatch")
