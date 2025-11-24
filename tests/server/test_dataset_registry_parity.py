"""Ensure dataset registry and limits wiring are consistent across backends."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.config.serving_models import ServingConfig
from codeintel.mcp.backend import DuckDBBackend
from codeintel.server.datasets import build_dataset_registry
from codeintel.services.factory import DatasetRegistryOptions, build_service_from_config
from codeintel.services.query_service import LocalQueryService
from codeintel.services.wiring import build_backend_resource
from codeintel.storage.gateway import open_memory_gateway


def test_dataset_registry_parity_across_factory_defaults() -> None:
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
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    now = "2024-01-01T00:00:00Z"
    gateway.core.insert_repo_map([(cfg.repo, cfg.commit, "{}", "{}", now)])
    svc = build_service_from_config(
        cfg,
        gateway=gateway,
        registry=DatasetRegistryOptions(tables=base_registry, validate=False),
    )
    if not isinstance(svc, LocalQueryService):
        pytest.fail("Expected LocalQueryService")
    if svc.dataset_tables != gateway.datasets.mapping:
        pytest.fail("Dataset registry mismatch")


def test_backend_resource_limits_and_registry_align() -> None:
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
    gateway = open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    now = "2024-01-01T00:00:00Z"
    gateway.core.insert_repo_map([(cfg.repo, cfg.commit, "{}", "{}", now)])
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
