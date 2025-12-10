"""Shared builders for LocalQueryService-backed FastAPI apps in tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI
from fastapi.testclient import TestClient

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import BackendResource, create_app
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.observability import ServiceObservability
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import build_duckdb_query_service
from tests._helpers.serving_harnesses import RecordingObservability

DEFAULT_LIMIT = 10
MAX_ROWS = 100


@dataclass(frozen=True)
class ServiceApp:
    """Aggregated service, backend, and app for delegate tests."""

    gateway: StorageGateway
    repo: str
    commit: str
    limits: BackendLimits
    service: LocalQueryService
    backend: DuckDBBackend
    app: FastAPI
    observability: ServiceObservability

    @contextmanager
    def client(self) -> Iterator[TestClient]:
        """Provide a TestClient bound to the configured app."""
        with TestClient(self.app) as test_client:
            yield test_client


def build_service_app(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    limits: BackendLimits | None = None,
    config_overrides: dict[str, Any] | None = None,
    observability: ServiceObservability | None = None,
) -> ServiceApp:
    """Construct a LocalQueryService, DuckDB backend, and FastAPI app for tests."""
    effective_limits = limits or BackendLimits(
        default_limit=DEFAULT_LIMIT,
        max_rows_per_call=MAX_ROWS,
    )
    effective_observability = observability or RecordingObservability()
    query = build_duckdb_query_service(
        gateway,
        repo=repo,
        commit=commit,
        limits=effective_limits,
    )
    service = LocalQueryService(
        query=query,
        dataset_tables=dict(gateway.datasets.mapping),
        observability=effective_observability,
    )
    backend = DuckDBBackend(
        gateway=gateway,
        repo=repo,
        commit=commit,
        limits=effective_limits,
        observability=effective_observability,
        service=service,
    )

    def load_config() -> ServingConfig:
        cfg_kwargs: dict[str, Any] = {
            "mode": "remote_api",
            "repo": repo,
            "commit": commit,
            "api_base_url": "http://test",
        }
        if config_overrides:
            cfg_kwargs.update(config_overrides)
        return ServingConfig(**cfg_kwargs)

    def backend_factory(cfg: ServingConfig, **_kwargs: object) -> BackendResource:
        _ = cfg
        return BackendResource(backend=backend, service=service, close=lambda: None)

    app = create_app(config_loader=load_config, backend_factory=backend_factory)

    return ServiceApp(
        gateway=gateway,
        repo=repo,
        commit=commit,
        limits=effective_limits,
        service=service,
        backend=backend,
        app=app,
        observability=effective_observability,
    )


__all__ = ["ServiceApp", "build_service_app"]
