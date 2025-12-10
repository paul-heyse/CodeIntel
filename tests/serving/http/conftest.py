"""Shared fixtures for HTTP route tests."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from fastapi import FastAPI

from codeintel.config.serving_models import ServingConfig
from codeintel.serving.backend import BackendLimits
from codeintel.serving.http.fastapi import BackendResource, create_app
from codeintel.serving.mcp.backend import DuckDBBackend
from codeintel.serving.services.query_service import LocalQueryService
from codeintel.storage.gateway import StorageGateway
from tests._helpers.gateway import build_duckdb_query_service


@pytest.fixture
def make_http_app() -> Callable[..., FastAPI]:
    """Build a FastAPI app factory wired to a DuckDB backend for HTTP route tests.

    Returns
    -------
    Callable[..., FastAPI]
        Factory that constructs configured FastAPI applications given gateway, snapshot,
        optional limits, config overrides, and auto-pipeline settings.
    """

    def _build(
        *,
        gateway: StorageGateway,
        snapshot: tuple[str, str],
        limits: BackendLimits | None = None,
        config_overrides: dict[str, Any] | None = None,
        auto_pipeline: bool = False,
    ) -> FastAPI:
        repo, commit = snapshot
        effective_limits = limits or BackendLimits(default_limit=10, max_rows_per_call=100)
        query = build_duckdb_query_service(
            gateway,
            repo=repo,
            commit=commit,
            limits=effective_limits,
        )
        service = LocalQueryService(
            query=query,
            dataset_tables=dict(gateway.datasets.mapping),
        )
        backend = DuckDBBackend(
            gateway=gateway,
            repo=repo,
            commit=commit,
            limits=effective_limits,
            observability=None,
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

        def backend_factory(_cfg: ServingConfig, **_kwargs: object) -> BackendResource:
            return BackendResource(backend=backend, service=service, close=lambda: None)

        return create_app(
            config_loader=load_config,
            backend_factory=backend_factory,
            auto_pipeline=auto_pipeline,
        )

    return _build


__all__ = ["make_http_app"]
