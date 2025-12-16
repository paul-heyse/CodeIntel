"""Environment-driven serving configuration.

This module defines a small, serving-only configuration surface used by the
semantic-first serving stack.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ServingSettings:
    """Serving layer configuration loaded from environment variables.

    Parameters
    ----------
    serve_dir
        Root directory for serving snapshots.
    hot_swap
        Enable automatic snapshot hot-swap on pointer change.
    pool_size
        Number of read-only DuckDB connections per worker.
    poll_interval_s
        Seconds between pointer file checks when hot_swap enabled.
    mcp_transport
        MCP transport mode: "stdio" or "http".
    host
        HTTP server bind address.
    port
        HTTP server port.
    auth_token
        Optional bearer token for remote serving.
    schema_enforcement
        Schema enforcement mode for semantic query columns: "strict", "warn", or "off".
    result_engine
        Result extraction engine: "polars" (preferred) or "pandas" (fallback).
    api_key
        Optional API key required by HTTP routes when set.
    cors_origins
        Optional CORS origins for browser clients (comma-separated in env).
    trusted_hosts
        Optional allowed hosts list for TrustedHostMiddleware (comma-separated in env).
    gzip_minimum_size
        Minimum response size (bytes) to enable gzip compression.
    enable_gzip
        Whether to enable gzip compression middleware.
    """

    serve_dir: Path
    hot_swap: bool = True
    pool_size: int = 4
    poll_interval_s: float = 1.0
    mcp_transport: str = "stdio"
    host: str = "127.0.0.1"
    port: int = 8000
    auth_token: str | None = None
    schema_enforcement: str = "strict"
    result_engine: str = "polars"
    api_key: str | None = None
    cors_origins: tuple[str, ...] = ()
    trusted_hosts: tuple[str, ...] = ()
    gzip_minimum_size: int = 500
    enable_gzip: bool = True

    @classmethod
    def from_env(cls) -> ServingSettings:
        """Load settings from environment variables.

        Returns
        -------
        ServingSettings
            Loaded settings.
        """
        serve_dir = Path(os.environ.get("CODEINTEL_SERVE_DIR", ".codeintel/serve")).resolve()
        cors_origins = _split_csv(os.environ.get("CODEINTEL_SERVE_CORS_ORIGINS", ""))
        trusted_hosts = _split_csv(os.environ.get("CODEINTEL_SERVE_TRUSTED_HOSTS", ""))
        return cls(
            serve_dir=serve_dir,
            hot_swap=os.environ.get("CODEINTEL_SERVE_HOTSWAP", "1") == "1",
            pool_size=int(os.environ.get("CODEINTEL_SERVE_POOL_SIZE", "4")),
            poll_interval_s=float(os.environ.get("CODEINTEL_SERVE_POLL_INTERVAL", "1.0")),
            mcp_transport=os.environ.get("CODEINTEL_MCP_TRANSPORT", "stdio"),
            host=os.environ.get("CODEINTEL_HOST", "127.0.0.1"),
            port=int(os.environ.get("CODEINTEL_PORT", "8000")),
            auth_token=os.environ.get("CODEINTEL_AUTH_TOKEN"),
            schema_enforcement=os.environ.get("CODEINTEL_SERVE_SCHEMA_ENFORCEMENT", "strict"),
            result_engine=os.environ.get("CODEINTEL_SERVE_RESULT_ENGINE", "polars"),
            api_key=os.environ.get("CODEINTEL_SERVE_API_KEY"),
            cors_origins=cors_origins,
            trusted_hosts=trusted_hosts,
            gzip_minimum_size=int(os.environ.get("CODEINTEL_SERVE_GZIP_MIN_SIZE", "500")),
            enable_gzip=os.environ.get("CODEINTEL_SERVE_GZIP", "1") == "1",
        )


__all__ = ["ServingSettings"]


def _split_csv(raw: str) -> tuple[str, ...]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(items)
