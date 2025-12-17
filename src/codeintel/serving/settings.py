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
    export_max_rows
        Maximum rows allowed for export endpoints.
    enable_export_endpoints
        Whether to enable the /export endpoints.
    mcp_enable_sampling
        Enable LLM sampling in MCP tools for large result summarization.
    mcp_sample_threshold
        Row count threshold above which LLM sampling is triggered.
    mcp_progress_reporting
        Enable progress updates in MCP tools via Context.report_progress().
    mcp_mask_errors
        Mask internal error details in MCP tool responses for security.
    mcp_max_concurrent_queries
        Maximum concurrent heavy queries allowed (for memory protection).
    mcp_enable_event_store
        Enable EventStore for SSE polling/resumability on HTTP transport.
    mcp_retry_interval_ms
        SSE retry interval in milliseconds for reconnecting clients.
    uvicorn_workers
        Number of Uvicorn worker processes. Use >1 for production.
    uvicorn_loop
        Event loop implementation: "auto", "asyncio", or "uvloop".
    uvicorn_http
        HTTP protocol implementation: "auto", "h11", or "httptools".
    uvicorn_limit_concurrency
        Maximum concurrent connections (None for unlimited).
    uvicorn_limit_max_requests
        Maximum requests per worker before restart (None for unlimited).
    uvicorn_timeout_keep_alive
        Keep-alive timeout in seconds.
    uvicorn_backlog
        Maximum pending connections in socket backlog.
    uvicorn_access_log
        Enable access logging.
    uvicorn_server_header
        Include Server header in responses (disable for security).
    uvicorn_proxy_headers
        Trust proxy headers (X-Forwarded-*) from allowed IPs.
    uvicorn_forwarded_allow_ips
        Comma-separated list of IPs allowed to set proxy headers.
    auth_required_for_remote
        Require authentication when binding to non-localhost interfaces.
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
    export_max_rows: int = 100_000
    enable_export_endpoints: bool = True

    # MCP Context Features
    mcp_enable_sampling: bool = False
    mcp_sample_threshold: int = 500
    mcp_progress_reporting: bool = True

    # MCP Error Handling
    mcp_mask_errors: bool = True

    # MCP Query Concurrency Control
    mcp_max_concurrent_queries: int = 2

    # MCP EventStore for SSE Resumability
    mcp_enable_event_store: bool = True
    mcp_retry_interval_ms: int = 1000

    # Uvicorn Production Configuration
    uvicorn_workers: int = 1
    uvicorn_loop: str = "auto"
    uvicorn_http: str = "auto"
    uvicorn_limit_concurrency: int | None = None
    uvicorn_limit_max_requests: int | None = None
    uvicorn_timeout_keep_alive: int = 30
    uvicorn_backlog: int = 2048
    uvicorn_access_log: bool = True
    uvicorn_server_header: bool = False
    uvicorn_proxy_headers: bool = False
    uvicorn_forwarded_allow_ips: str = "127.0.0.1"

    # Security: Auth Enforcement
    auth_required_for_remote: bool = True

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
            export_max_rows=int(os.environ.get("CODEINTEL_SERVE_EXPORT_MAX_ROWS", "100000")),
            enable_export_endpoints=os.environ.get("CODEINTEL_SERVE_ENABLE_EXPORT", "1") == "1",
            # MCP Context Features
            mcp_enable_sampling=os.environ.get("CODEINTEL_MCP_ENABLE_SAMPLING", "0") == "1",
            mcp_sample_threshold=int(os.environ.get("CODEINTEL_MCP_SAMPLE_THRESHOLD", "500")),
            mcp_progress_reporting=os.environ.get("CODEINTEL_MCP_PROGRESS", "1") == "1",
            # MCP Error Handling
            mcp_mask_errors=os.environ.get("CODEINTEL_MCP_MASK_ERRORS", "1") == "1",
            # MCP Query Concurrency Control
            mcp_max_concurrent_queries=int(
                os.environ.get("CODEINTEL_MCP_MAX_CONCURRENT_QUERIES", "2")
            ),
            # MCP EventStore for SSE Resumability
            mcp_enable_event_store=os.environ.get("CODEINTEL_MCP_EVENT_STORE", "1") == "1",
            mcp_retry_interval_ms=int(os.environ.get("CODEINTEL_MCP_RETRY_INTERVAL", "1000")),
            # Uvicorn Production Configuration
            uvicorn_workers=int(os.environ.get("CODEINTEL_UVICORN_WORKERS", "1")),
            uvicorn_loop=os.environ.get("CODEINTEL_UVICORN_LOOP", "auto"),
            uvicorn_http=os.environ.get("CODEINTEL_UVICORN_HTTP", "auto"),
            uvicorn_limit_concurrency=_parse_optional_int(
                os.environ.get("CODEINTEL_UVICORN_LIMIT_CONCURRENCY")
            ),
            uvicorn_limit_max_requests=_parse_optional_int(
                os.environ.get("CODEINTEL_UVICORN_LIMIT_MAX_REQUESTS")
            ),
            uvicorn_timeout_keep_alive=int(
                os.environ.get("CODEINTEL_UVICORN_TIMEOUT_KEEP_ALIVE", "30")
            ),
            uvicorn_backlog=int(os.environ.get("CODEINTEL_UVICORN_BACKLOG", "2048")),
            uvicorn_access_log=os.environ.get("CODEINTEL_UVICORN_ACCESS_LOG", "1") == "1",
            uvicorn_server_header=os.environ.get("CODEINTEL_UVICORN_SERVER_HEADER", "0") == "1",
            uvicorn_proxy_headers=os.environ.get("CODEINTEL_UVICORN_PROXY_HEADERS", "0") == "1",
            uvicorn_forwarded_allow_ips=os.environ.get(
                "CODEINTEL_UVICORN_FORWARDED_ALLOW_IPS", "127.0.0.1"
            ),
            # Security: Auth Enforcement
            auth_required_for_remote=os.environ.get("CODEINTEL_AUTH_REQUIRED_FOR_REMOTE", "1")
            == "1",
        )

    def validate_auth_for_host(self) -> None:
        """Validate that auth is configured when binding to non-localhost.

        Fail-fast security check: if binding to a public interface (0.0.0.0, ::),
        require either auth_token or api_key to be configured.

        Raises
        ------
        ValueError
            If bound to public interface without auth configured.
        """
        if not self.auth_required_for_remote:
            return

        public_hosts = {"0.0.0.0", "::", ""}
        if self.host in public_hosts:
            if not self.auth_token and not self.api_key:
                msg = (
                    f"Security error: Binding to {self.host!r} requires authentication. "
                    f"Set CODEINTEL_AUTH_TOKEN or CODEINTEL_SERVE_API_KEY, "
                    f"or set CODEINTEL_AUTH_REQUIRED_FOR_REMOTE=0 to disable this check."
                )
                raise ValueError(msg)


__all__ = ["ServingSettings"]


def _split_csv(raw: str) -> tuple[str, ...]:
    """Split comma-separated values into a tuple of strings.

    Parameters
    ----------
    raw
        Raw comma-separated string.

    Returns
    -------
    tuple[str, ...]
        Tuple of stripped, non-empty values.
    """
    items = [item.strip() for item in raw.split(",") if item.strip()]
    return tuple(items)


def _parse_optional_int(value: str | None) -> int | None:
    """Parse optional integer from environment variable.

    Parameters
    ----------
    value
        Raw string value or None.

    Returns
    -------
    int | None
        Parsed integer or None if input was None/empty.
    """
    if value is None or value.strip() == "":
        return None
    return int(value)
