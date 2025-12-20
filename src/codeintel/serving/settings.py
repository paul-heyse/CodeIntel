"""Environment-driven serving configuration.

This module defines a small, serving-only configuration surface used by the
semantic-first serving stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codeintel.core.env import (
    get_bool,
    get_float,
    get_int,
    get_path,
    get_str,
    is_set,
    split_csv,
)
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE


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
    export_batch_size
        Arrow record batch size used for streaming exports.
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
    mcp_enable_structured_logging
        Enable structured (JSON) middleware logs for MCP.
    mcp_rate_limit_rps
        Per-session sustained requests/sec allowed for MCP.
    mcp_rate_limit_burst
        Per-session burst capacity for MCP rate limiting.
    mcp_cache_listings
        Cache MCP listings (tools/list, resources/list, prompts/list) with TTL.
    mcp_cache_listings_ttl_seconds
        TTL for listing response caching.
    mcp_max_concurrent_exports
        Maximum concurrent export operations allowed.
    mcp_export_enable_tasks
        Enable background task capability for exports (SEP-1686).
    mcp_export_ttl_seconds
        TTL for export artifacts created via MCP (None disables expiry).
    mcp_export_cleanup_interval_seconds
        Interval (seconds) for periodic cleanup of expired export artifacts.
    mcp_export_max_full_read_bytes
        Maximum bytes allowed for reading an export payload via `codeintel://exports/{export_id}`.
    mcp_export_max_chunk_bytes
        Maximum bytes allowed per `.../bytes?...` chunk resource read.
    mcp_export_max_chunk_lines
        Maximum lines allowed per `.../lines?...` chunk resource read.
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
    mcp_enable_search
        Enable the code_search MCP tool.
    mcp_enable_explain
        Enable the semantic_explain MCP tool.
    mcp_enable_meta
        Enable the serving_meta MCP tool.
    mcp_enable_export
        Enable the semantic_export MCP tool.
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
    export_batch_size: int = DEFAULT_ARROW_BATCH_SIZE
    enable_export_endpoints: bool = True

    # MCP Context Features
    mcp_enable_sampling: bool = False
    mcp_sample_threshold: int = 500
    mcp_progress_reporting: bool = True

    # MCP Error Handling
    mcp_mask_errors: bool = True

    # MCP Query Concurrency Control
    mcp_max_concurrent_queries: int = 2
    mcp_max_concurrent_exports: int = 1

    # MCP EventStore for SSE Resumability
    mcp_enable_event_store: bool = True
    mcp_retry_interval_ms: int = 1000

    # MCP Middleware Features
    mcp_enable_structured_logging: bool = True
    mcp_rate_limit_rps: float = 20.0
    mcp_rate_limit_burst: int = 40
    mcp_cache_listings: bool = True
    mcp_cache_listings_ttl_seconds: int = 5

    # MCP Export Lifecycle (resources are not streaming)
    mcp_export_enable_tasks: bool = True
    mcp_export_ttl_seconds: int | None = 3600
    mcp_export_cleanup_interval_seconds: int = 60
    mcp_export_max_full_read_bytes: int = 1_000_000
    mcp_export_max_chunk_bytes: int = 1_000_000
    mcp_export_max_chunk_lines: int = 2_000

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

    # MCP Tool Feature Flags
    mcp_enable_search: bool = True
    mcp_enable_explain: bool = True
    mcp_enable_meta: bool = True
    mcp_enable_export: bool = True

    @classmethod
    def from_env(cls) -> ServingSettings:
        """Load settings from environment variables.

        Returns
        -------
        ServingSettings
            Loaded settings.
        """

        def get_required_bool(name: str, *, default: bool) -> bool:
            value = get_bool(name, default=default)
            return default if value is None else value

        def get_required_int(name: str, *, default: int) -> int:
            value = get_int(name, default=default)
            return default if value is None else value

        def get_required_float(name: str, *, default: float) -> float:
            value = get_float(name, default=default)
            return default if value is None else value

        def get_optional_int(name: str, *, default_when_unset: int | None = None) -> int | None:
            if not is_set(name):
                return default_when_unset
            return get_int(name, default=None)

        serve_dir = get_path("CODEINTEL_SERVE_DIR", default=Path(".codeintel/serve")) or Path(
            ".codeintel/serve"
        )
        cors_origins = split_csv(get_str("CODEINTEL_SERVE_CORS_ORIGINS", default=None))
        trusted_hosts = split_csv(get_str("CODEINTEL_SERVE_TRUSTED_HOSTS", default=None))
        return cls(
            serve_dir=serve_dir,
            hot_swap=get_required_bool("CODEINTEL_SERVE_HOTSWAP", default=True),
            pool_size=get_required_int("CODEINTEL_SERVE_POOL_SIZE", default=4),
            poll_interval_s=get_required_float("CODEINTEL_SERVE_POLL_INTERVAL", default=1.0),
            mcp_transport=get_str("CODEINTEL_MCP_TRANSPORT", default="stdio") or "stdio",
            host=get_str("CODEINTEL_HOST", default="127.0.0.1") or "127.0.0.1",
            port=get_required_int("CODEINTEL_PORT", default=8000),
            auth_token=get_str("CODEINTEL_AUTH_TOKEN", default=None),
            schema_enforcement=get_str("CODEINTEL_SERVE_SCHEMA_ENFORCEMENT", default="strict")
            or "strict",
            result_engine=get_str("CODEINTEL_SERVE_RESULT_ENGINE", default="polars") or "polars",
            api_key=get_str("CODEINTEL_SERVE_API_KEY", default=None),
            cors_origins=cors_origins,
            trusted_hosts=trusted_hosts,
            gzip_minimum_size=get_required_int("CODEINTEL_SERVE_GZIP_MIN_SIZE", default=500),
            enable_gzip=get_required_bool("CODEINTEL_SERVE_GZIP", default=True),
            export_max_rows=get_required_int("CODEINTEL_SERVE_EXPORT_MAX_ROWS", default=100_000),
            export_batch_size=get_required_int(
                "CODEINTEL_SERVE_EXPORT_BATCH_SIZE",
                default=DEFAULT_ARROW_BATCH_SIZE,
            ),
            enable_export_endpoints=get_required_bool(
                "CODEINTEL_SERVE_ENABLE_EXPORT", default=True
            ),
            # MCP Context Features
            mcp_enable_sampling=get_required_bool("CODEINTEL_MCP_ENABLE_SAMPLING", default=False),
            mcp_sample_threshold=get_required_int("CODEINTEL_MCP_SAMPLE_THRESHOLD", default=500),
            mcp_progress_reporting=get_required_bool("CODEINTEL_MCP_PROGRESS", default=True),
            # MCP Error Handling
            mcp_mask_errors=get_required_bool("CODEINTEL_MCP_MASK_ERRORS", default=True),
            # MCP Query Concurrency Control
            mcp_max_concurrent_queries=get_required_int(
                "CODEINTEL_MCP_MAX_CONCURRENT_QUERIES", default=2
            ),
            mcp_max_concurrent_exports=get_required_int(
                "CODEINTEL_MCP_MAX_CONCURRENT_EXPORTS", default=1
            ),
            # MCP EventStore for SSE Resumability
            mcp_enable_event_store=get_required_bool("CODEINTEL_MCP_EVENT_STORE", default=True),
            mcp_retry_interval_ms=get_required_int("CODEINTEL_MCP_RETRY_INTERVAL", default=1000),
            # MCP Middleware Features
            mcp_enable_structured_logging=get_required_bool(
                "CODEINTEL_MCP_ENABLE_STRUCTURED_LOGGING",
                default=True,
            ),
            mcp_rate_limit_rps=get_required_float("CODEINTEL_MCP_RATE_LIMIT_RPS", default=20.0),
            mcp_rate_limit_burst=get_required_int("CODEINTEL_MCP_RATE_LIMIT_BURST", default=40),
            mcp_cache_listings=get_required_bool("CODEINTEL_MCP_CACHE_LISTINGS", default=True),
            mcp_cache_listings_ttl_seconds=get_required_int(
                "CODEINTEL_MCP_CACHE_LISTINGS_TTL_SECONDS",
                default=5,
            ),
            # MCP Export Lifecycle
            mcp_export_enable_tasks=get_required_bool(
                "CODEINTEL_MCP_EXPORT_ENABLE_TASKS", default=True
            ),
            mcp_export_ttl_seconds=get_optional_int(
                "CODEINTEL_MCP_EXPORT_TTL_SECONDS",
                default_when_unset=3600,
            ),
            mcp_export_cleanup_interval_seconds=get_required_int(
                "CODEINTEL_MCP_EXPORT_CLEANUP_INTERVAL_SECONDS",
                default=60,
            ),
            mcp_export_max_full_read_bytes=get_required_int(
                "CODEINTEL_MCP_EXPORT_MAX_FULL_READ_BYTES",
                default=1_000_000,
            ),
            mcp_export_max_chunk_bytes=get_required_int(
                "CODEINTEL_MCP_EXPORT_MAX_CHUNK_BYTES",
                default=1_000_000,
            ),
            mcp_export_max_chunk_lines=get_required_int(
                "CODEINTEL_MCP_EXPORT_MAX_CHUNK_LINES",
                default=2_000,
            ),
            # Uvicorn Production Configuration
            uvicorn_workers=get_required_int("CODEINTEL_UVICORN_WORKERS", default=1),
            uvicorn_loop=get_str("CODEINTEL_UVICORN_LOOP", default="auto") or "auto",
            uvicorn_http=get_str("CODEINTEL_UVICORN_HTTP", default="auto") or "auto",
            uvicorn_limit_concurrency=get_optional_int("CODEINTEL_UVICORN_LIMIT_CONCURRENCY"),
            uvicorn_limit_max_requests=get_optional_int("CODEINTEL_UVICORN_LIMIT_MAX_REQUESTS"),
            uvicorn_timeout_keep_alive=get_required_int(
                "CODEINTEL_UVICORN_TIMEOUT_KEEP_ALIVE",
                default=30,
            ),
            uvicorn_backlog=get_required_int("CODEINTEL_UVICORN_BACKLOG", default=2048),
            uvicorn_access_log=get_required_bool("CODEINTEL_UVICORN_ACCESS_LOG", default=True),
            uvicorn_server_header=get_required_bool(
                "CODEINTEL_UVICORN_SERVER_HEADER", default=False
            ),
            uvicorn_proxy_headers=get_required_bool(
                "CODEINTEL_UVICORN_PROXY_HEADERS", default=False
            ),
            uvicorn_forwarded_allow_ips=get_str(
                "CODEINTEL_UVICORN_FORWARDED_ALLOW_IPS",
                default="127.0.0.1",
            )
            or "127.0.0.1",
            # Security: Auth Enforcement
            auth_required_for_remote=get_required_bool(
                "CODEINTEL_AUTH_REQUIRED_FOR_REMOTE", default=True
            ),
            # MCP Tool Feature Flags
            mcp_enable_search=get_required_bool("CODEINTEL_MCP_ENABLE_SEARCH", default=True),
            mcp_enable_explain=get_required_bool("CODEINTEL_MCP_ENABLE_EXPLAIN", default=True),
            mcp_enable_meta=get_required_bool("CODEINTEL_MCP_ENABLE_META", default=True),
            mcp_enable_export=get_required_bool("CODEINTEL_MCP_ENABLE_EXPORT", default=True),
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

        public_hosts = {"0.0.0.0", "::", ""}  # noqa: S104
        if self.host in public_hosts and not self.auth_token and not self.api_key:
            msg = (
                f"Security error: Binding to {self.host!r} requires authentication. "
                f"Set CODEINTEL_AUTH_TOKEN or CODEINTEL_SERVE_API_KEY, "
                f"or set CODEINTEL_AUTH_REQUIRED_FOR_REMOTE=0 to disable this check."
            )
            raise ValueError(msg)

    def validate_mcp_single_worker(self, *, mount_mcp: bool) -> None:
        """Validate the deployment contract for sessionful MCP.

        Parameters
        ----------
        mount_mcp
            Whether MCP is mounted/enabled for this serving process.

        Raises
        ------
        ValueError
            If MCP is mounted but the deployment is configured for >1 worker.
        """
        if not mount_mcp:
            return
        if self.uvicorn_workers != 1:
            msg = (
                "Invalid configuration: MCP is sessionful and requires uvicorn_workers=1. "
                f"Got uvicorn_workers={self.uvicorn_workers}."
            )
            raise ValueError(msg)


__all__ = ["ServingSettings"]
