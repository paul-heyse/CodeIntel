"""Environment-driven serving configuration."""

from __future__ import annotations

from pathlib import Path

from codeintel.core.config.settings import ServingSettings
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


def get_serving_settings() -> ServingSettings:
    """Load serving settings from environment variables.

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

    return ServingSettings(
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
        enable_export_endpoints=get_required_bool("CODEINTEL_SERVE_ENABLE_EXPORT", default=True),
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
            "CODEINTEL_MCP_EXPORT_ENABLE_TASKS",
            default=True,
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
        uvicorn_server_header=get_required_bool("CODEINTEL_UVICORN_SERVER_HEADER", default=False),
        uvicorn_proxy_headers=get_required_bool("CODEINTEL_UVICORN_PROXY_HEADERS", default=False),
        uvicorn_forwarded_allow_ips=get_str(
            "CODEINTEL_UVICORN_FORWARDED_ALLOW_IPS",
            default="127.0.0.1",
        )
        or "127.0.0.1",
        # Security: Auth Enforcement
        auth_required_for_remote=get_required_bool(
            "CODEINTEL_AUTH_REQUIRED_FOR_REMOTE",
            default=True,
        ),
        # MCP Tool Feature Flags
        mcp_enable_search=get_required_bool("CODEINTEL_MCP_ENABLE_SEARCH", default=True),
        mcp_enable_explain=get_required_bool("CODEINTEL_MCP_ENABLE_EXPLAIN", default=True),
        mcp_enable_meta=get_required_bool("CODEINTEL_MCP_ENABLE_META", default=True),
        mcp_enable_export=get_required_bool("CODEINTEL_MCP_ENABLE_EXPORT", default=True),
    )


__all__ = ["ServingSettings", "get_serving_settings"]
