"""Canonical runtime configuration loader for primitives and settings."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from codeintel.config.primitives import (
    BuildPaths,
    GraphBackendConfig,
    GraphFeatureFlags,
    ScanProfiles,
    SnapshotRef,
)
from codeintel.core.config.settings import (
    BuildSettings,
    ExportAuditSettings,
    HamiltonExecutionSettings,
    ServingSettings,
)
from codeintel.core.env import (
    get_bool,
    get_float,
    get_int,
    get_path,
    get_str,
    is_set,
    split_csv,
)
from codeintel.core.runtime import RuntimePrimitives
from codeintel.core.tools import ToolBinaries
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE


@dataclass(frozen=True)
class RuntimeSettings:
    """Bundle of runtime settings for build, execution, and serving."""

    build: BuildSettings
    execution: HamiltonExecutionSettings
    serving: ServingSettings


@dataclass(frozen=True)
class RuntimeBundle:
    """Runtime primitives and settings resolved for an entrypoint."""

    primitives: RuntimePrimitives
    settings: RuntimeSettings


@dataclass(frozen=True)
class RuntimeInputs:
    """Inputs required to build RuntimePrimitives."""

    snapshot: SnapshotRef
    paths: BuildPaths
    tools: ToolBinaries
    graph_backend: GraphBackendConfig
    graph_features: GraphFeatureFlags
    profiles: ScanProfiles | None = None


def _resolve_engine_version() -> str:
    override = os.environ.get("CODEINTEL_BUILD_ENGINE_VERSION", "").strip()
    if override:
        return override
    try:
        return version("codeintel")
    except PackageNotFoundError:
        return "unknown"


def _resolve_export_audit_log_path() -> Path | None:
    value = os.environ.get("CODEINTEL_EXPORT_AUDIT_LOG")
    if not value:
        return None
    return Path(value.strip())


def _resolve_export_audit_table_enabled() -> bool:
    return os.environ.get("CODEINTEL_EXPORT_AUDIT_TABLE") is not None


def _load_build_settings() -> BuildSettings:
    return BuildSettings(
        engine_version=_resolve_engine_version(),
        export_audit=ExportAuditSettings(
            log_path=_resolve_export_audit_log_path(),
            table_enabled=_resolve_export_audit_table_enabled(),
        ),
    )


def _load_execution_settings() -> HamiltonExecutionSettings:
    backend = get_str("HAMILTON_BACKEND", default="sequential") or "sequential"
    max_workers = get_int("HAMILTON_MAX_WORKERS", default=None)
    duckdb_extensions = split_csv(get_str("CODEINTEL_DUCKDB_EXTENSIONS", default=None))
    duckdb_threads = get_int("CODEINTEL_DUCKDB_THREADS", default=None)
    duckdb_memory_limit = get_str("CODEINTEL_DUCKDB_MEMORY_LIMIT", default=None)
    duckdb_temp_directory = get_path("CODEINTEL_DUCKDB_TEMP_DIRECTORY", default=None)
    duckdb_enable_profiling = get_bool("CODEINTEL_DUCKDB_ENABLE_PROFILING", default=None)
    duckdb_profiling_output = get_path("CODEINTEL_DUCKDB_PROFILING_OUTPUT", default=None)
    return HamiltonExecutionSettings(
        parallel_backend=backend,
        max_workers=max_workers,
        duckdb_extensions=duckdb_extensions,
        duckdb_threads=duckdb_threads,
        duckdb_memory_limit=duckdb_memory_limit,
        duckdb_temp_directory=duckdb_temp_directory,
        duckdb_enable_profiling=duckdb_enable_profiling,
        duckdb_profiling_output=duckdb_profiling_output,
    )


def _load_serving_settings() -> ServingSettings:
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


@lru_cache(maxsize=1)
def load_runtime_settings() -> RuntimeSettings:
    """Load runtime settings from environment variables.

    Returns
    -------
    RuntimeSettings
        Resolved runtime settings bundle.
    """
    return RuntimeSettings(
        build=_load_build_settings(),
        execution=_load_execution_settings(),
        serving=_load_serving_settings(),
    )


def build_runtime_primitives(inputs: RuntimeInputs) -> RuntimePrimitives:
    """Build RuntimePrimitives from component inputs.

    Returns
    -------
    RuntimePrimitives
        Runtime primitives constructed from the provided inputs.
    """
    return RuntimePrimitives(
        snapshot=inputs.snapshot,
        paths=inputs.paths,
        tools=inputs.tools,
        graph_backend=inputs.graph_backend,
        graph_features=inputs.graph_features,
        profiles=inputs.profiles,
    )


def load_runtime_bundle(inputs: RuntimeInputs) -> RuntimeBundle:
    """Build runtime primitives and load settings in one step.

    Returns
    -------
    RuntimeBundle
        Combined runtime primitives and settings bundle.
    """
    primitives = build_runtime_primitives(inputs)
    return RuntimeBundle(primitives=primitives, settings=load_runtime_settings())


__all__ = [
    "RuntimeBundle",
    "RuntimeInputs",
    "RuntimeSettings",
    "build_runtime_primitives",
    "load_runtime_bundle",
    "load_runtime_settings",
]
