"""Canonical runtime settings for CodeIntel subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path

from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE


@dataclass(frozen=True, slots=True)
class ExportAuditSettings:
    """Settings controlling export audit logging."""

    log_path: Path | None = None
    table_enabled: bool = False


@dataclass(frozen=True, slots=True)
class BuildSettings:
    """Build runtime settings injected into build execution."""

    engine_version: str
    export_audit: ExportAuditSettings = field(default_factory=ExportAuditSettings)


@dataclass(frozen=True, slots=True)
class HamiltonExecutionSettings:
    """Execution settings for Hamilton build runs."""

    parallel_backend: str = "sequential"
    max_workers: int | None = None
    duckdb_extensions: tuple[str, ...] = ()
    duckdb_threads: int | None = None
    duckdb_memory_limit: str | None = None
    duckdb_temp_directory: Path | None = None
    duckdb_enable_profiling: bool | None = None
    duckdb_profiling_output: Path | None = None


@dataclass(frozen=True, slots=True)
class ObservabilitySettings:
    """Observability settings for OpenTelemetry and storage tracing."""

    enabled: bool = True
    service_name: str | None = None
    otlp_endpoint: str | None = None
    export_traces: bool = True
    export_metrics: bool = True
    console_export: bool = False
    prometheus_enabled: bool = False
    duckdb_tracing_enabled: bool = True
    duckdb_require_parent_span: bool = True
    duckdb_statement_mode: str = "hash"
    duckdb_statement_hash_len: int = 16
    duckdb_query_summary_max_len: int = 255
    duckdb_query_summary_max_targets: int = 6
    duckdb_emit_legacy_db_attributes: bool = False
    duckdb_query_text_policy: str = "never"
    duckdb_query_text_max_len: int = 4096
    duckdb_query_text_strip_comments: bool = True
    duckdb_query_text_collapse_in_lists: bool = True
    duckdb_query_parameter_enabled: bool = False
    duckdb_query_parameter_keys: tuple[str, ...] = ()
    duckdb_query_parameter_hash_keys: tuple[str, ...] = ()
    duckdb_query_parameter_require_in_sql: bool = True
    duckdb_query_parameter_max_str_len: int = 80


@dataclass(frozen=True, slots=True)
class ServingSettings:
    """Serving layer configuration injected into runtime builders."""

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
    metrics_auth_required: bool = False

    # MCP Tool Feature Flags
    mcp_enable_search: bool = True
    mcp_enable_explain: bool = True
    mcp_enable_meta: bool = True
    mcp_enable_export: bool = True

    def validate_auth_for_host(self) -> None:
        """Validate that auth is configured when binding to non-localhost.

        Raises
        ------
        ValueError
            If bound to a public interface without auth configured.
        """
        if not self.auth_required_for_remote:
            return

        if _is_unspecified_host(self.host) and not self.auth_token and not self.api_key:
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


def _is_unspecified_host(host: str) -> bool:
    if not host:
        return True
    try:
        return ip_address(host).is_unspecified
    except ValueError:
        return False


__all__ = [
    "BuildSettings",
    "ExportAuditSettings",
    "HamiltonExecutionSettings",
    "ObservabilitySettings",
    "ServingSettings",
]
