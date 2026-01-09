"""Canonical runtime settings for CodeIntel subsystems."""

from __future__ import annotations

from dataclasses import dataclass, field
from ipaddress import ip_address
from pathlib import Path

from codeintel.core.columnar.schema import DEFAULT_SCHEMA_PROMOTE_OPTIONS, SchemaPromoteOptions
from codeintel.core.constants import (
    DEFAULT_ARROW_BATCH_READAHEAD,
    DEFAULT_ARROW_BATCH_SIZE,
    DEFAULT_ARROW_CACHE_METADATA,
    DEFAULT_ARROW_CPU_COUNT,
    DEFAULT_ARROW_FRAGMENT_READAHEAD,
    DEFAULT_ARROW_IO_THREAD_COUNT,
    DEFAULT_ARROW_PARQUET_BUFFER_SIZE,
    DEFAULT_ARROW_PARQUET_PRE_BUFFER,
    DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM,
    DEFAULT_ARROW_USE_THREADS,
)


@dataclass(frozen=True, slots=True)
class ExportAuditSettings:
    """Settings controlling export audit logging."""

    log_path: Path | None = None
    table_enabled: bool = False


@dataclass(frozen=True, slots=True)
class ArrowDatasetSettings:
    """Arrow dataset write tuning settings for build output."""

    compression: str | None = None
    row_group_size: int | None = 200_000
    data_page_size: int | None = 1_048_576
    max_rows_per_file: int | None = None
    dictionary_encode: bool = False
    dictionary_max_cardinality: int = 256
    unify_dictionaries: bool = False
    enable_sink_parquet: bool = True


@dataclass(frozen=True, slots=True)
class ArrowScanSettings:
    """Arrow dataset scan tuning settings."""

    batch_size: int = DEFAULT_ARROW_BATCH_SIZE
    batch_readahead: int | None = DEFAULT_ARROW_BATCH_READAHEAD
    fragment_readahead: int | None = DEFAULT_ARROW_FRAGMENT_READAHEAD
    cache_metadata: bool | None = DEFAULT_ARROW_CACHE_METADATA
    use_threads: bool | None = DEFAULT_ARROW_USE_THREADS
    parquet_pre_buffer: bool | None = DEFAULT_ARROW_PARQUET_PRE_BUFFER
    parquet_use_buffered_stream: bool | None = DEFAULT_ARROW_PARQUET_USE_BUFFERED_STREAM
    parquet_buffer_size: int | None = DEFAULT_ARROW_PARQUET_BUFFER_SIZE
    cpu_count: int | None = DEFAULT_ARROW_CPU_COUNT
    io_thread_count: int | None = DEFAULT_ARROW_IO_THREAD_COUNT
    profile: str | None = None


@dataclass(frozen=True, slots=True)
class ColumnarRuntimeSettings:
    """Columnar runtime profile settings."""

    profile: str | None = None


@dataclass(frozen=True, slots=True)
class BuildSettings:
    """Build runtime settings injected into build execution."""

    engine_version: str
    export_audit: ExportAuditSettings = field(default_factory=ExportAuditSettings)
    arrow_dataset: ArrowDatasetSettings = field(default_factory=ArrowDatasetSettings)
    arrow_scan: ArrowScanSettings = field(default_factory=ArrowScanSettings)
    polars_profile: bool = False
    polars_inspect: bool = False
    polars_query_opt_flags: tuple[str, ...] = ()
    polars_streaming: bool = True
    polars_streaming_fallback: bool = True
    schema_promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS
    dataset_row_index_name: str | None = None
    dataset_row_index_offset: int = 0


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
    dynamic_execution: bool = False
    dynamic_local_executor: str | None = None
    dynamic_remote_executor: str | None = None
    dynamic_remote_max_tasks: int | None = None
    materializers: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OtlpExporterSettings:
    """Settings for OTLP exporter configuration."""

    endpoint: str | None = None
    protocol: str | None = None
    headers: tuple[tuple[str, str], ...] = ()
    timeout_s: float | None = None
    compression: str | None = None
    certificate: str | None = None
    client_certificate: str | None = None
    client_key: str | None = None
    insecure: bool | None = None


@dataclass(frozen=True, slots=True)
class BatchProcessorSettings:
    """Batch processor tuning for trace/log exporters."""

    schedule_delay_ms: int | None = None
    max_queue_size: int | None = None
    max_export_batch_size: int | None = None
    export_timeout_ms: int | None = None


@dataclass(frozen=True, slots=True)
class MetricExportSettings:
    """Metric reader export interval and timeout settings."""

    export_interval_ms: int | None = None
    export_timeout_ms: int | None = None


@dataclass(frozen=True, slots=True)
class SpanLimitSettings:
    """Span limit settings for OpenTelemetry providers."""

    attribute_count_limit: int | None = None
    attribute_value_length_limit: int | None = None
    span_event_count_limit: int | None = None
    span_link_count_limit: int | None = None
    event_attribute_count_limit: int | None = None
    link_attribute_count_limit: int | None = None


@dataclass(frozen=True, slots=True)
class LogLimitSettings:
    """Log record limit settings for OpenTelemetry providers."""

    attribute_count_limit: int | None = None
    attribute_value_length_limit: int | None = None


@dataclass(frozen=True, slots=True)
class MetricViewSettings:
    """Histogram bucket overrides for metric views."""

    operation_duration_ms_buckets: tuple[float, ...] = ()
    query_duration_ms_buckets: tuple[float, ...] = ()
    http_duration_s_buckets: tuple[float, ...] = ()
    grpc_duration_s_buckets: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class GrpcObservabilitySettings:
    """gRPC observability configuration for grpcio-observability."""

    enabled: bool = False
    method_allowlist: tuple[str, ...] = ()
    target_allowlist: tuple[str, ...] = ()
    other_method_label: str = "other"
    other_target_label: str = "other"


@dataclass(frozen=True, slots=True)
class HamiltonTrackerSettings:
    """Hamilton UI tracker configuration."""

    enabled: bool = False
    project_id: str | None = None
    username: str | None = None
    dag_name: str | None = None
    tags: tuple[tuple[str, str], ...] = ()
    api_url: str | None = None
    ui_url: str | None = None
    capture_data_statistics: bool | None = None
    max_list_length: int | None = None
    max_dict_length: int | None = None


@dataclass(frozen=True, slots=True)
class ObservabilitySettings:
    """Observability settings for OpenTelemetry and storage tracing."""

    enabled: bool = True
    service_name: str | None = None
    service_version: str | None = None
    deployment_environment: str | None = None
    resource_attributes: tuple[tuple[str, str], ...] = ()
    propagators: tuple[str, ...] = ()
    traces_sampler: str | None = None
    traces_sampler_arg: float | None = None
    config_file: Path | None = None
    otlp: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)
    otlp_traces: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)
    otlp_metrics: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)
    otlp_logs: OtlpExporterSettings = field(default_factory=OtlpExporterSettings)
    export_traces: bool = True
    export_metrics: bool = True
    export_logs: bool = False
    console_export: bool = False
    prometheus_enabled: bool = False
    logs_auto_instrument: bool = False
    log_correlation: bool = False
    logs_trace_filter: bool = False
    traces_batch: BatchProcessorSettings = field(default_factory=BatchProcessorSettings)
    logs_batch: BatchProcessorSettings = field(default_factory=BatchProcessorSettings)
    metrics_export: MetricExportSettings = field(default_factory=MetricExportSettings)
    span_limits: SpanLimitSettings = field(default_factory=SpanLimitSettings)
    log_limits: LogLimitSettings = field(default_factory=LogLimitSettings)
    metrics_exemplar_filter: str | None = None
    metric_views: MetricViewSettings = field(default_factory=MetricViewSettings)
    teardown_enabled: bool = True
    teardown_task_sample_limit: int = 5
    teardown_thread_sample_limit: int = 5
    teardown_subprocess_sample_limit: int = 5
    cli_enabled: bool = True
    cli_args_allowlist: tuple[str, ...] = ()
    cli_args_capture_mode: str = "names-only"
    cli_arg_names_max: int = 25
    http_route_max_len: int = 120
    mcp_tool_name_max_len: int = 80
    operation_attribute_allowlist_overrides: tuple[tuple[str, tuple[str, ...]], ...] = ()
    grpc_observability: GrpcObservabilitySettings = field(default_factory=GrpcObservabilitySettings)
    hamilton_tracker: HamiltonTrackerSettings = field(default_factory=HamiltonTrackerSettings)
    duckdb_tracing_enabled: bool = True
    duckdb_require_parent_span: bool = True
    duckdb_statement_mode: str = "hash"
    duckdb_statement_hash_len: int = 16
    duckdb_query_summary_max_len: int = 255
    duckdb_query_summary_max_targets: int = 6
    duckdb_query_summary_emit_ellipsis: bool = True
    duckdb_query_summary_hash_suspicious_targets: bool = True
    duckdb_query_summary_hash_len: int = 12
    duckdb_query_summary_hash_min_len: int = 64
    duckdb_query_summary_include_subquery_operations: bool = True
    duckdb_query_summary_include_multi_statement: bool = True
    db_query_summary_span_name_hook: bool = False
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
class CliSettings:
    """CLI-specific settings resolved from the runtime loader."""

    config_path: Path | None = None


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
    query_engine: str = "auto"
    result_engine: str = "polars"
    query_timeout_s: float | None = None
    api_key: str | None = None
    cors_origins: tuple[str, ...] = ()
    trusted_hosts: tuple[str, ...] = ()
    gzip_minimum_size: int = 500
    enable_gzip: bool = True
    export_max_rows: int = 100_000
    export_batch_size: int = DEFAULT_ARROW_BATCH_SIZE
    export_timeout_s: float | None = None
    enable_export_endpoints: bool = True
    export_metrics_enabled: bool = False
    dataset_scan_metrics_enabled: bool = False
    dataset_fragment_readahead: int | None = None
    dataset_batch_readahead: int | None = None
    dataset_use_threads: bool | None = None
    dataset_unify_schemas: bool = False
    dataset_schema_promote_options: SchemaPromoteOptions = DEFAULT_SCHEMA_PROMOTE_OPTIONS

    # Arrow IPC Control Plane
    ipc_enable_options: bool = False
    ipc_compression: str | None = None
    ipc_use_threads: bool = True
    ipc_unify_dictionaries: bool = True
    ipc_metadata_version: str | None = None
    ipc_max_recursion_depth: int | None = None
    ipc_read_use_threads: bool | None = None

    # Polars Execution Controls
    polars_profile: bool = False
    polars_inspect: bool = False
    polars_query_opt_flags: tuple[str, ...] = ()
    polars_streaming: bool = True
    polars_streaming_fallback: bool = True
    polars_maintain_order: bool = True
    polars_use_arrow_scanner: bool = False
    polars_set_sorted: bool = False
    polars_unify_dictionaries: bool = False
    polars_collect_all: bool = False
    polars_sink_batches: bool = False
    polars_collect_schema: bool = False

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
    "ArrowDatasetSettings",
    "ArrowScanSettings",
    "BatchProcessorSettings",
    "BuildSettings",
    "CliSettings",
    "ColumnarRuntimeSettings",
    "ExportAuditSettings",
    "GrpcObservabilitySettings",
    "HamiltonExecutionSettings",
    "HamiltonTrackerSettings",
    "LogLimitSettings",
    "MetricExportSettings",
    "MetricViewSettings",
    "ObservabilitySettings",
    "OtlpExporterSettings",
    "ServingSettings",
    "SpanLimitSettings",
]
