"""Canonical runtime configuration loader for primitives and settings."""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
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
    BatchProcessorSettings,
    BuildSettings,
    CliSettings,
    ExportAuditSettings,
    GrpcObservabilitySettings,
    HamiltonExecutionSettings,
    HamiltonTrackerSettings,
    IcebergSettings,
    LogLimitSettings,
    MetricExportSettings,
    MetricViewSettings,
    ObservabilitySettings,
    OtlpExporterSettings,
    ServingSettings,
    SpanLimitSettings,
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
from codeintel.core.execution.context import ExecutionContext, RunContext
from codeintel.core.runtime import RuntimeBundle, RuntimePrimitives, RuntimeSettings, VariantConfig
from codeintel.core.tools import ToolBinaries
from codeintel.observability.semconv_keys import CODEINTEL_COMMIT, CODEINTEL_REPO
from codeintel.observability.test_mode import apply_test_telemetry_settings
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE

LOG = logging.getLogger(__name__)


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


def _parse_kv_pairs(value: str | None) -> tuple[tuple[str, str], ...]:
    if not value:
        return ()
    pairs: list[tuple[str, str]] = []
    for raw in split_csv(value):
        if "=" not in raw:
            continue
        key, val = raw.split("=", maxsplit=1)
        key = key.strip()
        val = val.strip()
        if not key:
            continue
        pairs.append((key, val))
    return tuple(pairs)


def _iceberg_default_enabled(*, deployment_environment: str | None, configured: bool) -> bool:
    if _is_prod_environment(deployment_environment):
        return False
    return configured


def _load_iceberg_settings() -> IcebergSettings:
    deployment_environment = get_str("CODEINTEL_DEPLOYMENT_ENVIRONMENT", default=None)
    catalog_name = get_str("CODEINTEL_ICEBERG_CATALOG_NAME", default="default") or "default"
    catalog_type = get_str("CODEINTEL_ICEBERG_CATALOG_TYPE", default="sql")
    if catalog_type is not None:
        catalog_type = catalog_type.strip() or "sql"
    catalog_uri = get_str("CODEINTEL_ICEBERG_CATALOG_URI", default=None)
    catalog_warehouse = get_str("CODEINTEL_ICEBERG_CATALOG_WAREHOUSE", default=None)
    catalog_properties = _parse_kv_pairs(
        get_str("CODEINTEL_ICEBERG_CATALOG_PROPERTIES", default=None)
    )
    config_path = get_path("CODEINTEL_ICEBERG_CONFIG_PATH", default=None)
    io_impl = get_str("CODEINTEL_ICEBERG_IO_IMPL", default=None)
    io_options = _parse_kv_pairs(get_str("CODEINTEL_ICEBERG_IO_OPTIONS", default=None))
    location_provider_impl = get_str(
        "CODEINTEL_ICEBERG_LOCATION_PROVIDER_IMPL",
        default=None,
    )
    write_data_path = get_str("CODEINTEL_ICEBERG_WRITE_DATA_PATH", default=None)
    write_metadata_path = get_str("CODEINTEL_ICEBERG_WRITE_METADATA_PATH", default=None)
    object_store_partitioned_paths = (
        get_bool("CODEINTEL_ICEBERG_OBJECT_STORE_PARTITIONED_PATHS", default=None)
        if is_set("CODEINTEL_ICEBERG_OBJECT_STORE_PARTITIONED_PATHS")
        else None
    )
    configured = any([catalog_uri, catalog_warehouse, config_path])
    default_enabled = _iceberg_default_enabled(
        deployment_environment=deployment_environment,
        configured=configured,
    )
    read_enabled = bool(get_bool("CODEINTEL_ICEBERG_READ_ENABLED", default=default_enabled))
    write_enabled = bool(get_bool("CODEINTEL_ICEBERG_WRITE_ENABLED", default=default_enabled))
    tombstones_enabled = bool(
        get_bool("CODEINTEL_ICEBERG_TOMBSTONES_ENABLED", default=default_enabled)
    )
    flight_enabled = bool(get_bool("CODEINTEL_ICEBERG_FLIGHT_ENABLED", default=False))
    read_ref = get_str("CODEINTEL_ICEBERG_READ_REF", default=None)
    enforced_prefixes = split_csv(get_str("CODEINTEL_ICEBERG_ENFORCE_PREFIXES", default=None))
    return IcebergSettings(
        read_enabled=read_enabled,
        write_enabled=write_enabled,
        tombstones_enabled=tombstones_enabled,
        flight_enabled=flight_enabled,
        read_ref=read_ref,
        enforced_table_prefixes=tuple(enforced_prefixes),
        catalog_name=catalog_name,
        catalog_type=catalog_type,
        catalog_uri=catalog_uri,
        catalog_warehouse=catalog_warehouse,
        catalog_properties=catalog_properties,
        config_path=config_path,
        io_impl=io_impl,
        io_options=io_options,
        location_provider_impl=location_provider_impl,
        write_data_path=write_data_path,
        write_metadata_path=write_metadata_path,
        object_store_partitioned_paths=object_store_partitioned_paths,
    )


def _load_build_settings() -> BuildSettings:
    def optional_bool(name: str) -> bool | None:
        if not is_set(name):
            return None
        return get_bool(name, default=None)

    polars_streaming = optional_bool("CODEINTEL_BUILD_POLARS_STREAMING")
    polars_streaming_fallback = optional_bool("CODEINTEL_BUILD_POLARS_STREAMING_FALLBACK")
    polars_flags = split_csv(get_str("CODEINTEL_BUILD_POLARS_QUERY_OPT_FLAGS", default=None))
    iceberg = _load_iceberg_settings()
    return BuildSettings(
        engine_version=_resolve_engine_version(),
        export_audit=ExportAuditSettings(
            log_path=_resolve_export_audit_log_path(),
            table_enabled=_resolve_export_audit_table_enabled(),
        ),
        iceberg=iceberg,
        polars_profile=bool(optional_bool("CODEINTEL_BUILD_POLARS_PROFILE") or False),
        polars_inspect=bool(optional_bool("CODEINTEL_BUILD_POLARS_INSPECT") or False),
        polars_query_opt_flags=tuple(polars_flags),
        polars_streaming=polars_streaming if polars_streaming is not None else True,
        polars_streaming_fallback=polars_streaming_fallback
        if polars_streaming_fallback is not None
        else True,
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
    dynamic_execution = bool(get_bool("CODEINTEL_HAMILTON_DYNAMIC_EXECUTION", default=False))
    dynamic_local_executor = get_str("CODEINTEL_HAMILTON_DYNAMIC_LOCAL_EXECUTOR", default=None)
    dynamic_remote_executor = get_str("CODEINTEL_HAMILTON_DYNAMIC_REMOTE_EXECUTOR", default=None)
    dynamic_remote_max_tasks = get_int(
        "CODEINTEL_HAMILTON_DYNAMIC_REMOTE_MAX_TASKS",
        default=None,
    )
    materializers = split_csv(get_str("CODEINTEL_HAMILTON_MATERIALIZERS", default=None))
    return HamiltonExecutionSettings(
        parallel_backend=backend,
        max_workers=max_workers,
        duckdb_extensions=duckdb_extensions,
        duckdb_threads=duckdb_threads,
        duckdb_memory_limit=duckdb_memory_limit,
        duckdb_temp_directory=duckdb_temp_directory,
        duckdb_enable_profiling=duckdb_enable_profiling,
        duckdb_profiling_output=duckdb_profiling_output,
        dynamic_execution=dynamic_execution,
        dynamic_local_executor=dynamic_local_executor,
        dynamic_remote_executor=dynamic_remote_executor,
        dynamic_remote_max_tasks=dynamic_remote_max_tasks,
        materializers=materializers,
    )


def _load_cli_settings() -> CliSettings:
    config_path = get_path("CODEINTEL_CONFIG_PATH", default=None)
    return CliSettings(config_path=config_path)


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

    def get_optional_float(name: str, *, default_when_unset: float | None = None) -> float | None:
        if not is_set(name):
            return default_when_unset
        return get_float(name, default=None)

    iceberg = _load_iceberg_settings()

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
        query_engine=get_str("CODEINTEL_SERVE_QUERY_ENGINE", default="auto") or "auto",
        result_engine=get_str("CODEINTEL_SERVE_RESULT_ENGINE", default="polars") or "polars",
        query_timeout_s=get_optional_float("CODEINTEL_SERVE_QUERY_TIMEOUT_S"),
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
        export_timeout_s=get_optional_float("CODEINTEL_SERVE_EXPORT_TIMEOUT_S"),
        enable_export_endpoints=get_required_bool("CODEINTEL_SERVE_ENABLE_EXPORT", default=True),
        export_metrics_enabled=get_required_bool(
            "CODEINTEL_SERVE_EXPORT_METRICS",
            default=False,
        ),
        dataset_scan_metrics_enabled=get_required_bool(
            "CODEINTEL_SERVE_DATASET_SCAN_METRICS",
            default=False,
        ),
        dataset_fragment_readahead=get_optional_int(
            "CODEINTEL_SERVE_DATASET_FRAGMENT_READAHEAD",
        ),
        iceberg=iceberg,
        ipc_enable_options=get_required_bool("CODEINTEL_SERVE_IPC_ENABLE_OPTIONS", default=False),
        ipc_compression=get_str("CODEINTEL_SERVE_IPC_COMPRESSION", default=None),
        ipc_use_threads=get_required_bool("CODEINTEL_SERVE_IPC_USE_THREADS", default=True),
        ipc_unify_dictionaries=get_required_bool(
            "CODEINTEL_SERVE_IPC_UNIFY_DICTIONARIES",
            default=True,
        ),
        ipc_metadata_version=get_str("CODEINTEL_SERVE_IPC_METADATA_VERSION", default=None),
        ipc_max_recursion_depth=get_optional_int(
            "CODEINTEL_SERVE_IPC_MAX_RECURSION_DEPTH",
        ),
        ipc_read_use_threads=get_bool("CODEINTEL_SERVE_IPC_READ_USE_THREADS", default=None),
        polars_profile=get_required_bool("CODEINTEL_SERVE_POLARS_PROFILE", default=False),
        polars_inspect=get_required_bool("CODEINTEL_SERVE_POLARS_INSPECT", default=False),
        polars_query_opt_flags=_obs_parse_csv(
            get_str("CODEINTEL_SERVE_POLARS_QUERY_OPT_FLAGS", default=None)
        ),
        polars_streaming=get_required_bool("CODEINTEL_SERVE_POLARS_STREAMING", default=True),
        polars_streaming_fallback=get_required_bool(
            "CODEINTEL_SERVE_POLARS_STREAMING_FALLBACK",
            default=True,
        ),
        polars_use_arrow_scanner=get_required_bool(
            "CODEINTEL_SERVE_POLARS_USE_ARROW_SCANNER",
            default=False,
        ),
        polars_set_sorted=get_required_bool("CODEINTEL_SERVE_POLARS_SET_SORTED", default=False),
        polars_unify_dictionaries=get_required_bool(
            "CODEINTEL_SERVE_POLARS_UNIFY_DICTIONARIES",
            default=False,
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
        metrics_auth_required=get_required_bool(
            "CODEINTEL_METRICS_REQUIRE_AUTH",
            default=False,
        ),
        # MCP Tool Feature Flags
        mcp_enable_search=get_required_bool("CODEINTEL_MCP_ENABLE_SEARCH", default=True),
        mcp_enable_explain=get_required_bool("CODEINTEL_MCP_ENABLE_EXPLAIN", default=True),
        mcp_enable_meta=get_required_bool("CODEINTEL_MCP_ENABLE_META", default=True),
        mcp_enable_export=get_required_bool("CODEINTEL_MCP_ENABLE_EXPORT", default=True),
    )


def _load_variant_settings() -> VariantConfig:
    raw_json = os.environ.get("CODEINTEL_VARIANTS_JSON", "").strip()
    if raw_json:
        payload = json.loads(raw_json)
        if not isinstance(payload, dict):
            msg = "CODEINTEL_VARIANTS_JSON must decode to an object"
            raise ValueError(msg)
        return VariantConfig.from_mapping(payload).validate()
    return VariantConfig().validate()


def _obs_opt_str(name: str) -> str | None:
    value = get_str(name, default=None)
    return value.strip() if value else None


def _obs_safe_get_bool(name: str, *, default: bool | None) -> bool | None:
    try:
        return get_bool(name, default=default)
    except ValueError as exc:
        LOG.warning("Invalid boolean for %s: %s", name, exc)
        return default


def _obs_safe_get_int(
    name: str,
    *,
    default: int | None,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int | None:
    try:
        return get_int(name, default=default, min_value=min_value, max_value=max_value)
    except ValueError as exc:
        LOG.warning("Invalid integer for %s: %s", name, exc)
        return default


def _obs_safe_get_float(
    name: str,
    *,
    default: float | None,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float | None:
    try:
        return get_float(name, default=default, min_value=min_value, max_value=max_value)
    except ValueError as exc:
        LOG.warning("Invalid float for %s: %s", name, exc)
        return default


def _obs_parse_csv(value: str | None) -> tuple[str, ...]:
    if not value:
        return ()
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(items)


def _obs_parse_kv_pairs(value: str | None) -> tuple[tuple[str, str], ...]:
    if not value:
        return ()
    pairs: list[tuple[str, str]] = []
    for raw in value.split(","):
        part = raw.strip()
        if not part:
            continue
        if "=" not in part:
            LOG.warning("Skipping invalid key/value pair in %s: %s", value, part)
            continue
        key, val = part.split("=", 1)
        key = key.strip()
        val = val.strip()
        if not key or not val:
            LOG.warning("Skipping invalid key/value pair in %s: %s", value, part)
            continue
        pairs.append((key, val))
    return tuple(pairs)


def _obs_parse_float_csv(name: str) -> tuple[float, ...]:
    raw = _obs_opt_str(name)
    if not raw:
        return ()
    values: list[float] = []
    for item in raw.split(","):
        part = item.strip()
        if not part:
            continue
        try:
            values.append(float(part))
        except ValueError:
            LOG.warning("Invalid float value in %s: %s", name, part)
            return ()
    return tuple(values)


def _obs_normalize_protocol(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized in {"grpc", "http/protobuf"}:
        return normalized
    LOG.warning("Unsupported OTLP protocol %s; ignoring", value)
    return None


def _obs_normalize_compression(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().lower()
    if normalized:
        return normalized
    return None


def _obs_parse_otlp_settings(prefix: str) -> OtlpExporterSettings:
    return OtlpExporterSettings(
        endpoint=_obs_opt_str(f"{prefix}_ENDPOINT"),
        protocol=_obs_normalize_protocol(_obs_opt_str(f"{prefix}_PROTOCOL")),
        headers=_obs_parse_kv_pairs(_obs_opt_str(f"{prefix}_HEADERS")),
        timeout_s=_obs_safe_get_float(f"{prefix}_TIMEOUT", default=None, min_value=0.0),
        compression=_obs_normalize_compression(_obs_opt_str(f"{prefix}_COMPRESSION")),
        certificate=_obs_opt_str(f"{prefix}_CERTIFICATE"),
        client_certificate=_obs_opt_str(f"{prefix}_CLIENT_CERTIFICATE"),
        client_key=_obs_opt_str(f"{prefix}_CLIENT_KEY"),
        insecure=_obs_safe_get_bool(f"{prefix}_INSECURE", default=None),
    )


def _obs_opt_non_negative_int(name: str, default: int) -> int:
    value = _obs_safe_get_int(name, default=default, min_value=0)
    if value is None:
        return default
    return int(value)


def _obs_statement_mode() -> str:
    statement_mode = get_str("CODEINTEL_OTEL_DB_STATEMENT_MODE", default=None)
    statement_mode_value = statement_mode.strip().lower() if statement_mode else "hash"
    if statement_mode_value not in {"full", "hash", "operation", "none"}:
        return "hash"
    return statement_mode_value


def _obs_query_text_policy() -> str:
    query_text_policy = get_str("CODEINTEL_OTEL_DB_QUERY_TEXT_POLICY", default="never")
    query_text_policy_value = query_text_policy.strip().lower() if query_text_policy else "never"
    if query_text_policy_value not in {
        "never",
        "parameterized",
        "redacted",
        "parameterized_or_redacted",
        "full",
    }:
        return "never"
    return query_text_policy_value


def _obs_cli_args_capture_mode() -> str:
    cli_args_capture_mode = (
        _obs_opt_str("CODEINTEL_OBSERVABILITY_CLI_ARGS_CAPTURE_MODE") or "names-only"
    )
    cli_args_capture_mode_value = cli_args_capture_mode.strip().lower()
    if cli_args_capture_mode_value not in {"names-only", "allowlist"}:
        return "names-only"
    return cli_args_capture_mode_value


def _obs_resolve_export_logs() -> bool:
    logs_exporter_raw = _obs_opt_str("OTEL_LOGS_EXPORTER")
    if logs_exporter_raw:
        logs_exporter = logs_exporter_raw.strip().lower()
        return logs_exporter != "none"
    return bool(_obs_safe_get_bool("CODEINTEL_EXPORT_LOGS", default=False))


def _obs_parse_operation_allowlist_overrides() -> tuple[tuple[str, tuple[str, ...]], ...]:
    raw = _obs_opt_str("CODEINTEL_OBSERVABILITY_OPERATION_ALLOWLIST_OVERRIDES")
    if not raw:
        return ()
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        LOG.warning("Invalid operation allowlist overrides JSON: %s", exc)
        return ()
    if not isinstance(payload, dict):
        LOG.warning("Operation allowlist overrides must be a JSON object")
        return ()
    overrides: list[tuple[str, tuple[str, ...]]] = []
    for key, value in payload.items():
        if not isinstance(key, str):
            continue
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            overrides.append((key, tuple(value)))
    if not overrides:
        return ()
    return tuple(overrides)


def _obs_positive_int(name: str, default: int) -> int:
    value = _obs_safe_get_int(name, default=default, min_value=0)
    if value is None:
        return default
    return int(value)


def _obs_resolve_hamilton_tracker() -> tuple[
    bool,
    str | None,
    str | None,
    tuple[tuple[str, str], ...],
]:
    tracker_enabled = bool(_obs_safe_get_bool("CODEINTEL_HAMILTON_TRACKER_ENABLED", default=False))
    project_id = _obs_opt_str("HAMILTON_PROJECT_ID")
    username = _obs_opt_str("HAMILTON_USERNAME")
    if not tracker_enabled and project_id and username:
        tracker_enabled = True
    tags_raw = _obs_opt_str("HAMILTON_TAGS") or _obs_opt_str("CODEINTEL_HAMILTON_TAGS")
    return tracker_enabled, project_id, username, _obs_parse_kv_pairs(tags_raw)


def _is_prod_environment(value: str | None) -> bool:
    """Return True when the deployment environment is production.

    Parameters
    ----------
    value
        Deployment environment string to evaluate.

    Returns
    -------
    bool
        True when the environment denotes production.
    """
    if value is None:
        return False
    normalized = value.strip().lower()
    return normalized in {"prod", "production"}


def _obs_resource_attributes() -> tuple[tuple[str, str], ...]:
    """Return resource attributes with CodeIntel-specific defaults applied.

    Returns
    -------
    tuple[tuple[str, str], ...]
        Resource attribute key/value pairs.
    """
    resource_attributes = dict(_obs_parse_kv_pairs(_obs_opt_str("OTEL_RESOURCE_ATTRIBUTES")))
    repo = _obs_opt_str("CODEINTEL_REPO")
    if repo and CODEINTEL_REPO not in resource_attributes:
        resource_attributes[CODEINTEL_REPO] = repo
    commit = _obs_opt_str("CODEINTEL_COMMIT")
    if commit and CODEINTEL_COMMIT not in resource_attributes:
        resource_attributes[CODEINTEL_COMMIT] = commit
    return tuple(resource_attributes.items())


def _obs_tracker_capture_policy(
    deployment_environment: str | None,
) -> tuple[bool | None, int | None, int | None]:
    """Resolve Hamilton tracker capture defaults based on environment.

    Parameters
    ----------
    deployment_environment
        Deployment environment name for safe defaults.

    Returns
    -------
    tuple[bool | None, int | None, int | None]
        capture_data_statistics, max_list_length, max_dict_length.
    """
    capture_data_statistics = _obs_safe_get_bool(
        "HAMILTON_CAPTURE_DATA_STATISTICS",
        default=None,
    )
    max_list_length = _obs_safe_get_int(
        "HAMILTON_MAX_LIST_LENGTH_CAPTURE",
        default=None,
        min_value=0,
    )
    max_dict_length = _obs_safe_get_int(
        "HAMILTON_MAX_DICT_LENGTH_CAPTURE",
        default=None,
        min_value=0,
    )
    if _is_prod_environment(deployment_environment):
        if capture_data_statistics is None:
            capture_data_statistics = False
        if max_list_length is None:
            max_list_length = 20
        if max_dict_length is None:
            max_dict_length = 50
    return capture_data_statistics, max_list_length, max_dict_length


def _load_observability_settings() -> ObservabilitySettings:
    config_file = get_path("OTEL_EXPERIMENTAL_CONFIG_FILE", default=None)
    sdk_disabled = _obs_safe_get_bool("OTEL_SDK_DISABLED", default=False)
    enabled = True if config_file is not None else not bool(sdk_disabled)

    statement_mode_value = _obs_statement_mode()
    query_text_policy_value = _obs_query_text_policy()
    export_logs = _obs_resolve_export_logs()
    deployment_environment = _obs_opt_str("CODEINTEL_DEPLOYMENT_ENVIRONMENT")
    tracker_enabled, hamilton_project_id, hamilton_username, hamilton_tags = (
        _obs_resolve_hamilton_tracker()
    )
    capture_data_statistics, max_list_length, max_dict_length = _obs_tracker_capture_policy(
        deployment_environment
    )

    settings = ObservabilitySettings(
        enabled=enabled,
        service_name=_obs_opt_str("OTEL_SERVICE_NAME"),
        service_version=_obs_opt_str("CODEINTEL_SERVICE_VERSION"),
        deployment_environment=deployment_environment,
        resource_attributes=_obs_resource_attributes(),
        propagators=split_csv(_obs_opt_str("OTEL_PROPAGATORS")),
        traces_sampler=_obs_opt_str("OTEL_TRACES_SAMPLER"),
        traces_sampler_arg=_obs_safe_get_float(
            "OTEL_TRACES_SAMPLER_ARG",
            default=None,
            min_value=0.0,
        ),
        config_file=config_file,
        otlp=_obs_parse_otlp_settings("OTEL_EXPORTER_OTLP"),
        otlp_traces=_obs_parse_otlp_settings("OTEL_EXPORTER_OTLP_TRACES"),
        otlp_metrics=_obs_parse_otlp_settings("OTEL_EXPORTER_OTLP_METRICS"),
        otlp_logs=_obs_parse_otlp_settings("OTEL_EXPORTER_OTLP_LOGS"),
        export_traces=bool(_obs_safe_get_bool("CODEINTEL_EXPORT_TRACES", default=True)),
        export_metrics=bool(_obs_safe_get_bool("CODEINTEL_EXPORT_METRICS", default=True)),
        export_logs=export_logs,
        console_export=bool(_obs_safe_get_bool("CODEINTEL_CONSOLE_TELEMETRY", default=False)),
        prometheus_enabled=bool(_obs_safe_get_bool("CODEINTEL_PROMETHEUS_METRICS", default=False)),
        logs_auto_instrument=bool(
            _obs_safe_get_bool("OTEL_PYTHON_LOGGING_AUTO_INSTRUMENTATION_ENABLED", default=False)
        ),
        log_correlation=bool(_obs_safe_get_bool("OTEL_PYTHON_LOG_CORRELATION", default=False)),
        logs_trace_filter=bool(
            _obs_safe_get_bool("CODEINTEL_OTEL_LOGS_TRACE_FILTER", default=False)
        ),
        traces_batch=BatchProcessorSettings(
            schedule_delay_ms=_obs_safe_get_int(
                "OTEL_BSP_SCHEDULE_DELAY",
                default=None,
                min_value=0,
            ),
            max_queue_size=_obs_safe_get_int(
                "OTEL_BSP_MAX_QUEUE_SIZE",
                default=None,
                min_value=0,
            ),
            max_export_batch_size=_obs_safe_get_int(
                "OTEL_BSP_MAX_EXPORT_BATCH_SIZE",
                default=None,
                min_value=0,
            ),
            export_timeout_ms=_obs_safe_get_int(
                "OTEL_BSP_EXPORT_TIMEOUT",
                default=None,
                min_value=0,
            ),
        ),
        logs_batch=BatchProcessorSettings(
            schedule_delay_ms=_obs_safe_get_int(
                "OTEL_BLRP_SCHEDULE_DELAY",
                default=None,
                min_value=0,
            ),
            max_queue_size=_obs_safe_get_int(
                "OTEL_BLRP_MAX_QUEUE_SIZE",
                default=None,
                min_value=0,
            ),
            max_export_batch_size=_obs_safe_get_int(
                "OTEL_BLRP_MAX_EXPORT_BATCH_SIZE",
                default=None,
                min_value=0,
            ),
            export_timeout_ms=_obs_safe_get_int(
                "OTEL_BLRP_EXPORT_TIMEOUT",
                default=None,
                min_value=0,
            ),
        ),
        metrics_export=MetricExportSettings(
            export_interval_ms=_obs_safe_get_int(
                "OTEL_METRIC_EXPORT_INTERVAL",
                default=None,
                min_value=0,
            ),
            export_timeout_ms=_obs_safe_get_int(
                "OTEL_METRIC_EXPORT_TIMEOUT",
                default=None,
                min_value=0,
            ),
        ),
        span_limits=SpanLimitSettings(
            attribute_count_limit=_obs_safe_get_int(
                "OTEL_ATTRIBUTE_COUNT_LIMIT",
                default=None,
                min_value=0,
            ),
            attribute_value_length_limit=_obs_safe_get_int(
                "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT",
                default=None,
                min_value=0,
            ),
            span_event_count_limit=_obs_safe_get_int(
                "OTEL_SPAN_EVENT_COUNT_LIMIT",
                default=None,
                min_value=0,
            ),
            span_link_count_limit=_obs_safe_get_int(
                "OTEL_SPAN_LINK_COUNT_LIMIT",
                default=None,
                min_value=0,
            ),
            event_attribute_count_limit=_obs_safe_get_int(
                "OTEL_EVENT_ATTRIBUTE_COUNT_LIMIT",
                default=None,
                min_value=0,
            ),
            link_attribute_count_limit=_obs_safe_get_int(
                "OTEL_LINK_ATTRIBUTE_COUNT_LIMIT",
                default=None,
                min_value=0,
            ),
        ),
        log_limits=LogLimitSettings(
            attribute_count_limit=_obs_safe_get_int(
                "OTEL_LOGRECORD_ATTRIBUTE_COUNT_LIMIT",
                default=None,
                min_value=0,
            ),
            attribute_value_length_limit=_obs_safe_get_int(
                "OTEL_LOGRECORD_ATTRIBUTE_VALUE_LENGTH_LIMIT",
                default=None,
                min_value=0,
            ),
        ),
        metrics_exemplar_filter=_obs_opt_str("OTEL_METRICS_EXEMPLAR_FILTER"),
        metric_views=MetricViewSettings(
            operation_duration_ms_buckets=_obs_parse_float_csv(
                "CODEINTEL_OTEL_METRIC_BUCKETS_OPERATION_DURATION_MS"
            ),
            query_duration_ms_buckets=_obs_parse_float_csv(
                "CODEINTEL_OTEL_METRIC_BUCKETS_QUERY_DURATION_MS"
            ),
            http_duration_s_buckets=_obs_parse_float_csv(
                "CODEINTEL_OTEL_METRIC_BUCKETS_HTTP_DURATION_S"
            ),
            grpc_duration_s_buckets=_obs_parse_float_csv(
                "CODEINTEL_OTEL_METRIC_BUCKETS_GRPC_DURATION_S"
            ),
        ),
        teardown_enabled=bool(
            _obs_safe_get_bool("CODEINTEL_OBSERVABILITY_TEARDOWN_ENABLED", default=True)
        ),
        teardown_task_sample_limit=_obs_opt_non_negative_int(
            "CODEINTEL_OBSERVABILITY_TEARDOWN_TASK_SAMPLE_LIMIT",
            default=5,
        ),
        teardown_thread_sample_limit=_obs_opt_non_negative_int(
            "CODEINTEL_OBSERVABILITY_TEARDOWN_THREAD_SAMPLE_LIMIT",
            default=5,
        ),
        teardown_subprocess_sample_limit=_obs_opt_non_negative_int(
            "CODEINTEL_OBSERVABILITY_TEARDOWN_SUBPROCESS_SAMPLE_LIMIT",
            default=5,
        ),
        cli_enabled=bool(_obs_safe_get_bool("CODEINTEL_OBSERVABILITY_CLI_ENABLED", default=True)),
        cli_args_allowlist=_obs_parse_csv(
            get_str("CODEINTEL_OBSERVABILITY_CLI_ARGS_ALLOWLIST", default=None)
        ),
        cli_args_capture_mode=_obs_cli_args_capture_mode(),
        cli_arg_names_max=_obs_positive_int(
            "CODEINTEL_OBSERVABILITY_CLI_ARG_NAMES_MAX",
            default=25,
        ),
        http_route_max_len=_obs_positive_int(
            "CODEINTEL_OBSERVABILITY_HTTP_ROUTE_MAX_LEN",
            default=120,
        ),
        mcp_tool_name_max_len=_obs_positive_int(
            "CODEINTEL_OBSERVABILITY_MCP_TOOL_NAME_MAX_LEN",
            default=80,
        ),
        operation_attribute_allowlist_overrides=_obs_parse_operation_allowlist_overrides(),
        grpc_observability=GrpcObservabilitySettings(
            enabled=bool(_obs_safe_get_bool("CODEINTEL_GRPC_OBSERVABILITY_ENABLED", default=False)),
            method_allowlist=_obs_parse_csv(_obs_opt_str("CODEINTEL_GRPC_METHOD_ALLOWLIST")),
            target_allowlist=_obs_parse_csv(_obs_opt_str("CODEINTEL_GRPC_TARGET_ALLOWLIST")),
            other_method_label=_obs_opt_str("CODEINTEL_GRPC_OTHER_METHOD_LABEL") or "other",
            other_target_label=_obs_opt_str("CODEINTEL_GRPC_OTHER_TARGET_LABEL") or "other",
        ),
        hamilton_tracker=HamiltonTrackerSettings(
            enabled=tracker_enabled,
            project_id=hamilton_project_id,
            username=hamilton_username,
            dag_name=_obs_opt_str("HAMILTON_DAG_NAME"),
            tags=hamilton_tags,
            api_url=_obs_opt_str("HAMILTON_API_URL"),
            ui_url=_obs_opt_str("HAMILTON_UI_URL"),
            capture_data_statistics=capture_data_statistics,
            max_list_length=max_list_length,
            max_dict_length=max_dict_length,
        ),
        duckdb_tracing_enabled=bool(
            _obs_safe_get_bool("CODEINTEL_OTEL_DUCKDB_TRACING", default=True)
        ),
        duckdb_require_parent_span=bool(
            _obs_safe_get_bool("CODEINTEL_OTEL_DUCKDB_REQUIRE_PARENT", default=True)
        ),
        duckdb_statement_mode=statement_mode_value,
        duckdb_statement_hash_len=int(
            _obs_safe_get_int("CODEINTEL_OTEL_DB_STATEMENT_HASH_LEN", default=16, min_value=1) or 16
        ),
        duckdb_query_summary_max_len=int(
            _obs_safe_get_int("CODEINTEL_OTEL_DB_QUERY_SUMMARY_MAX_LEN", default=255, min_value=1)
            or 255
        ),
        duckdb_query_summary_max_targets=int(
            _obs_safe_get_int("CODEINTEL_OTEL_DB_QUERY_SUMMARY_MAX_TARGETS", default=6, min_value=1)
            or 6
        ),
        duckdb_query_summary_emit_ellipsis=bool(
            _obs_safe_get_bool("CODEINTEL_OTEL_DB_QUERY_SUMMARY_EMIT_ELLIPSIS", default=True)
        ),
        duckdb_query_summary_hash_suspicious_targets=bool(
            _obs_safe_get_bool("CODEINTEL_OTEL_DB_QUERY_SUMMARY_HASH_SUSPICIOUS", default=True)
        ),
        duckdb_query_summary_hash_len=int(
            _obs_safe_get_int("CODEINTEL_OTEL_DB_QUERY_SUMMARY_HASH_LEN", default=12, min_value=1)
            or 12
        ),
        duckdb_query_summary_hash_min_len=int(
            _obs_safe_get_int(
                "CODEINTEL_OTEL_DB_QUERY_SUMMARY_HASH_MIN_LEN",
                default=64,
                min_value=1,
            )
            or 64
        ),
        duckdb_query_summary_include_subquery_operations=bool(
            _obs_safe_get_bool("CODEINTEL_OTEL_DB_QUERY_SUMMARY_INCLUDE_SUBQUERY_OPS", default=True)
        ),
        duckdb_query_summary_include_multi_statement=bool(
            _obs_safe_get_bool(
                "CODEINTEL_OTEL_DB_QUERY_SUMMARY_INCLUDE_MULTI_STATEMENT",
                default=True,
            )
        ),
        db_query_summary_span_name_hook=bool(
            _obs_safe_get_bool("CODEINTEL_OTEL_DB_QUERY_SUMMARY_SPAN_NAME_HOOK", default=False)
        ),
        duckdb_query_text_policy=query_text_policy_value,
        duckdb_query_text_max_len=int(
            _obs_safe_get_int("CODEINTEL_OTEL_DB_QUERY_TEXT_MAX_LEN", default=4096, min_value=1)
            or 4096
        ),
        duckdb_query_text_strip_comments=bool(
            _obs_safe_get_bool("CODEINTEL_OTEL_DB_QUERY_TEXT_STRIP_COMMENTS", default=True)
        ),
        duckdb_query_text_collapse_in_lists=bool(
            _obs_safe_get_bool("CODEINTEL_OTEL_DB_QUERY_TEXT_COLLAPSE_IN_LISTS", default=True)
        ),
        duckdb_query_parameter_enabled=bool(
            _obs_safe_get_bool("CODEINTEL_OTEL_DB_QUERY_PARAMETER_ENABLED", default=False)
        ),
        duckdb_query_parameter_keys=_obs_parse_csv(
            get_str("CODEINTEL_OTEL_DB_QUERY_PARAMETER_KEYS", default=None)
        ),
        duckdb_query_parameter_hash_keys=_obs_parse_csv(
            get_str("CODEINTEL_OTEL_DB_QUERY_PARAMETER_HASH_KEYS", default=None)
        ),
        duckdb_query_parameter_require_in_sql=bool(
            _obs_safe_get_bool("CODEINTEL_OTEL_DB_QUERY_PARAMETER_REQUIRE_IN_SQL", default=True)
        ),
        duckdb_query_parameter_max_str_len=int(
            _obs_safe_get_int(
                "CODEINTEL_OTEL_DB_QUERY_PARAMETER_MAX_STRLEN",
                default=80,
                min_value=1,
            )
            or 80
        ),
    )
    return apply_test_telemetry_settings(settings)


def load_runtime_settings() -> RuntimeSettings:
    """Load runtime settings from environment variables.

    Returns
    -------
    RuntimeSettings
        Resolved runtime settings bundle.
    """
    return RuntimeSettings(
        build=_load_build_settings(),
        cli=_load_cli_settings(),
        execution=_load_execution_settings(),
        serving=_load_serving_settings(),
        observability=_load_observability_settings(),
        variants=_load_variant_settings(),
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


def load_execution_context(*, primitives: RuntimePrimitives, run: RunContext) -> ExecutionContext:
    """Build an ExecutionContext using runtime settings from the loader.

    Parameters
    ----------
    primitives
        Runtime primitives resolved for the entrypoint.
    run
        Run context metadata for this execution.

    Returns
    -------
    ExecutionContext
        Unified execution context for the run.
    """
    bundle = RuntimeBundle(primitives=primitives, settings=load_runtime_settings())
    return ExecutionContext.from_runtime_bundle(bundle=bundle, run=run)


__all__ = [
    "RuntimeInputs",
    "build_runtime_primitives",
    "load_execution_context",
    "load_runtime_bundle",
    "load_runtime_settings",
]
