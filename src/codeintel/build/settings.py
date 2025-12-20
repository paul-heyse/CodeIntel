"""Build settings for runtime configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from codeintel.core.config.settings import (
    BuildSettings,
    ExportAuditSettings,
    HamiltonExecutionSettings,
)
from codeintel.core.env import get_bool, get_int, get_path, get_str, split_csv

__all__ = ["BuildSettings", "get_build_settings", "get_hamilton_execution_settings"]


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


@lru_cache(maxsize=1)
def get_build_settings() -> BuildSettings:
    """Return cached build settings resolved from the environment.

    Returns
    -------
    BuildSettings
        Resolved build settings.
    """
    return BuildSettings(
        engine_version=_resolve_engine_version(),
        export_audit=ExportAuditSettings(
            log_path=_resolve_export_audit_log_path(),
            table_enabled=_resolve_export_audit_table_enabled(),
        ),
    )


def get_hamilton_execution_settings() -> HamiltonExecutionSettings:
    """Return Hamilton execution settings resolved from the environment.

    Returns
    -------
    HamiltonExecutionSettings
        Execution settings derived from environment variables.
    """
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
