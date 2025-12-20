"""Build settings for runtime configuration."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

__all__ = [
    "BuildSettings",
    "get_build_settings",
]


@dataclass(frozen=True, slots=True)
class BuildSettings:
    """Runtime configuration values for build behavior."""

    engine_version: str
    export_audit_log_path: str | None
    export_audit_table_enabled: bool


def _resolve_engine_version() -> str:
    override = os.environ.get("CODEINTEL_BUILD_ENGINE_VERSION", "").strip()
    if override:
        return override
    try:
        return version("codeintel")
    except PackageNotFoundError:
        return "unknown"


def _resolve_export_audit_log_path() -> str | None:
    value = os.environ.get("CODEINTEL_EXPORT_AUDIT_LOG")
    return value.strip() if value else None


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
        export_audit_log_path=_resolve_export_audit_log_path(),
        export_audit_table_enabled=_resolve_export_audit_table_enabled(),
    )
