"""Build-wide default settings."""

from codeintel.core.config.settings import (
    ArrowScanSettings,
    BuildSettings,
    ColumnarRuntimeSettings,
)
from codeintel.core.runtime.loader import load_runtime_settings

DEFAULT_PROFILE_NAME = "full"


def get_build_settings() -> BuildSettings:
    """Return resolved build settings from runtime configuration.

    Returns
    -------
    BuildSettings
        Loaded build settings.
    """
    return load_runtime_settings().build


def get_arrow_scan_settings() -> ArrowScanSettings:
    """Return resolved Arrow scan settings from runtime configuration.

    Returns
    -------
    ArrowScanSettings
        Loaded Arrow scan settings.
    """
    return load_runtime_settings().build.arrow_scan


def get_columnar_runtime_settings() -> ColumnarRuntimeSettings:
    """Return resolved columnar runtime settings from runtime configuration.

    Returns
    -------
    ColumnarRuntimeSettings
        Loaded columnar runtime settings.
    """
    return load_runtime_settings().columnar


__all__ = [
    "DEFAULT_PROFILE_NAME",
    "ArrowScanSettings",
    "BuildSettings",
    "ColumnarRuntimeSettings",
    "get_arrow_scan_settings",
    "get_build_settings",
    "get_columnar_runtime_settings",
]
