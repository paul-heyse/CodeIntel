"""Build settings for runtime configuration."""

from __future__ import annotations

from codeintel.core.config.settings import BuildSettings, HamiltonExecutionSettings
from codeintel.core.runtime.loader import load_runtime_settings

__all__ = ["BuildSettings", "get_build_settings", "get_hamilton_execution_settings"]


def get_build_settings() -> BuildSettings:
    """Return cached build settings resolved from the environment.

    Returns
    -------
    BuildSettings
        Resolved build settings.
    """
    return load_runtime_settings().build


def get_hamilton_execution_settings() -> HamiltonExecutionSettings:
    """Return Hamilton execution settings resolved from the environment.

    Returns
    -------
    HamiltonExecutionSettings
        Execution settings derived from environment variables.
    """
    return load_runtime_settings().execution
