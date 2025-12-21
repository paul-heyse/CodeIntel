"""Environment-driven serving configuration."""

from __future__ import annotations

from codeintel.core.config.settings import ServingSettings
from codeintel.core.runtime.loader import load_runtime_settings


def get_serving_settings() -> ServingSettings:
    """Load serving settings from environment variables.

    Returns
    -------
    ServingSettings
        Loaded settings.
    """
    return load_runtime_settings().serving


__all__ = ["ServingSettings", "get_serving_settings"]
