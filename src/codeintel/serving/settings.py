"""Environment-driven serving configuration."""

from __future__ import annotations

from codeintel.core.config.settings import ServingSettings
from codeintel.core.config.view import SettingsView


def get_serving_settings() -> ServingSettings:
    """Load serving settings from environment variables.

    Returns
    -------
    ServingSettings
        Loaded settings.
    """
    return SettingsView.from_runtime().require_serving()


__all__ = ["ServingSettings", "get_serving_settings"]
