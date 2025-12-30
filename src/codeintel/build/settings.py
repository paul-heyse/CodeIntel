"""Build-wide default settings."""

from codeintel.core.config.settings import BuildSettings
from codeintel.core.config.view import SettingsView

DEFAULT_PROFILE_NAME = "full"


def get_build_settings() -> BuildSettings:
    """Return resolved build settings from runtime configuration.

    Returns
    -------
    BuildSettings
        Loaded build settings.
    """
    return SettingsView.from_runtime().build


__all__ = ["DEFAULT_PROFILE_NAME", "BuildSettings", "get_build_settings"]
