"""Environment variable parsing for CLI configuration.

Parse environment variables with support for nested paths:
- CODEINTEL_OUTPUT_FORMAT -> output_format
- CODEINTEL_PROGRESS_ENABLED -> progress.enabled
- CODEINTEL_TELEMETRY_ENDPOINT -> telemetry.endpoint
"""

from __future__ import annotations

import os
from pathlib import Path

from codeintel.cli.core.parsing import parse_bool

ENV_MAPPINGS: dict[str, tuple[str, type]] = {
    "OUTPUT_FORMAT": ("output_format", str),
    "COLOR": ("color", bool),
    "LOG_LEVEL": ("log_level", str),
    "PROGRESS_ENABLED": ("progress.enabled", bool),
    "PROGRESS_THRESHOLD": ("progress.threshold", float),
    "RETRY_MAX_ATTEMPTS": ("retry.max_attempts", int),
    "RETRY_INITIAL_DELAY": ("retry.initial_delay", float),
    "RETRY_BACKOFF_FACTOR": ("retry.backoff_factor", float),
    "RETRY_MAX_DELAY": ("retry.max_delay", float),
    "STORAGE_DB_PATH": ("storage.db_path", Path),
    "STORAGE_CACHE_DIR": ("storage.cache_dir", Path),
    "STORAGE_MAX_CONNECTIONS": ("storage.max_connections", int),
    "PROJECT_NAME": ("project.name", str),
    "PROJECT_REPO": ("project.repo", str),
    "PROJECT_ROOT": ("project.root", Path),
    "PROJECT_COMMIT": ("project.commit", str),
}


def load_env_config(prefix: str = "CODEINTEL_") -> dict[str, object]:
    """Load configuration from environment variables.

    Parameters
    ----------
    prefix
        Environment variable prefix.

    Returns
    -------
    dict[str, object]
        Nested configuration dictionary.
    """
    config: dict[str, object] = {}

    for env_suffix, (config_path, value_type) in ENV_MAPPINGS.items():
        env_var = f"{prefix}{env_suffix}"
        value = os.environ.get(env_var)

        if value is not None:
            converted = _convert_value(value, value_type)
            _set_nested(config, config_path, value=converted)

    return config


def _convert_value(value: str, target_type: type) -> str | bool | int | float:
    """Convert string value to target type.

    Parameters
    ----------
    value
        String value from environment.
    target_type
        Target Python type.

    Returns
    -------
    str | bool | int | float
        Converted value.
    """
    if target_type is bool:
        return parse_bool(value)
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)

    return value


def _set_nested(
    config: dict[str, object],
    path: str,
    *,
    value: str | bool | float,
) -> None:
    """Set a nested value in config dictionary.

    Parameters
    ----------
    config
        Configuration dictionary to modify.
    path
        Dot-separated path (e.g., "progress.enabled").
    value
        Value to set.
    """
    parts = path.split(".")
    target = config

    for part in parts[:-1]:
        if part not in target:
            target[part] = {}
        nested = target[part]
        if isinstance(nested, dict):
            target = nested

    target[parts[-1]] = value


__all__ = [
    "ENV_MAPPINGS",
    "load_env_config",
]
