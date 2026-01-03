"""Configuration loading from multiple sources.

Load CLI configuration with proper precedence:
1. Built-in defaults (lowest priority)
2. Config file (codeintel.toml)
3. Command-line flags (highest priority)
"""

from __future__ import annotations

import logging
import os
import tomllib
from pathlib import Path
from typing import cast

import msgspec

from codeintel.cli.config.model import CliConfig, ConfigLoadError, ConfigValidationError
from codeintel.cli.config.validation import validate_config
from codeintel.cli.core.parsing import parse_bool_or_none

LOG = logging.getLogger(__name__)

TOML_CONFIG_PATHS = [
    Path("codeintel.toml"),
    Path.home() / ".codeintel" / "config.toml",
]

_ENV_OVERRIDE_KEYS: dict[str, str] = {
    "CODEINTEL_COLOR": "color",
    "CODEINTEL_LOG_LEVEL": "log_level",
    "CODEINTEL_OUTPUT_FORMAT": "output_format",
    "CODEINTEL_TELEMETRY_ENABLED": "telemetry.enabled",
    "CODEINTEL_TELEMETRY_ENDPOINT": "telemetry.endpoint",
    "CODEINTEL_TELEMETRY_SERVICE_NAME": "telemetry.service_name",
}


def load_config(
    config_file: Path | None = None,
    cli_overrides: dict[str, object] | None = None,
    *,
    validate: bool = True,
) -> CliConfig:
    """Load configuration from all sources with precedence.

    Parameters
    ----------
    config_file
        Explicit config file path.
    cli_overrides
        Command-line overrides.
    validate
        Whether to validate the configuration.

    Returns
    -------
    CliConfig
        Merged configuration.

    Raises
    ------
    ConfigLoadError
        If validation is enabled and configuration is invalid.
    """
    sources: list[str] = ["defaults"]
    merged: dict[str, object] = {}

    file_config, file_source = _load_config_file(config_file)
    if file_config:
        merged = _deep_merge(merged, file_config)
        sources.append(file_source)

    env_overrides = _load_env_overrides()
    if env_overrides:
        merged = _deep_merge(merged, env_overrides)
        sources.append("env")

    if cli_overrides:
        flat_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
        merged = _deep_merge(merged, flat_overrides)
        sources.append("cli-flags")

    config = dict_to_config(merged, sources=tuple(sources))

    if validate:
        errors = validate_config(config)
        if errors:
            error_msg = f"Configuration validation failed with {len(errors)} error(s)"
            raise ConfigLoadError(error_msg, errors=errors)

    return config


def dict_to_config(
    data: dict[str, object],
    sources: tuple[str, ...] = (),
) -> CliConfig:
    """Convert dictionary to CliConfig with strict coercion.

    Parameters
    ----------
    data
        Raw configuration dictionary.
    sources
        Sources that contributed to this config.

    Returns
    -------
    CliConfig
        Typed configuration instance.

    Raises
    ------
    ConfigLoadError
        If the configuration cannot be coerced into CliConfig.
    """
    try:
        config = msgspec.convert(
            data,
            type=CliConfig,
            strict=True,
            dec_hook=_decode_config_value,
        )
    except msgspec.ValidationError as exc:
        error = ConfigValidationError(
            path="config",
            message=str(exc),
            code="invalid_type",
            value=None,
        )
        message = "Configuration coercion failed"
        raise ConfigLoadError(message, errors=[error]) from exc

    if sources:
        return config.__replace__(_sources=sources)
    return config


def _load_env_overrides() -> dict[str, object]:
    overrides: dict[str, object] = {}
    for env_key, config_key in _ENV_OVERRIDE_KEYS.items():
        value = os.environ.get(env_key)
        if value is None:
            continue
        stripped = value.strip()
        if not stripped:
            continue
        _set_nested_override(overrides, config_key, stripped)
    return overrides


def _set_nested_override(
    overrides: dict[str, object],
    key: str,
    value: object,
) -> None:
    if "." not in key:
        overrides[key] = value
        return
    parts = key.split(".")
    cursor = overrides
    for part in parts[:-1]:
        existing = cursor.get(part)
        if isinstance(existing, dict):
            cursor = existing
            continue
        next_node: dict[str, object] = {}
        cursor[part] = next_node
        cursor = next_node
    cursor[parts[-1]] = value


def _decode_config_value(target_type: type[object], value: object) -> object:
    if target_type is Path:
        if isinstance(value, Path):
            return value
        if isinstance(value, str):
            return Path(value)
        msg = f"Expected string for path, got {type(value).__name__}"
        raise TypeError(msg)
    if target_type is bool and isinstance(value, str):
        parsed = parse_bool_or_none(value, default=None)
        if parsed is None:
            msg = f"Invalid boolean value: {value}"
            raise ValueError(msg)
        return parsed
    if target_type is int and isinstance(value, str):
        return int(value)
    if target_type is float and isinstance(value, str):
        return float(value)
    return value


def config_to_dict(config: CliConfig) -> dict[str, object]:
    """Convert CliConfig to dictionary for serialization.

    Parameters
    ----------
    config
        Configuration to convert.

    Returns
    -------
    dict[str, object]
        Dictionary representation matching JSON Schema structure.
    """
    result: dict[str, object] = {
        "output_format": config.output_format,
        "color": config.color,
        "log_level": config.log_level,
        "progress": {
            "enabled": config.progress.enabled,
            "threshold": config.progress.threshold,
        },
        "telemetry": _build_telemetry_dict(config),
        "retry": {
            "max_attempts": config.retry.max_attempts,
            "initial_delay": config.retry.initial_delay,
            "backoff_factor": config.retry.backoff_factor,
            "max_delay": config.retry.max_delay,
        },
    }

    storage_dict = _build_storage_dict(config)
    if storage_dict:
        result["storage"] = storage_dict

    project_dict = _build_project_dict(config)
    if project_dict:
        result["project"] = project_dict

    return result


def _build_telemetry_dict(config: CliConfig) -> dict[str, object]:
    """Build telemetry section dictionary.

    Parameters
    ----------
    config
        Configuration to convert.

    Returns
    -------
    dict[str, object]
        Telemetry dictionary.
    """
    result: dict[str, object] = {
        "enabled": config.telemetry.enabled,
        "service_name": config.telemetry.service_name,
    }
    if config.telemetry.endpoint:
        result["endpoint"] = config.telemetry.endpoint
    return result


def _build_storage_dict(config: CliConfig) -> dict[str, object]:
    """Build storage section dictionary.

    Parameters
    ----------
    config
        Configuration to convert.

    Returns
    -------
    dict[str, object]
        Storage dictionary (empty if no values set).
    """
    if not config.storage.db_path and not config.storage.cache_dir:
        return {}
    result: dict[str, object] = {"max_connections": config.storage.max_connections}
    if config.storage.db_path:
        result["db_path"] = str(config.storage.db_path)
    if config.storage.cache_dir:
        result["cache_dir"] = str(config.storage.cache_dir)
    return result


def _build_project_dict(config: CliConfig) -> dict[str, object]:
    """Build project section dictionary.

    Parameters
    ----------
    config
        Configuration to convert.

    Returns
    -------
    dict[str, object]
        Project dictionary (empty if no values set).
    """
    if not config.project.name and not config.project.repo and not config.project.root:
        return {}
    result: dict[str, object] = {}
    if config.project.name:
        result["name"] = config.project.name
    if config.project.repo:
        result["repo"] = config.project.repo
    if config.project.root:
        result["root"] = str(config.project.root)
    if config.project.commit:
        result["commit"] = config.project.commit
    return result


def apply_overrides(config: CliConfig, overrides: dict[str, object]) -> CliConfig:
    """Apply overrides to create new config.

    Parameters
    ----------
    config
        Base configuration.
    overrides
        Field overrides (supports dot notation).

    Returns
    -------
    CliConfig
        New configuration with overrides applied.
    """
    data = config_to_dict(config)

    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            target: dict[str, object] = data
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                nested = target[part]
                if isinstance(nested, dict):
                    target = nested
            target[parts[-1]] = value
        else:
            data[key] = value

    return dict_to_config(data, sources=tuple(config.config_sources))


def _load_config_file(
    explicit_path: Path | None,
) -> tuple[dict[str, object] | None, str]:
    """Load configuration from file.

    Parameters
    ----------
    explicit_path
        Explicit path or None to search defaults.

    Returns
    -------
    tuple[dict[str, object] | None, str]
        Loaded config and source description.
    """
    if explicit_path and explicit_path.exists():
        config = _parse_config_file(explicit_path)
        return config, f"file:{explicit_path}"

    for path in TOML_CONFIG_PATHS:
        if path.exists():
            LOG.debug("Loading config from %s", path)
            config = _parse_config_file(path)
            return config, f"file:{path}"

    return None, ""


def _parse_config_file(path: Path) -> dict[str, object]:
    """Parse a configuration file.

    Parameters
    ----------
    path
        Path to config file.

    Returns
    -------
    dict[str, object]
        Parsed configuration.
    """
    with path.open("rb") as handle:
        parsed = tomllib.load(handle)

    if not isinstance(parsed, dict):
        return {}
    return cast("dict[str, object]", parsed)


def _deep_merge(base: dict[str, object], override: dict[str, object]) -> dict[str, object]:
    """Deep merge two dictionaries.

    Parameters
    ----------
    base
        Base dictionary.
    override
        Override dictionary.

    Returns
    -------
    dict[str, object]
        Merged dictionary.
    """
    result: dict[str, object] = dict(base)
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = value
    return result


__all__ = [
    "TOML_CONFIG_PATHS",
    "apply_overrides",
    "config_to_dict",
    "dict_to_config",
    "load_config",
]
