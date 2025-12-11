"""Configuration loading from multiple sources.

Load CLI configuration with proper precedence:
1. Built-in defaults (lowest priority)
2. Config file (~/.codeintel/config.yaml)
3. Environment variables
4. Command-line flags (highest priority)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import cast

import yaml

from codeintel.cli.config.env import load_env_config
from codeintel.cli.config.model import (
    CliConfig,
    ConfigLoadError,
    LogLevel,
    OutputFormat,
    PluginsConfigSection,
    ProgressConfig,
    ProjectConfigSection,
    RetryConfig,
    StorageConfigSection,
    TelemetryConfig,
)
from codeintel.cli.config.validation import validate_config
from codeintel.cli.core.parsing import parse_bool

LOG = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = [
    Path.home() / ".codeintel" / "config.yaml",
    Path.home() / ".codeintel" / "config.json",
    Path(".codeintel.yaml"),
    Path(".codeintel.json"),
]

# Valid enum values
VALID_OUTPUT_FORMATS = {"text", "json"}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


def load_config(
    config_file: Path | None = None,
    env_prefix: str = "CODEINTEL_",
    cli_overrides: dict[str, object] | None = None,
    *,
    validate: bool = True,
) -> CliConfig:
    """Load configuration from all sources with precedence.

    Parameters
    ----------
    config_file
        Explicit config file path.
    env_prefix
        Prefix for environment variables.
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

    # 1. Config file
    file_config, file_source = _load_config_file(config_file)
    if file_config:
        merged = _deep_merge(merged, file_config)
        sources.append(file_source)

    # 2. Environment variables
    env_config = load_env_config(env_prefix)
    if env_config:
        merged = _deep_merge(merged, env_config)
        sources.append("environment")

    # 3. CLI overrides
    if cli_overrides:
        flat_overrides = {k: v for k, v in cli_overrides.items() if v is not None}
        merged = _deep_merge(merged, flat_overrides)
        sources.append("cli-flags")

    # Build config from merged dict
    config = dict_to_config(merged, sources=tuple(sources))

    # 4. Validate if requested
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
    """Convert dictionary to CliConfig with type coercion.

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
    """
    progress = _parse_progress(data)
    telemetry = _parse_telemetry(data)
    retry = _parse_retry(data)
    storage = _parse_storage(data)
    project = _parse_project(data)
    plugins = _parse_plugins(data)

    output_format_raw = _get_string(data, "output_format", "text")
    output_format_value: OutputFormat = cast(
        "OutputFormat",
        output_format_raw if output_format_raw in VALID_OUTPUT_FORMATS else "text",
    )
    log_level_raw = _get_string(data, "log_level", "WARNING")
    log_level_value: LogLevel = cast(
        "LogLevel", log_level_raw if log_level_raw in VALID_LOG_LEVELS else "WARNING"
    )

    return CliConfig(
        output_format=output_format_value,
        color=_get_bool(data, "color", default=True),
        log_level=log_level_value,
        progress=progress,
        telemetry=telemetry,
        retry=retry,
        storage=storage,
        project=project,
        plugins=plugins,
        _sources=sources,
    )


def _parse_progress(data: dict[str, object]) -> ProgressConfig:
    """Parse progress config section.

    Parameters
    ----------
    data
        Raw configuration dictionary.

    Returns
    -------
    ProgressConfig
        Parsed progress configuration.
    """
    progress_data = data.get("progress", {})
    if isinstance(progress_data, dict):
        return ProgressConfig(
            enabled=_get_bool(progress_data, "enabled", default=True),
            threshold=_get_float(progress_data, "threshold", default=2.0),
        )
    return ProgressConfig()


def _parse_telemetry(data: dict[str, object]) -> TelemetryConfig:
    """Parse telemetry config section.

    Parameters
    ----------
    data
        Raw configuration dictionary.

    Returns
    -------
    TelemetryConfig
        Parsed telemetry configuration.
    """
    telemetry_data = data.get("telemetry", {})
    if isinstance(telemetry_data, dict):
        return TelemetryConfig(
            enabled=_get_bool(telemetry_data, "enabled", default=True),
            endpoint=_get_optional_string(telemetry_data, "endpoint"),
            service_name=_get_string(telemetry_data, "service_name", "codeintel-cli"),
        )
    return TelemetryConfig()


def _parse_retry(data: dict[str, object]) -> RetryConfig:
    """Parse retry config section.

    Parameters
    ----------
    data
        Raw configuration dictionary.

    Returns
    -------
    RetryConfig
        Parsed retry configuration.
    """
    retry_data = data.get("retry", {})
    if isinstance(retry_data, dict):
        return RetryConfig(
            max_attempts=_get_int(retry_data, "max_attempts", default=3),
            initial_delay=_get_float(retry_data, "initial_delay", default=0.5),
            backoff_factor=_get_float(retry_data, "backoff_factor", default=2.0),
            max_delay=_get_float(retry_data, "max_delay", default=30.0),
        )
    return RetryConfig()


def _parse_storage(data: dict[str, object]) -> StorageConfigSection:
    """Parse storage config section.

    Parameters
    ----------
    data
        Raw configuration dictionary.

    Returns
    -------
    StorageConfigSection
        Parsed storage configuration.
    """
    storage_data = data.get("storage", {})
    if isinstance(storage_data, dict):
        db_path = _get_optional_string(storage_data, "db_path")
        cache_dir = _get_optional_string(storage_data, "cache_dir")
        return StorageConfigSection(
            db_path=Path(db_path) if db_path else None,
            cache_dir=Path(cache_dir) if cache_dir else None,
            max_connections=_get_int(storage_data, "max_connections", default=5),
        )
    return StorageConfigSection()


def _parse_project(data: dict[str, object]) -> ProjectConfigSection:
    """Parse project config section.

    Parameters
    ----------
    data
        Raw configuration dictionary.

    Returns
    -------
    ProjectConfigSection
        Parsed project configuration.
    """
    project_data = data.get("project", {})
    if isinstance(project_data, dict):
        root = _get_optional_string(project_data, "root")
        return ProjectConfigSection(
            name=_get_optional_string(project_data, "name"),
            repo=_get_optional_string(project_data, "repo"),
            root=Path(root) if root else None,
            commit=_get_optional_string(project_data, "commit"),
        )
    return ProjectConfigSection()


def _parse_plugins(data: dict[str, object]) -> PluginsConfigSection:
    """Parse plugins config section.

    Parameters
    ----------
    data
        Raw configuration dictionary.

    Returns
    -------
    PluginsConfigSection
        Parsed plugins configuration.
    """
    plugins_data = data.get("plugins", {})
    if isinstance(plugins_data, dict):
        directories = plugins_data.get("directories", [])
        disabled = plugins_data.get("disabled", [])
        if isinstance(directories, list) and isinstance(disabled, list):
            return PluginsConfigSection(
                directories=tuple(Path(str(d)) for d in directories),
                disabled=tuple(str(d) for d in disabled),
            )
    return PluginsConfigSection()


def _get_string(data: dict[str, object], key: str, default: str) -> str:
    """Get string value from dict with default.

    Parameters
    ----------
    data
        Dictionary to get value from.
    key
        Key to look up.
    default
        Default value if not found.

    Returns
    -------
    str
        String value.
    """
    value = data.get(key, default)
    return str(value) if value is not None else default


def _get_optional_string(data: dict[str, object], key: str) -> str | None:
    """Get optional string value from dict.

    Parameters
    ----------
    data
        Dictionary to get value from.
    key
        Key to look up.

    Returns
    -------
    str | None
        String value or None.
    """
    value = data.get(key)
    return str(value) if value is not None else None


def _get_bool(data: dict[str, object], key: str, *, default: bool) -> bool:
    """Get boolean value from dict with default.

    Parameters
    ----------
    data
        Dictionary to get value from.
    key
        Key to look up.
    default
        Default value if not found.

    Returns
    -------
    bool
        Boolean value.
    """
    value = data.get(key, default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return parse_bool(value)
    return bool(value)


def _get_int(data: dict[str, object], key: str, *, default: int) -> int:
    """Get integer value from dict with default.

    Parameters
    ----------
    data
        Dictionary to get value from.
    key
        Key to look up.
    default
        Default value if not found.

    Returns
    -------
    int
        Integer value.
    """
    value = data.get(key, default)
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    return default


def _get_float(data: dict[str, object], key: str, *, default: float) -> float:
    """Get float value from dict with default.

    Parameters
    ----------
    data
        Dictionary to get value from.
    key
        Key to look up.
    default
        Default value if not found.

    Returns
    -------
    float
        Float value.
    """
    value = data.get(key, default)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    return default


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

    plugins_dict = _build_plugins_dict(config)
    if plugins_dict:
        result["plugins"] = plugins_dict

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


def _build_plugins_dict(config: CliConfig) -> dict[str, object]:
    """Build plugins section dictionary.

    Parameters
    ----------
    config
        Configuration to convert.

    Returns
    -------
    dict[str, object]
        Plugins dictionary (empty if no values set).
    """
    if not config.plugins.directories and not config.plugins.disabled:
        return {}
    result: dict[str, object] = {}
    if config.plugins.directories:
        result["directories"] = [str(d) for d in config.plugins.directories]
    if config.plugins.disabled:
        result["disabled"] = list(config.plugins.disabled)
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
    # Convert to dict, apply overrides, convert back
    data = config_to_dict(config)

    for key, value in overrides.items():
        if "." in key:
            # Handle nested keys like "progress.enabled"
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

    return dict_to_config(data, sources=config._sources)  # noqa: SLF001


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

    for path in DEFAULT_CONFIG_PATHS:
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
    content = path.read_text(encoding="utf-8")
    is_yaml = path.suffix in {".yaml", ".yml"}
    parsed = yaml.safe_load(content) if is_yaml else json.loads(content)

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
    "DEFAULT_CONFIG_PATHS",
    "apply_overrides",
    "config_to_dict",
    "dict_to_config",
    "load_config",
]
