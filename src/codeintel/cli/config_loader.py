"""Configuration loading and application.

Load CLI configuration from multiple sources with proper precedence:
1. Built-in defaults (lowest priority)
2. Config file (~/.codeintel/config.yaml)
3. Environment variables
4. Command-line flags (highest priority)
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from codeintel.cli.cli_config_schema import validate_with_json_schema
from codeintel.cli.cli_resilience import RetryPolicy

LOG = logging.getLogger(__name__)

DEFAULT_CONFIG_PATHS = [
    Path.home() / ".codeintel" / "config.yaml",
    Path.home() / ".codeintel" / "config.json",
    Path(".codeintel.yaml"),
    Path(".codeintel.json"),
]


@dataclass
class ResolvedConfig:
    """Fully resolved CLI configuration.

    Parameters
    ----------
    output_format
        Default output format.
    color
        Enable colored output.
    progress
        Show progress bars.
    progress_threshold
        Minimum duration (seconds) before showing progress.
    retry_policy
        Default retry policy.
    telemetry_enabled
        Enable telemetry.
    log_level
        Logging level.
    project_root
        Default project root.
    config_sources
        List of sources that contributed to this config.
    """

    output_format: str = "text"
    color: bool = True
    progress: bool = True
    progress_threshold: float = 2.0
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    telemetry_enabled: bool = True
    log_level: str = "WARNING"
    project_root: Path | None = None
    config_sources: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation.
        """
        return {
            "output_format": self.output_format,
            "color": self.color,
            "progress": self.progress,
            "progress_threshold": self.progress_threshold,
            "telemetry_enabled": self.telemetry_enabled,
            "log_level": self.log_level,
            "project_root": str(self.project_root) if self.project_root else None,
            "sources": self.config_sources,
            "retry": {
                "max_attempts": self.retry_policy.max_attempts,
                "initial_delay": self.retry_policy.initial_delay,
                "backoff_factor": self.retry_policy.backoff_factor,
            },
        }


class ConfigValidationError(Exception):
    """Configuration validation error.

    Parameters
    ----------
    errors
        List of validation error descriptions.
    """

    def __init__(self, errors: list[str]) -> None:
        """Initialize error."""
        self.errors = errors
        super().__init__(f"Configuration validation failed: {len(errors)} error(s)")


def load_config(
    *,
    config_file: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
    validate: bool = False,
) -> ResolvedConfig:
    """Load configuration from all sources.

    Parameters
    ----------
    config_file
        Explicit config file path.
    cli_overrides
        Command-line overrides.
    validate
        Whether to validate against JSON Schema.

    Returns
    -------
    ResolvedConfig
        Merged configuration.

    Raises
    ------
    ConfigValidationError
        If validation is enabled and configuration is invalid.
    """
    sources: list[str] = []
    merged: dict[str, Any] = {}

    # 1. Built-in defaults
    merged.update(_get_defaults())
    sources.append("defaults")

    # 2. Config file
    file_config, file_source = _load_config_file(config_file)
    if file_config:
        merged.update(file_config)
        sources.append(file_source)

    # 3. Environment variables
    env_config = _load_env_config()
    if env_config:
        merged.update(env_config)
        sources.append("environment")

    # 4. CLI overrides
    if cli_overrides:
        merged.update({k: v for k, v in cli_overrides.items() if v is not None})
        sources.append("cli-flags")

    # 5. Validate if requested
    if validate:
        errors = validate_with_json_schema(merged)
        if errors:
            raise ConfigValidationError([str(e) for e in errors])

    return _build_resolved_config(merged, sources)


def _get_defaults() -> dict[str, Any]:
    """Get built-in default values.

    Returns
    -------
    dict[str, Any]
        Default configuration.
    """
    return {
        "output_format": "text",
        "color": True,
        "progress": True,
        "progress_threshold": 2.0,
        "telemetry_enabled": True,
        "log_level": "WARNING",
    }


def _load_config_file(
    explicit_path: Path | None,
) -> tuple[dict[str, Any] | None, str]:
    """Load configuration from file.

    Parameters
    ----------
    explicit_path
        Explicit path or None to search defaults.

    Returns
    -------
    tuple[dict[str, Any] | None, str]
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


def _parse_config_file(path: Path) -> dict[str, Any]:
    """Parse a configuration file.

    Parameters
    ----------
    path
        Path to config file.

    Returns
    -------
    dict[str, Any]
        Parsed configuration.
    """
    content = path.read_text(encoding="utf-8")
    is_yaml = path.suffix in {".yaml", ".yml"}
    data = yaml.safe_load(content) if is_yaml else json.loads(content)

    if not isinstance(data, dict):
        return {}
    return data


def _load_env_config() -> dict[str, Any]:
    """Load configuration from environment variables.

    Returns
    -------
    dict[str, Any]
        Environment-based config.
    """
    config: dict[str, Any] = {}

    env_mappings: dict[str, str | tuple[str, type]] = {
        "CODEINTEL_OUTPUT_FORMAT": "output_format",
        "CODEINTEL_COLOR": ("color", bool),
        "CODEINTEL_PROGRESS": ("progress", bool),
        "CODEINTEL_TELEMETRY": ("telemetry_enabled", bool),
        "CODEINTEL_LOG_LEVEL": "log_level",
        "CODEINTEL_PROJECT_ROOT": ("project_root", Path),
    }

    for env_var, mapping in env_mappings.items():
        value = os.environ.get(env_var)
        if value is not None:
            if isinstance(mapping, tuple):
                key, converter = mapping
                if converter is bool:
                    config[key] = _parse_bool(value)
                elif converter is Path:
                    config[key] = Path(value)
                else:
                    config[key] = value
            else:
                config[mapping] = value

    return config


def _parse_bool(value: str) -> bool:
    """Parse boolean from string.

    Parameters
    ----------
    value
        String value.

    Returns
    -------
    bool
        Parsed boolean.
    """
    return value.lower() in {"true", "1", "yes", "on"}


def _build_resolved_config(
    merged: dict[str, Any],
    sources: list[str],
) -> ResolvedConfig:
    """Build ResolvedConfig from merged dict.

    Parameters
    ----------
    merged
        Merged configuration dict.
    sources
        Sources that contributed.

    Returns
    -------
    ResolvedConfig
        Resolved configuration.
    """
    retry_config = merged.get("retry", {})
    if isinstance(retry_config, dict):
        retry_policy = RetryPolicy(
            max_attempts=retry_config.get("max_attempts", 3),
            initial_delay=retry_config.get("initial_delay", 0.5),
            backoff_factor=retry_config.get("backoff_factor", 2.0),
        )
    else:
        retry_policy = RetryPolicy()

    project_root = merged.get("project_root")
    if isinstance(project_root, str):
        project_root = Path(project_root)
    elif not isinstance(project_root, (Path, type(None))):
        project_root = None

    return ResolvedConfig(
        output_format=str(merged.get("output_format", "text")),
        color=bool(merged.get("color", True)),
        progress=bool(merged.get("progress", True)),
        progress_threshold=float(merged.get("progress_threshold", 2.0)),
        retry_policy=retry_policy,
        telemetry_enabled=bool(merged.get("telemetry_enabled", True)),
        log_level=str(merged.get("log_level", "WARNING")),
        project_root=project_root,
        config_sources=sources,
    )


__all__ = [
    "DEFAULT_CONFIG_PATHS",
    "ConfigValidationError",
    "ResolvedConfig",
    "load_config",
]
