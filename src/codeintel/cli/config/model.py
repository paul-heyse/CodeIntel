"""Unified CLI configuration model.

This module defines the single source of truth for CLI configuration.
The model:
1. Defines all configuration with typed defaults
2. Generates JSON Schema 2020-12 via to_json_schema()
3. Validates at load time via from_dict()

Note: To avoid circular imports, configuration loading, schema generation,
and validation are provided as standalone functions in the config package.
Use CliConfig.from_sources() which delegates to config.load_config().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Literal

_PATH = Path


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
OutputFormat = Literal["text", "json"]


@dataclass(frozen=True)
class ProgressConfig:
    """Progress display configuration.

    Parameters
    ----------
    enabled
        Whether to show progress bars for long operations.
    threshold
        Minimum seconds before showing progress bar.
    """

    enabled: bool = True
    threshold: float = 2.0


@dataclass(frozen=True)
class TelemetryConfig:
    """Telemetry and observability configuration.

    Parameters
    ----------
    enabled
        Whether to collect and send telemetry.
    endpoint
        OTLP collector endpoint URL.
    service_name
        Service name for traces and metrics.
    """

    enabled: bool = True
    endpoint: str | None = None
    service_name: str = "codeintel-cli"


@dataclass(frozen=True)
class RetryConfig:
    """Retry policy configuration.

    Parameters
    ----------
    max_attempts
        Maximum retry attempts for retryable operations.
    initial_delay
        Initial retry delay in seconds.
    backoff_factor
        Exponential backoff multiplier.
    max_delay
        Maximum retry delay in seconds.
    """

    max_attempts: int = 3
    initial_delay: float = 0.5
    backoff_factor: float = 2.0
    max_delay: float = 30.0


@dataclass(frozen=True)
class StorageConfigSection:
    """Storage backend configuration.

    Parameters
    ----------
    db_path
        Path to DuckDB database file.
    cache_dir
        Directory for cached data.
    max_connections
        Maximum database connections.
    """

    db_path: Path | None = None
    cache_dir: Path | None = None
    max_connections: int = 5


@dataclass(frozen=True)
class ProjectConfigSection:
    """Project identification configuration.

    Parameters
    ----------
    name
        Project name.
    repo
        Repository identifier.
    root
        Project root directory path.
    commit
        Current commit SHA.
    """

    name: str | None = None
    repo: str | None = None
    root: Path | None = None
    commit: str | None = None


@dataclass(frozen=True)
class PluginsConfigSection:
    """Plugin system configuration.

    Parameters
    ----------
    directories
        Additional directories to search for plugins.
    disabled
        Plugin names to disable.
    """

    directories: tuple[Path, ...] = ()
    disabled: tuple[str, ...] = ()


@dataclass(frozen=True)
class CliConfig:
    """Complete CLI configuration - single source of truth.

    This model:
    1. Defines all configuration with typed defaults
    2. Generates JSON Schema 2020-12 via to_json_schema()
    3. Validates at load time via from_dict()

    Parameters
    ----------
    output_format
        Default output format for CLI commands.
    color
        Enable colored output in terminal.
    log_level
        Logging level for CLI output.
    progress
        Progress display configuration.
    telemetry
        Telemetry and observability configuration.
    retry
        Retry policy configuration.
    storage
        Storage backend configuration.
    project
        Project identification configuration.
    plugins
        Plugin system configuration.

    Examples
    --------
    >>> from codeintel.cli.config import load_config
    >>> config = load_config()
    >>> config.progress.enabled
    True
    >>> config.telemetry.service_name
    'codeintel-cli'
    """

    output_format: OutputFormat = "text"
    color: bool = True
    log_level: LogLevel = "WARNING"

    progress: ProgressConfig = field(default_factory=ProgressConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    retry: RetryConfig = field(default_factory=RetryConfig)
    storage: StorageConfigSection = field(default_factory=StorageConfigSection)
    project: ProjectConfigSection = field(default_factory=ProjectConfigSection)
    plugins: PluginsConfigSection = field(default_factory=PluginsConfigSection)

    _sources: tuple[str, ...] = field(default=(), repr=False, compare=False)

    SCHEMA_ID: ClassVar[str] = "https://codeintel.dev/schemas/cli-config.json"
    SCHEMA_TITLE: ClassVar[str] = "CodeIntel CLI Configuration"

    @property
    def config_sources(self) -> list[str]:
        """Get list of config sources.

        Returns
        -------
        list[str]
            Sources that contributed to this config.
        """
        return list(self._sources)


@dataclass(frozen=True)
class ConfigValidationError:
    """Configuration validation error.

    Parameters
    ----------
    path
        Dot-separated path to the invalid field.
    message
        Human-readable error message.
    code
        Machine-readable error code.
    value
        The invalid value (if safe to include).
    """

    path: str
    message: str
    code: str
    value: object = None


class ConfigLoadError(Exception):
    """Configuration loading failed.

    Parameters
    ----------
    message
        Error description.
    errors
        List of validation errors.
    """

    def __init__(self, message: str, errors: list[ConfigValidationError] | None = None) -> None:
        """Initialize error."""
        super().__init__(message)
        self.errors = errors or []


__all__ = [
    "CliConfig",
    "ConfigLoadError",
    "ConfigValidationError",
    "LogLevel",
    "OutputFormat",
    "PluginsConfigSection",
    "ProgressConfig",
    "ProjectConfigSection",
    "RetryConfig",
    "StorageConfigSection",
    "TelemetryConfig",
]
