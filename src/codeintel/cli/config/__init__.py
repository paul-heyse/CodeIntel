"""Unified CLI configuration package.

This package provides the single source of truth for CLI configuration:

- ``ConfigService``: Unified configuration service
- ``CliConfig``: The root configuration model with nested sections
- ``load_config``: Load configuration from files, environment, and CLI flags
- ``generate_schema``: Generate JSON Schema 2020-12 from the model
- ``validate_config``: Validate configuration against constraints

Examples
--------
>>> from codeintel.cli.config import CliConfig, load_config
>>> config = load_config()
>>> config.progress.enabled
True
>>> config.telemetry.service_name
'codeintel-cli'

>>>
>>> from codeintel.cli.config import generate_schema
>>> schema = generate_schema(CliConfig)
>>> schema["$schema"]
'https://json-schema.org/draft/2020-12/schema'
"""

from __future__ import annotations

from codeintel.cli.config.env import ENV_MAPPINGS, load_env_config
from codeintel.cli.config.loader import (
    DEFAULT_CONFIG_PATHS,
    apply_overrides,
    config_to_dict,
    dict_to_config,
    load_config,
)
from codeintel.cli.config.model import (
    CliConfig,
    ConfigLoadError,
    ConfigValidationError,
    LogLevel,
    OutputFormat,
    PluginsConfigSection,
    ProgressConfig,
    ProjectConfigSection,
    RetryConfig,
    StorageConfigSection,
    TelemetryConfig,
)
from codeintel.cli.config.schema import export_schema, generate_schema
from codeintel.cli.config.service import (
    CONFIG_ENV_PREFIX,
    CONFIG_PATH_ENV_VAR,
    TOML_CONFIG_PATHS,
    ConfigService,
    build_config_from_options,
    build_graph_backend_config,
    build_graph_feature_flags_from_env,
)
from codeintel.cli.config.validation import validate_config, validate_with_json_schema

__all__ = [
    "CONFIG_ENV_PREFIX",
    "CONFIG_PATH_ENV_VAR",
    "DEFAULT_CONFIG_PATHS",
    "ENV_MAPPINGS",
    "TOML_CONFIG_PATHS",
    "CliConfig",
    "ConfigLoadError",
    "ConfigService",
    "ConfigValidationError",
    "LogLevel",
    "OutputFormat",
    "PluginsConfigSection",
    "ProgressConfig",
    "ProjectConfigSection",
    "RetryConfig",
    "StorageConfigSection",
    "TelemetryConfig",
    "apply_overrides",
    "build_config_from_options",
    "build_graph_backend_config",
    "build_graph_feature_flags_from_env",
    "config_to_dict",
    "dict_to_config",
    "export_schema",
    "generate_schema",
    "load_config",
    "load_env_config",
    "validate_config",
    "validate_with_json_schema",
]
