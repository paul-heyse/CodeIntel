"""Configuration schema definitions and validation.

This module defines the expected structure of configuration files
and provides validation using both JSON Schema 2020-12 for IDE support
and the cli_validation infrastructure for runtime validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

import yaml

from codeintel.cli.cli_validation import (
    FloatValidator,
    IntValidator,
    PathValidator,
    StringValidator,
    ValidationError,
    ValidationResult,
    ValidationSchema,
)

# Optional jsonschema import - used for JSON Schema 2020-12 validation
try:
    import jsonschema as _jsonschema_module
except ImportError:
    _jsonschema_module = None  # type: ignore[assignment]

# =============================================================================
# JSON Schema 2020-12 Definition for IDE Autocomplete Support
# =============================================================================

CLI_CONFIG_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "https://codeintel.dev/schemas/cli-config.json",
    "title": "CodeIntel CLI Configuration",
    "description": "Configuration schema for the CodeIntel CLI",
    "type": "object",
    "properties": {
        "output_format": {
            "type": "string",
            "enum": ["text", "json"],
            "default": "text",
            "description": "Default output format for CLI commands",
        },
        "color": {
            "type": "boolean",
            "default": True,
            "description": "Enable colored output in terminal",
        },
        "progress": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "Show progress bars for long operations",
                },
                "threshold": {
                    "type": "number",
                    "minimum": 0,
                    "default": 2.0,
                    "description": "Minimum seconds before showing progress bar",
                },
            },
            "additionalProperties": False,
        },
        "telemetry": {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable telemetry collection",
                },
                "endpoint": {
                    "type": "string",
                    "format": "uri",
                    "description": "OTLP collector endpoint URL",
                },
                "service_name": {
                    "type": "string",
                    "default": "codeintel-cli",
                    "description": "Service name for traces and metrics",
                },
            },
            "additionalProperties": False,
        },
        "retry": {
            "type": "object",
            "properties": {
                "max_attempts": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": 3,
                    "description": "Maximum retry attempts for retryable operations",
                },
                "initial_delay": {
                    "type": "number",
                    "minimum": 0,
                    "default": 0.5,
                    "description": "Initial retry delay in seconds",
                },
                "backoff_factor": {
                    "type": "number",
                    "minimum": 1,
                    "default": 2.0,
                    "description": "Exponential backoff multiplier",
                },
                "max_delay": {
                    "type": "number",
                    "minimum": 0,
                    "default": 30.0,
                    "description": "Maximum retry delay in seconds",
                },
            },
            "additionalProperties": False,
        },
        "log_level": {
            "type": "string",
            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            "default": "WARNING",
            "description": "Logging level for CLI output",
        },
        "project": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 100,
                    "description": "Project name",
                },
                "repo": {
                    "type": "string",
                    "pattern": "^[a-zA-Z0-9_\\-\\.\\/]+$",
                    "description": "Repository identifier",
                },
                "root": {
                    "type": "string",
                    "description": "Project root directory path",
                },
                "commit": {
                    "type": "string",
                    "pattern": "^[a-fA-F0-9]{7,40}$",
                    "description": "Current commit SHA (7-40 hex characters)",
                },
            },
            "required": ["name", "repo", "root"],
            "additionalProperties": False,
        },
        "storage": {
            "type": "object",
            "properties": {
                "db_path": {
                    "type": "string",
                    "description": "Path to DuckDB database file",
                },
                "cache_dir": {
                    "type": "string",
                    "description": "Directory for cached data",
                },
                "max_connections": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100,
                    "default": 5,
                    "description": "Maximum database connections",
                },
            },
            "required": ["db_path", "cache_dir"],
            "additionalProperties": False,
        },
        "plugins": {
            "type": "object",
            "properties": {
                "directories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Additional directories to search for plugins",
                },
                "disabled": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Plugin names to disable",
                },
            },
            "additionalProperties": False,
        },
    },
    "additionalProperties": False,
}

# -----------------------------------------------------------------------------
# Common Validators for Configuration
# -----------------------------------------------------------------------------


REPO_NAME_VALIDATOR = StringValidator(
    min_length=1,
    max_length=200,
    pattern=r"^[a-zA-Z0-9_\-\.\/]+$",
)

COMMIT_SHA_VALIDATOR = StringValidator(
    min_length=7,
    max_length=40,
    pattern=r"^[a-fA-F0-9]+$",
)

LOG_LEVEL_VALIDATOR = StringValidator(
    allowed_values={"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"},
)


# -----------------------------------------------------------------------------
# Configuration Sections
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StorageConfig:
    """Storage configuration section.

    Parameters
    ----------
    db_path
        Path to DuckDB database.
    cache_dir
        Directory for cached data.
    max_connections
        Maximum database connections.
    """

    db_path: Path
    cache_dir: Path
    max_connections: int = 5

    @staticmethod
    def schema() -> ValidationSchema:
        """Get the validation schema.

        Returns
        -------
        ValidationSchema
            Schema for validating storage configuration.
        """
        return (
            ValidationSchema()
            .add("db_path", PathValidator())
            .add("cache_dir", PathValidator())
            .add("max_connections", IntValidator(min_value=1, max_value=100))
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult[StorageConfig]:
        """Create from dictionary with validation.

        Parameters
        ----------
        data
            Raw configuration data.

        Returns
        -------
        ValidationResult[StorageConfig]
            Validated configuration or errors.
        """
        # Add defaults for optional fields
        data_with_defaults = {
            "max_connections": 5,
            **data,
        }

        schema = cls.schema()
        result = schema.validate(data_with_defaults)
        if not result.is_valid:
            return ValidationResult.fail(result.errors)

        validated = result.value
        if validated is None:
            return ValidationResult.fail(
                [
                    ValidationError(
                        field="storage",
                        message="Validation returned no value",
                        code="internal_error",
                    ),
                ]
            )

        db_path = validated.get("db_path")
        cache_dir = validated.get("cache_dir")
        max_connections = validated.get("max_connections", 5)

        if not isinstance(db_path, Path):
            return ValidationResult.fail(
                [
                    ValidationError(
                        field="db_path",
                        message="db_path must be a Path",
                        code="invalid_type",
                    ),
                ]
            )

        if not isinstance(cache_dir, Path):
            return ValidationResult.fail(
                [
                    ValidationError(
                        field="cache_dir",
                        message="cache_dir must be a Path",
                        code="invalid_type",
                    ),
                ]
            )

        if not isinstance(max_connections, int):
            return ValidationResult.fail(
                [
                    ValidationError(
                        field="max_connections",
                        message="max_connections must be an integer",
                        code="invalid_type",
                    ),
                ]
            )

        return ValidationResult.ok(
            cls(
                db_path=db_path,
                cache_dir=cache_dir,
                max_connections=max_connections,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "db_path": str(self.db_path),
            "cache_dir": str(self.cache_dir),
            "max_connections": self.max_connections,
        }


@dataclass(frozen=True)
class ProjectConfig:
    """Project configuration section.

    Parameters
    ----------
    name
        Project name.
    repo
        Repository identifier.
    commit
        Current commit SHA.
    root
        Project root directory.
    """

    name: str
    repo: str
    root: Path
    commit: str | None = None

    @staticmethod
    def schema() -> ValidationSchema:
        """Get the validation schema.

        Returns
        -------
        ValidationSchema
            Schema for validating project configuration.
        """
        return (
            ValidationSchema()
            .add("name", StringValidator(min_length=1, max_length=100))
            .add("repo", REPO_NAME_VALIDATOR)
            .add("root", PathValidator(must_exist=True, must_be_dir=True))
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult[ProjectConfig]:
        """Create from dictionary with validation.

        Parameters
        ----------
        data
            Raw configuration data.

        Returns
        -------
        ValidationResult[ProjectConfig]
            Validated configuration or errors.
        """
        errors: list[ValidationError] = []

        # Validate required fields
        schema = cls.schema()
        required_data = {k: v for k, v in data.items() if k in {"name", "repo", "root"}}
        required_result = schema.validate(required_data)

        if not required_result.is_valid:
            errors.extend(required_result.errors)

        # Validate optional commit if present
        if "commit" in data and data["commit"] is not None:
            commit_result = COMMIT_SHA_VALIDATOR.validate(data["commit"], "commit")
            if not commit_result.is_valid:
                errors.extend(commit_result.errors)

        if errors:
            return ValidationResult.fail(errors)

        validated = required_result.value
        if validated is None:
            return ValidationResult.fail(
                [
                    ValidationError(
                        field="project",
                        message="Validation returned no value",
                        code="internal_error",
                    ),
                ]
            )

        name = validated.get("name")
        repo = validated.get("repo")
        root = validated.get("root")

        if not isinstance(name, str):
            return ValidationResult.fail(
                [
                    ValidationError(
                        field="name", message="name must be a string", code="invalid_type"
                    ),
                ]
            )

        if not isinstance(repo, str):
            return ValidationResult.fail(
                [
                    ValidationError(
                        field="repo", message="repo must be a string", code="invalid_type"
                    ),
                ]
            )

        if not isinstance(root, Path):
            return ValidationResult.fail(
                [
                    ValidationError(
                        field="root", message="root must be a Path", code="invalid_type"
                    ),
                ]
            )

        return ValidationResult.ok(
            cls(
                name=name,
                repo=repo,
                commit=data.get("commit"),
                root=root,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "name": self.name,
            "repo": self.repo,
            "root": str(self.root),
        }
        if self.commit:
            result["commit"] = self.commit
        return result


@dataclass(frozen=True)
class FullConfig:
    """Complete configuration with all sections.

    Parameters
    ----------
    project
        Project configuration.
    storage
        Storage configuration.
    log_level
        Logging level.
    """

    project: ProjectConfig
    storage: StorageConfig
    log_level: str = "INFO"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult[FullConfig]:
        """Create from dictionary with validation.

        Parameters
        ----------
        data
            Raw configuration data.

        Returns
        -------
        ValidationResult[FullConfig]
            Validated configuration or errors.
        """
        errors: list[ValidationError] = []
        project: ProjectConfig | None = None
        storage: StorageConfig | None = None

        # Validate project section
        if "project" not in data:
            errors.append(
                ValidationError(
                    field="project",
                    message="Required section 'project' is missing",
                    code="missing_section",
                ),
            )
        else:
            project_result = ProjectConfig.from_dict(data["project"])
            if not project_result.is_valid:
                errors.extend(project_result.errors)
            else:
                project = project_result.value

        # Validate storage section
        if "storage" not in data:
            errors.append(
                ValidationError(
                    field="storage",
                    message="Required section 'storage' is missing",
                    code="missing_section",
                ),
            )
        else:
            storage_result = StorageConfig.from_dict(data["storage"])
            if not storage_result.is_valid:
                errors.extend(storage_result.errors)
            else:
                storage = storage_result.value

        # Validate log_level if present
        if "log_level" in data:
            log_result = LOG_LEVEL_VALIDATOR.validate(data["log_level"], "log_level")
            if not log_result.is_valid:
                errors.extend(log_result.errors)

        if errors:
            return ValidationResult.fail(errors)

        if project is None or storage is None:
            return ValidationResult.fail(
                [
                    ValidationError(
                        field="config",
                        message="Missing required sections",
                        code="missing_section",
                    ),
                ]
            )

        return ValidationResult.ok(
            cls(
                project=project,
                storage=storage,
                log_level=data.get("log_level", "INFO"),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "project": self.project.to_dict(),
            "storage": self.storage.to_dict(),
            "log_level": self.log_level,
        }


# -----------------------------------------------------------------------------
# File Validation
# -----------------------------------------------------------------------------


def validate_config_file(path: Path) -> ValidationResult[FullConfig]:
    """Validate a configuration file.

    Parameters
    ----------
    path
        Path to configuration file (YAML or JSON).

    Returns
    -------
    ValidationResult[FullConfig]
        Validated configuration or errors.
    """
    if not path.exists():
        return ValidationResult.fail(
            [
                ValidationError(
                    field="path",
                    message=f"Configuration file not found: {path}",
                    code="file_not_found",
                    value=str(path),
                ),
            ]
        )

    content = path.read_text(encoding="utf-8")

    is_yaml = path.suffix in {".yaml", ".yml"}
    try:
        data = yaml.safe_load(content) if is_yaml else json.loads(content)
    except (json.JSONDecodeError, yaml.YAMLError, OSError) as e:
        return ValidationResult.fail(
            [
                ValidationError(
                    field="path",
                    message=f"Failed to parse configuration file: {e}",
                    code="parse_error",
                    value=str(path),
                ),
            ]
        )

    if not isinstance(data, dict):
        return ValidationResult.fail(
            [
                ValidationError(
                    field="config",
                    message="Configuration must be a dictionary",
                    code="invalid_type",
                ),
            ]
        )

    return FullConfig.from_dict(data)


def format_validation_errors(errors: list[ValidationError]) -> str:
    """Format validation errors for display.

    Parameters
    ----------
    errors
        List of validation errors.

    Returns
    -------
    str
        Formatted error message.
    """
    lines = ["Configuration validation failed:"]
    for error in errors:
        lines.append(f"  • {error.field}: {error.message}")
        if error.value:
            lines.append(f"    Got: {error.value}")
    return "\n".join(lines)


# =============================================================================
# JSON Schema Validation (for schema-based validation)
# =============================================================================


@dataclass
class JsonSchemaValidationError:
    """JSON Schema validation error.

    Parameters
    ----------
    path
        JSON path to error location.
    message
        Error message.
    value
        Invalid value (optional).
    """

    path: str
    message: str
    value: Any = None

    def __str__(self) -> str:
        """Format error message.

        Returns
        -------
        str
            Formatted error string.
        """
        if self.value is not None:
            return f"{self.path}: {self.message} (got: {self.value!r})"
        return f"{self.path}: {self.message}"


def _get_jsonschema() -> ModuleType | None:
    """Get jsonschema module if available.

    Returns
    -------
    ModuleType | None
        The jsonschema module or None if not available.
    """
    return _jsonschema_module


def validate_with_json_schema(
    config: dict[str, Any],
) -> list[JsonSchemaValidationError]:
    """Validate configuration against JSON Schema.

    Parameters
    ----------
    config
        Configuration to validate.

    Returns
    -------
    list[JsonSchemaValidationError]
        Validation errors (empty if valid).
    """
    errors: list[JsonSchemaValidationError] = []

    jsonschema_mod = _get_jsonschema()
    if jsonschema_mod is not None:
        # Use getattr to access Draft202012Validator dynamically
        validator_cls = getattr(jsonschema_mod, "Draft202012Validator", None)
        if validator_cls is not None:
            validator = validator_cls(CLI_CONFIG_JSON_SCHEMA)
            for error in validator.iter_errors(config):
                path = ".".join(str(p) for p in error.path) or "(root)"
                errors.append(
                    JsonSchemaValidationError(
                        path=path,
                        message=error.message,
                        value=error.instance if error.path else None,
                    )
                )
            return errors

    # Fallback to basic validation without jsonschema library
    errors.extend(_basic_json_schema_validate(config, CLI_CONFIG_JSON_SCHEMA, ""))
    return errors


def _validate_type(
    value: object,
    schema_type: str,
    path: str,
) -> JsonSchemaValidationError | None:
    """Validate value against a primitive type.

    Parameters
    ----------
    value
        Value to validate.
    schema_type
        Expected JSON Schema type.
    path
        Current JSON path.

    Returns
    -------
    JsonSchemaValidationError | None
        Error if validation fails, None otherwise.
    """
    type_checks: dict[str, tuple[type | tuple[type, ...], str]] = {
        "string": (str, "Expected string"),
        "boolean": (bool, "Expected boolean"),
        "number": ((int, float), "Expected number"),
        "integer": (int, "Expected integer"),
        "array": (list, "Expected array"),
    }

    if schema_type in type_checks:
        expected_type, message = type_checks[schema_type]
        if not isinstance(value, expected_type):
            return JsonSchemaValidationError(path=path, message=message, value=value)

    return None


def _basic_json_schema_validate(
    value: object,
    schema: dict[str, Any],
    path: str,
) -> list[JsonSchemaValidationError]:
    """Validate value against JSON Schema without jsonschema library.

    Parameters
    ----------
    value
        Value to validate.
    schema
        Schema to validate against.
    path
        Current JSON path.

    Returns
    -------
    list[JsonSchemaValidationError]
        Validation errors.
    """
    errors: list[JsonSchemaValidationError] = []
    schema_type = schema.get("type")

    if schema_type == "object" and isinstance(value, dict):
        errors.extend(_validate_object(value, schema, path))
    elif schema_type is not None:
        type_error = _validate_type(value, schema_type, path)
        if type_error:
            errors.append(type_error)

    # Check enum constraint
    if "enum" in schema and value not in schema["enum"]:
        errors.append(
            JsonSchemaValidationError(
                path=path,
                message=f"Must be one of: {schema['enum']}",
                value=value,
            )
        )

    return errors


def _validate_object(
    value: dict[str, Any],
    schema: dict[str, Any],
    path: str,
) -> list[JsonSchemaValidationError]:
    """Validate object against schema properties.

    Parameters
    ----------
    value
        Object value to validate.
    schema
        Schema with properties definition.
    path
        Current JSON path.

    Returns
    -------
    list[JsonSchemaValidationError]
        Validation errors.
    """
    errors: list[JsonSchemaValidationError] = []
    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties", True)

    for key, val in value.items():
        key_path = f"{path}.{key}" if path else key
        if key in properties:
            errors.extend(_basic_json_schema_validate(val, properties[key], key_path))
        elif not additional:
            errors.append(JsonSchemaValidationError(path=key_path, message="Unknown property"))

    return errors


def write_json_schema(path: Path) -> None:
    """Write JSON Schema to file.

    Parameters
    ----------
    path
        Output path for schema file.
    """
    path.write_text(json.dumps(CLI_CONFIG_JSON_SCHEMA, indent=2), encoding="utf-8")


def get_json_schema_url() -> str:
    """Get URL for schema file.

    Returns
    -------
    str
        Schema URL.
    """
    return "https://codeintel.dev/schemas/cli-config.json"


# =============================================================================
# Extended Configuration Sections
# =============================================================================


@dataclass(frozen=True)
class RetryConfig:
    """Retry configuration section.

    Parameters
    ----------
    max_attempts
        Maximum retry attempts.
    initial_delay
        Initial delay between retries in seconds.
    backoff_factor
        Exponential backoff multiplier.
    max_delay
        Maximum delay between retries in seconds.
    """

    max_attempts: int = 3
    initial_delay: float = 0.5
    backoff_factor: float = 2.0
    max_delay: float = 30.0

    @staticmethod
    def schema() -> ValidationSchema:
        """Get the validation schema.

        Returns
        -------
        ValidationSchema
            Schema for validating retry configuration.
        """
        return (
            ValidationSchema()
            .add("max_attempts", IntValidator(min_value=1, max_value=10))
            .add("initial_delay", FloatValidator(min_value=0.0))
            .add("backoff_factor", FloatValidator(min_value=1.0))
            .add("max_delay", FloatValidator(min_value=0.0))
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult[RetryConfig]:
        """Create from dictionary with validation.

        Parameters
        ----------
        data
            Raw configuration data.

        Returns
        -------
        ValidationResult[RetryConfig]
            Validated configuration or errors.
        """
        defaults: dict[str, int | float] = {
            "max_attempts": 3,
            "initial_delay": 0.5,
            "backoff_factor": 2.0,
            "max_delay": 30.0,
        }
        merged = {**defaults, **data}

        schema = cls.schema()
        result = schema.validate(merged)
        if not result.is_valid:
            return ValidationResult.fail(result.errors)

        validated: dict[str, Any] = result.value or {}

        # Extract with type assertions since schema validated types
        max_attempts_val = validated.get("max_attempts", 3)
        initial_delay_val = validated.get("initial_delay", 0.5)
        backoff_factor_val = validated.get("backoff_factor", 2.0)
        max_delay_val = validated.get("max_delay", 30.0)

        return ValidationResult.ok(
            cls(
                max_attempts=int(max_attempts_val)
                if isinstance(max_attempts_val, (int, float))
                else 3,
                initial_delay=float(initial_delay_val)
                if isinstance(initial_delay_val, (int, float))
                else 0.5,
                backoff_factor=float(backoff_factor_val)
                if isinstance(backoff_factor_val, (int, float))
                else 2.0,
                max_delay=float(max_delay_val) if isinstance(max_delay_val, (int, float)) else 30.0,
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "max_attempts": self.max_attempts,
            "initial_delay": self.initial_delay,
            "backoff_factor": self.backoff_factor,
            "max_delay": self.max_delay,
        }


@dataclass(frozen=True)
class TelemetryConfigSection:
    """Telemetry configuration section.

    Parameters
    ----------
    enabled
        Whether telemetry is enabled.
    endpoint
        OTLP collector endpoint.
    service_name
        Service name for traces.
    """

    enabled: bool = True
    endpoint: str | None = None
    service_name: str = "codeintel-cli"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult[TelemetryConfigSection]:
        """Create from dictionary with validation.

        Parameters
        ----------
        data
            Raw configuration data.

        Returns
        -------
        ValidationResult[TelemetryConfigSection]
            Validated configuration or errors.
        """
        errors: list[ValidationError] = []

        enabled = data.get("enabled", True)
        if not isinstance(enabled, bool):
            errors.append(
                ValidationError(
                    field="telemetry.enabled",
                    message="Must be a boolean",
                    code="invalid_type",
                )
            )

        endpoint = data.get("endpoint")
        if endpoint is not None and not isinstance(endpoint, str):
            errors.append(
                ValidationError(
                    field="telemetry.endpoint",
                    message="Must be a string",
                    code="invalid_type",
                )
            )

        service_name = data.get("service_name", "codeintel-cli")
        if not isinstance(service_name, str):
            errors.append(
                ValidationError(
                    field="telemetry.service_name",
                    message="Must be a string",
                    code="invalid_type",
                )
            )

        if errors:
            return ValidationResult.fail(errors)

        return ValidationResult.ok(
            cls(
                enabled=enabled if isinstance(enabled, bool) else True,
                endpoint=endpoint if isinstance(endpoint, str) else None,
                service_name=service_name if isinstance(service_name, str) else "codeintel-cli",
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        result: dict[str, object] = {
            "enabled": self.enabled,
            "service_name": self.service_name,
        }
        if self.endpoint:
            result["endpoint"] = self.endpoint
        return result


@dataclass(frozen=True)
class PluginsConfig:
    """Plugins configuration section.

    Parameters
    ----------
    directories
        Additional plugin directories.
    disabled
        Plugins to disable.
    """

    directories: tuple[str, ...] = ()
    disabled: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ValidationResult[PluginsConfig]:
        """Create from dictionary with validation.

        Parameters
        ----------
        data
            Raw configuration data.

        Returns
        -------
        ValidationResult[PluginsConfig]
            Validated configuration or errors.
        """
        errors: list[ValidationError] = []

        directories = data.get("directories", [])
        if not isinstance(directories, list):
            errors.append(
                ValidationError(
                    field="plugins.directories",
                    message="Must be an array",
                    code="invalid_type",
                )
            )
            directories = []

        disabled = data.get("disabled", [])
        if not isinstance(disabled, list):
            errors.append(
                ValidationError(
                    field="plugins.disabled",
                    message="Must be an array",
                    code="invalid_type",
                )
            )
            disabled = []

        if errors:
            return ValidationResult.fail(errors)

        return ValidationResult.ok(
            cls(
                directories=tuple(str(d) for d in directories),
                disabled=tuple(str(d) for d in disabled),
            ),
        )

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for serialization.

        Returns
        -------
        dict[str, object]
            Dictionary representation.
        """
        return {
            "directories": list(self.directories),
            "disabled": list(self.disabled),
        }


__all__ = [
    "CLI_CONFIG_JSON_SCHEMA",
    "COMMIT_SHA_VALIDATOR",
    "LOG_LEVEL_VALIDATOR",
    "REPO_NAME_VALIDATOR",
    "FullConfig",
    "JsonSchemaValidationError",
    "PluginsConfig",
    "ProjectConfig",
    "RetryConfig",
    "StorageConfig",
    "TelemetryConfigSection",
    "format_validation_errors",
    "get_json_schema_url",
    "validate_config_file",
    "validate_with_json_schema",
    "write_json_schema",
]
