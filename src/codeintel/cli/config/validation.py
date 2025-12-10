"""Model-driven configuration validation.

Validate CliConfig instances against type constraints and business rules.
"""

from __future__ import annotations

import re

from types import ModuleType

from codeintel.cli.config.model import CliConfig, ConfigValidationError
from codeintel.cli.config.schema import generate_schema

# Validation constraints
VALID_OUTPUT_FORMATS = {"text", "json"}
VALID_LOG_LEVELS = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
MIN_RETRY_ATTEMPTS = 1
MAX_RETRY_ATTEMPTS = 10
MIN_STORAGE_CONNECTIONS = 1
MAX_STORAGE_CONNECTIONS = 100
REPO_PATTERN = re.compile(r"^[a-zA-Z0-9_\-\./]+$")
COMMIT_PATTERN = re.compile(r"^[a-fA-F0-9]{7,40}$")

# Try to import jsonschema for JSON Schema validation
_jsonschema: ModuleType | None
try:
    import jsonschema as _jsonschema

    _HAS_JSONSCHEMA = True
except ImportError:
    _jsonschema = None
    _HAS_JSONSCHEMA = False


def validate_config(config: CliConfig) -> list[ConfigValidationError]:
    """Validate configuration against all constraints.

    Parameters
    ----------
    config
        Configuration to validate.

    Returns
    -------
    list[ConfigValidationError]
        List of validation errors (empty if valid).
    """
    errors: list[ConfigValidationError] = []
    errors.extend(_validate_top_level(config))
    errors.extend(_validate_progress(config))
    errors.extend(_validate_retry(config))
    errors.extend(_validate_storage(config))
    errors.extend(_validate_project(config))
    return errors


def _validate_top_level(config: CliConfig) -> list[ConfigValidationError]:
    """Validate top-level configuration fields.

    Parameters
    ----------
    config
        Configuration to validate.

    Returns
    -------
    list[ConfigValidationError]
        List of validation errors.
    """
    errors: list[ConfigValidationError] = []

    if config.output_format not in VALID_OUTPUT_FORMATS:
        errors.append(
            ConfigValidationError(
                path="output_format",
                message="Must be 'text' or 'json'",
                code="invalid_enum",
                value=config.output_format,
            )
        )

    if config.log_level not in VALID_LOG_LEVELS:
        errors.append(
            ConfigValidationError(
                path="log_level",
                message="Must be one of: DEBUG, INFO, WARNING, ERROR, CRITICAL",
                code="invalid_enum",
                value=config.log_level,
            )
        )

    return errors


def _validate_progress(config: CliConfig) -> list[ConfigValidationError]:
    """Validate progress configuration section.

    Parameters
    ----------
    config
        Configuration to validate.

    Returns
    -------
    list[ConfigValidationError]
        List of validation errors.
    """
    errors: list[ConfigValidationError] = []

    if config.progress.threshold < 0:
        errors.append(
            ConfigValidationError(
                path="progress.threshold",
                message="Must be non-negative",
                code="min_value",
                value=config.progress.threshold,
            )
        )

    return errors


def _validate_retry(config: CliConfig) -> list[ConfigValidationError]:
    """Validate retry configuration section.

    Parameters
    ----------
    config
        Configuration to validate.

    Returns
    -------
    list[ConfigValidationError]
        List of validation errors.
    """
    errors: list[ConfigValidationError] = []

    if config.retry.max_attempts < MIN_RETRY_ATTEMPTS:
        errors.append(
            ConfigValidationError(
                path="retry.max_attempts",
                message=f"Must be at least {MIN_RETRY_ATTEMPTS}",
                code="min_value",
                value=config.retry.max_attempts,
            )
        )

    if config.retry.max_attempts > MAX_RETRY_ATTEMPTS:
        errors.append(
            ConfigValidationError(
                path="retry.max_attempts",
                message=f"Must be at most {MAX_RETRY_ATTEMPTS}",
                code="max_value",
                value=config.retry.max_attempts,
            )
        )

    if config.retry.initial_delay < 0:
        errors.append(
            ConfigValidationError(
                path="retry.initial_delay",
                message="Must be non-negative",
                code="min_value",
                value=config.retry.initial_delay,
            )
        )

    if config.retry.backoff_factor < 1:
        errors.append(
            ConfigValidationError(
                path="retry.backoff_factor",
                message="Must be at least 1",
                code="min_value",
                value=config.retry.backoff_factor,
            )
        )

    if config.retry.max_delay < 0:
        errors.append(
            ConfigValidationError(
                path="retry.max_delay",
                message="Must be non-negative",
                code="min_value",
                value=config.retry.max_delay,
            )
        )

    return errors


def _validate_storage(config: CliConfig) -> list[ConfigValidationError]:
    """Validate storage configuration section.

    Parameters
    ----------
    config
        Configuration to validate.

    Returns
    -------
    list[ConfigValidationError]
        List of validation errors.
    """
    errors: list[ConfigValidationError] = []

    if config.storage.max_connections < MIN_STORAGE_CONNECTIONS:
        errors.append(
            ConfigValidationError(
                path="storage.max_connections",
                message=f"Must be at least {MIN_STORAGE_CONNECTIONS}",
                code="min_value",
                value=config.storage.max_connections,
            )
        )

    if config.storage.max_connections > MAX_STORAGE_CONNECTIONS:
        errors.append(
            ConfigValidationError(
                path="storage.max_connections",
                message=f"Must be at most {MAX_STORAGE_CONNECTIONS}",
                code="max_value",
                value=config.storage.max_connections,
            )
        )

    return errors


def _validate_project(config: CliConfig) -> list[ConfigValidationError]:
    """Validate project configuration section.

    Parameters
    ----------
    config
        Configuration to validate.

    Returns
    -------
    list[ConfigValidationError]
        List of validation errors.
    """
    errors: list[ConfigValidationError] = []

    if config.project.repo and not REPO_PATTERN.match(config.project.repo):
        errors.append(
            ConfigValidationError(
                path="project.repo",
                message="Must match pattern: ^[a-zA-Z0-9_\\-\\./]+$",
                code="pattern",
                value=config.project.repo,
            )
        )

    if config.project.commit and not COMMIT_PATTERN.match(config.project.commit):
        errors.append(
            ConfigValidationError(
                path="project.commit",
                message="Must be a valid commit SHA (7-40 hex characters)",
                code="pattern",
                value=config.project.commit,
            )
        )

    return errors


def validate_with_json_schema(
    config: dict[str, object],
) -> list[ConfigValidationError]:
    """Validate configuration dictionary against JSON Schema.

    Parameters
    ----------
    config
        Configuration dictionary to validate.

    Returns
    -------
    list[ConfigValidationError]
        List of validation errors (empty if valid).

    Notes
    -----
    This function uses jsonschema for validation if available.
    If jsonschema is not installed, returns an empty list (no validation).
    """
    if not _HAS_JSONSCHEMA or _jsonschema is None:
        # jsonschema not available, skip JSON Schema validation
        return []

    schema = generate_schema(CliConfig)
    errors: list[ConfigValidationError] = []

    try:
        _jsonschema.validate(config, schema)
    except _jsonschema.ValidationError as e:
        errors.append(
            ConfigValidationError(
                path=".".join(str(p) for p in e.absolute_path),
                message=e.message,
                code="json_schema",
                value=e.instance,
            )
        )

    return errors


__all__ = [
    "validate_config",
    "validate_with_json_schema",
]
