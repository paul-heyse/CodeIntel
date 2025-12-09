"""Configuration schema definitions and validation.

This module defines the expected structure of configuration files
and provides validation using the cli_validation infrastructure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from codeintel.cli.cli_validation import (
    IntValidator,
    PathValidator,
    StringValidator,
    ValidationError,
    ValidationResult,
    ValidationSchema,
)

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


__all__ = [
    "COMMIT_SHA_VALIDATOR",
    "LOG_LEVEL_VALIDATOR",
    "REPO_NAME_VALIDATOR",
    "FullConfig",
    "ProjectConfig",
    "StorageConfig",
    "format_validation_errors",
    "validate_config_file",
]
