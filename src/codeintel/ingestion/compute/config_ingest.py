"""Configuration file ingestion step with port injection.

This module provides a pure domain logic implementation for flattening
and ingesting configuration files, using ports for all I/O operations.
"""

from __future__ import annotations

import configparser
import json
import logging
import tomllib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import yaml

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.core.columnar.rows import (
    ColumnarRows,
    columnar_buffer_for_table_key,
    empty_table_for_table,
    table_for_columnar_rows,
)
from codeintel.ingestion.compute.base import BaseExtractStep
from codeintel.ingestion.context import IngestionContext, resolve_repo_commit

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    import pyarrow as pa

    from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)
CONFIG_VALUES_TABLE_KEY = "analytics.config_values"


def flatten_dict(
    d: dict[str, Any],
    parent_key: str = "",
    sep: str = ".",
) -> list[tuple[str, Any]]:
    """Flatten a nested dictionary into key-value pairs.

    Parameters
    ----------
    d
        Dictionary to flatten.
    parent_key
        Prefix for keys.
    sep
        Separator between key parts.

    Returns
    -------
    list[tuple[str, Any]]
        Flattened key-value pairs.
    """
    items: list[tuple[str, Any]] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep))
        elif isinstance(v, list):
            items.extend(flatten_list_items(v, new_key, sep))
        else:
            items.append((new_key, v))
    return items


def flatten_list_items(
    items_list: list[Any],
    parent_key: str = "",
    sep: str = ".",
) -> list[tuple[str, Any]]:
    """Flatten list items into key-value pairs.

    Parameters
    ----------
    items_list
        List to flatten.
    parent_key
        Prefix for keys.
    sep
        Separator between key parts.

    Returns
    -------
    list[tuple[str, Any]]
        Flattened key-value pairs from list items.
    """
    result: list[tuple[str, Any]] = []
    for i, item in enumerate(items_list):
        if isinstance(item, dict):
            result.extend(flatten_dict(item, f"{parent_key}[{i}]", sep))
        else:
            result.append((f"{parent_key}[{i}]", item))
    return result


def parse_toml(content: str) -> list[tuple[str, Any]] | None:
    """Parse TOML content.

    Parameters
    ----------
    content
        TOML content string.

    Returns
    -------
    list[tuple[str, Any]] | None
        Flattened key-value pairs or None on failure.
    """
    try:
        data = tomllib.loads(content)
    except tomllib.TOMLDecodeError as exc:
        log.debug("Failed to parse TOML: %s", exc)
        return None
    else:
        return flatten_dict(data)


def parse_yaml(content: str) -> list[tuple[str, Any]] | None:
    """Parse YAML content.

    Parameters
    ----------
    content
        YAML content string.

    Returns
    -------
    list[tuple[str, Any]] | None
        Flattened key-value pairs or None on failure.
    """
    try:
        data = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        log.debug("Failed to parse YAML: %s", exc)
        return None
    else:
        if isinstance(data, dict):
            return flatten_dict(data)
        return None


def parse_ini(content: str) -> list[tuple[str, Any]] | None:
    """Parse INI/CFG content.

    Parameters
    ----------
    content
        INI content string.

    Returns
    -------
    list[tuple[str, Any]] | None
        Flattened key-value pairs or None on failure.
    """
    try:
        parser = configparser.ConfigParser()
        parser.read_string(content)
    except configparser.Error as exc:
        log.debug("Failed to parse INI: %s", exc)
        return None
    else:
        return [
            (f"{section}.{key}", value)
            for section in parser.sections()
            for key, value in parser.items(section)
        ]


def parse_json(content: str) -> list[tuple[str, Any]] | None:
    """Parse JSON content.

    Parameters
    ----------
    content
        JSON content string.

    Returns
    -------
    list[tuple[str, Any]] | None
        Flattened key-value pairs or None on failure.
    """
    try:
        data = json.loads(content)
    except json.JSONDecodeError as exc:
        log.debug("Failed to parse JSON: %s", exc)
        return None
    else:
        if isinstance(data, dict):
            return flatten_dict(data)
        return None


def get_config_format(path: Path) -> str:
    """Get the configuration format from file extension.

    Parameters
    ----------
    path
        File path to check.

    Returns
    -------
    str
        Configuration format (yaml, json, toml, ini, env, or unknown).
    """
    suffix = path.suffix.lower()
    format_map = {
        ".yaml": "yaml",
        ".yml": "yaml",
        ".json": "json",
        ".toml": "toml",
        ".ini": "ini",
        ".cfg": "ini",
        ".env": "env",
    }
    return format_map.get(suffix, "unknown")


def parse_config_file(path: Path, content: str) -> list[tuple[str, Any]] | None:
    """Parse a configuration file and return flattened key-value pairs.

    Parameters
    ----------
    path
        File path for format detection.
    content
        File content.

    Returns
    -------
    list[tuple[str, Any]] | None
        Flattened key-value pairs, or None on parse failure.
    """
    suffix = path.suffix.lower()

    if suffix == ".toml":
        return parse_toml(content)

    if suffix in {".yaml", ".yml"}:
        return parse_yaml(content)

    if suffix in {".ini", ".cfg"}:
        return parse_ini(content)

    if suffix == ".json":
        return parse_json(content)

    return None


@dataclass(frozen=True)
class ConfigIngestResult:
    """Result bundle for config ingestion."""

    result: ExecutionResult
    rows: ColumnarRows = field(default_factory=dict)
    rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(CONFIG_VALUES_TABLE_KEY)
    )
    row_count: int = 0


class ConfigIngestStep(BaseExtractStep):
    """Configuration file ingestion step with port injection.

    This step flattens and ingests configuration files,
    using ports for all I/O operations.

    Parameters
    ----------
    discovery
        Discovery port for reading file content.
    """

    def execute(
        self,
        config_files: Sequence[ModuleRecord],
        *,
        repo: str | None = None,
        commit: str | None = None,
        context: IngestionContext | None = None,
    ) -> ConfigIngestResult:
        """Execute configuration file ingestion.

        Parameters
        ----------
        config_files
            Configuration files to process.
        repo
            Repository identifier.
        commit
            Commit identifier.
        context
            Optional ingestion context supplying repo/commit defaults.

        Returns
        -------
        ConfigIngestResult
            Result bundle with row tuples and execution status.
        """
        resolved_repo, resolved_commit = resolve_repo_commit(
            context=context,
            repo=repo,
            commit=commit,
        )
        try:
            buffer = columnar_buffer_for_table_key(CONFIG_VALUES_TABLE_KEY)
        except (KeyError, RuntimeError) as exc:
            return ConfigIngestResult(result=ExecutionResult.failed(str(exc)))
        errors: list[str] = []

        for record in config_files:
            content = self._discovery.read_text(record.file_path)
            if content is None:
                continue

            kvs = parse_config_file(record.file_path, content)
            if kvs is None:
                errors.append(f"Failed to parse {record.rel_path}")
                continue

            config_format = get_config_format(record.file_path)

            for key, _value in kvs:
                buffer.append(
                    {
                        "repo": resolved_repo,
                        "commit": resolved_commit,
                        "config_path": record.rel_path,
                        "format": config_format,
                        "key": key,
                        "reference_paths": "[]",
                        "reference_modules": "[]",
                        "reference_count": 0,
                    }
                )

        warnings = tuple(errors)
        if warnings and buffer.row_count == 0:
            warning_text = "; ".join(warnings)
            return ConfigIngestResult(
                result=ExecutionResult.failed(
                    f"Config ingest failed: {warning_text}",
                    warnings=warnings,
                ),
            )

        log.info(
            "Config ingest: repo=%s commit=%s files=%d values=%d",
            resolved_repo,
            resolved_commit,
            len(config_files),
            buffer.row_count,
        )

        rows_reader, row_count = table_for_columnar_rows(
            CONFIG_VALUES_TABLE_KEY,
            buffer.data,
            extras_policy="retain",
        )
        return ConfigIngestResult(
            result=ExecutionResult.ok(warnings=warnings),
            rows=buffer.data,
            rows_reader=rows_reader,
            row_count=row_count,
        )


__all__ = [
    "ConfigIngestResult",
    "ConfigIngestStep",
    "flatten_dict",
    "flatten_list_items",
    "get_config_format",
    "parse_config_file",
    "parse_ini",
    "parse_json",
    "parse_toml",
    "parse_yaml",
]
