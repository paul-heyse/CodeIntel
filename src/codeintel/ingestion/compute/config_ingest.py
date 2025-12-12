"""Configuration file ingestion step with port injection.

This module provides a pure domain logic implementation for flattening
and ingesting configuration files, using ports for all I/O operations.
"""

from __future__ import annotations

import configparser
import json
import logging
import tomllib
from typing import TYPE_CHECKING, Any

import yaml

from codeintel.ingestion.compute.base import StepResult

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort

log = logging.getLogger(__name__)


def _flatten_dict(
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
            items.extend(_flatten_dict(v, new_key, sep))
        elif isinstance(v, list):
            items.extend(_flatten_list_items(v, new_key, sep))
        else:
            items.append((new_key, v))
    return items


def _flatten_list_items(
    items_list: list[Any],
    parent_key: str,
    sep: str,
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
            result.extend(_flatten_dict(item, f"{parent_key}[{i}]", sep))
        else:
            result.append((f"{parent_key}[{i}]", item))
    return result


def _parse_toml(content: str) -> list[tuple[str, Any]] | None:
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
        return _flatten_dict(data)


def _parse_yaml(content: str) -> list[tuple[str, Any]] | None:
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
            return _flatten_dict(data)
        return None


def _parse_ini(content: str) -> list[tuple[str, Any]] | None:
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


def _parse_json(content: str) -> list[tuple[str, Any]] | None:
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
            return _flatten_dict(data)
        return None


def _get_config_format(path: Path) -> str:
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


def _parse_config_file(path: Path, content: str) -> list[tuple[str, Any]] | None:
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
        return _parse_toml(content)

    if suffix in {".yaml", ".yml"}:
        return _parse_yaml(content)

    if suffix in {".ini", ".cfg"}:
        return _parse_ini(content)

    if suffix == ".json":
        return _parse_json(content)

    return None


class ConfigIngestStep:
    """Configuration file ingestion step with port injection.

    This step flattens and ingests configuration files,
    using ports for all I/O operations.

    Parameters
    ----------
    storage
        Storage port for persisting data.
    discovery
        Discovery port for reading file content.
    """

    def __init__(
        self,
        storage: IngestStoragePort,
        discovery: ModuleDiscoveryPort,
    ) -> None:
        """Initialize the step.

        Parameters
        ----------
        storage
            Storage port for persisting data.
        discovery
            Discovery port for reading file content.
        """
        self._storage = storage
        self._discovery = discovery

    def execute(
        self,
        config_files: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
    ) -> StepResult:
        """Execute configuration file ingestion.

        Parameters
        ----------
        config_files
            Configuration files to process.
        repo
            Repository identifier.
        commit
            Commit identifier.

        Returns
        -------
        StepResult
            Execution result with row counts.
        """
        all_rows: list[list[object]] = []
        errors: list[str] = []

        for record in config_files:
            content = self._discovery.read_text(record.file_path)
            if content is None:
                continue

            kvs = _parse_config_file(record.file_path, content)
            if kvs is None:
                errors.append(f"Failed to parse {record.rel_path}")
                continue

            config_format = _get_config_format(record.file_path)

            for key, _value in kvs:
                all_rows.append(
                    [
                        repo,
                        commit,
                        record.rel_path,
                        config_format,
                        key,
                        "[]",
                        "[]",
                        0,
                    ]
                )

        table_counts: dict[str, int] = {}
        total_rows = 0

        if all_rows:
            scope = f"{repo}@{commit}"
            result = self._storage.write_batch("analytics.config_values", all_rows, scope=scope)
            table_counts["analytics.config_values"] = result.rows_written
            total_rows = result.rows_written

        log.info(
            "Config ingest: repo=%s commit=%s files=%d values=%d",
            repo,
            commit,
            len(config_files),
            len(all_rows),
        )

        return StepResult(rows_written=total_rows, table_counts=table_counts, errors=errors)


def flatten_dict(
    data: dict[str, Any], parent_key: str = "", sep: str = "."
) -> list[tuple[str, Any]]:
    """Public wrapper for flattening nested dictionaries.

    Parameters
    ----------
    data
        Dictionary to flatten.
    parent_key
        Prefix for keys.
    sep
        Separator between key parts.

    Returns
    -------
    list[tuple[str, Any]]
        Flattened key/value pairs.
    """
    return _flatten_dict(data, parent_key, sep)


def flatten_list_items(
    items: list[Any], parent_key: str = "", sep: str = "."
) -> list[tuple[str, Any]]:
    """Public wrapper for flattening list structures.

    Parameters
    ----------
    items
        List to flatten.
    parent_key
        Prefix for keys.
    sep
        Separator between key parts.

    Returns
    -------
    list[tuple[str, Any]]
        Flattened key/value pairs.
    """
    return _flatten_list_items(items, parent_key, sep)


def parse_config_file(file_path: Path, content: str) -> list[tuple[str, Any]] | None:
    """Public wrapper for parsing config files by extension.

    Parameters
    ----------
    file_path
        File path for format detection.
    content
        File content.

    Returns
    -------
    list[tuple[str, Any]] | None
        Parsed key/value pairs or None when parsing fails.
    """
    return _parse_config_file(file_path, content)


def parse_ini(content: str) -> list[tuple[str, Any]] | None:
    """Public wrapper for parsing INI config content.

    Parameters
    ----------
    content
        INI content string.

    Returns
    -------
    list[tuple[str, Any]] | None
        Parsed key/value pairs or None on failure.
    """
    return _parse_ini(content)


def parse_json(content: str) -> list[tuple[str, Any]] | None:
    """Public wrapper for parsing JSON config content.

    Parameters
    ----------
    content
        JSON content string.

    Returns
    -------
    list[tuple[str, Any]] | None
        Parsed key/value pairs or None on failure.
    """
    return _parse_json(content)


def parse_toml(content: str) -> list[tuple[str, Any]] | None:
    """Public wrapper for parsing TOML config content.

    Parameters
    ----------
    content
        TOML content string.

    Returns
    -------
    list[tuple[str, Any]] | None
        Parsed key/value pairs or None on failure.
    """
    return _parse_toml(content)


def parse_yaml(content: str) -> list[tuple[str, Any]] | None:
    """Public wrapper for parsing YAML config content.

    Parameters
    ----------
    content
        YAML content string.

    Returns
    -------
    list[tuple[str, Any]] | None
        Parsed key/value pairs or None on failure.
    """
    return _parse_yaml(content)


__all__ = [
    "ConfigIngestStep",
    "flatten_dict",
    "flatten_list_items",
    "parse_config_file",
    "parse_ini",
    "parse_json",
    "parse_toml",
    "parse_yaml",
]
