"""Ingest configuration keypaths into analytics.config_values."""

from __future__ import annotations

import configparser
import json
import logging
import tomllib  # Python 3.11+
from collections.abc import Iterable
from pathlib import Path

import duckdb
import yaml

log = logging.getLogger(__name__)

CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".env"}


def _iter_config_files(repo_root: Path) -> Iterable[Path]:
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in CONFIG_EXTENSIONS:
            yield path


def _detect_format(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".yaml", ".yml"}:
        return "yaml"
    if ext == ".toml":
        return "toml"
    if ext == ".json":
        return "json"
    if ext in {".ini", ".cfg"}:
        return "ini"
    if ext == ".env":
        return "env"
    return "other"


def _load_config(path: Path, fmt: str) -> dict:
    try:
        text = path.read_text(encoding="utf8")
    except (OSError, UnicodeDecodeError):
        return {}

    try:
        if fmt == "yaml":
            return yaml.safe_load(text) or {}
        if fmt == "toml":
            return tomllib.loads(text) or {}
        if fmt == "json":
            return json.loads(text) or {}
        if fmt == "ini":
            parser = configparser.ConfigParser()
            parser.read_string(text)
            data = {}
            for section in parser.sections():
                data[section] = dict(parser.items(section))
            return data
        if fmt == "env":
            data = {}
            for line in text.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                data[key.strip()] = value.strip()
            return data
    except Exception as exc:
        log.warning("Failed to parse config file %s (%s): %s", path, fmt, exc)
        return {}

    return {}


def _flatten_config(
    obj: object,
    prefix: str = "",
) -> list[tuple[str, object]]:
    """
    Flatten a nested dict or list into dotted keypaths.

    Examples
    --------
    >>> _flatten_config({"service": {"database": {"host": "db"}}})
    [('service.database.host', 'db')]
    >>> _flatten_config(["a", {"b": 2}])
    [('0', 'a'), ('1.b', 2)]

    Parameters
    ----------
    obj :
        Arbitrary nested mapping or sequence parsed from a config file.
    prefix : str, optional
        Prefix applied to generated keypaths; used during recursion.

    Returns
    -------
    list of tuple[str, object]
        Flattened keypaths paired with their leaf values.
    """
    items: list[tuple[str, object]] = []

    if isinstance(obj, dict):
        for key, value in obj.items():
            key_str = str(key)
            new_prefix = f"{prefix}.{key_str}" if prefix else key_str
            items.extend(_flatten_config(value, new_prefix))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            new_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            items.extend(_flatten_config(value, new_prefix))
    else:
        items.append((prefix, obj))

    return items


def ingest_config_values(
    con: duckdb.DuckDBPyConnection,
    repo_root: Path,
) -> None:
    """
    Populate analytics.config_values from configuration files.

    For each config file:
      - detect format
      - parse into a dict-like structure
      - flatten into keypaths
      - insert rows with empty reference_paths / reference_modules

    A later analytics step can fill reference_paths/modules by scanning
    AST/uses.
    """
    repo_root = repo_root.resolve()

    # Clear table (single repo per DB assumption)
    con.execute("DELETE FROM analytics.config_values")

    insert_sql = """
        INSERT INTO analytics.config_values (
            config_path, format, key,
            reference_paths, reference_modules, reference_count
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """

    count = 0

    for path in _iter_config_files(repo_root):
        rel_path = path.relative_to(repo_root).as_posix()
        fmt = _detect_format(path)
        data = _load_config(path, fmt)
        if not data:
            continue

        for keypath, value in _flatten_config(data):
            # We don't store the value itself; only keypaths + references.
            con.execute(
                insert_sql,
                [
                    rel_path,
                    fmt,
                    keypath,
                    [],  # reference_paths
                    [],  # reference_modules
                    0,  # reference_count
                ],
            )
            count += 1

    log.info("config_values ingested: %d keys across config files", count)
