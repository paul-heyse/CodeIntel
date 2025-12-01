"""Ingest configuration keypaths into analytics.config_values."""

from __future__ import annotations

import configparser
import json
import logging
import tomllib  # Python 3.11+
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import cast

import yaml

from codeintel.config import ConfigIngestStepConfig
from codeintel.config.dataset_contract import ConfigValueRow, config_value_to_tuple
from codeintel.ingestion.change_tracker import (
    ChangeTracker,
    IncrementalIngestOps,
    SupportsFullRebuild,
    run_incremental_ingest,
)
from codeintel.ingestion.common import ModuleRecord, run_batch
from codeintel.ingestion.paths import repo_relpath
from codeintel.ingestion.source_scanner import (
    ScanProfile,
    SourceScanner,
    default_config_profile,
    profile_from_env,
)
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)

CONFIG_EXTENSIONS = {".yaml", ".yml", ".toml", ".json", ".ini", ".cfg", ".env"}


def _iter_config_files(profile: ScanProfile) -> Iterable[Path]:
    scanner = SourceScanner(profile)
    yield from scanner.iter_files()


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


def _load_config(path: Path, fmt: str) -> dict[str, object]:
    try:
        text = path.read_text(encoding="utf8")
    except (OSError, UnicodeDecodeError):
        return {}

    parser_map: dict[str, Callable[[str, Path], dict[str, object]]] = {
        "yaml": _parse_yaml,
        "toml": _parse_toml,
        "json": _parse_json,
        "ini": _parse_ini,
        "env": _parse_env,
    }
    parser = parser_map.get(fmt)

    if parser is None:
        return {}
    return parser(text, path)


def _parse_yaml(text: str, path: Path) -> dict[str, object]:
    try:
        return yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        log.warning("Failed to parse YAML config %s: %s", path, exc)
        return {}


def _parse_toml(text: str, path: Path) -> dict[str, object]:
    try:
        return tomllib.loads(text) or {}
    except tomllib.TOMLDecodeError as exc:
        log.warning("Failed to parse TOML config %s: %s", path, exc)
        return {}


def _parse_json(text: str, path: Path) -> dict[str, object]:
    try:
        return json.loads(text) or {}
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse JSON config %s: %s", path, exc)
        return {}


def _parse_ini(text: str, path: Path) -> dict[str, object]:
    parser = configparser.ConfigParser()
    try:
        parser.read_string(text)
    except configparser.Error as exc:
        log.warning("Failed to parse INI config %s: %s", path, exc)
        return {}
    data: dict[str, dict[str, str]] = {}
    for section in parser.sections():
        data[section] = dict(parser.items(section))
    return cast("dict[str, object]", data)


def _parse_env(text: str, path: Path) -> dict[str, object]:
    del path
    data: dict[str, str] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        data[key.strip()] = value.strip()
    return cast("dict[str, object]", data)


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


def _collect_config_rows(
    cfg: ConfigIngestStepConfig,
    config_profile: ScanProfile,
) -> list[ConfigValueRow]:
    """
    Collect flattened config rows for the provided snapshot and profile.

    Parameters
    ----------
    cfg :
        Configuration for the ingestion step.
    config_profile :
        Scan profile controlling config file discovery.

    Returns
    -------
    list[ConfigValueRow]
        Flattened config keypaths for the current snapshot.
    """
    repo_root = cfg.repo_root
    rows: list[ConfigValueRow] = []
    for path in _iter_config_files(config_profile):
        if path.suffix.lower() not in CONFIG_EXTENSIONS:
            continue
        rel_path = repo_relpath(repo_root, path)
        fmt = _detect_format(path)
        data = _load_config(path, fmt)
        if not data:
            continue

        for keypath, _ in _flatten_config(data):
            rows.append(
                ConfigValueRow(
                    repo=cfg.repo,
                    commit=cfg.commit,
                    config_path=rel_path,
                    format=fmt,
                    key=keypath,
                    reference_paths=[],
                    reference_modules=[],
                    reference_count=0,
                )
            )
    return rows


@dataclass
class ConfigIngestOps(IncrementalIngestOps[ConfigValueRow], SupportsFullRebuild):
    """Implement incremental ingest wrapper for config values."""

    cfg: ConfigIngestStepConfig
    config_profile: ScanProfile
    rows: list[ConfigValueRow]
    dataset_name: str = field(init=False, default="analytics.config_values")

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """
        Config ingestion relies on a full rebuild regardless of module filters.

        Returns
        -------
        bool
            Always True to include all modules for rebuild evaluation.
        """
        del module
        return True

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """No-op; run_full_rebuild handles deletion."""

    @staticmethod
    def process_module(module: ModuleRecord) -> Iterable[ConfigValueRow]:
        """
        Return config rows for a module.

        Returns
        -------
        Iterable[ConfigValueRow]
            Always empty because config ingestion runs as a full rebuild.
        """
        del module
        return []

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[ConfigValueRow]) -> None:
        """No-op; run_full_rebuild performs insertion."""

    def run_full_rebuild(self, tracker: ChangeTracker) -> bool:
        """
        Perform a full rebuild of analytics.config_values.

        Returns
        -------
        bool
            True to signal the harness that ingestion completed.
        """
        rows = self.rows or _collect_config_rows(self.cfg, self.config_profile)

        run_batch(
            tracker.gateway,
            "analytics.config_values",
            [config_value_to_tuple(r) for r in rows],
            delete_params=[self.cfg.repo, self.cfg.commit],
        )
        log.info("config_values ingested: %d keys across config files", len(rows))
        return True


def ingest_config_values(
    gateway: StorageGateway,
    cfg: ConfigIngestStepConfig,
    config_profile: ScanProfile | None = None,
    *,
    tracker: ChangeTracker | None = None,
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

    Parameters
    ----------
    gateway:
        StorageGateway providing access to the DuckDB database.
    cfg:
        Repository context for config ingestion.
    config_profile:
        Optional scan profile to honor ignore/include rules while walking files.
    tracker : ChangeTracker | None
        Optional tracker enabling harness-based rebuild semantics.
    """
    repo_root = cfg.repo_root
    repo = cfg.repo
    commit = cfg.commit
    profile = config_profile or profile_from_env(default_config_profile(repo_root))
    rows = _collect_config_rows(cfg, profile)

    if tracker is not None:
        ops = ConfigIngestOps(cfg=cfg, config_profile=profile, rows=rows)
        tracker_for_config = replace(
            tracker,
            change_request=replace(tracker.change_request, full_rebuild=True),
        )
        run_incremental_ingest(tracker_for_config, ops)
        return

    run_batch(
        gateway,
        "analytics.config_values",
        [config_value_to_tuple(r) for r in rows],
        delete_params=[repo, commit],
    )
    log.info("config_values ingested: %d keys across config files", len(rows))
