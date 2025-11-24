"""Scan repository sources into core.modules and related metadata tables."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path
from typing import TypedDict

import yaml

from codeintel.config.models import RepoScanConfig
from codeintel.ingestion.common import run_batch
from codeintel.ingestion.source_scanner import ScanConfig, SourceScanner
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schemas import apply_all_schemas
from codeintel.utils.paths import relpath_to_module, repo_relpath

log = logging.getLogger(__name__)


@dataclass
class ModuleRow:
    """Representation of a single discovered module."""

    module: str
    path: str
    repo: str
    commit: str
    language: str = "python"
    tags: list[str] = field(default_factory=list)
    owners: list[str] = field(default_factory=list)


class TagEntry(TypedDict, total=False):
    """Classification rules applied to repository paths."""

    tag: str
    description: str | None
    includes: list[str]
    excludes: list[str]
    matches: list[str]


def _load_tags_index(tags_index_path: Path) -> list[TagEntry]:
    if not tags_index_path.is_file():
        log.info("tags_index.yaml not found at %s", tags_index_path)
        return []

    with tags_index_path.open("r", encoding="utf8") as f:
        data = yaml.safe_load(f) or []

    if not isinstance(data, list):
        log.warning("tags_index.yaml does not contain a list; ignoring.")
        return []

    return data


def _write_tags_index_table(
    gateway: StorageGateway,
    tags_entries: list[TagEntry],
) -> None:
    # This table does not carry repo/commit; assume one-repo-per-DB.
    values = []
    for entry in tags_entries:
        tag = entry.get("tag")
        description = entry.get("description")
        includes = entry.get("includes") or []
        excludes = entry.get("excludes") or []
        matches = entry.get("matches") or []
        values.append([tag, description, includes, excludes, matches])

    run_batch(gateway, "analytics.tags_index", values, delete_params=[], scope="tags_index")


def _tags_for_path(rel_path: str, tags_entries: list[TagEntry]) -> list[str]:
    tags: list[str] = []

    for entry in tags_entries:
        tag = entry.get("tag")
        if not tag:
            continue

        matches = entry.get("matches") or []
        includes = entry.get("includes") or []
        excludes = entry.get("excludes") or []

        if rel_path in matches:
            tags.append(tag)
            continue

        if includes and any(fnmatch(rel_path, pattern) for pattern in includes):
            if excludes and any(fnmatch(rel_path, pattern) for pattern in excludes):
                continue
            tags.append(tag)

    return sorted(set(tags))


def ingest_repo(
    gateway: StorageGateway,
    cfg: RepoScanConfig,
    scan_config: ScanConfig | None = None,
    *,
    apply_schema: bool = False,
) -> None:
    """
    Scan the repository and populate module metadata tables.

      - core.modules
      - core.repo_map
      - analytics.tags_index

    Assumes the DuckDB schemas & tables have already been created.

    Parameters
    ----------
    gateway:
        StorageGateway providing access to the target DuckDB database.
    cfg:
        Repository context and optional tags_index override.
    scan_config:
        Optional shared scan configuration; defaults to Python-only scanning.
    apply_schema:
        When True, apply all schemas prior to ingestion (destructive). Default False.
    """
    con = gateway.con
    repo_root = cfg.repo_root
    scan_cfg = scan_config or ScanConfig(repo_root=repo_root)
    tags_index_path = cfg.tags_index_path or (repo_root / "tags_index.yaml")

    log.info("Scanning repo %s at %s", cfg.repo, repo_root)
    if apply_schema:
        message = "apply_schema=True will drop/recreate tables; use sparingly"
        log.warning(message)
        apply_all_schemas(con)

    tags_entries = _load_tags_index(tags_index_path)
    if tags_entries:
        _write_tags_index_table(gateway, tags_entries)

    modules: dict[str, ModuleRow] = {}

    scanner = SourceScanner(scan_cfg)

    for record in scanner.iter_files(log):
        path = record.path
        rel_path = repo_relpath(repo_root, path)
        module = relpath_to_module(rel_path)
        tags = _tags_for_path(rel_path, tags_entries)

        modules[module] = ModuleRow(
            module=module,
            path=rel_path,
            repo=cfg.repo,
            commit=cfg.commit,
            language="python",
            tags=tags,
            owners=[],  # owner inference can be added later
        )

    log.info("Discovered %d Python modules", len(modules))

    module_values = [
        [
            m.module,
            m.path,
            m.repo,
            m.commit,
            m.language,
            m.tags or [],
            m.owners or [],
        ]
        for m in modules.values()
    ]
    run_batch(
        gateway,
        "core.modules",
        module_values,
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    modules_json = {m.module: m.path for m in modules.values()}
    now = datetime.now(UTC)
    run_batch(
        gateway,
        "core.repo_map",
        [[cfg.repo, cfg.commit, modules_json, {}, now]],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info("repo_map and modules written for %s@%s", cfg.repo, cfg.commit)
