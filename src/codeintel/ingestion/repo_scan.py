"""Scan repository sources into core.modules and related metadata tables."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from fnmatch import fnmatch
from pathlib import Path

import duckdb
import yaml

log = logging.getLogger(__name__)

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    ".tox",
    "__pycache__",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
}


@dataclass
class ModuleRow:
    """
    Representation of a single discovered module.

    Parameters
    ----------
    module : str
        Dotted module path derived from the file location.
    path : str
        Repository-relative file path.
    repo : str
        Repository slug.
    commit : str
        Commit SHA for the snapshot being ingested.
    language : str, optional
        Language identifier, defaults to "python".
    tags : list[str] | None, optional
        Optional classification tags for the module.
    owners : list[str] | None, optional
        Optional list of owners for the module.
    """

    module: str
    path: str
    repo: str
    commit: str
    language: str = "python"
    tags: list[str] = None
    owners: list[str] = None


def _iter_python_files(repo_root: Path) -> Iterable[Path]:
    for path in repo_root.rglob("*.py"):
        rel_parts = path.relative_to(repo_root).parts
        if any(part in IGNORE_DIRS for part in rel_parts):
            continue
        yield path


def _path_to_module(repo_root: Path, path: Path) -> str:
    rel = path.relative_to(repo_root).with_suffix("")
    return ".".join(rel.parts)


def relpath_to_module(rel_path: Path | str) -> str:
    """
    Convert a repository-relative Python path to a dotted module name.

    Parameters
    ----------
    rel_path:
        Repo-relative path to a .py file.

    Returns
    -------
    str
        Dotted module path.
    """
    return ".".join(Path(rel_path).with_suffix("").parts)


def _load_tags_index(tags_index_path: Path) -> list[dict]:
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
    con: duckdb.DuckDBPyConnection,
    tags_entries: list[dict],
) -> None:
    # This table does not carry repo/commit; assume one-repo-per-DB.
    con.execute("DELETE FROM analytics.tags_index")

    insert_sql = """
        INSERT INTO analytics.tags_index (tag, description, includes, excludes, matches)
        VALUES (?, ?, ?, ?, ?)
    """

    for entry in tags_entries:
        tag = entry.get("tag")
        description = entry.get("description")
        includes = entry.get("includes") or []
        excludes = entry.get("excludes") or []
        matches = entry.get("matches") or []
        con.execute(insert_sql, [tag, description, includes, excludes, matches])


def _tags_for_path(rel_path: str, tags_entries: list[dict]) -> list[str]:
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
            if not excludes or not any(fnmatch(rel_path, pattern) for pattern in excludes):
                tags.append(tag)

    return sorted(set(tags))


def ingest_repo(
    con: duckdb.DuckDBPyConnection,
    repo_root: Path,
    repo: str,
    commit: str,
    *,
    tags_index_path: Path | None = None,
) -> None:
    """
    Scan the repository and populate module metadata tables.

      - core.modules
      - core.repo_map
      - analytics.tags_index

    Assumes the DuckDB schemas & tables have already been created.

    Parameters
    ----------
    con:
        DuckDB connection.
    repo_root:
        Filesystem path to the repo root.
    repo:
        Repo identifier slug.
    commit:
        Commit SHA for this analysis run.
    tags_index_path:
        Optional explicit path to tags_index.yaml. Defaults to
        `<repo_root>/tags_index.yaml`.
    """
    repo_root = repo_root.resolve()
    tags_index_path = tags_index_path or (repo_root / "tags_index.yaml")

    log.info("Scanning repo %s at %s", repo, repo_root)

    tags_entries = _load_tags_index(tags_index_path)
    if tags_entries:
        _write_tags_index_table(con, tags_entries)

    modules: dict[str, ModuleRow] = {}

    for path in _iter_python_files(repo_root):
        rel_path = path.relative_to(repo_root).as_posix()
        module = _path_to_module(repo_root, path)
        tags = _tags_for_path(rel_path, tags_entries)

        modules[module] = ModuleRow(
            module=module,
            path=rel_path,
            repo=repo,
            commit=commit,
            language="python",
            tags=tags,
            owners=[],  # owner inference can be added later
        )

    log.info("Discovered %d Python modules", len(modules))

    # Truncate existing rows for this repo/commit.
    con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
        [repo, commit],
    )

    insert_modules = """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """

    for m in modules.values():
        con.execute(
            insert_modules,
            [
                m.module,
                m.path,
                m.repo,
                m.commit,
                m.language,
                m.tags or [],
                m.owners or [],
            ],
        )

    # core.repo_map: single row per (repo, commit)
    con.execute(
        "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
        [repo, commit],
    )

    modules_json = {m.module: m.path for m in modules.values()}
    now = datetime.now(UTC)

    con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        [repo, commit, modules_json, {}, now],
    )

    log.info("repo_map and modules written for %s@%s", repo, commit)
