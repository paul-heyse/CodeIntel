"""Extract structured docstrings with Griffe and persist to DuckDB."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
from docstring_parser import DocstringStyle, ParseError, parse
from griffe import GriffeError, LoadingError
from griffe.dataclasses import Class, Function, Module, Object
from griffe.loader import GriffeLoader

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocstringConfig:
    """Configuration for docstring ingestion."""

    repo_root: Path
    repo: str
    commit: str


def ingest_docstrings(con: duckdb.DuckDBPyConnection, cfg: DocstringConfig) -> None:
    """
    Extract docstrings for all Python modules in core.modules and persist them.

    Parameters
    ----------
    con : duckdb.DuckDBPyConnection
        Active DuckDB connection.
    cfg : DocstringConfig
        Repository context for this ingestion run.
    """
    repo_root = cfg.repo_root.resolve()
    modules = con.execute(
        """
        SELECT path, module
        FROM core.modules
        WHERE repo = ? AND commit = ? AND language = 'python'
        """,
        [cfg.repo, cfg.commit],
    ).fetchall()
    if not modules:
        log.info("No python modules found for %s@%s; skipping docstrings", cfg.repo, cfg.commit)
        return

    loader = GriffeLoader(search_paths=[str(repo_root)])
    rows: list[tuple[Any, ...]] = []
    now = datetime.now(UTC)

    for rel_path, module_name in modules:
        rel_path_str = str(rel_path).replace("\\", "/")
        file_path = repo_root / rel_path_str
        if not file_path.is_file():
            log.warning("Docstring ingest skipped missing file: %s", file_path)
            continue

        try:
            module_obj = loader.load_file(str(file_path))
        except (FileNotFoundError, OSError, LoadingError, GriffeError) as exc:
            log.warning("Failed to load %s with griffe: %s", file_path, exc)
            continue

        for obj in _iter_doc_objects(module_obj):
            record = _build_record(
                obj=obj,
                rel_path=rel_path_str,
                module_name=module_name,
                cfg=cfg,
                created_at=now,
            )
            if record is not None:
                rows.append(record)

    con.execute(
        """
        DELETE FROM core.docstrings
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
    )

    if rows:
        con.executemany(
            """
            INSERT INTO core.docstrings (
                repo, commit, rel_path, module, qualname, kind,
                lineno, end_lineno, raw_docstring, style, short_desc,
                long_desc, params, returns, raises, examples, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    log.info("Docstrings ingested: %d rows for %s@%s", len(rows), cfg.repo, cfg.commit)


def _iter_doc_objects(obj: Object) -> Iterator[Object]:
    yield obj
    for member in obj.members.values():
        if isinstance(member, (Module, Class, Function)):
            yield from _iter_doc_objects(member)


def _build_record(
    obj: Object,
    rel_path: str,
    module_name: str,
    cfg: DocstringConfig,
    created_at: datetime,
) -> tuple[Any, ...] | None:
    kind = _object_kind(obj)
    qualname = _qualname(obj, module_name)
    location = getattr(obj, "location", None)
    lineno = getattr(location, "lineno", None) if location else None
    end_lineno = getattr(location, "endlineno", None) if location else None

    raw = obj.docstring.value if obj.docstring is not None else None
    parsed = _parse_docstring(raw)

    return (
        cfg.repo,
        cfg.commit,
        rel_path,
        module_name,
        qualname,
        kind,
        lineno,
        end_lineno,
        raw,
        parsed["style"],
        parsed["short_desc"],
        parsed["long_desc"],
        parsed["params"],
        parsed["returns"],
        parsed["raises"],
        parsed["examples"],
        created_at,
    )


def _object_kind(obj: Object) -> str:
    if isinstance(obj, Module):
        return "module"
    if isinstance(obj, Class):
        return "class"
    if isinstance(obj, Function):
        return "method" if isinstance(obj.parent, Class) else "function"
    return "unknown"


def _qualname(obj: Object, module_name: str) -> str:
    path = getattr(obj, "path", None)
    if path:
        return str(path)
    if isinstance(obj, Module):
        return module_name
    return f"{module_name}.{getattr(obj, 'name', '<unknown>')}"


def _parse_docstring(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {
            "style": None,
            "short_desc": None,
            "long_desc": None,
            "params": [],
            "returns": None,
            "raises": [],
            "examples": [],
        }
    try:
        parsed = parse(raw, style=DocstringStyle.AUTO)
    except ParseError:
        return {
            "style": "unknown",
            "short_desc": None,
            "long_desc": None,
            "params": [],
            "returns": None,
            "raises": [],
            "examples": [],
        }

    params = [
        {
            "name": p.arg_name,
            "type": p.type_name,
            "desc": p.description,
            "default": p.default,
        }
        for p in parsed.params
    ]
    returns = None
    if parsed.returns is not None:
        returns = {
            "type": parsed.returns.type_name,
            "desc": parsed.returns.description,
        }
    raises = [
        {
            "type": r.type_name,
            "desc": r.description,
        }
        for r in parsed.raises
    ]
    examples: list[str] = []
    for ex in parsed.examples:
        text = (
            getattr(ex, "description", None)
            or getattr(ex, "snippet", None)
            or getattr(ex, "example", None)
            or getattr(ex, "text", None)
        )
        if text:
            examples.append(text)

    return {
        "style": "auto",
        "short_desc": parsed.short_description,
        "long_desc": parsed.long_description,
        "params": params,
        "returns": returns,
        "raises": raises,
        "examples": examples,
    }
