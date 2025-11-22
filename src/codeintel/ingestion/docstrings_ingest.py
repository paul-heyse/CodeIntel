"""Extract structured docstrings with Griffe and persist to DuckDB."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import duckdb
from docstring_parser import DocstringStyle, ParseError, parse

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocstringConfig:
    """Configuration for docstring ingestion."""

    repo_root: Path
    repo: str
    commit: str


class DocstringProto(Protocol):
    """Lightweight view of a griffe docstring."""

    value: str | None


class DocLocationProto(Protocol):
    """Subset of griffe location metadata."""

    lineno: int | None
    endlineno: int | None


class DocObject(Protocol):
    """Protocol capturing the griffe object interface we rely on."""

    members: dict[str, DocObject]
    path: str | None
    docstring: DocstringProto | None
    location: DocLocationProto | None
    parent: DocObject | None
    name: str


class LoaderProto(Protocol):
    """Protocol for the griffe loader we call."""

    def load_file(self, path: str) -> DocObject:
        """Load a Python file into a griffe object tree."""
        ...


@dataclass(frozen=True)
class GriffeHandles:
    """Loaded griffe classes and loader instance."""

    module_cls: type[DocObject]
    class_cls: type[DocObject]
    function_cls: type[DocObject]
    loader: LoaderProto
    griffe_error: type[Exception]
    loading_error: type[Exception]


@dataclass(frozen=True)
class DocstringContext:
    """Shared ingestion context for building docstring rows."""

    cfg: DocstringConfig
    created_at: datetime
    handles: GriffeHandles


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

    handles = _load_griffe(repo_root)
    if handles is None:
        return

    rows: list[tuple[object, ...]] = []
    ctx = DocstringContext(cfg=cfg, created_at=datetime.now(UTC), handles=handles)

    for rel_path, module_name in modules:
        rel_path_str = str(rel_path).replace("\\", "/")
        file_path = repo_root / rel_path_str
        if not file_path.is_file():
            log.warning("Docstring ingest skipped missing file: %s", file_path)
            continue

        try:
            module_obj = handles.loader.load_file(str(file_path))
        except (FileNotFoundError, OSError, handles.loading_error, handles.griffe_error) as exc:
            log.warning("Failed to load %s with griffe: %s", file_path, exc)
            continue
        for obj in _iter_doc_objects(
            module_obj,
            handles.module_cls,
            handles.class_cls,
            handles.function_cls,
        ):
            record = _build_record(
                obj=obj,
                rel_path=rel_path_str,
                module_name=module_name,
                ctx=ctx,
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


def _load_griffe(repo_root: Path) -> GriffeHandles | None:
    """
    Attempt to import griffe and return callable handles.

    Returns
    -------
    GriffeHandles | None
        Loaded classes and loader instance, or None if griffe is missing.
    """
    try:
        griffe_module = importlib.import_module("griffe")
        griffe_dataclasses = importlib.import_module("griffe.dataclasses")
        griffe_loader = importlib.import_module("griffe.loader")
    except ImportError:
        log.warning("griffe dependency is missing; skipping docstring ingestion.")
        return None

    module_cls = cast("type[DocObject]", griffe_dataclasses.Module)
    class_cls = cast("type[DocObject]", griffe_dataclasses.Class)
    function_cls = cast("type[DocObject]", griffe_dataclasses.Function)
    loader = cast("LoaderProto", griffe_loader.GriffeLoader(search_paths=[str(repo_root)]))
    griffe_error = cast("type[Exception]", griffe_module.GriffeError)
    loading_error = cast("type[Exception]", griffe_module.LoadingError)
    return GriffeHandles(
        module_cls=module_cls,
        class_cls=class_cls,
        function_cls=function_cls,
        loader=loader,
        griffe_error=griffe_error,
        loading_error=loading_error,
    )


def _iter_doc_objects(
    obj: DocObject,
    module_cls: type[DocObject],
    class_cls: type[DocObject],
    function_cls: type[DocObject],
) -> Iterator[DocObject]:
    yield obj
    members = getattr(obj, "members", {})
    for member in members.values():
        if isinstance(member, (module_cls, class_cls, function_cls)):
            yield from _iter_doc_objects(member, module_cls, class_cls, function_cls)


def _build_record(
    obj: DocObject,
    *,
    rel_path: str,
    module_name: str,
    ctx: DocstringContext,
) -> tuple[object, ...] | None:
    kind = _object_kind(obj, ctx.handles)
    qualname = _qualname(obj, module_name, ctx.handles.module_cls)
    location = getattr(obj, "location", None)
    lineno = getattr(location, "lineno", None) if location else None
    end_lineno = getattr(location, "endlineno", None) if location else None

    raw = obj.docstring.value if obj.docstring is not None else None
    parsed = _parse_docstring(raw)

    return (
        ctx.cfg.repo,
        ctx.cfg.commit,
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
        ctx.created_at,
    )


def _object_kind(obj: DocObject, handles: GriffeHandles) -> str:
    if isinstance(obj, handles.module_cls):
        return "module"
    if isinstance(obj, handles.class_cls):
        return "class"
    if isinstance(obj, handles.function_cls):
        return (
            "method" if isinstance(getattr(obj, "parent", None), handles.class_cls) else "function"
        )
    return "unknown"


def _qualname(obj: DocObject, module_name: str, module_cls: type[DocObject]) -> str:
    path = getattr(obj, "path", None)
    if path:
        return str(path)
    if isinstance(obj, module_cls):
        return module_name
    return f"{module_name}.{getattr(obj, 'name', '<unknown>')}"


def _parse_docstring(raw: str | None) -> dict[str, object]:
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
