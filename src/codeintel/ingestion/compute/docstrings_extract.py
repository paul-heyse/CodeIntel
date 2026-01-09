"""Docstring extraction step with port injection.

This module provides a pure domain logic implementation for extracting
and parsing structured docstrings, using ports for all I/O operations.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from docstring_parser import DocstringStyle, ParseError, parse

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.core.columnar.rows import (
    ColumnarRows,
    columnar_buffer_for_table_key,
    empty_table_for_table,
    table_for_columnar_rows,
)
from codeintel.ingestion.compute.base import BaseExtractStep, persist_arrow_tables
from codeintel.ingestion.context import IngestionContext, resolve_repo_commit

if TYPE_CHECKING:
    import pyarrow as pa

    from codeintel.ingestion.infrastructure.py_frontend import PyFrontend
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort

log = logging.getLogger(__name__)
DOCSTRINGS_TABLE_KEY = "core.docstrings"
DocstringRow = dict[str, object]


@dataclass(frozen=True)
class DocstringContext:
    """Shared ingestion context for building docstring rows."""

    repo: str
    commit: str
    created_at: datetime


class ParsedDocstring(TypedDict):
    """Normalized docstring parts parsed from raw text."""

    style: str | None
    short_desc: str | None
    long_desc: str | None
    params: list[dict[str, str | None]] | None
    returns: dict[str, str | None] | None
    raises: list[dict[str, str | None]] | None
    examples: list[str] | None


type DocstringNode = ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef


def _detect_style(raw: str) -> str | None:
    """Detect the docstring style (numpy, google, sphinx, etc.).

    Returns
    -------
    str | None
        Style name or None if undetected.
    """
    for style in (DocstringStyle.NUMPYDOC, DocstringStyle.GOOGLE, DocstringStyle.EPYDOC):
        try:
            parsed = parse(raw, style=style)
            if parsed.params or parsed.returns or parsed.raises:
                name = style.name.lower()
                return "numpy" if name == "numpydoc" else name
        except ParseError:
            continue
    return None


def _parse_docstring(raw: str) -> ParsedDocstring:
    """Parse a docstring into structured components.

    Returns
    -------
    ParsedDocstring
        Parsed docstring structure.
    """
    try:
        parsed = parse(raw)
        params = [
            {"name": p.arg_name, "type_name": p.type_name, "description": p.description}
            for p in parsed.params
        ]
        returns_obj = None
        if parsed.returns:
            returns_obj = {
                "type_name": parsed.returns.type_name,
                "description": parsed.returns.description,
            }
        raises = [{"type_name": r.type_name, "description": r.description} for r in parsed.raises]
        examples = [e.description for e in parsed.examples if e.description]

        return ParsedDocstring(
            style=_detect_style(raw),
            short_desc=parsed.short_description,
            long_desc=parsed.long_description,
            params=params or None,
            returns=returns_obj,
            raises=raises or None,
            examples=examples or None,
        )
    except (ParseError, ValueError) as exc:
        log.debug("Failed to parse docstring: %s", exc)
        return ParsedDocstring(
            style=None,
            short_desc=raw.split("\n", maxsplit=1)[0] if raw else None,
            long_desc=None,
            params=None,
            returns=None,
            raises=None,
            examples=None,
        )


class DocstringVisitor(ast.NodeVisitor):
    """Traverse AST to collect docstrings from modules, classes, and functions."""

    def __init__(self, rel_path: str, module_name: str, ctx: DocstringContext) -> None:
        """Initialize visitor.

        Parameters
        ----------
        rel_path
            Relative path to the file.
        module_name
            Python module name.
        ctx
            Docstring extraction context.
        """
        self.rel_path = rel_path
        self.module_name = module_name
        self.rows: list[DocstringRow] = []
        self.scope_stack: list[str] = []
        self.ctx = ctx

    def visit_Module(self, node: ast.Module) -> None:
        """Record a module-level docstring before traversing children."""
        self._record_docstring(node, "module")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Record a class docstring while tracking nested scope."""
        self._record_docstring(node, "class")
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Record a function docstring and traverse its body."""
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Record an async function docstring and traverse its body."""
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        """Process a function or async function node."""
        kind = "method" if self.scope_stack else "function"
        self._record_docstring(node, kind)
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _record_docstring(self, node: DocstringNode, kind: str) -> None:
        """Record a docstring row for a node."""
        raw_doc = ast.get_docstring(node, clean=False)
        if not raw_doc:
            return

        if kind == "module":
            qualname = self.module_name
        else:
            name = getattr(node, "name", "<unknown>")
            if self.scope_stack:
                qualname = f"{self.module_name}." + ".".join([*self.scope_stack, name])
            else:
                qualname = f"{self.module_name}.{name}"

        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)
        parsed: ParsedDocstring = _parse_docstring(raw_doc)

        self.rows.append(
            {
                "repo": self.ctx.repo,
                "commit": self.ctx.commit,
                "rel_path": self.rel_path,
                "module": self.module_name,
                "qualname": qualname,
                "kind": kind,
                "lineno": lineno,
                "end_lineno": end_lineno,
                "raw_docstring": raw_doc,
                "style": parsed["style"],
                "short_desc": parsed["short_desc"],
                "long_desc": parsed["long_desc"],
                "params": parsed["params"],
                "returns": parsed["returns"],
                "raises": parsed["raises"],
                "examples": parsed["examples"],
                "created_at": self.ctx.created_at,
            }
        )


def _extract_module_docstrings(
    module: ModuleRecord,
    source: str,
    ctx: DocstringContext,
    *,
    tree: ast.AST | None = None,
) -> list[DocstringRow]:
    """Extract docstrings from module source.

    Parameters
    ----------
    module
        Module record with metadata.
    source
        Module source code.
    ctx
        Docstring extraction context.
    tree
        Pre-parsed AST when already available.

    Returns
    -------
    list[DocstringRow]
        Extracted docstring rows.
    """
    if tree is None:
        try:
            tree = ast.parse(source, filename=str(module.file_path))
        except (SyntaxError, ValueError) as exc:
            log.warning("Failed to parse %s: %s", module.file_path, exc)
            return []

    visitor = DocstringVisitor(
        rel_path=module.rel_path,
        module_name=module.module_name,
        ctx=ctx,
    )
    try:
        visitor.visit(tree)
    except (RecursionError, ValueError) as exc:
        log.warning("AST visit failed for %s: %s", module.file_path, exc)
        return []

    return visitor.rows


@dataclass(frozen=True)
class DocstringsExtractResult:
    """Result bundle for docstring extraction."""

    result: ExecutionResult
    rows: ColumnarRows = field(default_factory=dict)
    rows_reader: pa.Table = field(
        default_factory=lambda: empty_table_for_table(DOCSTRINGS_TABLE_KEY)
    )
    row_count: int = 0


class DocstringsExtractStep(BaseExtractStep):
    """Docstring extraction step with port injection.

    This step extracts structured docstrings from modules,
    using ports for all I/O operations.

    Parameters
    ----------
    discovery
        Discovery port for reading module source.
    """

    def __init__(
        self,
        discovery: ModuleDiscoveryPort,
        *,
        frontend: PyFrontend | None = None,
    ) -> None:
        super().__init__(discovery=discovery, frontend=frontend)

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str | None = None,
        commit: str | None = None,
        context: IngestionContext | None = None,
        storage: IngestStoragePort | None = None,
    ) -> DocstringsExtractResult:
        """Execute docstring extraction on the provided modules.

        Parameters
        ----------
        modules
            Modules to process.
        repo
            Repository identifier.
        commit
            Commit identifier.
        context
            Optional ingestion context supplying repo/commit defaults.
        storage
            Optional storage port for persisting Arrow outputs.

        Returns
        -------
        DocstringsExtractResult
            Result bundle with row tuples and execution status.
        """
        resolved_repo, resolved_commit = resolve_repo_commit(
            context=context,
            repo=repo,
            commit=commit,
        )
        ctx = DocstringContext(
            repo=resolved_repo,
            commit=resolved_commit,
            created_at=datetime.now(UTC),
        )

        try:
            buffer = columnar_buffer_for_table_key(DOCSTRINGS_TABLE_KEY)
        except (KeyError, RuntimeError) as exc:
            return DocstringsExtractResult(result=ExecutionResult.failed(str(exc)))

        for module, source in self._iter_python_sources(modules):
            tree = self._frontend.get_ast(module) if self._frontend is not None else None
            docstrings = _extract_module_docstrings(module, source, ctx, tree=tree)
            for row in docstrings:
                buffer.append(row)

        log.info(
            "Docstring extraction: repo=%s commit=%s rows=%d",
            resolved_repo,
            resolved_commit,
            buffer.row_count,
        )

        rows_reader, row_count = table_for_columnar_rows(
            DOCSTRINGS_TABLE_KEY,
            buffer.data,
            extras_policy="retain",
        )
        scope = f"{resolved_repo}@{resolved_commit}"
        persist_arrow_tables(
            storage,
            {DOCSTRINGS_TABLE_KEY: rows_reader},
            scope=scope,
        )
        return DocstringsExtractResult(
            result=ExecutionResult.ok(),
            rows=buffer.data,
            rows_reader=rows_reader,
            row_count=row_count,
        )


__all__ = [
    "DocstringContext",
    "DocstringVisitor",
    "DocstringsExtractResult",
    "DocstringsExtractStep",
    "ParsedDocstring",
]
