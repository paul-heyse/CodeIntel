"""Docstring extraction step with port injection.

This module provides a pure domain logic implementation for extracting
and parsing structured docstrings, using ports for all I/O operations.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, TypedDict

from docstring_parser import DocstringStyle, ParseError, parse

from codeintel.config.datasets.columns import load_columns_by_table, serialize_row
from codeintel.core.schemas.generated_types import DocstringRow
from codeintel.ingestion.compute.base import BaseExtractStep, StepResult

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)


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
    params: object
    returns: object
    raises: object
    examples: object


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
        parsed = _parse_docstring(raw_doc)

        self.rows.append(
            DocstringRow(
                repo=self.ctx.repo,
                commit=self.ctx.commit,
                rel_path=self.rel_path,
                module=self.module_name,
                qualname=qualname,
                kind=kind,
                lineno=lineno,
                end_lineno=end_lineno,
                raw_docstring=raw_doc,
                style=parsed["style"],
                short_desc=parsed["short_desc"],
                long_desc=parsed["long_desc"],
                params=parsed["params"],
                returns=parsed["returns"],
                raises=parsed["raises"],
                examples=parsed["examples"],
                created_at=self.ctx.created_at,
            )
        )


def _extract_module_docstrings(
    module: ModuleRecord,
    source: str,
    ctx: DocstringContext,
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

    Returns
    -------
    list[DocstringRow]
        Extracted docstring rows.
    """
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


class DocstringsExtractStep(BaseExtractStep):
    """Docstring extraction step with port injection.

    This step extracts structured docstrings from modules,
    using ports for all I/O operations.

    Parameters
    ----------
    storage
        Storage port for persisting data.
    discovery
        Discovery port for reading module source.
    """

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
    ) -> StepResult:
        """Execute docstring extraction on the provided modules.

        Parameters
        ----------
        modules
            Modules to process.
        repo
            Repository identifier.
        commit
            Commit identifier.

        Returns
        -------
        StepResult
            Execution result with row counts.
        """
        ctx = DocstringContext(
            repo=repo,
            commit=commit,
            created_at=datetime.now(UTC),
        )

        columns = load_columns_by_table().get("core.docstrings", [])
        if not columns:
            return StepResult.fail("core.docstrings missing from schema provider")
        all_rows: list[tuple[object, ...]] = []
        errors: list[str] = []
        processed_paths: list[str] = []

        for module, source in self._iter_python_sources(modules):
            processed_paths.append(module.rel_path)
            docstrings = _extract_module_docstrings(module, source, ctx)
            all_rows.extend(serialize_row(ds, columns) for ds in docstrings)

        if processed_paths:
            self._storage.delete_by_paths(
                "core.docstrings",
                processed_paths,
                path_column="rel_path",
                repo=repo,
                commit=commit,
            )

        table_counts = self._write_and_count("core.docstrings", all_rows, repo=repo, commit=commit)
        total_rows = table_counts.get("core.docstrings", 0)

        log.info(
            "Docstring extraction: repo=%s commit=%s rows=%d",
            repo,
            commit,
            len(all_rows),
        )

        return StepResult(
            rows_written=total_rows,
            table_counts=table_counts,
            errors=errors,
        )


__all__ = [
    "DocstringContext",
    "DocstringVisitor",
    "DocstringsExtractStep",
    "ParsedDocstring",
]
