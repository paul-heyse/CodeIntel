"""Extract structured docstrings with AST and docstring-parser and persist to DuckDB."""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TypedDict

from docstring_parser import DocstringStyle, ParseError, parse

from codeintel.config.models import DocstringConfig
from codeintel.ingestion.common import (
    iter_modules,
    load_module_map,
    read_module_source,
    run_batch,
    should_skip_empty,
)
from codeintel.ingestion.source_scanner import ScanConfig
from codeintel.models.rows import DocstringRow, docstring_row_to_tuple
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DocstringContext:
    """Shared ingestion context for building docstring rows."""

    cfg: DocstringConfig
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


class DocstringVisitor(ast.NodeVisitor):
    """Traverse AST to collect docstrings from modules, classes, and functions."""

    def __init__(self, rel_path: str, module_name: str, ctx: DocstringContext) -> None:
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
        kind = "method" if self.scope_stack else "function"
        self._record_docstring(node, kind)
        self.scope_stack.append(node.name)
        self.generic_visit(node)
        self.scope_stack.pop()

    def _record_docstring(self, node: DocstringNode, kind: str) -> None:
        raw_doc = ast.get_docstring(node, clean=False)
        if not raw_doc:
            return

        # Build qualname
        if kind == "module":
            qualname = self.module_name
        else:
            # node has .name attribute for class/func
            name = getattr(node, "name", "<unknown>")
            if self.scope_stack:
                # For items inside scope_stack (which includes current node name for recursive visits,
                # but we are visiting *this* node now, so its name is NOT in stack yet for `visit_ClassDef` logic above...
                # Wait, visit methods append BEFORE generic_visit.
                # But we record docstring BEFORE append.
                # So stack contains parents.
                qualname = f"{self.module_name}." + ".".join([*self.scope_stack, name])
            else:
                qualname = f"{self.module_name}.{name}"

        # Line numbers
        # getattr used because Module might not have lineno in some py versions or edge cases,
        # though standard AST usually does for these types.
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", None)

        parsed = _parse_docstring(raw_doc)

        self.rows.append(
            DocstringRow(
                repo=self.ctx.cfg.repo,
                commit=self.ctx.cfg.commit,
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


def ingest_docstrings(
    gateway: StorageGateway,
    cfg: DocstringConfig,
    scan_config: ScanConfig | None = None,
) -> None:
    """
    Extract docstrings for all Python modules in core.modules and persist them.

    Parameters
    ----------
    gateway :
        StorageGateway providing access to the DuckDB database.
    cfg : DocstringConfig
        Repository context for this ingestion run.
    scan_config : ScanConfig | None
        Optional scan configuration controlling iteration cadence.
    """
    con = gateway.con
    repo_root = cfg.repo_root.resolve()
    module_map = load_module_map(gateway, cfg.repo, cfg.commit, language="python", logger=log)
    if should_skip_empty(module_map, logger=log):
        return

    rows: list[DocstringRow] = []
    ctx = DocstringContext(cfg=cfg, created_at=datetime.now(UTC))

    for record in iter_modules(
        module_map,
        repo_root,
        logger=log,
        scan_config=scan_config,
    ):
        source = read_module_source(record, logger=log)
        if source is None:
            continue

        try:
            tree = ast.parse(source, filename=str(record.file_path))
        except SyntaxError:
            log.warning("Failed to parse AST for docstrings: %s", record.file_path)
            continue
        visitor = DocstringVisitor(
            rel_path=record.rel_path, module_name=record.module_name, ctx=ctx
        )
        visitor.visit(tree)
        rows.extend(visitor.rows)

    run_batch(
        gateway,
        "core.docstrings",
        [docstring_row_to_tuple(row) for row in rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
    log.info("Docstrings ingested: %d rows for %s@%s", len(rows), cfg.repo, cfg.commit)


def _parse_docstring(raw: str | None) -> ParsedDocstring:
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
