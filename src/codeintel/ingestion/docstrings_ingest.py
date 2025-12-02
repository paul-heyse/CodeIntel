"""Extract structured docstrings with AST and docstring-parser and persist to DuckDB."""

from __future__ import annotations

import ast
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

from docstring_parser import DocstringStyle, ParseError, parse

from codeintel.config.builder import DocstringStepConfig
from codeintel.config.datasets import DocstringRow, docstring_row_to_tuple
from codeintel.ingestion.common import ModuleRecord, iter_modules, read_module_source, run_batch
from codeintel.ingestion.pipeline import (
    IngestPipeline,
    PipelineConfig,
    PipelineResult,
    execute_pipeline,
)
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.storage.module_index import load_module_map

if TYPE_CHECKING:
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.storage.gateway import StorageGateway

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


def _delete_existing_docstrings(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    rel_paths: list[str],
) -> None:
    """
    Remove existing docstring rows for a set of module paths.

    When rel_paths is empty, delete all rows for repo@commit.
    """
    if rel_paths:
        gateway.con.execute(
            """
            DELETE FROM core.docstrings
            WHERE repo = ? AND commit = ? AND rel_path IN (SELECT * FROM UNNEST(?))
            """,
            [repo, commit, rel_paths],
        )
        return

    run_batch(
        gateway,
        "core.docstrings",
        [],
        delete_params=[repo, commit],
    )


class DocstringPipeline:
    """Pipeline implementation for docstring extraction."""

    def __init__(self, *, repo: str, commit: str) -> None:
        self._repo = repo
        self._commit = commit
        self._ctx: DocstringContext = field(init=False)

    def _get_ctx(self) -> DocstringContext:
        """
        Get or create the docstring context.

        Returns
        -------
        DocstringContext
            Shared context for the extraction run.
        """
        return DocstringContext(
            repo=self._repo,
            commit=self._commit,
            created_at=datetime.now(UTC),
        )

    @property
    def dataset_name(self) -> str:
        """Return the dataset name for this pipeline."""
        return "core.docstrings"

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """
        Restrict ingestion to Python source modules.

        Parameters
        ----------
        module
            Module metadata describing the candidate file.

        Returns
        -------
        bool
            True when the module path ends with .py.
        """
        return module.rel_path.endswith(".py")

    def process_module(self, module: ModuleRecord) -> Iterable[DocstringRow]:
        """
        Parse a module and emit docstring rows.

        Parameters
        ----------
        module
            Module metadata describing the file to analyze.

        Returns
        -------
        Iterable[DocstringRow]
            Parsed docstring rows (empty on failure).
        """
        source = read_module_source(module, logger=log)
        if source is None:
            return []

        try:
            tree = ast.parse(source, filename=str(module.file_path))
        except SyntaxError:
            log.warning("Failed to parse AST for docstrings: %s", module.file_path)
            return []

        ctx = self._get_ctx()
        visitor = DocstringVisitor(
            rel_path=module.rel_path,
            module_name=module.module_name,
            ctx=ctx,
        )
        visitor.visit(tree)
        return visitor.rows

    def persist_rows(self, gateway: StorageGateway, rows: Sequence[DocstringRow]) -> int:
        """
        Persist parsed docstring rows.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rows
            Rows to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if not rows:
            return 0

        run_batch(
            gateway,
            "core.docstrings",
            [docstring_row_to_tuple(row) for row in rows],
            delete_params=None,
            scope=f"{self._repo}@{self._commit}",
        )
        return len(rows)

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """
        Delete rows corresponding to changed or deleted modules.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rel_paths
            Paths to delete.
        """
        _delete_existing_docstrings(
            gateway,
            repo=self._repo,
            commit=self._commit,
            rel_paths=list(rel_paths),
        )


# Type assertion that DocstringPipeline implements IngestPipeline
_: type[IngestPipeline[DocstringRow]] = DocstringPipeline


@dataclass
class DocstringIngestOps:
    """Backward-compatible ops class for incremental docstring ingestion.

    This class wraps the pipeline implementation to maintain API compatibility
    with existing code that uses run_incremental_ingest().
    """

    cfg: object  # DocstringStepConfig
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    dataset_name: str = field(init=False, default="core.docstrings")

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """
        Restrict ingestion to Python source modules.

        Returns
        -------
        bool
            True when the module path ends with .py.
        """
        return module.rel_path.endswith(".py")

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """Delete rows corresponding to changed or deleted modules."""
        if not isinstance(self.cfg, DocstringStepConfig):
            return
        _delete_existing_docstrings(
            gateway,
            repo=self.cfg.repo,
            commit=self.cfg.commit,
            rel_paths=list(rel_paths),
        )

    def process_module(self, module: ModuleRecord) -> Iterable[DocstringRow]:
        """
        Parse a module and emit docstring rows.

        Returns
        -------
        Iterable[DocstringRow]
            Extracted docstring rows for the module.
        """
        if not isinstance(self.cfg, DocstringStepConfig):
            return []

        source = read_module_source(module, logger=log)
        if source is None:
            return []

        try:
            tree = ast.parse(source, filename=str(module.file_path))
        except SyntaxError:
            log.warning("Failed to parse AST for docstrings: %s", module.file_path)
            return []

        ctx = DocstringContext(
            repo=self.cfg.repo,
            commit=self.cfg.commit,
            created_at=self.created_at,
        )
        visitor = DocstringVisitor(
            rel_path=module.rel_path,
            module_name=module.module_name,
            ctx=ctx,
        )
        visitor.visit(tree)
        return visitor.rows

    def insert_rows(self, gateway: StorageGateway, rows: Sequence[DocstringRow]) -> None:
        """Persist parsed docstring rows."""
        if not isinstance(self.cfg, DocstringStepConfig):
            return
        if not rows:
            return

        run_batch(
            gateway,
            "core.docstrings",
            [docstring_row_to_tuple(row) for row in rows],
            delete_params=None,
            scope=f"{self.cfg.repo}@{self.cfg.commit}",
        )


def ingest_docstrings(  # noqa: PLR0913
    gateway: StorageGateway,
    modules_or_cfg: Sequence[ModuleRecord] | object = (),
    code_profile: object = None,
    *,
    repo: str | None = None,
    commit: str | None = None,
    repo_root: Path | None = None,
    tracker: ChangeTracker | None = None,
    modules: Sequence[ModuleRecord] | None = None,
) -> PipelineResult:
    """
    Extract docstrings for all Python modules and persist them.

    Supports both new and legacy calling conventions for backward compatibility.

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    modules_or_cfg
        Either modules sequence (new API) or DocstringStepConfig (legacy API).
    code_profile
        Optional scan profile (legacy API).
    repo
        Repository identifier (new API).
    commit
        Commit identifier (new API).
    repo_root
        Repository root path (new API).
    tracker
        Optional change tracker for incremental mode.
    modules
        Alternative modules parameter (new API).

    Returns
    -------
    PipelineResult
        Execution result with row counts.

    Raises
    ------
    ValueError
        When repo, commit, or repo_root are not provided via either the
        cfg parameter or explicit keyword arguments.
    """
    # Handle legacy API: ingest_docstrings(gateway, cfg, code_profile, tracker=...)
    if isinstance(modules_or_cfg, DocstringStepConfig):
        cfg = modules_or_cfg
        profile: ScanProfile | None = None
        if isinstance(code_profile, ScanProfile):
            profile = code_profile

        module_map = load_module_map(
            gateway,
            cfg.repo,
            cfg.commit,
            language="python",
            logger=log,
        )

        actual_modules = list(
            iter_modules(
                module_map,
                cfg.repo_root,
                logger=log,
                scan_profile=profile,
            )
        )
        actual_repo = cfg.repo
        actual_commit = cfg.commit
        actual_repo_root = cfg.repo_root
    else:
        # New API
        if modules is not None:
            actual_modules = list(modules)
        elif isinstance(modules_or_cfg, Sequence):
            actual_modules = list(modules_or_cfg)
        else:
            actual_modules = []
        actual_repo = repo
        actual_commit = commit
        actual_repo_root = repo_root

    if actual_repo is None or actual_commit is None or actual_repo_root is None:
        message = "repo, commit, and repo_root are required"
        raise ValueError(message)

    pipeline = DocstringPipeline(repo=actual_repo, commit=actual_commit)

    result = execute_pipeline(
        pipeline,
        gateway,
        actual_modules,
        tracker=tracker,
        config=PipelineConfig(),
    )

    log.info(
        "Docstrings ingested: %d rows for %s@%s",
        result.rows_persisted,
        actual_repo,
        actual_commit,
    )

    return result


# Backward compatibility: keep old function signature
def ingest_docstrings_legacy(
    gateway: StorageGateway,
    cfg: object,
    code_profile: object = None,
    *,
    tracker: ChangeTracker | None = None,
) -> None:
    """
    Legacy entry point for docstring ingestion.

    Deprecated: Use ingest_docstrings() with explicit parameters instead.

    Raises
    ------
    TypeError
        When cfg is not a DocstringStepConfig instance.
    """
    if not isinstance(cfg, DocstringStepConfig):
        message = "cfg must be DocstringStepConfig"
        raise TypeError(message)

    # Cast code_profile to proper type
    profile: ScanProfile | None = None
    if isinstance(code_profile, ScanProfile):
        profile = code_profile

    module_map = load_module_map(
        gateway,
        cfg.repo,
        cfg.commit,
        language="python",
        logger=log,
    )

    modules = list(
        iter_modules(
            module_map,
            cfg.repo_root,
            logger=log,
            scan_profile=profile,
        )
    )

    ingest_docstrings(
        gateway,
        modules,
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        tracker=tracker,
    )


def _parse_docstring(raw: str | None) -> ParsedDocstring:
    """
    Parse a raw docstring into structured components.

    Parameters
    ----------
    raw
        Raw docstring text.

    Returns
    -------
    ParsedDocstring
        Structured docstring with extracted components.
    """
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
