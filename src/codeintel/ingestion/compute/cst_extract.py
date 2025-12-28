"""CST extraction step with port injection.

This module provides a pure domain logic implementation for extracting
LibCST concrete syntax trees, using ports for all I/O operations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import libcst as cst
from libcst import metadata

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.core.columnar.rows import ColumnarRows, columnar_buffer_for_table_key
from codeintel.ingestion.compute.base import BaseExtractStep
from codeintel.ingestion.infrastructure.cst_utils import (
    CstCaptureConfig,
    CstCaptureVisitor,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.ingestion.ports.discovery import ModuleRecord

log = logging.getLogger(__name__)
CST_NODES_TABLE_KEY = "core.cst_nodes"


ASYNC_FUNC_DEF = getattr(cst, "AsyncFunctionDef", cst.FunctionDef)


CstRow = tuple[str, str, str, dict[str, list[int]], str, tuple[str, ...], tuple[str, ...]]


CST_CAPTURE_CONFIG = CstCaptureConfig(
    kinds=(
        cst.Module,
        cst.FunctionDef,
        ASYNC_FUNC_DEF,
        cst.ClassDef,
        cst.Assign,
        cst.AnnAssign,
        cst.AugAssign,
        cst.Import,
        cst.ImportFrom,
        cst.Call,
        cst.Return,
        cst.Raise,
        cst.Yield,
        cst.If,
        cst.Else,
        cst.For,
        cst.While,
        cst.With,
        cst.Try,
        cst.ExceptHandler,
        cst.Match,
    ),
    snippet_limit=200,
)


@dataclass(frozen=True)
class ModuleCstResult:
    """Result from processing a single module's CST.

    Attributes
    ----------
    rel_path
        Relative path to the module.
    rows
        CST rows extracted.
    error
        Error message if parsing failed.
    """

    rel_path: str
    rows: list[CstRow]
    error: str | None = None


@dataclass(frozen=True)
class CstExtractResult:
    """Result bundle for CST extraction."""

    result: ExecutionResult
    rows: ColumnarRows = field(default_factory=dict)
    row_count: int = 0


class CstVisitor(CstCaptureVisitor):
    """Collect CST rows using shared capture helpers."""

    def __init__(self, rel_path: str, module_name: str, source: str) -> None:
        """Initialize visitor.

        Parameters
        ----------
        rel_path
            Relative path to the file.
        module_name
            Python module name.
        source
            Source code text.
        """
        super().__init__(rel_path, module_name, source, config=CST_CAPTURE_CONFIG)


def _extract_module_cst(
    module: ModuleRecord,
    source: str,
) -> ModuleCstResult:
    """Extract CST from module source.

    Parameters
    ----------
    module
        Module record with metadata.
    source
        Module source code.

    Returns
    -------
    ModuleCstResult
        Extraction result with rows or error.
    """
    try:
        wrapper = metadata.MetadataWrapper(
            cst.parse_module(source),
            unsafe_skip_copy=True,
        )
        visitor = CstVisitor(
            rel_path=module.rel_path,
            module_name=module.module_name,
            source=source,
        )
        wrapper.visit(visitor)
        return ModuleCstResult(rel_path=module.rel_path, rows=visitor.rows, error=None)
    except (cst.ParserSyntaxError, ValueError, TypeError, RuntimeError) as exc:
        return ModuleCstResult(rel_path=module.rel_path, rows=[], error=str(exc))


class CstExtractStep(BaseExtractStep):
    """CST extraction step with port injection.

    This step extracts LibCST concrete syntax trees from modules,
    using ports for all I/O operations.

    Parameters
    ----------
    discovery
        Discovery port for reading module source.
    """

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
    ) -> CstExtractResult:
        """Execute CST extraction on the provided modules.

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
        CstExtractResult
            Result bundle with row tuples and execution status.
        """
        try:
            buffer = columnar_buffer_for_table_key(CST_NODES_TABLE_KEY)
        except (KeyError, RuntimeError) as exc:
            return CstExtractResult(result=ExecutionResult.failed(str(exc)))
        warnings: list[str] = []

        for module, source in self._iter_python_sources(modules):
            result = _extract_module_cst(module, source)
            if result.error is not None:
                warnings.append(f"Failed to parse {module.rel_path}: {result.error}")
                log.warning("Failed to parse %s: %s", module.rel_path, result.error)
                continue

            for row in result.rows:
                rel_path, node_id, kind, span, snippet, parents, qnames = row
                buffer.append(
                    {
                        "path": rel_path,
                        "node_id": node_id,
                        "kind": kind,
                        "span": span,
                        "text_preview": snippet,
                        "parents": list(parents),
                        "qnames": list(qnames),
                    }
                )

        log.info(
            "CST extraction: repo=%s commit=%s rows=%d",
            repo,
            commit,
            buffer.row_count,
        )

        return CstExtractResult(
            result=ExecutionResult.ok(warnings=tuple(warnings)),
            rows=buffer.data,
            row_count=buffer.row_count,
        )


__all__ = [
    "CST_CAPTURE_CONFIG",
    "CstExtractResult",
    "CstExtractStep",
    "CstVisitor",
    "ModuleCstResult",
]
