"""CST extraction step with port injection.

This module provides a pure domain logic implementation for extracting
LibCST concrete syntax trees, using ports for all I/O operations.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import libcst as cst
from libcst import metadata

from codeintel.ingestion.cst_utils import CstCaptureConfig, CstCaptureVisitor
from codeintel.ingestion.steps.base import StepResult

if TYPE_CHECKING:
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort

log = logging.getLogger(__name__)

# Define the async function definition type for compatibility
ASYNC_FUNC_DEF = getattr(cst, "AsyncFunctionDef", cst.FunctionDef)

# Row type for CST nodes
CstRow = tuple[str, str, str, dict[str, list[int]], str, tuple[str, ...], tuple[str, ...]]

# Default CST capture configuration
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


class CstExtractStep:
    """CST extraction step with port injection.

    This step extracts LibCST concrete syntax trees from modules,
    using ports for all I/O operations.

    Parameters
    ----------
    storage
        Storage port for persisting data.
    discovery
        Discovery port for reading module source.
    """

    def __init__(
        self,
        storage: IngestStoragePort,
        discovery: ModuleDiscoveryPort,
    ) -> None:
        """Initialize the step.

        Parameters
        ----------
        storage
            Storage port for persisting data.
        discovery
            Discovery port for reading module source.
        """
        self._storage = storage
        self._discovery = discovery

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
    ) -> StepResult:
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
        StepResult
            Execution result with row counts.
        """
        all_rows: list[list[object]] = []
        errors: list[str] = []

        for module in modules:
            if not module.rel_path.endswith(".py"):
                continue

            source = self._discovery.read_module_source(module)
            if source is None:
                continue

            result = _extract_module_cst(module, source)
            if result.error is not None:
                errors.append(f"Failed to parse {module.rel_path}: {result.error}")
                log.warning("Failed to parse %s: %s", module.rel_path, result.error)
                continue

            # Normalize rows for storage
            for row in result.rows:
                rel_path, node_id, kind, span, snippet, parents, qnames = row
                all_rows.append(
                    [
                        rel_path,
                        node_id,
                        kind,
                        span,
                        snippet,
                        list(parents),
                        list(qnames),
                    ]
                )

        # Persist rows
        table_counts: dict[str, int] = {}
        total_rows = 0

        if all_rows:
            scope = f"{repo}@{commit}"
            result = self._storage.write_batch("core.cst_nodes", all_rows, scope=scope)
            table_counts["core.cst_nodes"] = result.rows_written
            total_rows = result.rows_written

        log.info(
            "CST extraction: repo=%s commit=%s rows=%d",
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
    "CST_CAPTURE_CONFIG",
    "CstExtractStep",
    "CstVisitor",
    "ModuleCstResult",
]
