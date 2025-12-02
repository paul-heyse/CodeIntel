"""Typing and diagnostics ingestion step with port injection.

This module provides a pure domain logic implementation for computing
typedness ratios and collecting static diagnostics, using ports for
all I/O operations.
"""

from __future__ import annotations

import ast
import asyncio
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.datasets import (
    StaticDiagnosticRow,
    TypednessRow,
    static_diagnostic_to_tuple,
    typedness_row_to_tuple,
)
from codeintel.ingestion.ports.tools import ToolStatus
from codeintel.ingestion.steps.base import StepResult

if TYPE_CHECKING:
    from codeintel.ingestion.ports.discovery import ModuleDiscoveryPort, ModuleRecord
    from codeintel.ingestion.ports.storage import IngestStoragePort
    from codeintel.ingestion.ports.tools import IngestToolPort

log = logging.getLogger(__name__)


@dataclass
class AnnotationInfo:
    """Ratio and count statistics summarizing annotations in a file.

    Attributes
    ----------
    params_ratio
        Fraction of parameters (excluding self/cls) with annotations.
    returns_ratio
        Fraction of functions with return annotations.
    untyped_defs
        Count of function definitions missing full annotations.
    """

    params_ratio: float
    returns_ratio: float
    untyped_defs: int


@dataclass
class DiagnosticCounts:
    """Error counts from diagnostic tools.

    Attributes
    ----------
    pyright
        Errors from pyright.
    pyrefly
        Errors from pyrefly.
    ruff
        Errors from ruff.
    """

    pyright: dict[str, int]
    pyrefly: dict[str, int]
    ruff: dict[str, int]


def _compute_annotation_info(source: str) -> AnnotationInfo | None:
    """Compute annotation statistics from source code.

    Parameters
    ----------
    source
        Python source code.

    Returns
    -------
    AnnotationInfo | None
        Annotation statistics or None on parse failure.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None

    total_params, annotated_params, func_count = 0, 0, 0
    return_annotated, untyped_defs = 0, 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_count += 1
            params = _collect_function_params(node)

            for arg in params:
                if arg.arg not in {"self", "cls"}:
                    total_params += 1
                    if arg.annotation is not None:
                        annotated_params += 1

            has_return = node.returns is not None
            if has_return:
                return_annotated += 1

            if not _is_fully_typed(params, has_return=has_return):
                untyped_defs += 1

    params_ratio = annotated_params / total_params if total_params else 1.0
    returns_ratio = return_annotated / func_count if func_count else 1.0

    return AnnotationInfo(
        params_ratio=params_ratio,
        returns_ratio=returns_ratio,
        untyped_defs=untyped_defs,
    )


def _collect_function_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.arg]:
    """Collect all parameters from a function definition.

    Parameters
    ----------
    node
        Function definition node.

    Returns
    -------
    list[ast.arg]
        All parameter arguments.
    """
    params: list[ast.arg] = []
    posonly = getattr(node.args, "posonlyargs", [])
    params.extend(posonly)
    params.extend(node.args.args)
    params.extend(node.args.kwonlyargs)
    return params


def _is_fully_typed(params: list[ast.arg], *, has_return: bool) -> bool:
    """Check if a function is fully typed.

    Parameters
    ----------
    params
        Function parameters.
    has_return
        Whether function has return annotation.

    Returns
    -------
    bool
        True if fully typed.
    """
    return all(
        arg.annotation is not None for arg in params if arg.arg not in {"self", "cls"}
    ) and has_return


async def _collect_diagnostic_counts(
    repo_root: Path,
    tools: IngestToolPort,
) -> DiagnosticCounts:
    """Collect error counts from all diagnostic tools.

    Parameters
    ----------
    repo_root
        Repository root directory.
    tools
        Tool port for running diagnostics.

    Returns
    -------
    DiagnosticCounts
        Error counts from each tool.
    """
    pyright_result = await tools.run_pyright(repo_root)
    pyrefly_result = await tools.run_pyrefly(repo_root)
    ruff_result = await tools.run_ruff(repo_root)

    return DiagnosticCounts(
        pyright=pyright_result.errors_by_path() if pyright_result.status == ToolStatus.OK else {},
        pyrefly=pyrefly_result.errors_by_path() if pyrefly_result.status == ToolStatus.OK else {},
        ruff=ruff_result.errors_by_path() if ruff_result.status == ToolStatus.OK else {},
    )


class TypingIngestStep:
    """Typing and diagnostics ingestion step with port injection.

    This step computes typedness ratios and collects static diagnostics,
    using ports for all I/O operations.

    Parameters
    ----------
    storage
        Storage port for persisting data.
    discovery
        Discovery port for reading module source.
    tools
        Optional tool port for running diagnostics.
    """

    def __init__(
        self,
        storage: IngestStoragePort,
        discovery: ModuleDiscoveryPort,
        tools: IngestToolPort | None = None,
    ) -> None:
        """Initialize the step.

        Parameters
        ----------
        storage
            Storage port for persisting data.
        discovery
            Discovery port for reading module source.
        tools
            Optional tool port for running diagnostics.
        """
        self._storage = storage
        self._discovery = discovery
        self._tools = tools

    async def execute_async(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
        repo_root: str,
        run_diagnostics: bool = True,
    ) -> StepResult:
        """Execute typing analysis on the provided modules.

        Parameters
        ----------
        modules
            Modules to process.
        repo
            Repository identifier.
        commit
            Commit identifier.
        repo_root
            Repository root path.
        run_diagnostics
            Whether to run external diagnostic tools.

        Returns
        -------
        StepResult
            Execution result with row counts.
        """
        created_at = datetime.now(UTC)

        # Collect diagnostic counts
        diag_counts = DiagnosticCounts(pyright={}, pyrefly={}, ruff={})
        if run_diagnostics and self._tools is not None:
            diag_counts = await _collect_diagnostic_counts(Path(repo_root), self._tools)

        # Process modules
        typedness_rows, diagnostic_rows = self._process_modules(
            modules, repo, commit, created_at, diag_counts
        )

        # Persist rows
        return self._persist_rows(typedness_rows, diagnostic_rows, repo, commit)

    def _process_modules(
        self,
        modules: Sequence[ModuleRecord],
        repo: str,
        commit: str,
        created_at: datetime,
        diag_counts: DiagnosticCounts,
    ) -> tuple[list[list[object]], list[list[object]]]:
        """Process modules and build rows.

        Parameters
        ----------
        modules
            Modules to process.
        repo
            Repository identifier.
        commit
            Commit identifier.
        created_at
            Timestamp.
        diag_counts
            Diagnostic counts from tools.

        Returns
        -------
        tuple[list[list[object]], list[list[object]]]
            Typedness rows and diagnostic rows.
        """
        typedness_rows: list[list[object]] = []
        diagnostic_rows: list[list[object]] = []

        for module in modules:
            if not module.rel_path.endswith(".py"):
                continue

            source = self._discovery.read_module_source(module)
            if source is None:
                continue

            info = _compute_annotation_info(source)
            if info is None:
                continue

            # Build typedness row
            row = TypednessRow(
                repo=repo,
                commit=commit,
                rel_path=module.rel_path,
                params_ratio=info.params_ratio,
                returns_ratio=info.returns_ratio,
                untyped_defs=info.untyped_defs,
                pyright_errors=diag_counts.pyright.get(module.rel_path, 0),
                pyrefly_errors=diag_counts.pyrefly.get(module.rel_path, 0),
                ruff_errors=diag_counts.ruff.get(module.rel_path, 0),
                created_at=created_at,
            )
            typedness_rows.append(list(typedness_row_to_tuple(row)))

            # Build diagnostic rows
            diagnostic_rows.extend(
                self._build_diagnostic_rows(module, repo, commit, created_at, diag_counts)
            )

        return typedness_rows, diagnostic_rows

    @staticmethod
    def _build_diagnostic_rows(
        module: ModuleRecord,
        repo: str,
        commit: str,
        created_at: datetime,
        diag_counts: DiagnosticCounts,
    ) -> list[list[object]]:
        """Build diagnostic rows for a module.

        Parameters
        ----------
        module
            Module being processed.
        repo
            Repository identifier.
        commit
            Commit identifier.
        created_at
            Timestamp.
        diag_counts
            Diagnostic counts.

        Returns
        -------
        list[list[object]]
            Diagnostic rows.
        """
        rows: list[list[object]] = []
        for tool, counts in [
            ("pyright", diag_counts.pyright),
            ("pyrefly", diag_counts.pyrefly),
            ("ruff", diag_counts.ruff),
        ]:
            count = counts.get(module.rel_path, 0)
            if count > 0:
                diag_row = StaticDiagnosticRow(
                    repo=repo,
                    commit=commit,
                    rel_path=module.rel_path,
                    tool=tool,
                    error_count=count,
                    created_at=created_at,
                )
                rows.append(list(static_diagnostic_to_tuple(diag_row)))
        return rows

    def _persist_rows(
        self,
        typedness_rows: list[list[object]],
        diagnostic_rows: list[list[object]],
        repo: str,
        commit: str,
    ) -> StepResult:
        """Persist rows to storage.

        Parameters
        ----------
        typedness_rows
            Typedness rows to persist.
        diagnostic_rows
            Diagnostic rows to persist.
        repo
            Repository identifier.
        commit
            Commit identifier.

        Returns
        -------
        StepResult
            Result with row counts.
        """
        table_counts: dict[str, int] = {}
        total_rows = 0

        if typedness_rows:
            scope = f"{repo}@{commit}"
            result = self._storage.write_batch("core.typedness", typedness_rows, scope=scope)
            table_counts["core.typedness"] = result.rows_written
            total_rows += result.rows_written

        if diagnostic_rows:
            result = self._storage.write_batch("core.static_diagnostics", diagnostic_rows)
            table_counts["core.static_diagnostics"] = result.rows_written
            total_rows += result.rows_written

        log.info(
            "Typing ingest: repo=%s commit=%s typedness=%d diagnostics=%d",
            repo,
            commit,
            len(typedness_rows),
            len(diagnostic_rows),
        )

        return StepResult(rows_written=total_rows, table_counts=table_counts)

    def execute(
        self,
        modules: Sequence[ModuleRecord],
        *,
        repo: str,
        commit: str,
        repo_root: str,
    ) -> StepResult:
        """Execute typing analysis synchronously (without diagnostics).

        Parameters
        ----------
        modules
            Modules to process.
        repo
            Repository identifier.
        commit
            Commit identifier.
        repo_root
            Repository root path.

        Returns
        -------
        StepResult
            Execution result with row counts.
        """
        return asyncio.get_event_loop().run_until_complete(
            self.execute_async(modules, repo=repo, commit=commit, repo_root=repo_root, run_diagnostics=False)
        )


__all__ = ["AnnotationInfo", "TypingIngestStep"]
