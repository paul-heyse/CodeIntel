"""Ingest typedness ratios and static diagnostics for Python files."""

from __future__ import annotations

import ast
import asyncio
import fnmatch
import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config import TypingIngestStepConfig
from codeintel.config.datasets import (
    StaticDiagnosticRow,
    TypednessRow,
    static_diagnostic_to_tuple,
    typedness_row_to_tuple,
)
from codeintel.config.models import ToolsConfig
from codeintel.ingestion.common import ModuleRecord, iter_modules, run_batch
from codeintel.ingestion.pipeline import (
    IngestPipeline,
    PipelineConfig,
    PipelineResult,
    execute_pipeline,
)
from codeintel.ingestion.source_scanner import (
    ScanProfile,
    default_code_profile,
    profile_from_env,
)
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.storage.module_index import load_module_map

if TYPE_CHECKING:
    from codeintel.ingestion.change_tracker import ChangeTracker
    from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def _discover_python_modules(repo_root: Path, profile: ScanProfile) -> list[ModuleRecord]:
    """
    Discover Python modules on the filesystem as a fallback.

    Used when no modules are registered in core.modules.

    Parameters
    ----------
    repo_root
        Repository root path.
    profile
        Scan profile for filtering.

    Returns
    -------
    list[ModuleRecord]
        Discovered module records.
    """
    discovered: list[ModuleRecord] = []
    patterns = tuple(profile.include_globs) if profile is not None else ("**/*.py",)
    ignore_set = set(profile.ignore_dirs) if profile is not None else set()

    for root, dirs, files in repo_root.walk():
        # Skip ignored directories
        dirs[:] = [d for d in dirs if d not in ignore_set]

        rel_root = root.relative_to(repo_root)
        for file in files:
            if not file.endswith(".py"):
                continue
            rel_path = str(rel_root / file)
            if any(fnmatch.fnmatch(rel_path, pat) for pat in patterns):
                module_name = rel_path.replace("/", ".").removesuffix(".py")
                discovered.append(
                    ModuleRecord(
                        rel_path=rel_path,
                        module_name=module_name,
                        file_path=root / file,
                        index=len(discovered),
                        total=-1,  # Will be fixed up later
                    )
                )

    # Fix up totals on frozen dataclass (requires __setattr__ bypass)
    for mod in discovered:
        object.__setattr__(mod, "total", len(discovered))  # noqa: PLC2801

    return discovered


@dataclass
class AnnotationInfo:
    """
    Ratio and count statistics summarizing annotations in a file.

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


def _compute_annotation_info_for_file(path: Path) -> AnnotationInfo | None:
    """
    Parse a file and compute annotation statistics.

    Parameters
    ----------
    path
        Path to the Python source file.

    Returns
    -------
    AnnotationInfo | None
        Annotation statistics or None on parse failure.
    """
    try:
        source = path.read_text(encoding="utf8")
    except (OSError, UnicodeDecodeError):
        return None

    try:
        tree = ast.parse(source, filename=str(path))
    except SyntaxError:
        return None

    total_params = 0
    annotated_params = 0
    func_count = 0
    return_annotated = 0
    untyped_defs = 0

    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_count += 1

            params = []
            posonly = getattr(node.args, "posonlyargs", [])
            params.extend(posonly)
            params.extend(node.args.args)
            params.extend(node.args.kwonlyargs)

            for arg in params:
                if arg.arg in {"self", "cls"}:
                    continue
                total_params += 1
                if arg.annotation is not None:
                    annotated_params += 1

            has_return = node.returns is not None
            if has_return:
                return_annotated += 1

            fully_typed = (
                all(arg.annotation is not None for arg in params if arg.arg not in {"self", "cls"})
                and has_return
            )
            if not fully_typed:
                untyped_defs += 1

    params_ratio = annotated_params / total_params if total_params else 1.0
    returns_ratio = return_annotated / func_count if func_count else 1.0

    return AnnotationInfo(
        params_ratio=params_ratio,
        returns_ratio=returns_ratio,
        untyped_defs=untyped_defs,
    )


async def _collect_error_maps(
    repo_root: Path,
    service: ToolService,
) -> dict[str, dict[str, int]]:
    """
    Run all diagnostic tools and collect error counts.

    Parameters
    ----------
    repo_root
        Repository root for tool invocation.
    service
        Tool service for running diagnostics.

    Returns
    -------
    dict[str, dict[str, int]]
        Error counts keyed by tool name and file path.
    """
    pyrefly_map, pyright_map, ruff_map = await asyncio.gather(
        service.run_pyrefly(repo_root),
        service.run_pyright(repo_root),
        service.run_ruff(repo_root),
    )
    return {
        "pyrefly": dict(pyrefly_map),
        "pyright": dict(pyright_map),
        "ruff": dict(ruff_map),
    }


@dataclass
class TypingIngestResult:
    """Bundle typedness and diagnostic rows for a module."""

    typedness: TypednessRow | None
    diagnostics: StaticDiagnosticRow | None


def _delete_existing_typing_rows(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    rel_paths: list[str],
) -> None:
    """
    Delete existing typedness and diagnostic rows for selected paths.

    When rel_paths is empty, delete all rows for repo@commit.
    """
    if rel_paths:
        gateway.con.execute(
            """
            DELETE FROM analytics.typedness
            WHERE repo = ? AND commit = ? AND path IN (SELECT * FROM UNNEST(?))
            """,
            [repo, commit, rel_paths],
        )
        gateway.con.execute(
            """
            DELETE FROM analytics.static_diagnostics
            WHERE repo = ? AND commit = ? AND rel_path IN (SELECT * FROM UNNEST(?))
            """,
            [repo, commit, rel_paths],
        )
        return

    run_batch(
        gateway,
        "analytics.typedness",
        [],
        delete_params=[repo, commit],
    )
    run_batch(
        gateway,
        "analytics.static_diagnostics",
        [],
        delete_params=[repo, commit],
    )


class TypingPipeline:
    """Pipeline implementation for typedness and diagnostics extraction."""

    def __init__(
        self,
        *,
        repo: str,
        commit: str,
        repo_root: Path,
        error_maps: dict[str, dict[str, int]],
    ) -> None:
        self._repo = repo
        self._commit = commit
        self._repo_root = repo_root
        self._error_maps = error_maps

    @property
    def dataset_name(self) -> str:
        """Return the dataset name for this pipeline."""
        return "analytics.typedness"

    @staticmethod
    def module_filter(module: ModuleRecord) -> bool:
        """
        Determine whether a module should be processed.

        Parameters
        ----------
        module
            Module metadata describing the candidate file.

        Returns
        -------
        bool
            True when the module is a Python source file.
        """
        return module.rel_path.endswith(".py")

    def process_module(self, module: ModuleRecord) -> Iterable[TypingIngestResult]:
        """
        Compute typedness and diagnostic rows for a single module.

        Parameters
        ----------
        module
            Module metadata describing the file to analyze.

        Returns
        -------
        Iterable[TypingIngestResult]
            Rows summarizing annotations and static diagnostics.
        """
        path = module.file_path
        info = _compute_annotation_info_for_file(path) or AnnotationInfo(
            params_ratio=0.0,
            returns_ratio=0.0,
            untyped_defs=0,
        )

        pf_errors = self._error_maps["pyrefly"].get(module.rel_path, 0)
        py_errors = self._error_maps["pyright"].get(module.rel_path, 0)
        ruff_errors = self._error_maps["ruff"].get(module.rel_path, 0)
        total_errors = pf_errors + py_errors

        typedness = TypednessRow(
            repo=self._repo,
            commit=self._commit,
            path=module.rel_path,
            type_error_count=total_errors,
            annotation_ratio={
                "params": info.params_ratio,
                "returns": info.returns_ratio,
            },
            untyped_defs=info.untyped_defs,
            overlay_needed=bool(total_errors > 0 or info.untyped_defs > 0),
        )

        diagnostics = StaticDiagnosticRow(
            repo=self._repo,
            commit=self._commit,
            rel_path=module.rel_path,
            pyrefly_errors=pf_errors,
            pyright_errors=py_errors,
            ruff_errors=ruff_errors,
            total_errors=total_errors,
            has_errors=total_errors > 0,
        )

        return [TypingIngestResult(typedness=typedness, diagnostics=diagnostics)]

    def persist_rows(self, gateway: StorageGateway, rows: Sequence[TypingIngestResult]) -> int:
        """
        Insert typedness and diagnostic rows.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rows
            Extraction results to persist.

        Returns
        -------
        int
            Number of rows persisted.
        """
        if not rows:
            return 0

        rel_paths: set[str] = set()
        typedness_rows: list[TypednessRow] = []
        diag_rows: list[StaticDiagnosticRow] = []

        for result in rows:
            if result.typedness is not None:
                typedness_rows.append(result.typedness)
                rel_paths.add(result.typedness["path"])
            if result.diagnostics is not None:
                diag_rows.append(result.diagnostics)
                rel_paths.add(result.diagnostics["rel_path"])

        if rel_paths:
            _delete_existing_typing_rows(
                gateway,
                repo=self._repo,
                commit=self._commit,
                rel_paths=sorted(rel_paths),
            )

        total = 0
        if typedness_rows:
            run_batch(
                gateway,
                "analytics.typedness",
                [typedness_row_to_tuple(row) for row in typedness_rows],
                delete_params=None,
                scope=f"{self._repo}@{self._commit}",
            )
            total += len(typedness_rows)

        if diag_rows:
            run_batch(
                gateway,
                "analytics.static_diagnostics",
                [static_diagnostic_to_tuple(row) for row in diag_rows],
                delete_params=None,
                scope=f"{self._repo}@{self._commit}",
            )

        return total

    def delete_rows(self, gateway: StorageGateway, rel_paths: Sequence[str]) -> None:
        """
        Delete rows for modules scheduled for removal.

        Parameters
        ----------
        gateway
            Storage gateway for database access.
        rel_paths
            Relative paths to delete.
        """
        _delete_existing_typing_rows(
            gateway,
            repo=self._repo,
            commit=self._commit,
            rel_paths=list(rel_paths),
        )


# Type assertion that TypingPipeline implements IngestPipeline
_: type[IngestPipeline[TypingIngestResult]] = TypingPipeline


def ingest_typing_signals(  # noqa: PLR0913
    gateway: StorageGateway,
    modules_or_cfg: Sequence[ModuleRecord] | object = (),
    *,
    cfg: object = None,
    repo: str | None = None,
    commit: str | None = None,
    repo_root: Path | None = None,
    code_profile: ScanProfile | None = None,
    tool_service: ToolService | None = None,
    tracker: ChangeTracker | None = None,
    modules: Sequence[ModuleRecord] | None = None,
) -> PipelineResult:
    """
    Populate per-file typedness and static diagnostics.

    Supports both new and legacy calling conventions for backward compatibility.

    Populates:
    - analytics.typedness
    - analytics.static_diagnostics

    Parameters
    ----------
    gateway
        Storage gateway for database access.
    modules_or_cfg
        Either modules sequence (new API) or first positional arg.
    cfg
        Legacy TypingIngestStepConfig parameter.
    repo
        Repository identifier (new API).
    commit
        Commit identifier (new API).
    repo_root
        Repository root path (new API).
    code_profile
        Optional scan profile for filtering.
    tool_service
        Optional tool service for diagnostics.
    tracker
        Optional change tracker for incremental mode.
    modules
        Alternative modules parameter (new API).

    Returns
    -------
    PipelineResult
        Execution result with counts and timing.

    Raises
    ------
    ValueError
        When repo, commit, or repo_root are not provided via either the
        cfg parameter or explicit keyword arguments.

    Notes
    -----
    Pyrefly drives static error counts; annotation_ratio is computed from
    Python AST (params and returns).
    """
    # Handle legacy API: ingest_typing_signals(gateway, cfg=cfg, ...)
    actual_cfg: TypingIngestStepConfig | None = None
    if isinstance(cfg, TypingIngestStepConfig):
        actual_cfg = cfg
    elif isinstance(modules_or_cfg, TypingIngestStepConfig):
        actual_cfg = modules_or_cfg

    if actual_cfg is not None:
        profile = code_profile or profile_from_env(default_code_profile(actual_cfg.repo_root))
        module_map = load_module_map(
            gateway,
            actual_cfg.repo,
            actual_cfg.commit,
            language="python",
            logger=log,
        )

        if module_map:
            actual_modules = list(
                iter_modules(
                    module_map,
                    actual_cfg.repo_root,
                    logger=log,
                    scan_profile=profile,
                )
            )
        else:
            # Fallback: scan filesystem for Python files if no modules in database
            actual_modules = _discover_python_modules(actual_cfg.repo_root, profile)

        actual_repo = actual_cfg.repo
        actual_commit = actual_cfg.commit
        actual_repo_root = actual_cfg.repo_root
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

    service = tool_service
    if service is None:
        tools_config = ToolsConfig.model_validate({})
        shared_runner = ToolRunner(
            tools_config=tools_config, cache_dir=actual_repo_root / "build" / ".tool_cache"
        )
        service = ToolService(shared_runner, tools_config)

    error_maps = asyncio.run(_collect_error_maps(actual_repo_root, service))

    pipeline = TypingPipeline(
        repo=actual_repo,
        commit=actual_commit,
        repo_root=actual_repo_root,
        error_maps=error_maps,
    )

    return execute_pipeline(
        pipeline,
        gateway,
        actual_modules,
        tracker=tracker,
        config=PipelineConfig(),
    )


# Backward compatibility: keep old function signature
def ingest_typing_signals_legacy(
    gateway: StorageGateway,
    cfg: TypingIngestStepConfig,
    *,
    code_profile: ScanProfile | None = None,
    tool_service: ToolService | None = None,
    tracker: ChangeTracker | None = None,
) -> None:
    """
    Legacy entry point for typing ingestion.

    Deprecated: Use ingest_typing_signals() with explicit parameters instead.
    """
    profile = code_profile or profile_from_env(default_code_profile(cfg.repo_root))

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

    ingest_typing_signals(
        gateway,
        modules,
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        code_profile=profile,
        tool_service=tool_service,
        tracker=tracker,
    )
