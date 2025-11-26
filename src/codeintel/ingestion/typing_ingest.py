"""Ingest typedness ratios and static diagnostics for Python files."""

from __future__ import annotations

import ast
import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from codeintel.config import TypingIngestStepConfig
from codeintel.config.models import ToolsConfig
from codeintel.ingestion.common import run_batch
from codeintel.ingestion.source_scanner import (
    ScanProfile,
    SourceScanner,
    default_code_profile,
    profile_from_env,
)
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.models.rows import (
    StaticDiagnosticRow,
    TypednessRow,
    static_diagnostic_to_tuple,
    typedness_row_to_tuple,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.utils.paths import repo_relpath

log = logging.getLogger(__name__)


@dataclass
class AnnotationInfo:
    """
    Ratio and count statistics summarizing annotations in a file.

    Parameters
    ----------
    params_ratio : float
        Fraction of parameters (excluding self/cls) with annotations.
    returns_ratio : float
        Fraction of functions with return annotations.
    untyped_defs : int
        Count of function definitions missing full annotations.
    """

    params_ratio: float
    returns_ratio: float
    untyped_defs: int


def _iter_python_files(profile: ScanProfile) -> Iterable[Path]:
    scanner = SourceScanner(profile)
    yield from scanner.iter_files()


def _compute_annotation_info_for_file(path: Path) -> AnnotationInfo | None:
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
            # Python 3.8+ supports posonlyargs; older code will ignore.
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


def ingest_typing_signals(
    gateway: StorageGateway,
    cfg: TypingIngestStepConfig,
    *,
    code_profile: ScanProfile | None = None,
    tools: ToolsConfig | None = None,
    tool_service: ToolService | None = None,
) -> None:
    """
    Populate per-file typedness and static diagnostics.

      - analytics.typedness
      - analytics.static_diagnostics

    Notes
    -----
      * Pyrefly drives static error counts; annotation_ratio is computed from Python AST
        (params & returns).
    """
    repo_root = cfg.repo_root
    profile = code_profile or profile_from_env(default_code_profile(repo_root))
    active_tools = tools or ToolsConfig.model_validate({})
    active_service = tool_service
    if active_service is None:
        shared_runner = ToolRunner(
            tools_config=active_tools, cache_dir=repo_root / "build" / ".tool_cache"
        )
        active_service = ToolService(shared_runner, active_tools)

    # Compute annotation info for each Python file
    annotation_info: dict[str, AnnotationInfo] = {}
    for path in _iter_python_files(profile):
        rel_path = repo_relpath(repo_root, path)
        info = _compute_annotation_info_for_file(path)
        if info is not None:
            annotation_info[rel_path] = info

    error_maps = asyncio.run(_collect_error_maps(repo_root, active_service))
    path_set = (
        set(annotation_info)
        | set(error_maps["pyrefly"])
        | set(error_maps["pyright"])
        | set(error_maps["ruff"])
    )

    typedness_rows: list[TypednessRow] = []
    diag_rows: list[StaticDiagnosticRow] = []
    default_info = AnnotationInfo(params_ratio=0.0, returns_ratio=0.0, untyped_defs=0)
    for rel_path in sorted(path_set):
        info = annotation_info.get(rel_path, default_info)
        pf_errors = error_maps["pyrefly"].get(rel_path, 0)
        py_errors = error_maps["pyright"].get(rel_path, 0)
        total_errors = pf_errors + py_errors

        typedness_rows.append(
            TypednessRow(
                repo=cfg.repo,
                commit=cfg.commit,
                path=rel_path,
                type_error_count=total_errors,
                annotation_ratio={
                    "params": info.params_ratio,
                    "returns": info.returns_ratio,
                },
                untyped_defs=info.untyped_defs,
                overlay_needed=bool(total_errors > 0 or info.untyped_defs > 0),
            )
        )

        diag_rows.append(
            StaticDiagnosticRow(
                repo=cfg.repo,
                commit=cfg.commit,
                rel_path=rel_path,
                pyrefly_errors=pf_errors,
                pyright_errors=py_errors,
                ruff_errors=error_maps["ruff"].get(rel_path, 0),
                total_errors=total_errors,
                has_errors=total_errors > 0,
            )
        )

    run_batch(
        gateway,
        "analytics.typedness",
        [typedness_row_to_tuple(row) for row in typedness_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
    run_batch(
        gateway,
        "analytics.static_diagnostics",
        [static_diagnostic_to_tuple(row) for row in diag_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info(
        "Typedness & static diagnostics ingested for %d files in %s@%s",
        len(path_set),
        cfg.repo,
        cfg.commit,
    )
