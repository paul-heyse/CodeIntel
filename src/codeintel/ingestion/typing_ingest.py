"""Ingest typedness ratios and static diagnostics for Python files."""

from __future__ import annotations

import ast
import asyncio
import json
import logging
import shutil
import tempfile
from asyncio.subprocess import PIPE
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from codeintel.config.models import ToolsConfig, TypingIngestConfig
from codeintel.ingestion.common import run_batch
from codeintel.ingestion.source_scanner import ScanConfig, SourceScanner
from codeintel.ingestion.tool_runner import ToolResult, ToolRunner
from codeintel.models.rows import (
    StaticDiagnosticRow,
    TypednessRow,
    static_diagnostic_to_tuple,
    typedness_row_to_tuple,
)
from codeintel.storage.gateway import StorageGateway
from codeintel.types import PyreflyError
from codeintel.utils.paths import repo_relpath

log = logging.getLogger(__name__)
MISSING_BINARY_EXIT_CODE = 127
PYRIGHT_BIN = "pyright"
RUFF_BIN = "ruff"

IGNORE_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    ".tox",
    "__pycache__",
    "build",
    "dist",
    ".mypy_cache",
    ".pytest_cache",
}


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


def _iter_python_files(repo_root: Path, scan_cfg: ScanConfig | None) -> Iterable[Path]:
    scanner = SourceScanner(
        scan_cfg or ScanConfig(repo_root=repo_root, ignore_dirs=tuple(sorted(IGNORE_DIRS)))
    )
    for record in scanner.iter_files(log):
        yield record.path


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


def _run_command(args: list[str], cwd: Path | None = None) -> tuple[int, str, str]:
    async def _exec() -> tuple[int, str, str]:
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=str(cwd) if cwd is not None else None,
            stdout=PIPE,
            stderr=PIPE,
        )
        stdout_b, stderr_b = await proc.communicate()
        return (
            proc.returncode if proc.returncode is not None else 1,
            stdout_b.decode(),
            stderr_b.decode(),
        )

    try:
        return asyncio.run(_exec())
    except FileNotFoundError as exc:
        return MISSING_BINARY_EXIT_CODE, "", str(exc)


def _resolve_pyrefly_bin() -> str:
    return shutil.which("pyrefly") or "pyrefly"


def _run_pyrefly(repo_root: Path, runner: ToolRunner) -> dict[str, int]:
    """
    Run pyrefly and aggregate error counts per file.

    Parameters
    ----------
    repo_root : Path
        Root directory to scan with pyrefly.
    runner : ToolRunner
        Shared tool runner for invoking pyrefly.
    runner : ToolRunner
        Shared tool runner for invoking pyrefly.

    Returns
    -------
    dict[str, int]
        Mapping from repository-relative paths to error counts.
    """
    repo_root = repo_root.resolve()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_path = Path(tmp.name)

    output_path = runner.cache_dir / "pyrefly.json"
    args = [
        _resolve_pyrefly_bin(),
        "check",
        str(repo_root),
        "--output-format",
        "json",
        "--output",
        str(output_path),
        "--summary",
        "none",
        "--count-errors=0",
    ]

    result: ToolResult = runner.run("pyrefly", args, cwd=repo_root, output_path=output_path)
    if result.returncode == MISSING_BINARY_EXIT_CODE:
        log.warning("pyrefly binary not found; treating all files as 0 errors")
        output_path.unlink(missing_ok=True)
        return {}

    payload = runner.load_json(output_path) or {}
    if not payload and result.returncode != 0:
        log.warning(
            "pyrefly check exited with code %s and no output; stdout=%s stderr=%s",
            result.returncode,
            result.stdout.strip(),
            result.stderr.strip(),
        )
        output_path.unlink(missing_ok=True)
        return {}

    errors_field = payload.get("errors") if isinstance(payload, dict) else None
    errors: list[PyreflyError] = errors_field if isinstance(errors_field, list) else []
    errors_by_file: dict[str, int] = {}
    for diag in errors:
        if diag.get("severity") != "error":
            continue
        file_name = diag.get("path")
        if not file_name:
            continue
        file_path = Path(str(file_name)).resolve()
        try:
            rel_path = repo_relpath(repo_root, file_path)
        except ValueError:
            continue
        errors_by_file[rel_path] = errors_by_file.get(rel_path, 0) + 1

    return errors_by_file


def _run_pyright(
    repo_root: Path, runner: ToolRunner, *, pyright_bin: str = PYRIGHT_BIN
) -> dict[str, int]:
    """
    Run pyright and aggregate error counts per file.

    Returns
    -------
    dict[str, int]
        Mapping from repository-relative paths to error counts.
    """
    args = [
        pyright_bin,
        "--outputjson",
        str(repo_root),
    ]
    result = runner.run("pyright", args, cwd=repo_root)
    if result.returncode == MISSING_BINARY_EXIT_CODE:
        log.warning("pyright binary not found; treating all files as 0 errors")
        return {}
    payload = {}
    try:
        payload = json.loads(result.stdout) if result.stdout else {}
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse pyright JSON output: %s", exc)
        return {}

    diagnostics = payload.get("generalDiagnostics") if isinstance(payload, dict) else None
    if not isinstance(diagnostics, list):
        return {}
    errors_by_file: dict[str, int] = {}
    for diag in diagnostics:
        if not isinstance(diag, dict):
            continue
        if diag.get("severity") != "error":
            continue
        file_name = diag.get("file")
        if not file_name:
            continue
        rel_path = repo_relpath(repo_root, Path(str(file_name)))
        errors_by_file[rel_path] = errors_by_file.get(rel_path, 0) + 1
    return errors_by_file


def _run_ruff(repo_root: Path, runner: ToolRunner) -> dict[str, int]:
    """
    Run ruff and aggregate error counts per file.

    Returns
    -------
    dict[str, int]
        Mapping from repository-relative paths to error counts.
    """
    args = [
        RUFF_BIN,
        "check",
        str(repo_root),
        "--output-format",
        "json",
    ]
    result = runner.run("ruff", args, cwd=repo_root)
    if result.returncode == MISSING_BINARY_EXIT_CODE:
        log.warning("ruff binary not found; treating all files as 0 errors")
        return {}
    try:
        payload = json.loads(result.stdout) if result.stdout else []
    except json.JSONDecodeError as exc:
        log.warning("Failed to parse ruff JSON output: %s", exc)
        return {}
    if not isinstance(payload, list):
        return {}
    errors_by_file: dict[str, int] = {}
    for diag in payload:
        if not isinstance(diag, dict):
            continue
        file_name = diag.get("filename")
        if not file_name:
            continue
        rel_path = repo_relpath(repo_root, Path(str(file_name)))
        errors_by_file[rel_path] = errors_by_file.get(rel_path, 0) + 1
    return errors_by_file


def ingest_typing_signals(
    gateway: StorageGateway,
    cfg: TypingIngestConfig,
    *,
    scan_config: ScanConfig | None = None,
    runner: ToolRunner | None = None,
    tools: ToolsConfig | None = None,
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
    con = gateway.con
    repo_root = cfg.repo_root
    scan_cfg = scan_config or ScanConfig(
        repo_root=repo_root, ignore_dirs=tuple(sorted(IGNORE_DIRS))
    )
    shared_runner = runner or ToolRunner(cache_dir=repo_root / "build" / ".tool_cache")
    pyright_bin = tools.pyright_bin if tools is not None else PYRIGHT_BIN

    # Compute annotation info for each Python file
    annotation_info: dict[str, AnnotationInfo] = {}
    for path in _iter_python_files(repo_root, scan_cfg):
        rel_path = repo_relpath(repo_root, path)
        info = _compute_annotation_info_for_file(path)
        if info is not None:
            annotation_info[rel_path] = info

    error_maps = {
        "pyrefly": _run_pyrefly(repo_root, shared_runner),
        "pyright": _run_pyright(repo_root, shared_runner, pyright_bin=pyright_bin),
        "ruff": _run_ruff(repo_root, shared_runner),
    }
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
