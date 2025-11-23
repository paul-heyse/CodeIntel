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

import duckdb

from codeintel.config.models import TypingIngestConfig
from codeintel.ingestion.common import run_batch
from codeintel.models.rows import (
    StaticDiagnosticRow,
    TypednessRow,
    static_diagnostic_to_tuple,
    typedness_row_to_tuple,
)
from codeintel.types import PyreflyError
from codeintel.utils.paths import repo_relpath

log = logging.getLogger(__name__)
MISSING_BINARY_EXIT_CODE = 127

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


def _iter_python_files(repo_root: Path) -> Iterable[Path]:
    search_root = repo_root / "src"
    if not search_root.is_dir():
        search_root = repo_root

    for path in search_root.rglob("*.py"):
        rel_parts = path.relative_to(repo_root).parts
        if any(part in IGNORE_DIRS for part in rel_parts):
            continue
        yield path


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


def _run_pyrefly(repo_root: Path) -> dict[str, int]:
    """
    Run pyrefly and aggregate error counts per file.

    Parameters
    ----------
    repo_root : Path
        Root directory to scan with pyrefly.

    Returns
    -------
    dict[str, int]
        Mapping from repository-relative paths to error counts.
    """
    repo_root = repo_root.resolve()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        output_path = Path(tmp.name)

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

    code, stdout, stderr = _run_command(args, cwd=repo_root)
    if code == MISSING_BINARY_EXIT_CODE:
        log.warning("pyrefly binary not found; treating all files as 0 errors")
        output_path.unlink(missing_ok=True)
        return {}

    # Try to parse output regardless of exit code, as code 1 usually just means errors were found.
    try:
        if output_path.exists() and output_path.stat().st_size > 0:
            payload = json.loads(output_path.read_text(encoding="utf8"))
        else:
            # If no output file but non-zero exit, log warning
            if code != 0:
                log.warning(
                    "pyrefly check exited with code %s and no output; stdout=%s stderr=%s",
                    code,
                    stdout.strip(),
                    stderr.strip(),
                )
            return {}
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("Failed to read pyrefly JSON output: %s", exc)
        output_path.unlink(missing_ok=True)
        return {}
    finally:
        output_path.unlink(missing_ok=True)

    errors: list[PyreflyError] = payload.get("errors") or []
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


def ingest_typing_signals(
    con: duckdb.DuckDBPyConnection,
    cfg: TypingIngestConfig,
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

    # Compute annotation info for each Python file
    annotation_info: dict[str, AnnotationInfo] = {}
    for path in _iter_python_files(repo_root):
        rel_path = repo_relpath(repo_root, path)
        info = _compute_annotation_info_for_file(path)
        if info is not None:
            annotation_info[rel_path] = info

    pyrefly_errors = _run_pyrefly(repo_root)

    all_paths = set(annotation_info.keys()) | set(pyrefly_errors.keys())

    typedness_rows: list[TypednessRow] = []
    diag_rows: list[StaticDiagnosticRow] = []
    for rel_path in sorted(all_paths):
        info = annotation_info.get(
            rel_path, AnnotationInfo(params_ratio=0.0, returns_ratio=0.0, untyped_defs=0)
        )
        py_errors = 0
        pf_errors = pyrefly_errors.get(rel_path, 0)
        total_errors = pf_errors
        has_errors = total_errors > 0

        annotation_ratio = {
            "params": info.params_ratio,
            "returns": info.returns_ratio,
        }
        overlay_needed = bool(total_errors > 0 or info.untyped_defs > 0)

        typedness_rows.append(
            TypednessRow(
                path=rel_path,
                type_error_count=total_errors,
                annotation_ratio=annotation_ratio,
                untyped_defs=info.untyped_defs,
                overlay_needed=overlay_needed,
            )
        )

        diag_rows.append(
            StaticDiagnosticRow(
                rel_path=rel_path,
                pyrefly_errors=pf_errors,
                pyright_errors=py_errors,
                total_errors=total_errors,
                has_errors=has_errors,
            )
        )

    run_batch(
        con,
        "analytics.typedness",
        [typedness_row_to_tuple(row) for row in typedness_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )
    run_batch(
        con,
        "analytics.static_diagnostics",
        [static_diagnostic_to_tuple(row) for row in diag_rows],
        delete_params=[cfg.repo, cfg.commit],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info(
        "Typedness & static diagnostics ingested for %d files in %s@%s",
        len(all_paths),
        cfg.repo,
        cfg.commit,
    )
