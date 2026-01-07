"""SCIP CLI argument helpers."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path


def _normalize_target_only_paths(paths: Sequence[str]) -> list[str]:
    normalized: list[str] = []
    seen: set[str] = set()
    for path in paths:
        candidate = path.strip()
        if not candidate:
            continue
        candidate = candidate.replace("\\", "/").lstrip("/").rstrip("/")
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        normalized.append(candidate)
    return normalized


def build_scip_python_args(
    *,
    target_base: Path,
    output_scip: Path,
    project_name: str,
    rel_paths: Sequence[str] | None = None,
    scope_paths: Sequence[str] | None = None,
    environment_json: Path | None = None,
) -> list[str]:
    """Build scip-python CLI arguments.

    Parameters
    ----------
    target_base
        Project root passed to scip-python.
    output_scip
        Output index.scip path.
    project_name
        Project name used for SCIP identity.
    rel_paths
        Optional list of repo-relative module paths to index.
    scope_paths
        Optional repo-relative prefixes to index when rel_paths is not supplied.
    environment_json
        Optional scip-python --environment JSON file.

    Returns
    -------
    list[str]
        Argument list for scip-python.
    """
    args = [
        "index",
        str(target_base),
        "--output",
        str(output_scip),
        "--project-name",
        project_name,
    ]
    if environment_json is not None:
        args.extend(["--environment", str(environment_json)])
    target_only_paths = rel_paths if rel_paths is not None else scope_paths
    if target_only_paths:
        for rel_path in _normalize_target_only_paths(target_only_paths):
            args.extend(["--target-only", rel_path])
    return args


def ensure_pip_available() -> None:
    """Raise if pip is unavailable for scip-python environment discovery."""
    if shutil.which("pip") is not None or shutil.which("pip3") is not None:
        return
    message = (
        "scip-python requires pip on PATH unless --environment is provided. "
        "Install pip (uv add --dev pip) or set environment_json."
    )
    raise ValueError(message)


__all__ = ["build_scip_python_args", "ensure_pip_available"]
