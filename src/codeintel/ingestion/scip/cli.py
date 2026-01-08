"""SCIP CLI argument helpers."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
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


@dataclass(frozen=True)
class ScipPythonArgs:
    """Arguments for scip-python CLI invocations."""

    target_base: Path
    output_scip: Path
    project_name: str
    target_paths: Sequence[str] | None = None
    environment_json: Path | None = None
    project_version: str | None = None
    project_namespace: str | None = None


def build_scip_python_args(args: ScipPythonArgs) -> list[str]:
    """Build scip-python CLI arguments.

    Parameters
    ----------
    args
        scip-python argument bundle.

    Returns
    -------
    list[str]
        Argument list for scip-python.
    """
    argv = [
        "index",
        str(args.target_base),
        "--output",
        str(args.output_scip),
        "--project-name",
        args.project_name,
    ]
    if args.project_version is not None:
        argv.extend(["--project-version", args.project_version])
    if args.project_namespace is not None:
        argv.extend(["--project-namespace", args.project_namespace])
    if args.environment_json is not None:
        argv.extend(["--environment", str(args.environment_json)])
    if args.target_paths:
        for rel_path in _normalize_target_only_paths(args.target_paths):
            argv.extend(["--target-only", rel_path])
    return argv


def ensure_pip_available() -> None:
    """Raise if pip is unavailable for scip-python environment discovery.

    Raises
    ------
    ValueError
        If neither pip nor pip3 is found on PATH.
    """
    if shutil.which("pip") is not None or shutil.which("pip3") is not None:
        return
    message = (
        "scip-python requires pip on PATH unless --environment is provided. "
        "Install pip (uv add --dev pip) or set environment_json."
    )
    raise ValueError(message)


__all__ = ["ScipPythonArgs", "build_scip_python_args", "ensure_pip_available"]
