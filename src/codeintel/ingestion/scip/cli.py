"""SCIP CLI argument helpers."""

from __future__ import annotations

import shutil
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from codeintel.ingestion.scip.environment import pip_available


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


@contextmanager
def stage_pyright_config(
    *,
    target_base: Path,
    pyright_config_path: Path | None,
) -> Iterator[Path | None]:
    """Stage a pyrightconfig.json for scip-python in the target base.

    Yields
    ------
    Path | None
        Path to the staged config, or None when no staging occurs.

    Raises
    ------
    ValueError
        If the provided pyright config path is invalid.
    """
    if pyright_config_path is None:
        yield None
        return
    if not pyright_config_path.is_file():
        message = f"pyright_config_path does not exist: {pyright_config_path}"
        raise ValueError(message)
    dest_path = target_base / "pyrightconfig.json"
    if dest_path.exists() and not dest_path.is_file():
        message = f"pyrightconfig.json is not a file: {dest_path}"
        raise ValueError(message)
    if pyright_config_path.resolve() == dest_path.resolve():
        yield dest_path
        return

    backup: bytes | None = None
    if dest_path.exists():
        backup = dest_path.read_bytes()
    shutil.copyfile(pyright_config_path, dest_path)
    try:
        yield dest_path
    finally:
        if backup is None:
            dest_path.unlink(missing_ok=True)
        else:
            dest_path.write_bytes(backup)


def ensure_pip_available() -> None:
    """Raise if pip is unavailable for scip-python environment discovery.

    Raises
    ------
    ValueError
        If neither pip nor pip3 is found on PATH.
    """
    if pip_available():
        return
    message = (
        "scip-python requires pip on PATH unless --environment is provided. "
        "Install pip (uv add --dev pip) or set environment_json."
    )
    raise ValueError(message)


__all__ = [
    "ScipPythonArgs",
    "build_scip_python_args",
    "ensure_pip_available",
    "stage_pyright_config",
]
