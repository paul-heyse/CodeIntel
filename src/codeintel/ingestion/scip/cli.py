"""SCIP CLI argument helpers."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path


def build_scip_python_args(
    *,
    target_base: Path,
    output_scip: Path,
    project_name: str,
    rel_paths: Sequence[str] | None = None,
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
    for rel_path in rel_paths or ():
        args.extend(["--target-only", rel_path])
    return args


__all__ = ["build_scip_python_args"]
