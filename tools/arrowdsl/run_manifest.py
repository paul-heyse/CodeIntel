"""Thin wrapper for Arrow DSL run manifest emission."""

from __future__ import annotations

from pathlib import Path

from codeintel.core.columnar.run_manifest import RunManifestOptions
from codeintel.core.columnar.run_manifest import write_run_manifest as _write_run_manifest


def write_run_manifest(
    output_dir: Path,
    *,
    options: RunManifestOptions | None = None,
) -> Path:
    """Write a run manifest using the core columnar helper.

    Returns
    -------
    pathlib.Path
        Path to the written manifest.
    """
    return _write_run_manifest(output_dir, options=options)


__all__ = ["write_run_manifest"]
