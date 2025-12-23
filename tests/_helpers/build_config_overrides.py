"""Helpers for writing and reloading build configuration in tests."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from codeintel.build.config import load_build_config
from tests._helpers.build import write_build_config

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness


def write_build_config_sections(
    repo_root: Path,
    sections: Mapping[str, Mapping[str, Any]],
) -> Path:
    """Write build config sections to the repo root.

    Parameters
    ----------
    repo_root
        Repository root where config should be written.
    sections
        Mapping of section names to config values.

    Returns
    -------
    Path
        Path to the written config file.
    """
    return write_build_config(repo_root, sections)


def reload_build_config(harness: HamiltonBuildHarness) -> HamiltonBuildHarness:
    """Reload BuildConfig from repo_root and apply to the harness.

    Returns
    -------
    HamiltonBuildHarness
        Harness instance with updated BuildConfig.
    """
    config = load_build_config(harness.ctx.repo_root)
    return harness.with_build_config(config)


__all__ = [
    "reload_build_config",
    "write_build_config_sections",
]
