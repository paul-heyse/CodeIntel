"""Configuration resolution utilities with environment overrides.

This module provides shared helpers for resolving configuration with
environment variable overrides. These functions consolidate logic that
was previously duplicated across CLI and pipeline execution modules.

Examples
--------
>>> from codeintel.config import resolve_tools_config, resolve_scan_profiles
>>> tools = resolve_tools_config()
>>> profiles = resolve_scan_profiles(Path("/repo"))
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from codeintel.config import ScanProfiles
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import GraphBackendConfig
from codeintel.ingestion.infrastructure.scanning import (
    ScanProfile,
    default_code_profile,
    default_config_profile,
    profile_from_env,
)

# Binary names that should be resolved to full paths
_BINARY_FIELDS = (
    "scip_python_bin",
    "scip_bin",
    "pyright_bin",
    "pyrefly_bin",
    "ruff_bin",
    "coverage_bin",
    "pytest_bin",
    "git_bin",
)


def resolve_tools_config(
    base: ToolsConfig | None = None,
    *,
    resolve_paths: bool = True,
) -> ToolsConfig:
    """Build a ToolsConfig applying environment overrides when present.

    Parameters
    ----------
    base
        Optional base configuration to extend.
    resolve_paths
        If True, resolve binary names to full paths via shutil.which.
        This ensures subprocess calls work even when PATH differs.

    Returns
    -------
    ToolsConfig
        Tools configuration with environment overrides applied.

    Examples
    --------
    >>> import os
    >>> os.environ["CODEINTEL_GIT_BIN"] = "/usr/bin/git"
    >>> cfg = resolve_tools_config()
    >>> cfg.git_bin
    '/usr/bin/git'
    """
    # Start with default values from ToolsConfig
    default_config = ToolsConfig()
    data = base.model_dump() if base is not None else default_config.model_dump()
    env_map = {
        "CODEINTEL_SCIP_PYTHON_BIN": "scip_python_bin",
        "CODEINTEL_SCIP_BIN": "scip_bin",
        "CODEINTEL_PYRIGHT_BIN": "pyright_bin",
        "CODEINTEL_PYREFLY_BIN": "pyrefly_bin",
        "CODEINTEL_RUFF_BIN": "ruff_bin",
        "CODEINTEL_COVERAGE_BIN": "coverage_bin",
        "CODEINTEL_PYTEST_BIN": "pytest_bin",
        "CODEINTEL_GIT_BIN": "git_bin",
        "CODEINTEL_COVERAGE_FILE": "coverage_file",
        "CODEINTEL_PYTEST_REPORT": "pytest_report_path",
    }
    for env_var, field in env_map.items():
        value = os.getenv(env_var)
        if value:
            data[field] = value

    # Resolve binary names to full paths for subprocess compatibility
    if resolve_paths:
        for field in _BINARY_FIELDS:
            bin_name = data.get(field)
            if bin_name and not Path(bin_name).is_absolute():
                full_path = shutil.which(bin_name)
                if full_path:
                    data[field] = full_path

    return ToolsConfig.model_validate(data)


def resolve_scan_profiles(
    repo_root: Path,
    code_profile: ScanProfile | None = None,
    config_profile: ScanProfile | None = None,
) -> ScanProfiles:
    """Resolve code and config scan profiles with environment overrides.

    Parameters
    ----------
    repo_root
        Repository root directory for default profile generation.
    code_profile
        Optional explicit code profile; uses env/defaults if None.
    config_profile
        Optional explicit config profile; uses env/defaults if None.

    Returns
    -------
    ScanProfiles
        Resolved code and config scan profiles.

    Examples
    --------
    >>> from pathlib import Path
    >>> profiles = resolve_scan_profiles(Path("/repo"))
    >>> profiles.code is not None
    True
    """
    resolved_code = code_profile or profile_from_env(default_code_profile(repo_root))
    resolved_config = config_profile or profile_from_env(default_config_profile(repo_root))
    return ScanProfiles(code=resolved_code, config=resolved_config)


def resolve_graph_backend(config: GraphBackendConfig | None = None) -> GraphBackendConfig:
    """Resolve graph backend configuration with defaults.

    Parameters
    ----------
    config
        Optional explicit configuration; returns defaults if None.

    Returns
    -------
    GraphBackendConfig
        Graph backend settings with defaults applied.

    Examples
    --------
    >>> cfg = resolve_graph_backend(None)
    >>> cfg is not None
    True
    """
    return config or GraphBackendConfig()


__all__ = [
    "resolve_graph_backend",
    "resolve_scan_profiles",
    "resolve_tools_config",
]
