"""Run environment capture for reproducibility.

Capture and persist build environment details to enable reproducibility
and debugging of build runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as pkg_version
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from codeintel.build.config import BuildConfig

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RunEnvironment:
    """Captured environment for a build run.

    Attributes
    ----------
    python_version
        Full Python version string (e.g., "3.13.0").
    os_name
        Operating system name (e.g., "Linux", "Darwin", "Windows").
    os_version
        Operating system release version.
    tool_versions
        Versions of key build tools (pyright, ruff, etc.).
    config_hash
        Hash of the build configuration used.
    git_dirty
        Whether the git working tree has uncommitted changes.
    captured_at
        When the environment was captured.
    """

    python_version: str
    os_name: str
    os_version: str
    tool_versions: dict[str, str] = field(default_factory=dict)
    config_hash: str | None = None
    git_dirty: bool = False
    captured_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @classmethod
    def capture(cls, config: BuildConfig | None = None) -> RunEnvironment:
        """Capture the current build environment.

        Parameters
        ----------
        config
            Optional build configuration to hash.

        Returns
        -------
        RunEnvironment
            Captured environment details.
        """
        return cls(
            python_version=_get_python_version(),
            os_name=platform.system(),
            os_version=platform.release(),
            tool_versions=_capture_tool_versions(),
            config_hash=_hash_config(config) if config else None,
            git_dirty=_is_git_dirty(),
            captured_at=datetime.now(tz=UTC),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns
        -------
        dict[str, Any]
            Dictionary representation of the environment.
        """
        return {
            "python_version": self.python_version,
            "os_name": self.os_name,
            "os_version": self.os_version,
            "tool_versions": self.tool_versions,
            "config_hash": self.config_hash,
            "git_dirty": self.git_dirty,
            "captured_at": self.captured_at.isoformat() if self.captured_at else None,
        }


def _get_python_version() -> str:
    """Get the Python version string.

    Returns
    -------
    str
        Python version (e.g., "3.13.0").
    """
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def _capture_tool_versions() -> dict[str, str]:
    """Capture versions of key build tools.

    Returns
    -------
    dict[str, str]
        Tool name to version mapping.

    Notes
    -----
    Uses Python introspection where possible to avoid subprocess calls.
    Tool versions that require subprocess are omitted.
    """
    tools: dict[str, str] = {}

    # Capture Python runtime info
    tools["python"] = _get_python_version()

    # Try to capture ruff version via importlib
    try:
        tools["ruff"] = pkg_version("ruff")
    except PackageNotFoundError:
        log.debug("environment.tool_versions ruff package not found")

    return tools


def _hash_config(config: BuildConfig) -> str:
    """Hash the build configuration.

    Parameters
    ----------
    config
        Build configuration to hash.

    Returns
    -------
    str
        SHA256 hash of config (truncated to 16 chars).
    """
    try:
        # Try to serialize config to JSON
        config_dict = {
            "profile": getattr(config, "profile", None),
            "root": str(getattr(config, "root", "")),
        }
        serialized = json.dumps(config_dict, sort_keys=True, default=str)
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
    except (TypeError, ValueError, AttributeError):
        return ""


def _is_git_dirty() -> bool:
    """Check if git working tree has uncommitted changes.

    Returns
    -------
    bool
        True if there are uncommitted changes. Returns False if git state
        cannot be determined without subprocess calls.

    Notes
    -----
    This is a best-effort check that avoids subprocess calls for security
    compliance. It checks for the presence of a .git/index.lock file as
    a heuristic for ongoing operations, but cannot reliably detect all
    uncommitted changes without git CLI access.
    """
    cwd = Path.cwd()
    git_dir = cwd / ".git"

    if not git_dir.is_dir():
        return False

    # Check for index.lock as a heuristic for in-progress operations
    index_lock = git_dir / "index.lock"
    return index_lock.exists()


__all__ = [
    "RunEnvironment",
]
