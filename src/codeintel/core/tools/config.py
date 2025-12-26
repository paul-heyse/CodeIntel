"""Tool configuration primitives for resolving executables and environments."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.core.tools.names import ToolName

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class ToolBinaries:
    """Immutable configuration for external tool executables and timeouts."""

    scip_python_bin: str = "scip-python"
    protoc_bin: str = "python"
    pyright_bin: str = "pyright"
    pyrefly_bin: str = "pyrefly"
    ruff_bin: str = "ruff"
    coverage_bin: str = "coverage"
    pytest_bin: str = "pytest"
    git_bin: str = "git"
    default_timeout_s: float = 300.0

    def resolve_path(self, tool: ToolName | str) -> str:
        """Return the configured executable name/path for a tool identifier.

        Parameters
        ----------
        tool
            Tool identifier (enum or string).

        Returns
        -------
        str
            Resolved executable name/path.
        """
        tool_name = tool.value if isinstance(tool, ToolName) else str(tool)
        mapping = {
            "scip-python": self.scip_python_bin,
            "protoc": self.protoc_bin,
            "pyright": self.pyright_bin,
            "pyrefly": self.pyrefly_bin,
            "coverage": self.coverage_bin,
            "pytest": self.pytest_bin,
            "ruff": self.ruff_bin,
            "git": self.git_bin,
        }
        return str(mapping.get(tool_name, tool_name))


def build_tool_env(
    binaries: ToolBinaries,
    tool: ToolName | str,
    *,
    base_env: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Build a process environment for tool execution.

    Parameters
    ----------
    binaries
        Tool binary configuration.
    tool
        Tool identifier (currently unused; reserved for per-tool env overrides).
    base_env
        Optional base environment mapping to overlay on top of the current process environment.

    Returns
    -------
    dict[str, str]
        Environment mapping to pass to a subprocess invocation.
    """
    _ = tool
    env: dict[str, str] = dict(os.environ)
    if base_env:
        env.update(base_env)
    env.setdefault("CODEINTEL_TOOL_TIMEOUT", str(int(binaries.default_timeout_s)))
    return env


__all__ = [
    "ToolBinaries",
    "build_tool_env",
]
