"""Resolve external tool binaries with repo-aware fallbacks."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.models import ToolsConfig
from codeintel.core.tools.names import ToolName

if TYPE_CHECKING:
    from collections.abc import Iterable


@dataclass(frozen=True)
class ToolResolveConfig:
    """Resolution context for tool binary lookup."""

    repo_root: Path | None
    extra_bin_dirs: tuple[Path, ...]

    @classmethod
    def from_env(cls) -> ToolResolveConfig:
        """Build resolve config from environment variables.

        Returns
        -------
        ToolResolveConfig
            Resolve configuration derived from environment variables.
        """
        repo_root = _env_path("CODEINTEL_REPO_ROOT") or Path.cwd()
        extra_bin_dirs = _env_paths("CODEINTEL_TOOL_BIN_DIR")
        return cls(repo_root=repo_root, extra_bin_dirs=extra_bin_dirs)

    def with_repo_root(self, repo_root: Path | None) -> ToolResolveConfig:
        """Return a copy with updated repo_root when provided.

        Returns
        -------
        ToolResolveConfig
            Resolve configuration with repo_root overridden.
        """
        if repo_root is None:
            return self
        return ToolResolveConfig(repo_root=repo_root, extra_bin_dirs=self.extra_bin_dirs)


@dataclass(frozen=True)
class ToolResolution:
    """Resolution outcome for a single tool binary."""

    tool: ToolName
    configured: str
    resolved: Path | None
    origin: str | None
    searched: tuple[str, ...]


def resolve_tool(
    tool: ToolName | str,
    *,
    config: ToolsConfig,
    resolve_cfg: ToolResolveConfig,
) -> ToolResolution:
    """Resolve a tool binary, honoring repo-local and extra paths.

    Returns
    -------
    ToolResolution
        Resolution record containing resolved path and search metadata.
    """
    tool_name = _coerce_tool(tool)
    configured = config.resolve_path(tool_name)
    searched: list[str] = []

    configured_path = Path(configured)
    searched.append("configured_path")
    if _is_executable(configured_path):
        return ToolResolution(
            tool=tool_name,
            configured=configured,
            resolved=configured_path,
            origin="configured_path",
            searched=tuple(searched),
        )

    searched.append("PATH")
    discovered = shutil.which(configured)
    if discovered:
        return ToolResolution(
            tool=tool_name,
            configured=configured,
            resolved=Path(discovered),
            origin="PATH",
            searched=tuple(searched),
        )

    node_modules_path = _node_modules_path(resolve_cfg.repo_root, configured)
    searched.append("node_modules")
    if _is_executable(node_modules_path):
        return ToolResolution(
            tool=tool_name,
            configured=configured,
            resolved=node_modules_path,
            origin="node_modules",
            searched=tuple(searched),
        )

    searched.append("extra_bin_dirs")
    for extra_dir in resolve_cfg.extra_bin_dirs:
        candidate = extra_dir / configured
        if _is_executable(candidate):
            return ToolResolution(
                tool=tool_name,
                configured=configured,
                resolved=candidate,
                origin=f"extra_bin_dir:{extra_dir}",
                searched=tuple(searched),
            )

    return ToolResolution(
        tool=tool_name,
        configured=configured,
        resolved=None,
        origin=None,
        searched=tuple(searched),
    )


def resolve_tools(
    tools: Iterable[ToolName | str],
    *,
    config: ToolsConfig,
    resolve_cfg: ToolResolveConfig,
) -> dict[ToolName, ToolResolution]:
    """Resolve multiple tools and return resolution metadata.

    Returns
    -------
    dict[ToolName, ToolResolution]
        Mapping of tool identifiers to resolution outcomes.
    """
    resolutions: dict[ToolName, ToolResolution] = {}
    for tool in tools:
        resolution = resolve_tool(tool, config=config, resolve_cfg=resolve_cfg)
        resolutions[resolution.tool] = resolution
    return resolutions


def _coerce_tool(tool: ToolName | str) -> ToolName:
    if isinstance(tool, ToolName):
        return tool
    return ToolName(str(tool))


def _env_path(env_var: str) -> Path | None:
    raw = os.environ.get(env_var)
    if not raw:
        return None
    return Path(raw).expanduser()


def _env_paths(env_var: str) -> tuple[Path, ...]:
    raw = os.environ.get(env_var)
    if not raw:
        return ()
    parts = [part for part in raw.split(os.pathsep) if part]
    return tuple(Path(part).expanduser() for part in parts)


def _is_executable(path: Path) -> bool:
    return path.is_file() and os.access(path, os.X_OK)


def _node_modules_path(repo_root: Path | None, executable: str) -> Path:
    if repo_root is None:
        return Path(executable)
    return repo_root / "node_modules" / ".bin" / executable


__all__ = [
    "ToolResolution",
    "ToolResolveConfig",
    "resolve_tool",
    "resolve_tools",
]
