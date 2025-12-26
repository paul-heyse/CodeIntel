"""Execution options for Hamilton build runs.

This module centralizes "how the build runs" (execution behavior toggles) into a single typed
object. It intentionally excludes resources/identity, which remain part of ``BuildEnv``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.hamilton.hooks import HookOptions
from codeintel.build.settings import DEFAULT_PROFILE_NAME

if TYPE_CHECKING:
    from codeintel.build.hamilton.env import BuildEnv


@dataclass(frozen=True, slots=True)
class BuildExecutionOptions:
    """Execution behavior options for a Hamilton build run."""

    profile: str | None = None
    parallel_backend: str = "sequential"
    max_workers: int | None = None
    enable_hamilton_cache: bool = False
    cache_dir: str | None = None
    enable_telemetry: bool = True
    enable_progress: bool = False
    enable_timing: bool = False

    def resolved_profile(self, *, env: BuildEnv) -> str:
        """Resolve the effective profile for this run.

        Parameters
        ----------
        env
            Build environment for the run.

        Returns
        -------
        str
            Effective profile name.
        """
        if self.profile is not None:
            return self.profile
        if env.profile is not None:
            return env.profile
        return DEFAULT_PROFILE_NAME

    def resolved_cache_dir(self, *, env: BuildEnv) -> Path:
        """Resolve the cache directory for this run.

        Parameters
        ----------
        env
            Build environment for the run.

        Notes
        -----
        This only matters when ``enable_hamilton_cache`` is True.

        Returns
        -------
        Path
            Effective cache directory.
        """
        if self.cache_dir is not None:
            return Path(self.cache_dir)
        return env.paths.build_dir / ".hamilton_cache"

    def hook_options(self, *, env: BuildEnv) -> HookOptions:
        """Construct HookOptions for this run.

        Parameters
        ----------
        env
            Build environment for the run.

        Validation is kept strict-only by default; telemetry/progress/timing follow the execution
        options toggles.

        Returns
        -------
        HookOptions
            Hook configuration derived from env and execution options.
        """
        return HookOptions(
            strict_contracts=env.strict_contracts,
            enable_validation=env.strict_contracts,
            enable_telemetry=self.enable_telemetry,
            enable_progress=self.enable_progress,
            enable_timing=self.enable_timing,
        )


__all__ = ["BuildExecutionOptions"]
