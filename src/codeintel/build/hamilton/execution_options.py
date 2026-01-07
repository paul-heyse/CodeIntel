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
    enable_hamilton_cache: bool = True
    cache_dir: str | None = None
    enable_telemetry: bool = True
    enable_io_telemetry: bool = True
    enable_progress: bool = False
    enable_timing: bool = False
    plugins_enabled: tuple[str, ...] | None = None
    plugins_disabled: tuple[str, ...] | None = None
    allow_workspace_modules: bool | None = None

    def validate(self) -> None:
        """Validate execution option values.

        Raises
        ------
        ValueError
            If any option value is invalid.
        """
        if self.parallel_backend not in _ALLOWED_PARALLEL_BACKENDS:
            msg = (
                "parallel_backend must be one of "
                f"{sorted(_ALLOWED_PARALLEL_BACKENDS)}; got {self.parallel_backend!r}"
            )
            raise ValueError(msg)
        if self.max_workers is not None and self.max_workers <= 0:
            msg = "max_workers must be a positive integer"
            raise ValueError(msg)
        if self.plugins_enabled is not None and not isinstance(self.plugins_enabled, tuple):
            msg = "plugins_enabled must be a tuple of plugin names"
            raise ValueError(msg)
        if self.plugins_disabled is not None and not isinstance(self.plugins_disabled, tuple):
            msg = "plugins_disabled must be a tuple of plugin names"
            raise ValueError(msg)
        if self.cache_dir is not None and not isinstance(self.cache_dir, str):
            msg = "cache_dir must be a string path"
            raise ValueError(msg)
        if self.profile is not None and not isinstance(self.profile, str):
            msg = "profile must be a string"
            raise ValueError(msg)

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

    def hook_options(
        self,
        *,
        telemetry_output_path: Path | None = None,
        io_telemetry_output_path: Path | None = None,
        progress_desc: str | None = None,
    ) -> HookOptions:
        """Construct HookOptions for this run.

        Telemetry/progress/timing follow the execution options toggles.

        Returns
        -------
        HookOptions
            Hook configuration derived from execution options.
        """
        resolved_progress_desc = progress_desc if progress_desc is not None else "Building targets"
        return HookOptions(
            enable_telemetry=self.enable_telemetry,
            enable_io_telemetry=self.enable_io_telemetry,
            enable_progress=self.enable_progress,
            enable_timing=self.enable_timing,
            progress_desc=resolved_progress_desc,
            telemetry_output_path=telemetry_output_path,
            io_telemetry_output_path=io_telemetry_output_path,
        )

    def plugin_overrides(self) -> dict[str, object]:
        """Return config overrides for plugin selection.

        Returns
        -------
        dict[str, object]
            Mapping of config overrides for plugin selection.
        """
        overrides: dict[str, object] = {}
        if self.plugins_enabled is not None:
            overrides["ci.plugins.enabled"] = list(self.plugins_enabled)
        if self.plugins_disabled is not None:
            overrides["ci.plugins.disabled"] = list(self.plugins_disabled)
        if self.allow_workspace_modules is not None:
            overrides["ci.plugins.allow_workspace_modules"] = self.allow_workspace_modules
        return overrides


_ALLOWED_PARALLEL_BACKENDS: frozenset[str] = frozenset(
    {
        "sequential",
        "threadpool",
        "ray",
        "dask",
        "auto",
    }
)


__all__ = ["BuildExecutionOptions"]
