"""Runtime resolution service with caching.

Consolidate runtime resolution from:
- ``resolution/runtime.py`` (canonical resolve_from_params)
- ``handlers/context.py`` (_resolve_runtime)
- ``deps/providers.py`` (_resolve_db_path)

Provide caching to avoid repeated resolution within a command execution.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar

import codeintel.cli.resolution.runtime as resolution_runtime
from codeintel.cli.resolution.params import RuntimeParams
from codeintel.cli.services.params import ParamService

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from codeintel.cli.resolution.types import ResolvedRuntime

LOG = logging.getLogger(__name__)


class RuntimeService:
    """Runtime resolution with caching.

    Lazily resolve runtime configuration and cache the result for the
    duration of command execution.

    Parameters
    ----------
    params
        ParamService or dict with resolution parameters.
    project_root
        Optional explicit project root override.
    db_path
        Optional explicit database path override.
    allow_fallback
        When True, fall back to explicit params when no project file found.
        When None, fallback is enabled only if explicit repo/commit/db_path params are set.

    Examples
    --------
    >>> service = RuntimeService.from_dict({"project_root": Path(".")})
    >>> runtime = service.runtime
    >>> runtime.db_path
    PosixPath('build/db/codeintel.duckdb')
    """

    SERVICE_NAME: ClassVar[str] = "runtime"

    def initialize(self) -> None:
        """Initialize the service (no-op, resolution is lazy)."""

    def shutdown(self) -> None:
        """Shut down the service by invalidating cached runtime."""
        self.invalidate()

    @property
    def is_ready(self) -> bool:
        """Check if service is ready.

        Returns
        -------
        bool
            Always True (resolution is lazy).
        """
        return True

    def __init__(
        self,
        params: ParamService | Mapping[str, object] | RuntimeParams | None = None,
        *,
        project_root: Path | None = None,
        db_path: Path | None = None,
        allow_fallback: bool | None = None,
    ) -> None:
        """Initialize runtime service."""
        if params is None:
            raw_params: dict[str, object] = {}
            runtime_params = RuntimeParams()
        elif isinstance(params, RuntimeParams):
            runtime_params = params
            raw_params = self._runtime_params_to_dict(runtime_params)
        elif isinstance(params, ParamService):
            raw_params = dict(params.raw)
            runtime_params = RuntimeParams.from_dict(raw_params)
        else:
            raw_params = dict(params)
            runtime_params = RuntimeParams.from_dict(raw_params)

        if project_root is not None:
            raw_params["project_root"] = project_root
            runtime_params = replace(runtime_params, project_root=project_root)
        if db_path is not None:
            raw_params["db_path"] = db_path
            runtime_params = replace(runtime_params, db_path=db_path)

        self._params = raw_params
        self._runtime_params = runtime_params

        self._allow_fallback = allow_fallback
        self._resolved: ResolvedRuntime | None = None

    @classmethod
    def from_dict(
        cls,
        params: dict[str, object],
        *,
        allow_fallback: bool | None = None,
    ) -> RuntimeService:
        """Create from dictionary.

        Parameters
        ----------
        params
            Resolution parameters.
        allow_fallback
            If True, fall back to explicit params.

        Returns
        -------
        RuntimeService
            Configured service.
        """
        return cls(params, allow_fallback=allow_fallback)

    @classmethod
    def from_param_service(
        cls,
        params: ParamService,
        *,
        project_root: Path | None = None,
        db_path: Path | None = None,
        allow_fallback: bool | None = None,
    ) -> RuntimeService:
        """Create from ParamService.

        Parameters
        ----------
        params
            Parameter service.
        project_root
            Optional project root override.
        db_path
            Optional database path override.
        allow_fallback
            Optional override for fallback behavior when no project file is found.

        Returns
        -------
        RuntimeService
            Configured service.
        """
        return cls(
            params,
            project_root=project_root,
            db_path=db_path,
            allow_fallback=allow_fallback,
        )

    @property
    def runtime(self) -> ResolvedRuntime:
        """Get resolved runtime (lazy, cached).

        The resolution may raise errors from the underlying
        :func:`~codeintel.cli.resolution.runtime.resolve_from_params` function.

        Returns
        -------
        ResolvedRuntime
            Fully resolved runtime.
        """
        if self._resolved is None:
            self._resolved = self._resolve()
        return self._resolved

    @property
    def is_resolved(self) -> bool:
        """Check if runtime has been resolved.

        Returns
        -------
        bool
            True if runtime is cached.
        """
        return self._resolved is not None

    @property
    def params(self) -> dict[str, object]:
        """Return a copy of the underlying parameters."""
        return dict(self._params)

    @property
    def runtime_params(self) -> RuntimeParams:
        """Return canonical runtime parameters.

        Returns
        -------
        RuntimeParams
            Canonical runtime parameters derived from inputs.
        """
        return self._runtime_params

    @property
    def db_path(self) -> Path:
        """Get database path.

        Returns
        -------
        Path
            Database file path.
        """
        return self.runtime.db_path

    @property
    def project_root(self) -> Path:
        """Get project root directory.

        Returns
        -------
        Path
            Project root.
        """
        return self.runtime.root

    def invalidate(self) -> None:
        """Clear cached runtime.

        Call this if underlying parameters change and re-resolution is needed.
        """
        self._resolved = None

    @staticmethod
    def _runtime_params_to_dict(params: RuntimeParams) -> dict[str, object]:
        return {
            "project_root": params.project_root,
            "repo": params.repo,
            "commit": params.commit,
            "db_path": params.db_path,
            "build_dir": params.build_dir,
            "repo_root": params.repo_root,
            "document_output_dir": params.document_output_dir,
            "backend": {
                "use_gpu": params.backend.use_gpu,
                "backend": params.backend.backend,
                "strict": params.backend.strict,
            },
        }

    def _resolve(self) -> ResolvedRuntime:
        """Perform resolution.

        Returns
        -------
        ResolvedRuntime
            Resolved runtime.
        """
        LOG.debug("Resolving runtime with params: %s", list(self._params.keys()))
        return resolution_runtime.resolve_from_params(
            self._runtime_params,
            allow_fallback=self._allow_fallback,
        )


__all__ = [
    "RuntimeService",
]
