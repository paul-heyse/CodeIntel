"""Runtime resolution service with caching.

Consolidate runtime resolution from:
- ``resolution/runtime.py`` (canonical resolve_from_params)
- ``handlers/context.py`` (_resolve_runtime)
- ``deps/providers.py`` (_resolve_db_path)

Provide caching to avoid repeated resolution within a command execution.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.cli.resolution.runtime import resolve_from_params
from codeintel.cli.services.params import ParamService

if TYPE_CHECKING:
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
        If True, fall back to explicit params when no project file found.

    Examples
    --------
    >>> service = RuntimeService.from_dict({"project_root": Path(".")})
    >>> runtime = service.runtime  # Lazily resolved
    >>> runtime.db_path  # doctest: +SKIP
    PosixPath('build/db/codeintel.duckdb')
    """

    def __init__(
        self,
        params: ParamService | Mapping[str, object] | None = None,
        *,
        project_root: Path | None = None,
        db_path: Path | None = None,
        allow_fallback: bool = True,
    ) -> None:
        """Initialize runtime service."""
        if params is None:
            self._params: dict[str, object] = {}
        elif isinstance(params, ParamService):
            self._params = dict(params.raw)
        else:
            self._params = dict(params)

        # Apply explicit overrides
        if project_root is not None:
            self._params["project_root"] = project_root
        if db_path is not None:
            self._params["db_path"] = db_path

        self._allow_fallback = allow_fallback
        self._resolved: ResolvedRuntime | None = None

    @classmethod
    def from_dict(
        cls,
        params: dict[str, object],
        *,
        allow_fallback: bool = True,
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

        Returns
        -------
        RuntimeService
            Configured service.
        """
        return cls(params, project_root=project_root, db_path=db_path)

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

    def _resolve(self) -> ResolvedRuntime:
        """Perform resolution.

        Returns
        -------
        ResolvedRuntime
            Resolved runtime.
        """
        LOG.debug("Resolving runtime with params: %s", list(self._params.keys()))
        return resolve_from_params(self._params, allow_fallback=self._allow_fallback)


__all__ = [
    "RuntimeService",
]
