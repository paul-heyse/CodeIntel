"""Lazy resource providers.

Resources are only initialized when first accessed, reducing startup overhead
and enabling commands to declare dependencies they may not always use.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from codeintel.cli.deps.protocols import StorageAccess
    from codeintel.storage.gateway import StorageGateway


class LazyStorageProvider:
    """Lazy-loading storage provider.

    Gateway is only opened when .gateway is first accessed, avoiding
    unnecessary database connections for commands that may not need storage.

    Parameters
    ----------
    project_root
        Optional project root for runtime resolution.
    db_path
        Optional explicit database path (overrides resolution).
    """

    def __init__(
        self,
        *,
        project_root: Path | None = None,
        db_path: Path | None = None,
    ) -> None:
        """Initialize lazy storage provider."""
        self._project_root = project_root
        self._db_path = db_path
        self._gateway: StorageGateway | None = None

    @property
    def gateway(self) -> StorageGateway:
        """Get or create storage gateway.

        Returns
        -------
        StorageGateway
            Open read-only storage gateway.
        """
        if self._gateway is None:
            self._gateway = self._open_gateway()
        return self._gateway

    @contextmanager
    def write_gateway(self) -> Iterator[StorageGateway]:
        """Context manager for write-enabled gateway.

        Yields
        ------
        StorageGateway
            Write-enabled storage gateway that will be closed on exit.
        """
        from codeintel.storage.gateway import StorageConfig, open_gateway

        db_path = self._resolve_db_path()
        config = StorageConfig(db_path=db_path, read_only=False)
        gw = open_gateway(config)
        try:
            yield gw
        finally:
            gw.close()

    def close(self) -> None:
        """Close gateway if open."""
        if self._gateway is not None:
            self._gateway.close()
            self._gateway = None

    def _open_gateway(self) -> StorageGateway:
        """Open read-only gateway.

        Returns
        -------
        StorageGateway
            Open read-only gateway.
        """
        from codeintel.storage.gateway import StorageConfig, open_gateway

        db_path = self._resolve_db_path()
        config = StorageConfig(db_path=db_path, read_only=True)
        return open_gateway(config)

    def _resolve_db_path(self) -> Path:
        """Resolve database path from config or explicit value.

        Returns
        -------
        Path
            Resolved database path.
        """
        if self._db_path is not None:
            return self._db_path

        from codeintel.cli.resolution.runtime import resolve_from_params

        params: dict[str, object] = {}
        if self._project_root is not None:
            params["project_root"] = self._project_root

        runtime = resolve_from_params(params)
        return runtime.db_path


class LazyServingProvider:
    """Lazy-loading serving layer provider.

    Service stack is only built when .invoke() is first called.

    Parameters
    ----------
    storage
        Storage access for serving operations.
    project_root
        Optional project root for runtime resolution.
    """

    def __init__(
        self,
        *,
        storage: StorageAccess | None = None,
        project_root: Path | None = None,
    ) -> None:
        """Initialize lazy serving provider."""
        self._storage = storage
        self._project_root = project_root

    def invoke(
        self,
        operation_id: str,
        params: dict[str, object],
        *,
        skip_prereqs: bool = False,
    ) -> dict[str, object]:
        """Invoke a serving operation.

        Parameters
        ----------
        operation_id
            Operation ID in the serving catalog.
        params
            Operation parameters.
        skip_prereqs
            If True, skip prerequisite pipeline execution.

        Returns
        -------
        dict[str, object]
            Operation result.

        Raises
        ------
        ValueError
            If operation not found or storage not available.
        """
        from codeintel.serving.auto_pipeline import run_operation_prereqs
        from codeintel.serving.bootstrap import build_service_stack
        from codeintel.serving.operations.catalog import get_operation

        op = get_operation(operation_id)
        if op is None:
            msg = f"Unknown serving operation: {operation_id}"
            raise ValueError(msg)

        if self._storage is None:
            msg = "Storage access required for serving operations"
            raise ValueError(msg)

        gateway = self._storage.gateway

        # Resolve runtime for serving config
        from codeintel.cli.resolution.runtime import resolve_from_params

        resolve_params: dict[str, object] = {}
        if self._project_root is not None:
            resolve_params["project_root"] = self._project_root
        runtime = resolve_from_params(resolve_params)

        # Run prerequisites if needed
        if not skip_prereqs:
            run_operation_prereqs(
                op_id=operation_id,
                gateway=gateway,
                snapshot=runtime.snapshot,
                paths=runtime.paths,
                tools=runtime.tools,
            )

        # Build service stack and invoke
        stack = build_service_stack(
            config=runtime.serving,
            gateway=gateway,
        )

        try:
            method = getattr(stack.service, op.backend_method, None)
            if method is None:
                msg = f"Backend method not found: {op.backend_method}"
                raise ValueError(msg)

            result = method(**params)

            # Serialize result to dictionary
            if hasattr(result, "model_dump"):
                return result.model_dump(mode="json")
            if hasattr(result, "__dict__"):
                return dict(result.__dict__)
            return {"result": result}
        finally:
            stack.close()


__all__ = [
    "LazyServingProvider",
    "LazyStorageProvider",
]
