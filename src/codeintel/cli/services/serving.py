"""Serving operation invocation service.

Consolidate the 3 near-identical serving invocation implementations:
- ``handlers/context.py:invoke_serving_operation``
- ``deps/providers.py:LazyServingProvider.invoke``
- ``commands/ops.py:_invoke_operation_for_result``

Provide a single, unified implementation for invoking serving operations.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

from codeintel.serving.auto_pipeline import run_operation_prereqs
from codeintel.serving.bootstrap import build_service_stack
from codeintel.serving.operations.catalog import get_operation

if TYPE_CHECKING:
    from codeintel.cli.services.runtime import RuntimeService
    from codeintel.cli.services.storage import StorageService

LOG = logging.getLogger(__name__)


class ServingError(Exception):
    """Error during serving operation invocation.

    Parameters
    ----------
    operation_id
        The operation that failed.
    message
        Error message.
    """

    def __init__(self, operation_id: str, message: str) -> None:
        """Initialize serving error."""
        super().__init__(message)
        self.operation_id = operation_id


class ServingService:
    """Serving layer operation invocation.

    Provide unified access to the serving operation catalog with proper
    prerequisite execution and result serialization.

    Parameters
    ----------
    runtime
        RuntimeService for configuration.
    storage
        StorageService for gateway access.

    Examples
    --------
    >>> service = ServingService(runtime_svc, storage_svc)  # doctest: +SKIP
    >>> result = service.invoke("function.summary", {"goid_h128": "abc123"})
    """

    def __init__(
        self,
        runtime: RuntimeService,
        storage: StorageService,
    ) -> None:
        """Initialize serving service."""
        self._runtime = runtime
        self._storage = storage

    @property
    def runtime_service(self) -> RuntimeService:
        """Return the underlying RuntimeService."""
        return self._runtime

    @property
    def storage_service(self) -> StorageService:
        """Return the underlying StorageService."""
        return self._storage

    def invoke(
        self,
        operation_id: str,
        params: dict[str, object],
        *,
        skip_prereqs: bool = False,
    ) -> dict[str, Any]:
        """Invoke a serving operation.

        Execute a serving operation through the unified stack:
        1. Look up operation in catalog
        2. Run prerequisites (unless skipped)
        3. Build service stack
        4. Invoke backend method
        5. Serialize result

        Parameters
        ----------
        operation_id
            Operation ID from the serving catalog.
        params
            Operation parameters.
        skip_prereqs
            If True, skip prerequisite pipeline execution.

        Returns
        -------
        dict[str, Any]
            Operation result as dictionary.

        Raises
        ------
        ServingError
            If operation not found or invocation fails.

        Examples
        --------
        >>> result = service.invoke(  # doctest: +SKIP
        ...     "function.summary",
        ...     {"goid_h128": "abc123"},
        ... )
        """
        # Look up operation
        op = get_operation(operation_id)
        if op is None:
            msg = f"Unknown serving operation: {operation_id}"
            raise ServingError(operation_id, msg)

        runtime = self._runtime.runtime
        gateway = self._storage.gateway

        # Run prerequisites if needed
        if not skip_prereqs:
            LOG.debug("Running prerequisites for operation: %s", operation_id)
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
                raise ServingError(operation_id, msg)

            LOG.debug("Invoking %s.%s with params: %s", operation_id, op.backend_method, params)
            result = method(**params)

            # Serialize result to dictionary
            return self._serialize_result(result)

        finally:
            stack.close()

    def invoke_batch(
        self,
        operation_id: str,
        params_list: list[dict[str, object]],
        *,
        skip_prereqs: bool = False,
    ) -> list[dict[str, Any]]:
        """Invoke a serving operation for multiple parameter sets.

        Parameters
        ----------
        operation_id
            Operation ID from the serving catalog.
        params_list
            List of parameter dictionaries.
        skip_prereqs
            If True, skip prerequisite pipeline execution.

        Returns
        -------
        list[dict[str, Any]]
            List of operation results.
        """
        results: list[dict[str, Any]] = []
        for params in params_list:
            try:
                result = self.invoke(operation_id, params, skip_prereqs=skip_prereqs)
                results.append({"success": True, "result": result, "input": params})
            except (ServingError, ValueError, TypeError) as exc:
                results.append({"success": False, "error": str(exc), "input": params})
        return results

    @staticmethod
    def _serialize_result(result: object) -> dict[str, Any]:
        """Serialize operation result to dictionary.

        Parameters
        ----------
        result
            Raw result from backend method.

        Returns
        -------
        dict[str, Any]
            Serialized result.
        """
        # Pydantic model
        model_dump = getattr(result, "model_dump", None)
        if callable(model_dump):
            # Pydantic's model_dump returns dict[str, Any]
            return cast("dict[str, Any]", model_dump(mode="json"))

        # Dataclass or object with __dict__
        if hasattr(result, "__dict__") and not isinstance(result, type):
            return dict(result.__dict__)

        # Primitive value
        return {"value": result}

    @staticmethod
    def serialize_result(result: object) -> dict[str, Any]:
        """Public wrapper for result serialization."""
        return ServingService._serialize_result(result)


__all__ = [
    "ServingError",
    "ServingService",
]
