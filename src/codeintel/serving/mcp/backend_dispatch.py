"""Backend dispatch mixin for unified method handling.

This module provides a mixin class that consolidates the repetitive dispatch
pattern used by both ``DuckDBBackend`` and ``HttpBackend`` in the MCP layer.

The Pattern
-----------
Before this mixin, each backend method followed this pattern:

**DuckDBBackend** (local):

    def get_function_summary(self, *, urn=None, goid_h128=None, ...):
        scope_payload = scope if isinstance(scope, GraphScopePayload) else None
        try:
            domain_result = self.service.get_function_summary(...)
        except ProblemError as exc:
            raise errors.McpError(exc.detail) from exc
        return FunctionSummaryResponse.from_domain(domain_result)

**HttpBackend** (remote):

    def get_function_summary(self, *, urn=None, goid_h128=None, ...):
        result = self.service.get_function_summary(
            scope=scope if isinstance(scope, GraphScopePayload) else None, ...
        )
        if isinstance(result, FunctionSummaryResponse):
            return result
        return FunctionSummaryResponse.from_domain(result)

With this mixin, both become:

    def get_function_summary(self, *, urn=None, goid_h128=None, ...):
        return self._dispatch(
            "get_function_summary",
            FunctionSummaryResponse,
            urn=urn, goid_h128=goid_h128, ..., scope=scope,
        )

See Also
--------
- ``codeintel.serving.services.conversion`` : Domain/response conversion utilities
- ``codeintel.serving.mcp.models.normalize_scope`` : Scope normalization
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, TypeVar

from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import normalize_scope
from codeintel.serving.services.conversion import to_response_result
from codeintel.serving.services.errors import ProblemError

if TYPE_CHECKING:
    from codeintel.serving.services.query_service import QueryService

R = TypeVar("R")


class BackendDispatchMixin(ABC):
    """Mixin providing common dispatch pattern for backend methods.

    This mixin consolidates the repetitive error handling and response conversion
    logic shared by ``DuckDBBackend`` and ``HttpBackend``. Subclasses must:

    1. Have a ``service`` attribute of type ``QueryService``
    2. Implement the ``is_local`` property

    The ``_dispatch()`` method then handles:

    - Scope normalization via ``normalize_scope()``
    - Error translation (``ProblemError`` → ``McpError``) for local backends
    - Response type coercion via ``to_response_result()``

    Example
    -------
    ::

        @dataclass
        class DuckDBBackend(BackendDispatchMixin, DatasetBackendMixin, QueryBackend):
            service: QueryService
            # ...

            @property
            def is_local(self) -> bool:
                return True

            def get_function_summary(self, *, urn=None, ...):
                _require_identifier(urn=urn, ...)
                return self._dispatch(
                    "get_function_summary",
                    FunctionSummaryResponse,
                    urn=urn, ..., scope=scope,
                )
    """

    if TYPE_CHECKING:
        # This is provided by the concrete dataclass, not by this mixin
        service: QueryService

    @property
    @abstractmethod
    def is_local(self) -> bool:
        """Return True if this is a local (DuckDB) backend.

        Returns
        -------
        bool
            True for DuckDBBackend, False for HttpBackend.
        """
        ...

    def _dispatch(
        self,
        method_name: str,
        response_type: type[R],
        **kwargs: object,
    ) -> R:
        """
        Dispatch a method call with error handling and response conversion.

        This method encapsulates the common dispatch pattern:

        1. Normalize ``scope`` parameter if present
        2. Call the corresponding service method
        3. Handle errors (for local backends)
        4. Convert result to response type

        Parameters
        ----------
        method_name
            Name of the method on ``self.service`` to call.
        response_type
            Pydantic response model type for conversion.
        **kwargs
            Keyword arguments to pass to the service method.
            If ``scope`` is present, it will be normalized.

        Returns
        -------
        R
            Response model instance.

        Raises
        ------
        errors.McpError
            When the underlying service reports a problem detail (local only).
        """
        # Normalize scope if present
        if "scope" in kwargs:
            kwargs["scope"] = normalize_scope(kwargs["scope"])

        method = getattr(self.service, method_name)

        if self.is_local:
            try:
                domain_result = method(**kwargs)
            except ProblemError as exc:
                raise errors.McpError(exc.detail) from exc
            # response_type is expected to have from_domain() - all MCP response models do
            return response_type.from_domain(domain_result)  # type: ignore[attr-defined]

        result = method(**kwargs)
        return to_response_result(result, response_type)


__all__ = ["BackendDispatchMixin"]
