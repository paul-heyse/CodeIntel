"""Response conversion utilities for domain/transport model interop.

This module provides unified conversion patterns for transforming between
domain models and transport (Pydantic response) models. It eliminates the
repeated conversion patterns that were duplicated across service and backend
methods.

Two Conversion Directions
-------------------------
1. **to_domain_result**: raw response → domain model (used by service layer)
2. **to_response_result**: domain model → response model (used by MCP backend)

Usage - Service Layer (raw → domain)
------------------------------------
Instead of:

    raw_resp = self._call("method_name", lambda: self.query.functions.method(...))
    if isinstance(raw_resp, dm.FunctionSummaryResult):
        return raw_resp
    if isinstance(raw_resp, FunctionSummaryResponse):
        return raw_resp.to_domain()
    return FunctionSummaryResponse.model_validate(raw_resp).to_domain()

Use:

    raw = self._call("method_name", lambda: self.query.functions.method(...))
    return to_domain_result(raw, dm.FunctionSummaryResult, FunctionSummaryResponse)

Usage - MCP Backend Layer (domain → response)
---------------------------------------------
Instead of:

    result = self.service.get_function_summary(...)
    if isinstance(result, FunctionSummaryResponse):
        return result
    return FunctionSummaryResponse.from_domain(result)

Use:

    result = self.service.get_function_summary(...)
    return to_response_result(result, FunctionSummaryResponse)

See Also
--------
- ``codeintel.serving.domain_models`` : Domain model definitions
- ``codeintel.serving.mcp.models`` : Transport (Pydantic) response models
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

D = TypeVar("D")  # Domain model type
D_co = TypeVar("D_co", covariant=True)  # Covariant domain model type for Protocol
R_co = TypeVar("R_co", covariant=True)  # Covariant response model type for Protocol


@runtime_checkable
class HasToDomain(Protocol[D_co]):
    """Protocol for response models with to_domain() method."""

    def to_domain(self) -> D_co:
        """
        Convert response model to domain model.

        Returns
        -------
        D_co
            Domain model instance.
        """
        ...

    @classmethod
    def model_validate(cls, obj: object) -> HasToDomain[D_co]:
        """
        Validate and construct from arbitrary object.

        Parameters
        ----------
        obj
            Raw object to validate.

        Returns
        -------
        HasToDomain[D_co]
            Validated response model instance.
        """
        ...


@runtime_checkable
class HasFromDomain(Protocol[R_co]):
    """Protocol for response models with from_domain() class method.

    This is the inverse of ``HasToDomain`` - used when converting domain models
    to transport (Pydantic) response models in the MCP backend layer.

    Example
    -------
    All MCP response models implement this protocol::

        class FunctionSummaryResponse(BaseModel):
            @classmethod
            def from_domain(cls, domain: dm.FunctionSummaryResult) -> Self:
                return cls(found=domain.found, summary=domain.summary, ...)
    """

    @classmethod
    def from_domain(cls, domain: object) -> R_co:
        """
        Create response model from domain model.

        Parameters
        ----------
        domain
            Domain model instance to convert.

        Returns
        -------
        R_co
            Response model instance.
        """
        ...


def to_domain_result(
    raw: object,
    domain_type: type[D],
    response_type: type[HasToDomain[D]],
) -> D:
    """
    Convert raw response to domain model with type coercion.

    This function handles three cases:

    1. ``raw`` is already the domain type → return as-is
    2. ``raw`` is the response type → call ``to_domain()``
    3. ``raw`` is dict/other → validate as response, then ``to_domain()``

    Parameters
    ----------
    raw
        Raw response from query layer or HTTP transport.
    domain_type
        Expected domain model type (e.g., ``dm.FunctionSummaryResult``).
    response_type
        Pydantic response model type (e.g., ``FunctionSummaryResponse``).

    Returns
    -------
    D
        Domain model instance.

    Examples
    --------
    >>> from codeintel.serving import domain_models as dm
    >>> from codeintel.serving.mcp.models import FunctionSummaryResponse
    >>> raw = {"found": True, "summary": {"urn": "..."}, "meta": {}}
    >>> result = to_domain_result(raw, dm.FunctionSummaryResult, FunctionSummaryResponse)
    >>> isinstance(result, dm.FunctionSummaryResult)
    True
    """
    if isinstance(raw, domain_type):
        return raw
    if isinstance(raw, response_type):
        return raw.to_domain()
    return response_type.model_validate(raw).to_domain()


def to_response_result[R](
    raw: object,
    response_type: type[R],
) -> R:
    """
    Convert domain model to response model with type coercion.

    This function handles two cases:

    1. ``raw`` is already the response type → return as-is
    2. ``raw`` is a domain model → call ``from_domain()``

    This is the inverse of ``to_domain_result()`` and is used by MCP backend
    methods to ensure consistent response serialization.

    Parameters
    ----------
    raw
        Raw result from service layer (domain model or already-converted response).
    response_type
        Expected Pydantic response model type (e.g., ``FunctionSummaryResponse``).

    Returns
    -------
    R
        Response model instance.

    Examples
    --------
    >>> from codeintel.serving import domain_models as dm
    >>> from codeintel.serving.mcp.models import FunctionSummaryResponse
    >>> domain = dm.FunctionSummaryResult(found=True, summary=None, meta=dm.ResponseMeta())
    >>> response = to_response_result(domain, FunctionSummaryResponse)
    >>> isinstance(response, FunctionSummaryResponse)
    True
    """
    if isinstance(raw, response_type):
        return raw
    # response_type is expected to have from_domain() at runtime - all MCP response models do
    return response_type.from_domain(raw)  # type: ignore[attr-defined]


__all__ = ["HasFromDomain", "HasToDomain", "to_domain_result", "to_response_result"]
