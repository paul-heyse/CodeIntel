"""Response conversion utilities for domain/transport model interop.

This module provides a unified conversion pattern for transforming raw responses
from the query layer into domain models. It eliminates the repeated 4-line
conversion pattern that was duplicated across all service methods.

Usage
-----
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

See Also
--------
- ``codeintel.serving.domain_models`` : Domain model definitions
- ``codeintel.serving.mcp.models`` : Transport (Pydantic) response models
"""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

D = TypeVar("D")  # Domain model type
D_co = TypeVar("D_co", covariant=True)  # Covariant domain model type for Protocol


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


__all__ = ["HasToDomain", "to_domain_result"]
