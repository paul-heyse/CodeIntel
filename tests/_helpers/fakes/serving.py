"""Fake serving/MCP scope fixtures for testing.

This module provides fake implementations for serving layer tests,
including scope recording stubs for query service testing.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.backend.pagination import BackendLimits
from codeintel.serving.backend.query_api import DuckDBQueryApi
from codeintel.serving.mcp.models import FunctionSummaryResponse
from codeintel.serving.services.query_service import LocalQueryService, ResponseMeta
from codeintel.storage.gateway import open_memory_gateway


@dataclass
class ScopeRecordingQuery:
    """Stub query that records scopes and optionally delegates to a callable.

    This class consolidates scope recording and delegation functionality:
    - Always records scopes passed to ``get_function_summary``
    - Optionally delegates to a provided callable for custom responses
    - Without a delegate, returns a minimal stub response
    """

    scopes: list[GraphRunScope | None] = field(default_factory=list)
    delegate: Callable[..., object] | None = field(default=None)

    def __post_init__(self) -> None:
        """Set up fake gateway and query API attributes."""
        self.gateway = open_memory_gateway(apply_schema=True, ensure_macros=True)
        self.repo = self.gateway.config.repo
        self.commit = self.gateway.config.commit
        self.limits = BackendLimits()
        self.graph_engine: Any | None = None
        self.functions = self
        self.modules = self
        self.subsystems = self
        self.datasets = self

    def get_function_summary(
        self,
        *,
        urn: str | None = None,
        goid_h128: int | None = None,
        rel_path: str | None = None,
        qualname: str | None = None,
        scope: GraphRunScope | None = None,
    ) -> dict[str, object] | FunctionSummaryResponse:
        """
        Record scope and return stub or delegated function summary.

        Parameters
        ----------
        urn
            Function URN.
        goid_h128
            GOID hash.
        rel_path
            Relative path.
        qualname
            Qualified name.
        scope
            Graph run scope to record.

        Returns
        -------
        dict[str, object] | FunctionSummaryResponse
            Stub response or validated response from delegate.
        """
        self.scopes.append(scope)

        if self.delegate is not None:
            result = self.delegate(
                urn=urn,
                goid_h128=goid_h128,
                rel_path=rel_path,
                qualname=qualname,
                scope=scope,
            )
            if isinstance(result, FunctionSummaryResponse):
                return result
            return FunctionSummaryResponse.model_validate(result)

        return {
            "found": True,
            "summary": None,
            "meta": ResponseMeta().model_dump(),
        }


@dataclass
class ServingScopePack:
    """Pack containing a scope-recording query and bound LocalQueryService."""

    query: ScopeRecordingQuery
    service: LocalQueryService


def build_serving_scope_pack() -> ServingScopePack:
    """
    Build a serving scope pack with recording query and LocalQueryService.

    Returns
    -------
    ServingScopePack
        Pack containing the query stub and service ready for tests.
    """
    stub = ScopeRecordingQuery()
    service = LocalQueryService(query=cast("DuckDBQueryApi", stub))
    return ServingScopePack(query=stub, service=service)


__all__ = ["ScopeRecordingQuery", "ServingScopePack", "build_serving_scope_pack"]
