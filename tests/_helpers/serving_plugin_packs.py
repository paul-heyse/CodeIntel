"""Reusable serving/MCP scope fixture packs."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import cast

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.backend.pagination import BackendLimits
from codeintel.serving.backend.query_api import DuckDBQueryApi
from codeintel.serving.services.query_service import LocalQueryService, ResponseMeta
from codeintel.storage.gateway import StorageGateway


@dataclass
class ScopeRecordingQuery:
    """Stub query that records scopes passed into service calls."""

    scopes: list[GraphRunScope | None] = field(default_factory=list)

    def __init__(self) -> None:
        self.scopes = []
        self.gateway = cast(
            "StorageGateway", SimpleNamespace(datasets=SimpleNamespace(mapping={}), con=None)
        )
        self.limits = BackendLimits()
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
    ) -> dict[str, object]:
        _ = (urn, goid_h128, rel_path, qualname)
        self.scopes.append(scope)
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
