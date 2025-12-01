"""Reusable serving/MCP scope fixture packs."""

from __future__ import annotations

from dataclasses import dataclass, field

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.serving.services.query_service import LocalQueryService, ResponseMeta


@dataclass
class ScopeRecordingQuery:
    """Stub query that records scopes passed into service calls."""

    scopes: list[GraphRunScope | None] = field(default_factory=list)

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
    query = ScopeRecordingQuery()
    service = LocalQueryService(query=query)
    return ServingScopePack(query=query, service=service)


__all__ = ["ScopeRecordingQuery", "ServingScopePack", "build_serving_scope_pack"]
