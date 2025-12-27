"""Typed planning data model for DAG-native plan outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PLAN_SCHEMA_VERSION: str = "v1"

PlanMode = Literal["predict", "audit"]
PlanCacheStatus = Literal["hit", "miss", "unknown"]
PlanPredictedAction = Literal["compute", "reuse", "blocked"]


@dataclass(frozen=True, slots=True)
class PlanRequest:
    """Plan request parameters injected into the planning DAG."""

    requested_targets: tuple[str, ...]
    mode: PlanMode
    include_node_details: bool
    include_io_details: bool
    include_cache_details: bool

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dictionary representation.

        Returns
        -------
        dict[str, object]
            JSON-ready payload for the plan request.
        """
        return {
            "requested_targets": list(self.requested_targets),
            "mode": self.mode,
            "include_node_details": self.include_node_details,
            "include_io_details": self.include_io_details,
            "include_cache_details": self.include_cache_details,
        }


@dataclass(frozen=True, slots=True)
class PlanNodeStat:
    """Per-node cache probe status for planning introspection."""

    node: str
    version: str
    cache_status: PlanCacheStatus

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dictionary representation.

        Returns
        -------
        dict[str, object]
            JSON-ready payload for the node stat.
        """
        return {
            "node": self.node,
            "version": self.version,
            "cache_status": self.cache_status,
        }


@dataclass(frozen=True, slots=True)
class PlanTargetEntry:
    """Plan entry describing a single target in the dependency closure."""

    target: str
    domain: str
    deps: tuple[str, ...]
    reads: tuple[str, ...]
    writes_tables: tuple[str, ...]
    writes_artifacts: tuple[str, ...]
    predicted_action: PlanPredictedAction
    block_reasons: tuple[str, ...]
    cache_hit_ratio: float | None
    miss_nodes: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dictionary representation.

        Returns
        -------
        dict[str, object]
            JSON-ready payload for the plan entry.
        """
        payload: dict[str, object] = {
            "target": self.target,
            "domain": self.domain,
            "deps": list(self.deps),
            "reads": list(self.reads),
            "writes_tables": list(self.writes_tables),
            "writes_artifacts": list(self.writes_artifacts),
            "predicted_action": self.predicted_action,
            "block_reasons": list(self.block_reasons),
            "miss_nodes": list(self.miss_nodes),
        }
        if self.cache_hit_ratio is not None:
            payload["cache_hit_ratio"] = self.cache_hit_ratio
        return payload


@dataclass(frozen=True, slots=True)
class BuildPlan:
    """Plan output emitted by the planning DAG."""

    request: PlanRequest
    closure: tuple[str, ...]
    entries: tuple[PlanTargetEntry, ...]
    created_at_utc: str
    build_fingerprint: str

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-ready dictionary representation.

        Returns
        -------
        dict[str, object]
            JSON-ready payload for the build plan.
        """
        return {
            "plan_schema_version": PLAN_SCHEMA_VERSION,
            "request": self.request.to_dict(),
            "closure": list(self.closure),
            "entries": [entry.to_dict() for entry in self.entries],
            "created_at_utc": self.created_at_utc,
            "build_fingerprint": self.build_fingerprint,
        }


__all__ = [
    "PLAN_SCHEMA_VERSION",
    "BuildPlan",
    "PlanCacheStatus",
    "PlanMode",
    "PlanNodeStat",
    "PlanPredictedAction",
    "PlanRequest",
    "PlanTargetEntry",
]
