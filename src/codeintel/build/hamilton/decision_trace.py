"""Decision trace utilities for Hamilton build runs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict, cast

from codeintel.build.hamilton.impl_kind import ImplKind
from codeintel.build.hamilton.planner import (
    HamiltonBuildPlan,
    PlanReason,
    PlanStatus,
)

DECISION_TRACE_TARGET_NAME = "decision_trace"
DECISION_TRACE_ARTIFACT_NAME = "build_decision_trace"
DECISION_TRACE_PATH_TEMPLATE = "{build_dir}/decision_trace.json"

if TYPE_CHECKING:
    from collections.abc import Mapping


class DecisionTracePayload(TypedDict):
    """Typed payload for decision trace JSON entries."""

    index: int
    target: str
    node: str
    module: str
    status: PlanStatus
    reason: PlanReason
    input_hash: str | None
    options_hash: str | None
    prior_input_hash: str | None
    dependencies: list[str]
    table_keys: list[str]
    artifact_keys: list[str]
    dep_hashes: dict[str, str]
    prior_dep_hashes: dict[str, str]
    impl_kind: ImplKind


@dataclass(frozen=True, slots=True)
class DecisionTraceRecord:
    """Serializable decision record for a planned target."""

    index: int
    target: str
    node: str
    module: str
    status: PlanStatus
    reason: PlanReason
    input_hash: str | None
    options_hash: str | None
    prior_input_hash: str | None
    dependencies: tuple[str, ...]
    table_keys: tuple[str, ...]
    artifact_keys: tuple[str, ...]
    dep_hashes: dict[str, str]
    prior_dep_hashes: dict[str, str]
    impl_kind: ImplKind

    def to_dict(self) -> DecisionTracePayload:
        """Return a JSON-ready dictionary.

        Returns
        -------
        dict[str, object]
            JSON-serializable mapping for this record.
        """
        return {
            "index": self.index,
            "target": self.target,
            "node": self.node,
            "module": self.module,
            "status": self.status,
            "reason": self.reason,
            "input_hash": self.input_hash,
            "options_hash": self.options_hash,
            "prior_input_hash": self.prior_input_hash,
            "dependencies": list(self.dependencies),
            "table_keys": list(self.table_keys),
            "artifact_keys": list(self.artifact_keys),
            "dep_hashes": dict(self.dep_hashes),
            "prior_dep_hashes": dict(self.prior_dep_hashes),
            "impl_kind": self.impl_kind,
        }


def build_decision_trace(plan: HamiltonBuildPlan) -> list[DecisionTraceRecord]:
    """Build decision trace records from a Hamilton plan.

    Returns
    -------
    list[DecisionTraceRecord]
        Ordered decision trace records for the plan entries.
    """
    records: list[DecisionTraceRecord] = []
    for idx, entry in enumerate(plan.entries):
        records.append(
            DecisionTraceRecord(
                index=idx,
                target=entry.target,
                node=entry.node,
                module=str(entry.module),
                status=entry.status,
                reason=entry.reason,
                input_hash=entry.input_hash,
                options_hash=entry.options_hash,
                prior_input_hash=entry.prior_input_hash,
                dependencies=entry.dependencies,
                table_keys=entry.table_keys,
                artifact_keys=entry.artifact_keys,
                dep_hashes=_sorted_mapping(entry.dep_hashes),
                prior_dep_hashes=_sorted_mapping(entry.prior_dep_hashes),
                impl_kind=entry.impl_kind,
            )
        )
    return records


def build_decision_trace_payload(plan: HamiltonBuildPlan) -> list[DecisionTracePayload]:
    """Return decision trace payload as a JSON-ready list.

    Returns
    -------
    list[dict[str, object]]
        JSON-serializable decision trace payload.
    """
    return [record.to_dict() for record in build_decision_trace(plan)]


def default_decision_trace_path(build_dir: Path) -> Path:
    """Return the default decision trace path within the build directory.

    Returns
    -------
    Path
        Filesystem path for the decision trace payload.
    """
    return build_dir / "decision_trace.json"


def write_decision_trace(path: Path, plan: HamiltonBuildPlan) -> None:
    """Write decision trace JSON to the provided path."""
    payload = build_decision_trace_payload(plan)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload_text = json.dumps(payload, indent=2)
    path.write_text(f"{payload_text}\n", encoding="utf-8")


def read_decision_trace(path: Path) -> list[DecisionTracePayload]:
    """Load decision trace JSON from disk.

    Returns
    -------
    list[dict[str, object]]
        Parsed decision trace payload.

    Raises
    ------
    TypeError
        If the payload is not a list.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        msg = f"Decision trace payload must be a list, got {type(data)}"
        raise TypeError(msg)
    return cast("list[DecisionTracePayload]", data)


def _sorted_mapping(mapping: Mapping[str, str]) -> dict[str, str]:
    return dict(sorted(mapping.items(), key=lambda item: item[0]))


__all__ = [
    "DECISION_TRACE_ARTIFACT_NAME",
    "DECISION_TRACE_PATH_TEMPLATE",
    "DECISION_TRACE_TARGET_NAME",
    "DecisionTracePayload",
    "DecisionTraceRecord",
    "build_decision_trace",
    "build_decision_trace_payload",
    "default_decision_trace_path",
    "read_decision_trace",
    "write_decision_trace",
]
