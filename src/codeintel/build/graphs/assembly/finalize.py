"""Finalize helpers for graph assembly."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.graphs.engine.datasets import GraphRunMetadata, persist_finalize_artifacts
from codeintel.build.tabular.plan_ops import Plan
from codeintel.core.columnar.arrowdsl import ExecutionPlan, run_pipeline
from codeintel.core.columnar.finalize_ops import FinalizeResult, finalize_spec_for_table

if TYPE_CHECKING:
    from codeintel.core.columnar.dedupe_ops import DedupeTier
    from codeintel.core.columnar.execution_context import ExecutionContext


@dataclass(frozen=True, slots=True)
class GraphFinalizeArtifacts:
    """Finalize artifact persistence configuration."""

    dataset_root: Path
    snapshot_id: str
    run_metadata: GraphRunMetadata | None = None


def finalize_graph_plan(
    plan: Plan,
    *,
    table_key: str,
    determinism: DedupeTier,
    ctx: ExecutionContext | None,
    artifacts: GraphFinalizeArtifacts | None = None,
) -> FinalizeResult:
    """Run a graph plan through finalize and optionally persist artifacts."""
    result = run_pipeline(
        plan=ExecutionPlan.from_plan(plan, determinism=determinism),
        finalize=finalize_spec_for_table(
            table_key,
            mode="tolerant",
            determinism=determinism,
        ),
        ctx=ctx,
    )
    if artifacts is not None:
        persist_finalize_artifacts(
            dataset_root=artifacts.dataset_root,
            snapshot_id=artifacts.snapshot_id,
            base_table_key=table_key,
            result=result,
            run_metadata=artifacts.run_metadata,
        )
    return result


__all__ = ["GraphFinalizeArtifacts", "finalize_graph_plan"]
