"""Native Hamilton target for the build decision trace artifact."""

from __future__ import annotations

import json

from hamilton.function_modifiers import source, value

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.decision_trace import (
    DECISION_TRACE_ARTIFACT_NAME,
    DECISION_TRACE_PATH_TEMPLATE,
    DECISION_TRACE_TARGET_NAME,
    build_decision_trace_payload,
)
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.materialization_records import (
    FileArtifactRecordContext,
    record_from_file_artifact_materialization,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.planner import compute_plan
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute
from codeintel.build.targets import TargetGraph

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord)


def _requested_targets(env: BuildEnv) -> tuple[str, ...]:
    context = env.execution_context
    if context is None:
        return ()
    return context.run.requested_datasets


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{DECISION_TRACE_ARTIFACT_NAME}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(DECISION_TRACE_TARGET_NAME),
    artifact_name=value(DECISION_TRACE_ARTIFACT_NAME),
    path_template=value(DECISION_TRACE_PATH_TEMPLATE),
)
@tag_compute(domain="export", target=DECISION_TRACE_TARGET_NAME, target_="decision_trace__content")
def decision_trace__content(env: BuildEnv, graph: TargetGraph) -> str | None:
    """Materialize the build decision trace as JSON.

    Returns
    -------
    str | None
        JSON payload for the decision trace, or None when skipped.
    """
    requested = _requested_targets(env)
    if not requested:
        return None
    plan = compute_plan(env=env, graph=graph, requested=requested, graph_source="hamilton")
    payload = build_decision_trace_payload(plan)
    return f"{json.dumps(payload, indent=2)}\n"


@codeintel_target(domain="export", target=DECISION_TRACE_TARGET_NAME)
def t__decision_trace(
    env: BuildEnv,
    graph: TargetGraph,
    m__artifact__build_decision_trace: MaterializationMetadata,
) -> TargetRunRecord:
    """Persist the decision trace artifact and emit a target record.

    Returns
    -------
    TargetRunRecord
        Record describing the decision trace materialization.
    """
    context = FileArtifactRecordContext(
        env=env,
        graph=graph,
        target_name=DECISION_TRACE_TARGET_NAME,
    )
    return record_from_file_artifact_materialization(
        context=context,
        expected_artifact_name=DECISION_TRACE_ARTIFACT_NAME,
        materialization=m__artifact__build_decision_trace,
    )


__all__ = [
    "decision_trace__content",
    "t__decision_trace",
]
