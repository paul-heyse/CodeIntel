"""Reusable helpers for tool-backed native Hamilton targets."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar, cast

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import record_from_materializations
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_artifact_materializations_collector,
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.patterns.savers import save_artifact, save_rows
from codeintel.build.hamilton.native.patterns.specs import ToolTargetSpec
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.native.tool_results import HasExecutionResult, ToolStepOutput
from codeintel.build.hamilton.nodes.module_attach import attach_node
from codeintel.build.hamilton.nodes.signature_tools import set_signature
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_compute, tag_tool
from codeintel.build.hashing import InputHashOptions
from codeintel.build.targets import TargetGraph
from codeintel.core.errors import CodeIntelError

if TYPE_CHECKING:
    from codeintel.build.hamilton.native.patterns.specs import ArtifactOutputSpec, TableOutputSpec

TPayload = TypeVar("TPayload")
TRowsByTable = Mapping[str, tuple[tuple[object, ...], ...]]

_RECOVERABLE_EXCEPTIONS = (
    ValueError,
    TypeError,
    KeyError,
    RuntimeError,
    OSError,
    CodeIntelError,
)


@dataclass(frozen=True, slots=True)
class IngestStep(Generic[TPayload]):
    """Standard ingest step wrapper for tool-backed targets."""

    result: ExecutionResult
    payload: TPayload | None = None


def run_tool_step(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    run: Callable[[], ToolStepOutput],
    hash_options: InputHashOptions | None = None,
    skip_reason: str | None = None,
) -> ToolStepOutput:
    """Execute a tool step with manifest-based skip handling."""
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        target_name,
        hash_options=hash_options,
    )
    if executor.should_skip():
        return ToolStepOutput(
            result=ExecutionResult.skip(skip_reason or f"{target_name} target skipped"),
        )
    try:
        output = run()
    except _RECOVERABLE_EXCEPTIONS as exc:
        return ToolStepOutput(result=ExecutionResult.failed(str(exc)))
    if not isinstance(output, ToolStepOutput):
        msg = f"{target_name} tool step returned invalid result: {type(output)}"
        return ToolStepOutput(result=ExecutionResult.failed(msg))
    return output


def run_tool_and_ingest(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    run: Callable[[], ToolStepOutput],
    ingest: Callable[[ToolStepOutput], IngestStep[TPayload]],
    hash_options: InputHashOptions | None = None,
    skip_reason: str | None = None,
) -> tuple[ToolStepOutput, IngestStep[TPayload]]:
    """Execute tool and ingest steps with consistent skip/error handling."""
    tool_output = run_tool_step(
        env=env,
        graph=graph,
        target_name=target_name,
        run=run,
        hash_options=hash_options,
        skip_reason=skip_reason,
    )
    if tool_output.result.skipped:
        return tool_output, IngestStep(result=ExecutionResult.skip("Tool step skipped"))
    if not tool_output.result.success:
        error = tool_output.result.error or f"{target_name} tool step failed"
        return tool_output, IngestStep(result=ExecutionResult.failed(error))
    try:
        ingest_output = ingest(tool_output)
    except _RECOVERABLE_EXCEPTIONS as exc:
        return tool_output, IngestStep(result=ExecutionResult.failed(str(exc)))
    if not isinstance(ingest_output, IngestStep):
        msg = f"{target_name} ingest step returned invalid result: {type(ingest_output)}"
        return tool_output, IngestStep(result=ExecutionResult.failed(msg))
    return tool_output, ingest_output


def finalize_target_from_materializations(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    tool_step: HasExecutionResult | None,
    ingest_step: HasExecutionResult | None,
    artifact_materializations: Mapping[str, MaterializationMetadata] | None,
    table_materializations: Mapping[str, MaterializationMetadata] | None,
    change_delta: Mapping[str, object] | None = None,
    hash_options: InputHashOptions | None = None,
) -> TargetRunRecord:
    """Finalize a target from saver metadata with standard failure gating."""
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        target_name,
        hash_options=hash_options,
    )
    if tool_step is not None and not tool_step.result.success:
        message = tool_step.result.error or f"{target_name} tool step failed"
        return executor.fail(RuntimeError(message))
    if ingest_step is not None:
        if not ingest_step.result.success and not ingest_step.result.skipped:
            message = ingest_step.result.error or f"{target_name} ingest step failed"
            return executor.fail(RuntimeError(message))

    return record_from_materializations(
        env=env,
        graph=graph,
        target_name=target_name,
        artifact_materializations=artifact_materializations,
        table_materializations=table_materializations,
        change_delta=change_delta,
    )


def attach_tool_target_template(
    module: object,
    *,
    spec: ToolTargetSpec,
    run_fn: Callable[..., ToolStepOutput],
    ingest_fn: Callable[..., IngestStep[TRowsByTable]] | None = None,
    hash_options_node: str | None = None,
) -> None:
    """Attach a tool-backed target scaffold to a module.

    This helper generates run/ingest nodes, per-output saver nodes, collectors,
    and the final target anchor using the provided spec.
    """
    if spec.tables and ingest_fn is None:
        msg = f"Tool target {spec.target_name} requires ingest_fn for table outputs"
        raise ValueError(msg)

    run_node = f"t__{spec.target_name}__run"
    ingest_node = f"t__{spec.target_name}__ingest"
    artifact_collector_node = f"{spec.target_name}__materializations"
    table_collector_node = f"{spec.target_name}__table_materializations"

    tagged_run = tag_tool(
        domain=spec.domain,
        target=spec.target_name,
        extra_tags=spec.tool_tags,
    )(run_fn)
    attach_node(module, node_name=run_node, fn=tagged_run)

    if ingest_fn is not None:
        tagged_ingest = tag_compute(domain=spec.domain, target=spec.target_name)(ingest_fn)
        attach_node(module, node_name=ingest_node, fn=tagged_ingest)

    for artifact_spec in spec.artifacts:
        _attach_artifact_node(
            module,
            artifact_spec=artifact_spec,
            domain=spec.domain,
            target_name=spec.target_name,
            run_node=run_node,
            hash_options_node=hash_options_node,
        )

    if ingest_fn is not None and spec.tables:
        for table_spec in spec.tables:
            _attach_table_rows_node(
                module,
                table_spec=table_spec,
                domain=spec.domain,
                target_name=spec.target_name,
                ingest_node=ingest_node,
                hash_options_node=hash_options_node,
            )

    artifact_collector = make_artifact_materializations_collector(
        domain=spec.domain,
        target=spec.target_name,
        artifacts=[artifact.name for artifact in spec.artifacts],
        node_name=artifact_collector_node,
    )
    attach_node(module, node_name=artifact_collector_node, fn=artifact_collector)

    table_collector = make_table_materializations_collector(
        domain=spec.domain,
        target=spec.target_name,
        table_keys=[table.table_key for table in spec.tables],
        node_name=table_collector_node,
    )
    attach_node(module, node_name=table_collector_node, fn=table_collector)

    anchor = _build_anchor(
        spec=spec,
        run_node=run_node,
        ingest_node=ingest_node if ingest_fn is not None else None,
        artifact_collector_node=artifact_collector_node,
        table_collector_node=table_collector_node,
        hash_options_node=hash_options_node,
    )
    attach_node(module, node_name=f"t__{spec.target_name}", fn=anchor)


def _attach_artifact_node(
    module: object,
    *,
    artifact_spec: ArtifactOutputSpec,
    domain: str,
    target_name: str,
    run_node: str,
    hash_options_node: str | None,
) -> None:
    def artifact_fn(**kwargs: object) -> Path | None:
        run_result = kwargs.get(run_node)
        if not isinstance(run_result, ToolStepOutput):
            msg = f"Expected ToolStepOutput for {run_node}, got {type(run_result)}"
            raise TypeError(msg)
        if run_result.result.skipped or not run_result.result.success:
            return None
        return run_result.path_for(artifact_spec.name)

    signature = inspect.Signature(
        [
            inspect.Parameter(
                run_node,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=ToolStepOutput,
            )
        ],
        return_annotation=Path | None,
    )
    artifact_fn = set_signature(artifact_fn, signature)
    artifact_fn.__name__ = f"{target_name}__{artifact_spec.name}_artifact"
    decorator = save_artifact(
        domain=domain,
        target=target_name,
        artifact_name=artifact_spec.name,
        path_template=artifact_spec.path_template,
        output_role=artifact_spec.output_role,
        hash_options_node=hash_options_node,
    )
    attach_node(module, node_name=artifact_fn.__name__, fn=decorator(artifact_fn))


def _attach_table_rows_node(
    module: object,
    *,
    table_spec: TableOutputSpec,
    domain: str,
    target_name: str,
    ingest_node: str,
    hash_options_node: str | None,
) -> None:
    def rows_fn(**kwargs: object) -> tuple[tuple[object, ...], ...] | None:
        ingest_result = kwargs.get(ingest_node)
        if not isinstance(ingest_result, IngestStep):
            msg = f"Expected IngestStep for {ingest_node}, got {type(ingest_result)}"
            raise TypeError(msg)
        if ingest_result.result.skipped or not ingest_result.result.success:
            return None
        payload = ingest_result.payload
        if payload is None:
            msg = f"Missing ingest payload for {table_spec.table_key}"
            raise ValueError(msg)
        rows = payload.get(table_spec.table_key)
        if rows is None:
            msg = f"Missing rows for {table_spec.table_key}"
            raise ValueError(msg)
        return rows

    signature = inspect.Signature(
        [
            inspect.Parameter(
                ingest_node,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=IngestStep[TRowsByTable],
            )
        ],
        return_annotation=tuple[tuple[object, ...], ...] | None,
    )
    rows_fn = set_signature(rows_fn, signature)
    node_name = f"{target_name}__{table_spec.table_key.replace('.', '__')}_rows"
    rows_fn.__name__ = node_name
    decorator = save_rows(
        domain=domain,
        target=target_name,
        table_key=table_spec.table_key,
        columns=table_spec.columns,
        output_role=table_spec.output_role,
        hash_options_node=hash_options_node,
    )
    attach_node(module, node_name=node_name, fn=decorator(rows_fn))


def _build_anchor(
    *,
    spec: ToolTargetSpec,
    run_node: str,
    ingest_node: str | None,
    artifact_collector_node: str,
    table_collector_node: str,
    hash_options_node: str | None,
) -> Callable[..., TargetRunRecord]:
    def anchor_fn(**kwargs: object) -> TargetRunRecord:
        env = kwargs.get("env")
        graph = kwargs.get("graph")
        if not isinstance(env, BuildEnv):
            msg = "Missing BuildEnv for target anchor"
            raise TypeError(msg)
        if not isinstance(graph, TargetGraph):
            msg = "Missing TargetGraph for target anchor"
            raise TypeError(msg)
        tool_step = kwargs.get(run_node)
        ingest_step = kwargs.get(ingest_node) if ingest_node is not None else None
        artifact_materializations = cast(
            "Mapping[str, MaterializationMetadata]",
            kwargs.get(artifact_collector_node),
        )
        table_materializations = cast(
            "Mapping[str, MaterializationMetadata]",
            kwargs.get(table_collector_node),
        )
        hash_options = None
        if hash_options_node is not None:
            hash_options = cast("InputHashOptions | None", kwargs.get(hash_options_node))
        return finalize_target_from_materializations(
            env=env,
            graph=graph,
            target_name=spec.target_name,
            tool_step=cast("HasExecutionResult | None", tool_step),
            ingest_step=cast("HasExecutionResult | None", ingest_step),
            artifact_materializations=artifact_materializations,
            table_materializations=table_materializations,
            hash_options=hash_options,
        )

    params = [
        inspect.Parameter(
            "env",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=BuildEnv,
        ),
        inspect.Parameter(
            "graph",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=TargetGraph,
        ),
        inspect.Parameter(
            run_node,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=ToolStepOutput,
        ),
    ]
    if ingest_node is not None:
        params.append(
            inspect.Parameter(
                ingest_node,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=IngestStep[TRowsByTable],
            )
        )
    params.append(
        inspect.Parameter(
            artifact_collector_node,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=dict[str, MaterializationMetadata],
        )
    )
    params.append(
        inspect.Parameter(
            table_collector_node,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=dict[str, MaterializationMetadata],
        )
    )
    if hash_options_node is not None:
        params.append(
            inspect.Parameter(
                hash_options_node,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=InputHashOptions,
            )
        )

    anchor_fn = set_signature(
        anchor_fn,
        inspect.Signature(params, return_annotation=TargetRunRecord),
    )
    anchor_fn.__name__ = f"t__{spec.target_name}"
    return codeintel_target(
        domain=spec.domain,
        target=spec.target_name,
        spec=spec.spec,
    )(anchor_fn)


__all__ = [
    "IngestStep",
    "attach_tool_target_template",
    "finalize_target_from_materializations",
    "run_tool_and_ingest",
    "run_tool_step",
]
