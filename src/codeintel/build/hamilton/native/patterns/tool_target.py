"""Reusable helpers for tool-backed native Hamilton targets."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, cast

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_artifact_materializations_collector,
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.patterns.savers import (
    ArtifactSaveSpec,
    SaverContext,
    TableSaveSpec,
    save_artifact,
    save_rows,
)
from codeintel.build.hamilton.native.patterns.specs import ToolTargetSpec
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.native.tool_results import HasExecutionResult, ToolStepOutput
from codeintel.build.hamilton.nodes.module_attach import tagged_attach_node
from codeintel.build.hamilton.nodes.signature_tools import set_signature
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tag_spec import TagSpec
from codeintel.build.hamilton.tagging import tag_compute, tag_tool
from codeintel.build.hashing import InputHashOptions
from codeintel.core.errors import CodeIntelError

if TYPE_CHECKING:
    from codeintel.build.hamilton.native.patterns.specs import ArtifactOutputSpec, TableOutputSpec

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
class ToolRunContext:
    """Execution context for tool step helpers."""

    env: BuildEnv
    catalog: DagCatalog
    target_name: str
    hash_options: InputHashOptions | None = None
    skip_reason: str | None = None


@dataclass(frozen=True, slots=True)
class ToolFinalizeContext:
    """Context for target finalization helpers."""

    env: BuildEnv
    catalog: DagCatalog
    target_name: str
    hash_options: InputHashOptions | None = None
    change_delta: Mapping[str, object] | None = None


@dataclass(frozen=True, slots=True)
class IngestStep[TPayload]:
    """Standard ingest step wrapper for tool-backed targets."""

    result: ExecutionResult
    payload: TPayload | None = None


@dataclass(frozen=True, slots=True)
class _TemplateContext:
    module: ModuleType
    domain: str
    target_name: str
    saver_context: SaverContext


@dataclass(frozen=True, slots=True)
class _AnchorInputs:
    spec: ToolTargetSpec
    run_node: str
    ingest_node: str | None
    artifact_collector_node: str
    table_collector_node: str
    hash_options_node: str | None


def run_tool_step(
    *,
    context: ToolRunContext,
    run: Callable[[], ToolStepOutput],
) -> ToolStepOutput:
    """Execute a tool step with manifest-based skip handling.

    Parameters
    ----------
    context
        Execution context for the tool step.
    run
        Callable that executes the tool step.

    Returns
    -------
    ToolStepOutput
        Output of the tool step, including skip or failure metadata.
    """
    executor = NativeTargetExecutor.for_target(
        context.env,
        context.catalog,
        context.target_name,
        hash_options=context.hash_options,
    )
    if executor.should_skip():
        return ToolStepOutput(
            result=ExecutionResult.skip(
                context.skip_reason or f"{context.target_name} target skipped"
            ),
        )
    try:
        output = run()
    except _RECOVERABLE_EXCEPTIONS as exc:
        return ToolStepOutput(result=ExecutionResult.failed(str(exc)))
    if not isinstance(output, ToolStepOutput):
        msg = f"{context.target_name} tool step returned invalid result: {type(output)}"
        return ToolStepOutput(result=ExecutionResult.failed(msg))
    return output


def run_tool_and_ingest[TPayload](
    *,
    context: ToolRunContext,
    run: Callable[[], ToolStepOutput],
    ingest: Callable[[ToolStepOutput], IngestStep[TPayload]],
) -> tuple[ToolStepOutput, IngestStep[TPayload]]:
    """Execute tool and ingest steps with consistent skip/error handling.

    Parameters
    ----------
    context
        Execution context for the tool step.
    run
        Callable that executes the tool step.
    ingest
        Callable that transforms the tool step output into ingest output.

    Returns
    -------
    tuple[ToolStepOutput, IngestStep[TPayload]]
        Tool step output and the ingest step result.
    """
    tool_output = run_tool_step(
        context=context,
        run=run,
    )
    if tool_output.result.skipped:
        return tool_output, IngestStep(result=ExecutionResult.skip("Tool step skipped"))
    if not tool_output.result.success:
        error = tool_output.result.error or f"{context.target_name} tool step failed"
        return tool_output, IngestStep(result=ExecutionResult.failed(error))
    try:
        ingest_output = ingest(tool_output)
    except _RECOVERABLE_EXCEPTIONS as exc:
        return tool_output, IngestStep(result=ExecutionResult.failed(str(exc)))
    if not isinstance(ingest_output, IngestStep):
        msg = f"{context.target_name} ingest step returned invalid result: {type(ingest_output)}"
        return tool_output, IngestStep(result=ExecutionResult.failed(msg))
    return tool_output, ingest_output


def finalize_target_from_materializations(
    *,
    context: ToolFinalizeContext,
    tool_step: HasExecutionResult | None,
    ingest_step: HasExecutionResult | None,
    artifact_materializations: Mapping[str, MaterializationResult] | None,
    table_materializations: Mapping[str, MaterializationResult] | None,
) -> TargetRunRecord:
    """Finalize a target from saver metadata with standard failure gating.

    Parameters
    ----------
    context
        Context containing build environment and target metadata.
    tool_step
        Tool step output used to gate finalization.
    ingest_step
        Ingest step output used to gate finalization.
    artifact_materializations
        Materialization results for artifact outputs.
    table_materializations
        Materialization results for table outputs.

    Returns
    -------
    TargetRunRecord
        Finalized target run record.
    """
    executor = NativeTargetExecutor.for_target(
        context.env,
        context.catalog,
        context.target_name,
        hash_options=context.hash_options,
    )
    if tool_step is not None and not tool_step.result.success:
        message = tool_step.result.error or f"{context.target_name} tool step failed"
        return executor.fail(RuntimeError(message))
    if (
        ingest_step is not None
        and not ingest_step.result.success
        and not ingest_step.result.skipped
    ):
        message = ingest_step.result.error or f"{context.target_name} ingest step failed"
        return executor.fail(RuntimeError(message))

    record_context = MaterializationRecordContext(
        env=context.env,
        catalog=context.catalog,
        target_name=context.target_name,
        change_delta=context.change_delta,
    )
    return record_from_materializations(
        context=record_context,
        artifact_materializations=artifact_materializations,
        table_materializations=table_materializations,
    )


def attach_tool_target_template(
    module: ModuleType,
    *,
    spec: ToolTargetSpec,
    run_fn: Callable[..., ToolStepOutput],
    ingest_fn: Callable[..., IngestStep[TRowsByTable]] | None = None,
    hash_options_node: str | None = None,
) -> None:
    """Attach a tool-backed target scaffold to a module.

    This helper generates run/ingest nodes, per-output saver nodes, collectors,
    and the final target anchor using the provided spec.

    Raises
    ------
    ValueError
        If table outputs are specified without an ingest function.
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
    tagged_attach_node(
        module,
        node_name=run_node,
        fn=tagged_run,
        tag_spec=TagSpec.for_tool(
            domain=spec.domain,
            target=spec.target_name,
            extra_tags=spec.tool_tags,
        ),
    )

    if ingest_fn is not None:
        tagged_ingest = tag_compute(domain=spec.domain, target=spec.target_name)(ingest_fn)
        tagged_attach_node(
            module,
            node_name=ingest_node,
            fn=tagged_ingest,
            tag_spec=TagSpec.for_compute(domain=spec.domain, target=spec.target_name),
        )

    saver_context = SaverContext(
        domain=spec.domain,
        target=spec.target_name,
        hash_options_node=hash_options_node,
    )
    template_context = _TemplateContext(
        module=module,
        domain=spec.domain,
        target_name=spec.target_name,
        saver_context=saver_context,
    )

    for artifact_spec in spec.artifacts:
        _attach_artifact_node(
            context=template_context,
            artifact_spec=artifact_spec,
            run_node=run_node,
        )

    if ingest_fn is not None and spec.tables:
        for table_spec in spec.tables:
            _attach_table_rows_node(
                context=template_context,
                table_spec=table_spec,
                ingest_node=ingest_node,
            )

    artifact_collector = make_artifact_materializations_collector(
        domain=spec.domain,
        target=spec.target_name,
        artifacts=[artifact.name for artifact in spec.artifacts],
        node_name=artifact_collector_node,
    )
    tagged_attach_node(
        module,
        node_name=artifact_collector_node,
        fn=artifact_collector,
        tag_spec=TagSpec.for_helper(domain=spec.domain, target=spec.target_name),
    )

    table_collector = make_table_materializations_collector(
        domain=spec.domain,
        target=spec.target_name,
        table_keys=[table.table_key for table in spec.tables],
        node_name=table_collector_node,
    )
    tagged_attach_node(
        module,
        node_name=table_collector_node,
        fn=table_collector,
        tag_spec=TagSpec.for_helper(domain=spec.domain, target=spec.target_name),
    )

    anchor = _build_anchor(
        inputs=_AnchorInputs(
            spec=spec,
            run_node=run_node,
            ingest_node=ingest_node if ingest_fn is not None else None,
            artifact_collector_node=artifact_collector_node,
            table_collector_node=table_collector_node,
            hash_options_node=hash_options_node,
        ),
    )
    tagged_attach_node(
        module,
        node_name=f"t__{spec.target_name}",
        fn=anchor,
        tag_spec=TagSpec.for_materialize(domain=spec.domain, target=spec.target_name),
    )


def _attach_artifact_node(
    *,
    context: _TemplateContext,
    artifact_spec: ArtifactOutputSpec,
    run_node: str,
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
    artifact_fn.__name__ = f"{context.target_name}__{artifact_spec.name}_artifact"
    decorator = save_artifact(
        context=context.saver_context,
        spec=ArtifactSaveSpec(
            artifact_name=artifact_spec.name,
            path_template=artifact_spec.path_template,
            output_role=artifact_spec.output_role,
        ),
    )
    tagged_attach_node(
        context.module,
        node_name=artifact_fn.__name__,
        fn=decorator(artifact_fn),
        tag_spec=TagSpec.for_compute(domain=context.domain, target=context.target_name),
    )


def _attach_table_rows_node(
    *,
    context: _TemplateContext,
    table_spec: TableOutputSpec,
    ingest_node: str,
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
    node_name = f"{context.target_name}__{table_spec.table_key.replace('.', '__')}_rows"
    rows_fn.__name__ = node_name
    decorator = save_rows(
        context=context.saver_context,
        spec=TableSaveSpec(
            table_key=table_spec.table_key,
            columns=table_spec.columns,
            output_role=table_spec.output_role,
        ),
    )
    tagged_attach_node(
        context.module,
        node_name=node_name,
        fn=decorator(rows_fn),
        tag_spec=TagSpec.for_compute(domain=context.domain, target=context.target_name),
    )


def _build_anchor(*, inputs: _AnchorInputs) -> Callable[..., TargetRunRecord]:
    def anchor_fn(**kwargs: object) -> TargetRunRecord:
        env = kwargs.get("env")
        catalog = kwargs.get("catalog")
        if not isinstance(env, BuildEnv):
            msg = "Missing BuildEnv for target anchor"
            raise TypeError(msg)
        if not isinstance(catalog, DagCatalog):
            msg = "Missing DagCatalog for target anchor"
            raise TypeError(msg)
        tool_step = kwargs.get(inputs.run_node)
        ingest_step = kwargs.get(inputs.ingest_node) if inputs.ingest_node is not None else None
        artifact_materializations = cast(
            "Mapping[str, MaterializationResult]",
            kwargs.get(inputs.artifact_collector_node),
        )
        table_materializations = cast(
            "Mapping[str, MaterializationResult]",
            kwargs.get(inputs.table_collector_node),
        )
        hash_options = None
        if inputs.hash_options_node is not None:
            hash_options = cast("InputHashOptions | None", kwargs.get(inputs.hash_options_node))
        return finalize_target_from_materializations(
            context=ToolFinalizeContext(
                env=env,
                catalog=catalog,
                target_name=inputs.spec.target_name,
                hash_options=hash_options,
            ),
            tool_step=cast("HasExecutionResult | None", tool_step),
            ingest_step=cast("HasExecutionResult | None", ingest_step),
            artifact_materializations=artifact_materializations,
            table_materializations=table_materializations,
        )

    params = [
        inspect.Parameter(
            "env",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=BuildEnv,
        ),
        inspect.Parameter(
            "catalog",
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=DagCatalog,
        ),
        inspect.Parameter(
            inputs.run_node,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=ToolStepOutput,
        ),
    ]
    if inputs.ingest_node is not None:
        params.append(
            inspect.Parameter(
                inputs.ingest_node,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=IngestStep[TRowsByTable],
            )
        )
    params.append(
        inspect.Parameter(
            inputs.artifact_collector_node,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=dict[str, MaterializationResult],
        )
    )
    params.append(
        inspect.Parameter(
            inputs.table_collector_node,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            annotation=dict[str, MaterializationResult],
        )
    )
    if inputs.hash_options_node is not None:
        params.append(
            inspect.Parameter(
                inputs.hash_options_node,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=InputHashOptions,
            )
        )

    anchor_fn = set_signature(
        anchor_fn,
        inspect.Signature(params, return_annotation=TargetRunRecord),
    )
    anchor_fn.__name__ = f"t__{inputs.spec.target_name}"
    return codeintel_target(
        domain=inputs.spec.domain,
        target=inputs.spec.target_name,
        spec=inputs.spec.spec,
    )(anchor_fn)


__all__ = [
    "IngestStep",
    "ToolFinalizeContext",
    "ToolRunContext",
    "attach_tool_target_template",
    "finalize_target_from_materializations",
    "run_tool_and_ingest",
    "run_tool_step",
]
