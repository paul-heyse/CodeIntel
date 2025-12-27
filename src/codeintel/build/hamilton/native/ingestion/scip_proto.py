"""Native Hamilton target for SCIP protobuf codegen."""

from __future__ import annotations

import asyncio
import logging
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.patterns import (
    ArtifactSaveSpec,
    SaverContext,
    ToolFinalizeContext,
    ToolRunContext,
    finalize_target_from_materializations,
    run_tool_step,
    save_artifact,
)
from codeintel.build.hamilton.native.target_decorators import (
    TargetSpecDescriptor,
    codeintel_target,
)
from codeintel.build.hamilton.native.tool_results import ToolStepOutput
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_compute, tag_helper, tag_tool
from codeintel.build.hashing import compute_options_hash
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.core.hashing import file_hash
from codeintel.core.tools import ToolName
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolNotFoundError,
    ToolRunOptions,
)

if TYPE_CHECKING:
    from codeintel.ingestion.engine.infrastructure import ToolRunResult

log = logging.getLogger(__name__)

SCIP_PROTO_TARGET = "scip_proto"
SCIP_PROTO_ARTIFACT = "scip_pb2"

ScipProtoRunResult = ToolStepOutput

SCIP_PROTO_SAVE_CONTEXT = SaverContext(
    domain="ingestion",
    target=SCIP_PROTO_TARGET,
)


def _proto_path(repo_root: Path) -> Path:
    return repo_root / "src" / "codeintel" / "ingestion" / "scip" / "proto" / "scip.proto"


def _proto_out_dir(env: BuildEnv) -> Path:
    return env.paths.scip_dir / "proto"


def _grpc_tools_version() -> str | None:
    try:
        return metadata.version("grpcio-tools")
    except metadata.PackageNotFoundError:
        return None


def _options_hash(env: BuildEnv) -> str | None:
    tools_config = env.providers.tool_runner.tools_config
    options = {
        "protoc_bin": tools_config.protoc_bin,
        "grpc_tools_version": _grpc_tools_version(),
        "python_version": sys.version,
    }
    return compute_options_hash(options)




def _run_codegen(
    env: BuildEnv,
    proto_path: Path,
    out_dir: Path,
    output_path: Path,
) -> ToolRunResult:
    args = [
        "-m",
        "grpc_tools.protoc",
        "-I",
        str(proto_path.parent),
        "--python_out",
        str(out_dir),
        str(proto_path),
    ]
    return asyncio.run(
        env.providers.tool_runner.run_async(
            ToolName.PROTOC,
            args,
            options=ToolRunOptions(
                cwd=env.snapshot.repo_root,
                output_path=output_path,
                timeout_s=env.providers.tool_runner.tools_config.default_timeout_s,
            ),
        )
    )


@tag_tool(domain="ingestion", target=SCIP_PROTO_TARGET)
def t__scip_proto__run(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ScipProtoRunResult:
    """Generate scip_pb2.py using grpc_tools.protoc.

    Returns
    -------
    ScipProtoRunResult
        Result describing the codegen outcome and output path.
    """
    proto_path = _proto_path(env.snapshot.repo_root)
    out_dir = _proto_out_dir(env)
    output_path = out_dir / "scip_pb2.py"

    context = ToolRunContext(
        env=env,
        catalog=catalog,
        target_name=SCIP_PROTO_TARGET,
    )

    def _execute() -> ScipProtoRunResult:
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            result = _run_codegen(env, proto_path, out_dir, output_path)
        except (ToolExecutionError, ToolNotFoundError, OSError, RuntimeError, ValueError) as exc:
            log.exception("SCIP protobuf codegen failed")
            return ScipProtoRunResult(result=ExecutionResult.failed(str(exc)))

        if not result.ok:
            message = result.stderr.strip() or "SCIP protobuf codegen failed"
            return ScipProtoRunResult(result=ExecutionResult.failed(message))

        return ScipProtoRunResult(
            result=ExecutionResult.ok(),
            outputs={SCIP_PROTO_ARTIFACT: output_path},
        )

    return run_tool_step(context=context, run=_execute)


@save_artifact(
    context=SCIP_PROTO_SAVE_CONTEXT,
    spec=ArtifactSaveSpec(
        artifact_name=SCIP_PROTO_ARTIFACT,
        path_template="{scip_dir}/proto/scip_pb2.py",
    ),
)
@tag_compute(domain="ingestion", target=SCIP_PROTO_TARGET, target_="scip__proto_artifact")
def scip__proto_artifact(t__scip_proto__run: ScipProtoRunResult) -> Path | None:
    """Expose scip_pb2.py path for artifact materialization.

    Returns
    -------
    Path | None
        Path to scip_pb2.py, or None when codegen failed.
    """
    if not t__scip_proto__run.result.success or t__scip_proto__run.result.skipped:
        return None
    return t__scip_proto__run.path_for(SCIP_PROTO_ARTIFACT)


@tag_compute(domain="ingestion", target=SCIP_PROTO_TARGET)
def scip__proto_module_path(
    env: BuildEnv,
    t__scip_proto__run: ScipProtoRunResult,
) -> Path | None:
    """Return the scip_pb2.py path for downstream parsing nodes.

    Returns
    -------
    Path | None
        Path to scip_pb2.py, or None when codegen failed.
    """
    _ = env
    if not t__scip_proto__run.result.success:
        return None
    return t__scip_proto__run.path_for(SCIP_PROTO_ARTIFACT)


@tag_helper(domain="ingestion", target=SCIP_PROTO_TARGET)
def scip_proto__materializations(
    m__artifact__scip_pb2: MaterializationResult,
) -> dict[str, MaterializationResult]:
    """Collect scip proto artifact materializations.

    Returns
    -------
    dict[str, MaterializationResult]
        Mapping of artifact names to materialization results.
    """
    return {SCIP_PROTO_ARTIFACT: m__artifact__scip_pb2}


@tag_helper(domain="ingestion", target=SCIP_PROTO_TARGET)
def scip_proto__finalize_context(
    env: BuildEnv,
    catalog: DagCatalog,
) -> ToolFinalizeContext:
    """Build finalization context for SCIP proto codegen.

    Returns
    -------
    ToolFinalizeContext
        Finalization context for SCIP proto.
    """
    return ToolFinalizeContext(
        env=env,
        catalog=catalog,
        target_name=SCIP_PROTO_TARGET,
    )


@codeintel_target(
    domain="ingestion",
    target=SCIP_PROTO_TARGET,
    spec=TargetSpecDescriptor(
        resources=TargetResources(
            tracker=True,
            tools=("protoc",),
        ),
        execution=TOOL_EXECUTION,
    ),
)
def t__scip_proto(
    scip_proto__finalize_context: ToolFinalizeContext,
    t__scip_proto__run: ScipProtoRunResult,
    scip_proto__materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Emit target run record for protobuf codegen.

    Returns
    -------
    TargetRunRecord
        Record describing the codegen materialization outcome.
    """
    return finalize_target_from_materializations(
        context=scip_proto__finalize_context,
        tool_step=t__scip_proto__run,
        ingest_step=None,
        artifact_materializations=scip_proto__materializations,
        table_materializations=None,
    )


__all__ = [
    "ScipProtoRunResult",
    "scip__proto_module_path",
    "t__scip_proto",
    "t__scip_proto__run",
]
