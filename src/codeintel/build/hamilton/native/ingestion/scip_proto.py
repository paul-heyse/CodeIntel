"""Native Hamilton target for SCIP protobuf codegen."""

from __future__ import annotations

import asyncio
import logging
import sys
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING

from hamilton.function_modifiers import source, value

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materializations,
)
from codeintel.build.hamilton.native.target_decorators import (
    TargetSpecDescriptor,
    codeintel_target,
)
from codeintel.build.hamilton.native.tool_results import ToolStepOutput
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_tool
from codeintel.build.hashing import InputHashOptions, compute_options_hash
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.build.targets import TargetGraph
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
def t__scip_proto__run(env: BuildEnv, graph: TargetGraph) -> ScipProtoRunResult:
    """Generate scip_pb2.py using grpc_tools.protoc.

    Returns
    -------
    ScipProtoRunResult
        Result describing the codegen outcome and output path.
    """
    proto_path = _proto_path(env.snapshot.repo_root)
    out_dir = _proto_out_dir(env)
    output_path = out_dir / "scip_pb2.py"
    hash_options = InputHashOptions(
        file_state_hash=file_hash(proto_path),
        options_hash=_options_hash(env),
    )
    executor = NativeTargetExecutor.for_target(
        env,
        graph,
        SCIP_PROTO_TARGET,
        hash_options=hash_options,
    )
    if executor.should_skip():
        return ScipProtoRunResult(
            result=ExecutionResult.skip("SCIP proto target skipped"),
            outputs={SCIP_PROTO_ARTIFACT: output_path},
        )

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


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node("artifact.scip_pb2"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_PROTO_TARGET),
    artifact_name=value(SCIP_PROTO_ARTIFACT),
    path_template=value("{scip_dir}/proto/scip_pb2.py"),
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


@tag_compute(domain="ingestion", target=SCIP_PROTO_TARGET)
def scip__proto_materializations(
    m__artifact__scip_pb2: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    """Collect scip proto artifact materializations.

    Returns
    -------
    dict[str, MaterializationMetadata]
        Mapping of artifact names to materialization metadata.
    """
    return {SCIP_PROTO_ARTIFACT: m__artifact__scip_pb2}


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
    env: BuildEnv,
    graph: TargetGraph,
    t__scip_proto__run: ScipProtoRunResult,
    scip__proto_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """Emit target run record for protobuf codegen.

    Returns
    -------
    TargetRunRecord
        Record describing the codegen materialization outcome.
    """
    executor = NativeTargetExecutor.for_target(env, graph, SCIP_PROTO_TARGET)
    if not t__scip_proto__run.result.success:
        return executor.fail(
            RuntimeError(t__scip_proto__run.result.error or "SCIP proto failed")
        )
    return record_from_file_artifact_materializations(
        env=env,
        graph=graph,
        target_name=SCIP_PROTO_TARGET,
        materializations=scip__proto_materializations,
    )


__all__ = [
    "ScipProtoRunResult",
    "scip__proto_module_path",
    "t__scip_proto",
    "t__scip_proto__run",
]
