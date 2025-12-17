"""Native Hamilton implementation for scip target.

This target produces the SCIP artifacts declared in the build contract:
- ``scip_index``: ``{scip_dir}/index.scip``
- ``scip_json``: ``{scip_dir}/index.json``

Tool execution is delegated to the ingestion tool runtime (ToolService),
while persistence is DAG-visible via ``FileArtifactSaver``.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hamilton.function_modifiers import source, tag, value
from hamilton.function_modifiers.adapters import SaveToDecorator

from codeintel.build.contracts import ArtifactSpec
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.hooks.manifest_hook import TargetRunRecord
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.executor import NativeTargetExecutor
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materializations,
)
from codeintel.build.hamilton.native.target_spec_helpers import (
    TargetSpecOptions,
    make_output_target,
)
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.build.targets import TargetGraph
from codeintel.ingestion.engine.infrastructure import ToolRunner
from codeintel.ingestion.engine.service import ToolService

log = logging.getLogger(__name__)

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, Path, TargetRunRecord)

SCIP_TARGET_NAME = "scip"
SCIP_ARTIFACT_INDEX = "scip_index"
SCIP_ARTIFACT_JSON = "scip_json"

SCIP_ARTIFACT_SPECS = (
    ArtifactSpec(SCIP_ARTIFACT_INDEX, "{scip_dir}/index.scip", "SCIP index file"),
    ArtifactSpec(SCIP_ARTIFACT_JSON, "{scip_dir}/index.json", "SCIP JSON export"),
)

TARGET_SPECS = (
    make_output_target(
        name=SCIP_TARGET_NAME,
        module="ingestion",
        description="SCIP index ingestion and GOID generation.",
        options=TargetSpecOptions(
            artifacts=SCIP_ARTIFACT_SPECS,
            resources=TargetResources(
                tracker=True,
                modules=True,
                tools=(
                    "scip-python",
                    "scip",
                ),
            ),
            execution=TOOL_EXECUTION,
        ),
    ),
)


@dataclass(frozen=True)
class ScipRunResult:
    """Result from running SCIP tooling.

    Attributes
    ----------
    success
        Whether execution completed successfully.
    skipped
        Whether execution was skipped due to manifest match.
    index_path
        Path to the generated index.scip.
    json_path
        Path to the generated index.json.
    error
        Error message on failure.
    """

    success: bool
    skipped: bool = False
    index_path: Path | None = None
    json_path: Path | None = None
    error: str | None = None


def _tool_service(env: BuildEnv) -> ToolService:
    runner = ToolRunner(
        tools_config=env.providers.tool_runner.tools_config,
        cache_dir=env.paths.build_dir / ".tool_cache",
    )
    return ToolService(runner)


@tag(domain="ingestion", target=SCIP_TARGET_NAME, node_type="tool")
def t__scip__run(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
) -> ScipRunResult:
    """Execute scip-python + scip print to produce SCIP artifacts.

    Returns
    -------
    ScipRunResult
        Run outcome including output paths or error details.
    """
    if t__modules.status != "succeeded":
        return ScipRunResult(
            success=False,
            error=f"Upstream modules target failed: {t__modules.error}",
        )

    executor = NativeTargetExecutor.for_target(env, graph, SCIP_TARGET_NAME)
    if executor.should_skip():
        return ScipRunResult(success=True, skipped=True)

    output_scip = env.paths.scip_dir / "index.scip"
    output_json = env.paths.scip_dir / "index.json"

    try:
        service = _tool_service(env)
        asyncio.run(
            service.run_scip_full(
                env.snapshot.repo_root,
                output_scip=output_scip,
                output_json=output_json,
            )
        )
        return ScipRunResult(
            success=True,
            index_path=output_scip,
            json_path=output_json,
        )
    except Exception as exc:
        log.exception("SCIP execution failed")
        return ScipRunResult(success=False, error=str(exc))


@SaveToDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node("artifact.scip_index"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_TARGET_NAME),
    artifact_name=value(SCIP_ARTIFACT_INDEX),
)
@tag(
    domain="ingestion",
    target=SCIP_TARGET_NAME,
    node_type="compute",
    target_="scip__index_artifact",
)
def scip__index_artifact(t__scip__run: ScipRunResult) -> Path | None:
    """Return the Path to index.scip for materialization.

    Returns
    -------
    Path | None
        Artifact path, or None when the run was skipped/failed.
    """
    if not t__scip__run.success or t__scip__run.skipped:
        return None
    return t__scip__run.index_path


@SaveToDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node("artifact.scip_json"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_TARGET_NAME),
    artifact_name=value(SCIP_ARTIFACT_JSON),
)
@tag(
    domain="ingestion",
    target=SCIP_TARGET_NAME,
    node_type="compute",
    target_="scip__json_artifact",
)
def scip__json_artifact(t__scip__run: ScipRunResult) -> Path | None:
    """Return the Path to index.json for materialization.

    Returns
    -------
    Path | None
        Artifact path, or None when the run was skipped/failed.
    """
    if not t__scip__run.success or t__scip__run.skipped:
        return None
    return t__scip__run.json_path


@tag(domain="ingestion", target=SCIP_TARGET_NAME, node_type="helper")
def scip__materializations(
    m__artifact__scip_index: dict[str, Any],
    m__artifact__scip_json: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Collect scip artifact materialization payloads into a single mapping.

    Returns
    -------
    dict[str, dict[str, Any]]
        Materialization metadata keyed by artifact name.
    """
    return {
        SCIP_ARTIFACT_INDEX: m__artifact__scip_index,
        SCIP_ARTIFACT_JSON: m__artifact__scip_json,
    }


@tag(domain="ingestion", target=SCIP_TARGET_NAME, node_type="materialize")
def t__scip(
    env: BuildEnv,
    graph: TargetGraph,
    t__scip__run: ScipRunResult,
    t__modules: TargetRunRecord,
    scip__materializations: dict[str, dict[str, Any]],
) -> TargetRunRecord:
    """Finalize scip target from artifact materialization metadata.

    Returns
    -------
    TargetRunRecord
        Record describing the materialization outcome.
    """
    executor = NativeTargetExecutor.for_target(env, graph, SCIP_TARGET_NAME)
    if executor.should_skip():
        return executor.skip()
    if t__modules.status != "succeeded":
        return executor.fail(RuntimeError(t__modules.error or "Upstream modules target failed"))
    if not t__scip__run.success:
        return executor.fail(RuntimeError(t__scip__run.error or "SCIP execution failed"))

    return record_from_file_artifact_materializations(
        env=env,
        graph=graph,
        target_name="scip",
        materializations=scip__materializations,
    )


__all__ = [
    "ScipRunResult",
    "t__scip",
    "t__scip__run",
]
