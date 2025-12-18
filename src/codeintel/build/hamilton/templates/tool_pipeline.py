"""Reusable subDAG pipeline for tool-invocation artifact targets.

This module defines a small, reusable Hamilton subDAG that turns tool output
(bytes, string, or Path) into a persisted file artifact (via ``FileArtifactSaver``)
and then into a ``TargetRunRecord``.

It is intended to be used with Hamilton's ``@subdag`` decorator to reduce
boilerplate in native target modules that invoke external tools (e.g., SCIP,
linters, code generators) and persist their output as artifacts.

Notes
-----
Hamilton namespaces nodes created by ``@subdag`` using dotted names
(``<namespace>.<node_name>``). This is acceptable for internal pipeline nodes;
the public target node (e.g., ``t__scip``) remains a stable identifier.
"""

from __future__ import annotations

from pathlib import Path

from hamilton.function_modifiers import source

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.native.materialization_records import (
    record_from_file_artifact_materialization,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize
from codeintel.build.targets import TargetGraph

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, Path)


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_="artifact_metadata",
    env=source("env"),
    graph=source("graph"),
    target_name=source("target_name"),
    artifact_name=source("artifact_name"),
)
@tag_compute()
def tool_output_to_save(
    tool_output: bytes | str | Path | None,
) -> bytes | str | Path | None:
    """Return the tool output to be persisted as an artifact.

    Parameters
    ----------
    tool_output
        Tool output to persist. Supports bytes, UTF-8 string, or Path to
        an existing file. None indicates no output (skip materialization).

    Returns
    -------
    bytes | str | Path | None
        The same tool output.
    """
    return tool_output


@tag_materialize()
def record(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    artifact_name: str,
    artifact_metadata: dict[str, object],
) -> TargetRunRecord:
    """Convert artifact saver metadata into a TargetRunRecord.

    Parameters
    ----------
    env
        Build environment for manifest persistence and expected output refs.
    graph
        Target graph used to resolve the OutputTarget contract.
    target_name
        Target name for which the record is being produced.
    artifact_name
        Artifact name expected to be written for this target.
    artifact_metadata
        Materialization metadata dict returned by the Hamilton saver node.

    Returns
    -------
    TargetRunRecord
        Record describing succeeded/skipped/failed completion.
    """
    return record_from_file_artifact_materialization(
        env=env,
        graph=graph,
        target_name=target_name,
        expected_artifact_name=artifact_name,
        materialization=artifact_metadata,
    )


__all__ = [
    "record",
    "tool_output_to_save",
]
