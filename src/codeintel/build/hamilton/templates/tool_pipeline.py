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
from types import ModuleType

from hamilton.function_modifiers.dependencies import source, value

from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.hamilton.native.materialization_records import (
    FileArtifactRecordContext,
    record_from_file_artifact_materialization,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.tagging import tag_compute, tag_materialize
from codeintel.build.targets import TargetGraph

_HAMILTON_TYPE_HINTS = (BuildEnv, TargetGraph, TargetRunRecord, Path)


def build_tool_pipeline_module(
    *,
    target_name: str,
    artifact_name: str,
    path_template: str,
) -> ModuleType:
    """Build a module that materializes tool output into an artifact record.

    Parameters
    ----------
    target_name
        Target name for contract attribution and manifest hashing.
    artifact_name
        Artifact name expected to be written for this target.
    path_template
        Path template used to resolve the artifact output location.

    Returns
    -------
    ModuleType
        Module exposing ``tool_output_to_save`` and ``record`` nodes.
    """
    module = ModuleType(f"tool_pipeline_{target_name}_{artifact_name}")

    @SaveToObjectMetadataDecorator(
        [FileArtifactSaver],
        output_name_="artifact_metadata",
        env=source("env"),
        graph=source("graph"),
        target_name=value(target_name),
        artifact_name=value(artifact_name),
        path_template=value(path_template),
    )
    @tag_compute(target=target_name)
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

    @tag_materialize(target=target_name)
    def record(
        env: BuildEnv,
        graph: TargetGraph,
        artifact_metadata: MaterializationMetadata,
    ) -> TargetRunRecord:
        """Convert artifact saver metadata into a TargetRunRecord.

        Parameters
        ----------
        env
            Build environment for manifest persistence and expected output refs.
        graph
            Target graph used to resolve the OutputTarget contract.
        artifact_metadata
            Materialization metadata dict returned by the Hamilton saver node.

        Returns
        -------
        TargetRunRecord
            Record describing succeeded/skipped/failed completion.
        """
        context = FileArtifactRecordContext(
            env=env,
            graph=graph,
            target_name=target_name,
        )
        return record_from_file_artifact_materialization(
            context=context,
            expected_artifact_name=artifact_name,
            materialization=artifact_metadata,
        )

    module.__dict__["tool_output_to_save"] = tool_output_to_save
    module.__dict__["record"] = record
    return module


__all__ = [
    "build_tool_pipeline_module",
]
