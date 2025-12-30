"""Shared DAG authoring patterns for native Hamilton targets."""

from __future__ import annotations

from codeintel.build.hamilton.native.patterns.access import (
    DataAccessSpec,
    load_access,
    load_query_spec,
    load_table_spec,
)
from codeintel.build.hamilton.native.patterns.loaders import load_query, load_table
from codeintel.build.hamilton.native.patterns.materialization_collectors import (
    make_artifact_materializations_collector,
    make_mixed_materializations_collector,
    make_table_materializations_collector,
)
from codeintel.build.hamilton.native.patterns.paths import (
    resolve_artifact_output_path,
    resolve_artifact_output_paths,
)
from codeintel.build.hamilton.native.patterns.savers import (
    ArtifactSaveSpec,
    DatasetSaveSpec,
    RelationTableSaveSpec,
    SaverContext,
    save_artifact,
    save_artifact_internal,
    save_dataset,
    save_relation_table,
)
from codeintel.build.hamilton.native.patterns.specs import (
    ArtifactOutputSpec,
    OutputRole,
    TableOutputSpec,
    ToolTargetSpec,
)
from codeintel.build.hamilton.native.patterns.target_builder import (
    IngestStep,
    ToolFinalizeContext,
    ToolRunContext,
    attach_tool_target_template,
    finalize_target_from_materializations,
    run_tool_and_ingest,
    run_tool_step,
)

__all__ = [
    "ArtifactOutputSpec",
    "ArtifactSaveSpec",
    "DataAccessSpec",
    "DatasetSaveSpec",
    "IngestStep",
    "OutputRole",
    "RelationTableSaveSpec",
    "SaverContext",
    "TableOutputSpec",
    "ToolFinalizeContext",
    "ToolRunContext",
    "ToolTargetSpec",
    "attach_tool_target_template",
    "finalize_target_from_materializations",
    "load_access",
    "load_query",
    "load_query_spec",
    "load_table",
    "load_table_spec",
    "make_artifact_materializations_collector",
    "make_mixed_materializations_collector",
    "make_table_materializations_collector",
    "resolve_artifact_output_path",
    "resolve_artifact_output_paths",
    "run_tool_and_ingest",
    "run_tool_step",
    "save_artifact",
    "save_artifact_internal",
    "save_dataset",
    "save_relation_table",
]
