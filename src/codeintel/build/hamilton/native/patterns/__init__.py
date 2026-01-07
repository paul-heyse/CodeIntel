"""Shared DAG authoring patterns for native Hamilton targets."""

from __future__ import annotations

from codeintel.build.hamilton.native.patterns.access import DataAccessSpec, load_table_spec
from codeintel.build.hamilton.native.patterns.loaders import load_table
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
from codeintel.build.hamilton.native.patterns.table_target import (
    DatasetSaveSpecOptions,
    MultiTableTargetContext,
    RelationTableSaveSpecOptions,
    TableTargetContext,
    TableTargetSpec,
    TableTargetTableContext,
    TableTargetTableSpec,
    attach_table_target_template,
    build_multi_table_target_spec,
    build_multi_table_target_spec_from_contexts,
    build_single_table_target_spec,
    build_table_target_specs,
)
from codeintel.build.hamilton.native.patterns.tool_target import (
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
    "DatasetSaveSpecOptions",
    "IngestStep",
    "MultiTableTargetContext",
    "OutputRole",
    "RelationTableSaveSpec",
    "RelationTableSaveSpecOptions",
    "SaverContext",
    "TableOutputSpec",
    "TableTargetContext",
    "TableTargetSpec",
    "TableTargetTableContext",
    "TableTargetTableSpec",
    "ToolFinalizeContext",
    "ToolRunContext",
    "ToolTargetSpec",
    "attach_table_target_template",
    "attach_tool_target_template",
    "build_multi_table_target_spec",
    "build_multi_table_target_spec_from_contexts",
    "build_single_table_target_spec",
    "build_table_target_specs",
    "finalize_target_from_materializations",
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
