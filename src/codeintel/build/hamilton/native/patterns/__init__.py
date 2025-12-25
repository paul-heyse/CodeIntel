"""Shared DAG authoring patterns for native Hamilton targets."""

from __future__ import annotations

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
    save_artifact,
    save_artifact_internal,
    save_ibis_table,
    save_rows,
    save_rows_internal,
)
from codeintel.build.hamilton.native.patterns.specs import (
    ArtifactOutputSpec,
    OutputRole,
    TableOutputSpec,
    ToolTargetSpec,
)
from codeintel.build.hamilton.native.patterns.tool_target import (
    IngestStep,
    attach_tool_target_template,
    finalize_target_from_materializations,
    run_tool_and_ingest,
    run_tool_step,
)

__all__ = [
    "ArtifactOutputSpec",
    "IngestStep",
    "OutputRole",
    "TableOutputSpec",
    "ToolTargetSpec",
    "attach_tool_target_template",
    "finalize_target_from_materializations",
    "make_artifact_materializations_collector",
    "make_mixed_materializations_collector",
    "make_table_materializations_collector",
    "resolve_artifact_output_path",
    "resolve_artifact_output_paths",
    "run_tool_and_ingest",
    "run_tool_step",
    "save_artifact",
    "save_artifact_internal",
    "save_ibis_table",
    "save_rows",
    "save_rows_internal",
]
