"""Pipeline specification types and registry.

This package contains pipeline specification definitions:

- PipelineSpec: Defines a pipeline's stages and steps
- PipelineStage: A stage within a pipeline
- StageModule: Module enum for stage classification
- PIPELINE_SPECS: Registry of all pipeline specifications
- get_pipeline_spec: Lookup a pipeline spec by name
- list_pipeline_specs: List all registered pipeline names
"""

from __future__ import annotations

from codeintel.pipeline.spec.model import (
    PIPELINE_SPECS,
    PipelineSpec,
    PipelineStage,
    StageModule,
    get_pipeline_spec,
    list_pipeline_specs,
)

__all__ = [
    "PIPELINE_SPECS",
    "PipelineSpec",
    "PipelineStage",
    "StageModule",
    "get_pipeline_spec",
    "list_pipeline_specs",
]
