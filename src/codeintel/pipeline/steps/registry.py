"""Pipeline step registry with query and execution APIs.

This module provides the registry for pipeline steps, backed by the
unified BasePluginRegistry infrastructure.

Use :func:`build_registry` to create a registry from step dictionaries,
or use :class:`StepPluginRegistry` directly for more control.
"""

from __future__ import annotations

from collections.abc import Mapping

from codeintel.pipeline.steps.base import PipelineStep
from codeintel.pipeline.steps.plugin_registry import (
    StepPlan,
    StepPluginRegistry,
    build_step_plugin_registry,
)


def build_registry(*step_dicts: Mapping[str, PipelineStep]) -> StepPluginRegistry:
    """Build a StepPluginRegistry from one or more step dictionaries.

    Parameters
    ----------
    step_dicts
        One or more mappings of step name to step instance.

    Returns
    -------
    StepPluginRegistry
        Registry containing all provided steps.

    Examples
    --------
    >>> registry = build_registry(INGESTION_STEPS, GRAPH_STEPS, ANALYTICS_STEPS)
    >>> registry.get("repo_scan")
    RepoScanStep(...)
    """
    merged: dict[str, PipelineStep] = {}
    for step_dict in step_dicts:
        merged.update(step_dict)
    return build_step_plugin_registry(merged)


__all__ = [
    "StepPlan",
    "StepPluginRegistry",
    "build_registry",
]
