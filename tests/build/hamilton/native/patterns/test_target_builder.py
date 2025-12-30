"""Tests for tool target scaffolding helpers."""

from __future__ import annotations

from types import ModuleType

import pytest

from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.patterns.specs import TableOutputSpec, ToolTargetSpec
from codeintel.build.hamilton.native.patterns.target_builder import attach_tool_target_template
from codeintel.build.hamilton.native.target_decorators import TargetSpecDescriptor
from codeintel.build.hamilton.native.tool_results import ToolStepOutput


def _run_step() -> ToolStepOutput:
    return ToolStepOutput(result=ExecutionResult.ok())


def test_attach_tool_target_template_sets_anchor_docstring() -> None:
    """Ensure the generated anchor includes the expected docstring."""
    module = ModuleType("codeintel.tests.target_builder")
    spec = ToolTargetSpec(
        domain="analytics",
        target_name="demo",
        spec=TargetSpecDescriptor(),
    )
    attach_tool_target_template(module, spec=spec, run_fn=_run_step)
    anchor = module.t__demo
    assert anchor.__doc__ == "Finalize demo target materialization."
    assert callable(module.t__demo__run)


def test_attach_tool_target_template_requires_ingest_for_tables() -> None:
    """Require an ingest function when table outputs are declared."""
    module = ModuleType("codeintel.tests.target_builder_tables")
    spec = ToolTargetSpec(
        domain="analytics",
        target_name="demo",
        spec=TargetSpecDescriptor(),
        tables=(TableOutputSpec(table_key="core.demo"),),
    )
    with pytest.raises(ValueError, match="requires ingest_fn"):
        attach_tool_target_template(module, spec=spec, run_fn=_run_step)
