"""Tests for PR-79: DAG-first defaults for graph_source."""

from __future__ import annotations

import pytest

from codeintel.cli.commands.build import BuildExplainCommand, BuildGraphCommand, BuildPlanCommand


def test_pr79_graph_source_default_is_hamilton() -> None:
    """Verify build plan/graph/explain default to Hamilton-derived deps."""
    if BuildPlanCommand().graph_source != "hamilton":
        pytest.fail("Expected BuildPlanCommand.graph_source default to be 'hamilton'")
    if BuildGraphCommand().graph_source != "hamilton":
        pytest.fail("Expected BuildGraphCommand.graph_source default to be 'hamilton'")
    if BuildExplainCommand(target="modules").graph_source != "hamilton":
        pytest.fail("Expected BuildExplainCommand.graph_source default to be 'hamilton'")
