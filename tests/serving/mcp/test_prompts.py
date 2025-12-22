"""Tests for MCP prompt templates."""

from __future__ import annotations

from pathlib import Path

from fastmcp import FastMCP

from codeintel.serving.mcp.prompts import list_prompt_names, register_prompts
from codeintel.serving.settings import ServingSettings
from tests._helpers.assertions.expectation_assertions import (
    expect_in,
    expect_true,
)

# Minimum number of prompts expected to be registered
MIN_PROMPT_COUNT = 4


def _get_prompt_names(mcp: FastMCP) -> set[str]:
    """Get registered prompt names from MCP server.

    Parameters
    ----------
    mcp
        FastMCP server instance.

    Returns
    -------
    set[str]
        Set of registered prompt names.

    """
    return list_prompt_names(mcp)


def test_register_prompts_adds_prompts() -> None:
    """Verify prompts are registered on MCP server."""
    mcp = FastMCP("Test")
    settings = ServingSettings(serve_dir=Path.cwd())
    register_prompts(mcp, settings=settings)

    # Check prompts are accessible via prompt manager
    prompts = _get_prompt_names(mcp)
    expect_true(len(prompts) >= MIN_PROMPT_COUNT)


def test_explore_codebase_prompt_registered() -> None:
    """Verify explore_codebase prompt is registered."""
    mcp = FastMCP("Test")
    settings = ServingSettings(serve_dir=Path.cwd())
    register_prompts(mcp, settings=settings)

    prompts = _get_prompt_names(mcp)
    expect_in("explore_codebase", prompts)


def test_wizard_export_data_prompt_registered() -> None:
    """Verify wizard_export_data prompt is registered."""
    mcp = FastMCP("Test")
    settings = ServingSettings(serve_dir=Path.cwd())
    register_prompts(mcp, settings=settings)

    prompts = _get_prompt_names(mcp)
    expect_in("wizard_export_data", prompts)


def test_wizard_query_view_prompt_registered() -> None:
    """Verify wizard_query_view prompt is registered."""
    mcp = FastMCP("Test")
    settings = ServingSettings(serve_dir=Path.cwd())
    register_prompts(mcp, settings=settings)

    prompts = _get_prompt_names(mcp)
    expect_in("wizard_query_view", prompts)


def test_what_changed_between_snapshots_prompt_registered() -> None:
    """Verify what_changed_between_snapshots prompt is registered."""
    mcp = FastMCP("Test")
    settings = ServingSettings(serve_dir=Path.cwd())
    register_prompts(mcp, settings=settings)

    prompts = _get_prompt_names(mcp)
    expect_in("what_changed_between_snapshots", prompts)


def test_prompts_have_all_expected_names() -> None:
    """Verify all expected prompts are registered."""
    mcp = FastMCP("Test")
    settings = ServingSettings(serve_dir=Path.cwd())
    register_prompts(mcp, settings=settings)

    prompts = _get_prompt_names(mcp)
    expected_names = {
        "explore_codebase",
        "wizard_export_data",
        "wizard_query_view",
        "what_changed_between_snapshots",
    }
    for name in expected_names:
        expect_in(name, prompts)
