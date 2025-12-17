"""Tests for MCP prompt templates."""

from __future__ import annotations

from fastmcp import FastMCP

from codeintel.serving.mcp.prompts import register_prompts
from tests._helpers.assertions.expectation_assertions import (
    expect_in,
    expect_true,
)

# Minimum number of prompts expected to be registered
MIN_PROMPT_COUNT = 5


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

    Notes
    -----
    Accesses internal FastMCP state for testing purposes only.
    """
    return set(mcp._prompt_manager._prompts.keys())  # noqa: SLF001


def test_register_prompts_adds_prompts() -> None:
    """Verify prompts are registered on MCP server."""
    mcp = FastMCP("Test")
    register_prompts(mcp)

    # Check prompts are accessible via prompt manager
    prompts = _get_prompt_names(mcp)
    expect_true(len(prompts) >= MIN_PROMPT_COUNT)


def test_explore_codebase_prompt_registered() -> None:
    """Verify explore_codebase prompt is registered."""
    mcp = FastMCP("Test")
    register_prompts(mcp)

    prompts = _get_prompt_names(mcp)
    expect_in("explore_codebase", prompts)


def test_find_function_prompt_registered() -> None:
    """Verify find_function prompt is registered."""
    mcp = FastMCP("Test")
    register_prompts(mcp)

    prompts = _get_prompt_names(mcp)
    expect_in("find_function", prompts)


def test_export_data_prompt_registered() -> None:
    """Verify export_data prompt is registered."""
    mcp = FastMCP("Test")
    register_prompts(mcp)

    prompts = _get_prompt_names(mcp)
    expect_in("export_data", prompts)


def test_analyze_metrics_prompt_registered() -> None:
    """Verify analyze_metrics prompt is registered."""
    mcp = FastMCP("Test")
    register_prompts(mcp)

    prompts = _get_prompt_names(mcp)
    expect_in("analyze_metrics", prompts)


def test_get_server_status_prompt_registered() -> None:
    """Verify get_server_status prompt is registered."""
    mcp = FastMCP("Test")
    register_prompts(mcp)

    prompts = _get_prompt_names(mcp)
    expect_in("get_server_status", prompts)


def test_prompts_have_all_expected_names() -> None:
    """Verify all expected prompts are registered."""
    mcp = FastMCP("Test")
    register_prompts(mcp)

    prompts = _get_prompt_names(mcp)
    expected_names = {
        "explore_codebase",
        "find_function",
        "export_data",
        "analyze_metrics",
        "get_server_status",
    }
    for name in expected_names:
        expect_in(name, prompts)
