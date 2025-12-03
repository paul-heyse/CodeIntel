"""Tests for GOID builder plugin and helper functions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import pandas as pd
import pytest

from codeintel.config import ConfigBuilder
from codeintel.graphs.plugins.builders import goid as goid_builder

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_ONE: Final[int] = 1
EXPECTED_FIVE: Final[int] = 5
EXPECTED_TEN: Final[int] = 10
EXPECTED_TWELVE: Final[int] = 12
EXPECTED_TWENTY: Final[int] = 20
EXPECTED_FIFTY: Final[int] = 50


# ===========================================================================
# Plugin Registration and Metadata Tests
# ===========================================================================


def test_get_goid_builder_plugin_returns_plugin() -> None:
    """get_goid_builder_plugin returns a valid plugin."""
    plugin = goid_builder.get_goid_builder_plugin()

    assert plugin is not None
    assert hasattr(plugin, "metadata")
    assert hasattr(plugin, "execute")


def test_goid_builder_plugin_metadata() -> None:
    """goid_builder_plugin has correct metadata."""
    plugin = goid_builder.goid_builder_plugin

    assert plugin.metadata.name == "goid_builder"
    assert plugin.metadata.stage == "goid"
    assert "core.goids" in plugin.metadata.produces_tables
    assert "core.goid_crosswalk" in plugin.metadata.produces_tables


def test_goid_builder_plugin_provides_goids() -> None:
    """goid_builder_plugin provides goids capability."""
    plugin = goid_builder.goid_builder_plugin

    assert "goids" in plugin.metadata.provides


def test_goid_builder_plugin_has_no_dependencies() -> None:
    """goid_builder_plugin has no dependencies."""
    plugin = goid_builder.goid_builder_plugin

    assert plugin.metadata.depends_on == ()


# ===========================================================================
# build_goid_entries_for_testing Tests
# ===========================================================================


def _make_row(**overrides: object) -> pd.Series:
    """Create a pandas Series row for GOID building tests.

    Parameters
    ----------
    **overrides
        Field values to override defaults.

    Returns
    -------
    pd.Series
        Row for build_goid_entries_for_testing.
    """
    defaults: dict[str, object] = {
        "path": "m.py",
        "node_type": "FunctionDef",
        "name": "foo",
        "qualname": "m.foo",
        "lineno": 10,
        "end_lineno": 12,
        "decorator_start_line": None,
        "decorator_end_line": None,
        "col_offset": 0,
        "end_col_offset": 0,
        "parent_qualname": None,
        "decorators": [],
        "docstring": None,
        "hash": "h1",
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def test_goid_start_line_includes_decorator_span() -> None:
    """GOID start_line should widen to the earliest decorator line."""
    row = _make_row(
        lineno=10,
        end_lineno=12,
        decorator_start_line=5,
        decorator_end_line=6,
    )
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    goid_row, crosswalk_row = goid_builder.build_goid_entries_for_testing(
        row, cfg, now, {"m.py": "m"}
    )

    expected_start_line = 5
    if goid_row["start_line"] != expected_start_line:
        message = f"start_line {goid_row['start_line']} != {expected_start_line}"
        pytest.fail(message)
    if crosswalk_row["start_line"] != expected_start_line:
        message = f"crosswalk start_line {crosswalk_row['start_line']} != {expected_start_line}"
        pytest.fail(message)


def test_goid_build_entries_without_decorator() -> None:
    """build_goid_entries_for_testing works without decorator lines."""
    row = _make_row(
        path="pkg/utils.py",
        qualname="pkg.utils.helper",
        lineno=10,
        end_lineno=12,
    )
    builder = ConfigBuilder.from_snapshot(repo="myrepo", commit="abc123", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    goid_row, crosswalk_row = goid_builder.build_goid_entries_for_testing(
        row, cfg, now, {"pkg/utils.py": "pkg.utils"}
    )

    # Without decorator, start_line should be lineno
    assert goid_row["start_line"] == EXPECTED_TEN
    assert goid_row["end_line"] == EXPECTED_TWELVE
    assert crosswalk_row["start_line"] == EXPECTED_TEN
    assert crosswalk_row["end_line"] == EXPECTED_TWELVE


def test_goid_build_entries_class_definition() -> None:
    """build_goid_entries_for_testing handles ClassDef node type."""
    row = _make_row(
        path="models.py",
        node_type="ClassDef",
        name="User",
        qualname="models.User",
        lineno=5,
        end_lineno=20,
    )
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    goid_row, _crosswalk_row = goid_builder.build_goid_entries_for_testing(
        row, cfg, now, {"models.py": "models"}
    )

    # Class should have proper kind set
    assert goid_row["kind"] == "class"
    assert goid_row["start_line"] == EXPECTED_FIVE
    assert goid_row["end_line"] == EXPECTED_TWENTY


def test_goid_build_entries_module_fallback() -> None:
    """build_goid_entries_for_testing uses relpath when module not in mapping."""
    row = _make_row(
        path="unknown/module.py",
        node_type="Module",
        name="module",
        qualname="module",
        lineno=1,
        end_lineno=50,
    )
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    # Empty mapping - should fallback to relpath_to_module
    goid_row, crosswalk_row = goid_builder.build_goid_entries_for_testing(row, cfg, now, {})

    # Crosswalk should have a module_path derived from relpath
    assert crosswalk_row["module_path"] is not None
    assert goid_row["start_line"] == EXPECTED_ONE
    assert goid_row["end_line"] == EXPECTED_FIFTY


def test_goid_build_entries_async_function() -> None:
    """build_goid_entries_for_testing handles AsyncFunctionDef node type."""
    row = _make_row(
        path="async_mod.py",
        node_type="AsyncFunctionDef",
        name="fetch",
        qualname="async_mod.fetch",
        lineno=10,
        end_lineno=15,
    )
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    goid_row, _crosswalk_row = goid_builder.build_goid_entries_for_testing(
        row, cfg, now, {"async_mod.py": "async_mod"}
    )

    # AsyncFunctionDef should be classified as function
    assert goid_row["kind"] == "function"


def test_goid_build_entries_method_in_class() -> None:
    """build_goid_entries_for_testing handles methods with parent_qualname."""
    row = _make_row(
        path="models.py",
        node_type="FunctionDef",
        name="save",
        qualname="models.User.save",
        lineno=15,
        end_lineno=20,
        parent_qualname="models.User",
    )
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    goid_row, crosswalk_row = goid_builder.build_goid_entries_for_testing(
        row, cfg, now, {"models.py": "models"}
    )

    # Method should be classified as method
    assert goid_row["kind"] == "method"
    assert crosswalk_row["ast_qualname"] == "models.User.save"


def test_goid_build_entries_preserves_repo_and_commit() -> None:
    """build_goid_entries_for_testing preserves repo and commit in output."""
    row = _make_row()
    builder = ConfigBuilder.from_snapshot(repo="my_repo", commit="abc123", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    goid_row, crosswalk_row = goid_builder.build_goid_entries_for_testing(
        row, cfg, now, {"m.py": "m"}
    )

    assert goid_row["repo"] == "my_repo"
    assert goid_row["commit"] == "abc123"
    assert crosswalk_row["repo"] == "my_repo"
    assert crosswalk_row["commit"] == "abc123"


def test_goid_build_entries_generates_urn() -> None:
    """build_goid_entries_for_testing generates URN."""
    row = _make_row()
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    goid_row, crosswalk_row = goid_builder.build_goid_entries_for_testing(
        row, cfg, now, {"m.py": "m"}
    )

    # URN should be generated and not empty
    assert goid_row["urn"] is not None
    assert len(goid_row["urn"]) > 0
    assert crosswalk_row["goid"] is not None


def test_goid_build_entries_handles_none_end_line() -> None:
    """build_goid_entries_for_testing handles None end_lineno."""
    row = _make_row(end_lineno=None)
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    goid_row, crosswalk_row = goid_builder.build_goid_entries_for_testing(
        row, cfg, now, {"m.py": "m"}
    )

    # Should handle None end_line gracefully
    assert goid_row["end_line"] is None
    assert crosswalk_row["end_line"] is None


def test_goid_build_entries_decorator_after_lineno() -> None:
    """build_goid_entries_for_testing uses lineno when decorator is after."""
    row = _make_row(
        lineno=5,
        end_lineno=10,
        decorator_start_line=10,
    )
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    goid_row, _crosswalk_row = goid_builder.build_goid_entries_for_testing(
        row, cfg, now, {"m.py": "m"}
    )

    # When decorator_start_line > lineno, use lineno
    assert goid_row["start_line"] == EXPECTED_FIVE


def test_goid_build_entries_zero_decorator() -> None:
    """build_goid_entries_for_testing ignores zero decorator_start_line."""
    row = _make_row(
        lineno=10,
        end_lineno=15,
        decorator_start_line=0,
    )
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    goid_row, _crosswalk_row = goid_builder.build_goid_entries_for_testing(
        row, cfg, now, {"m.py": "m"}
    )

    # Zero decorator should be ignored, use lineno
    assert goid_row["start_line"] == EXPECTED_TEN


def test_goid_build_entries_sets_language() -> None:
    """build_goid_entries_for_testing sets language from config."""
    row = _make_row()
    builder = ConfigBuilder.from_snapshot(repo="r", commit="c", repo_root=Path())
    cfg = builder.goid_builder(language="python")
    now = datetime.now(UTC)

    goid_row, crosswalk_row = goid_builder.build_goid_entries_for_testing(
        row, cfg, now, {"m.py": "m"}
    )

    assert goid_row["language"] == "python"
    assert crosswalk_row["lang"] == "python"
