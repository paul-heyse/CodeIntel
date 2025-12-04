"""Extended tests for GOID builder plugin.

This module provides additional test coverage for the GOID builder plugin
from `codeintel.graphs.plugins.builders.goid`, including:

- Helper functions (_safe_int, _compute_start_line)
- Build entry construction
- Plugin protocol compliance
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final

import pandas as pd
import pytest

from codeintel.config.steps_graphs import GoidBuilderStepConfig
from codeintel.graphs.core.protocol import GraphPluginProtocol
from codeintel.graphs.plugins.builders.goid import (
    build_goid_entries_for_testing,
    get_goid_builder_plugin,
    goid_builder_plugin,
)

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway
    from codeintel.storage.snapshot import SnapshotRef

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
EXPECTED_START_LINE: Final[int] = 10
EXPECTED_END_LINE: Final[int] = 20
DECORATOR_START_LINE: Final[int] = 8
TEST_REPO: Final[str] = "test-repo"
TEST_COMMIT: Final[str] = "test-commit-abc"
TEST_PATH: Final[str] = "src/pkg/module.py"
TEST_MODULE: Final[str] = "src.pkg.module"
TEST_QUALNAME: Final[str] = "MyClass.my_method"


# ---------------------------------------------------------------------------
# Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_ast_row() -> pd.Series:
    """Create a sample AST node row for testing."""
    return pd.Series({
        "path": TEST_PATH,
        "node_type": "FunctionDef",
        "name": "my_method",
        "qualname": TEST_QUALNAME,
        "lineno": EXPECTED_START_LINE,
        "end_lineno": EXPECTED_END_LINE,
        "decorator_start_line": None,
        "decorator_end_line": None,
        "parent_qualname": "MyClass",
    })


@pytest.fixture
def decorated_ast_row() -> pd.Series:
    """Create an AST node row with decorator for testing."""
    return pd.Series({
        "path": TEST_PATH,
        "node_type": "FunctionDef",
        "name": "decorated_method",
        "qualname": "MyClass.decorated_method",
        "lineno": EXPECTED_START_LINE,
        "end_lineno": EXPECTED_END_LINE,
        "decorator_start_line": DECORATOR_START_LINE,
        "decorator_end_line": 9,
        "parent_qualname": "MyClass",
    })


@pytest.fixture
def module_ast_row() -> pd.Series:
    """Create a module-level AST node row."""
    return pd.Series({
        "path": TEST_PATH,
        "node_type": "Module",
        "name": "module",
        "qualname": "module",
        "lineno": 1,
        "end_lineno": 100,
        "decorator_start_line": None,
        "decorator_end_line": None,
        "parent_qualname": None,
    })


@pytest.fixture
def class_ast_row() -> pd.Series:
    """Create a class definition AST node row."""
    return pd.Series({
        "path": TEST_PATH,
        "node_type": "ClassDef",
        "name": "MyClass",
        "qualname": "MyClass",
        "lineno": 5,
        "end_lineno": 50,
        "decorator_start_line": None,
        "decorator_end_line": None,
        "parent_qualname": None,
    })


@pytest.fixture
def async_function_ast_row() -> pd.Series:
    """Create an async function AST node row."""
    return pd.Series({
        "path": TEST_PATH,
        "node_type": "AsyncFunctionDef",
        "name": "async_method",
        "qualname": "MyClass.async_method",
        "lineno": 30,
        "end_lineno": 40,
        "decorator_start_line": None,
        "decorator_end_line": None,
        "parent_qualname": "MyClass",
    })


@pytest.fixture
def goid_config(graph_snapshot: SnapshotRef) -> GoidBuilderStepConfig:
    """Create a GOID builder config for testing."""
    return GoidBuilderStepConfig(snapshot=graph_snapshot)


# ===========================================================================
# Plugin Instance Tests
# ===========================================================================


def test_goid_builder_plugin_protocol() -> None:
    """GOID builder plugin implements GraphPluginProtocol."""
    assert isinstance(goid_builder_plugin, GraphPluginProtocol)


def test_get_goid_builder_plugin() -> None:
    """get_goid_builder_plugin returns goid_builder_plugin."""
    result = get_goid_builder_plugin()
    assert result is goid_builder_plugin


def test_goid_builder_plugin_name() -> None:
    """GOID builder plugin has correct name."""
    assert goid_builder_plugin.metadata.name == "goid_builder"


def test_goid_builder_plugin_stage() -> None:
    """GOID builder plugin has goid stage."""
    assert goid_builder_plugin.metadata.stage == "goid"


def test_goid_builder_plugin_kind() -> None:
    """GOID builder plugin is builder kind."""
    assert goid_builder_plugin.metadata.kind == "builder"


def test_goid_builder_plugin_provides() -> None:
    """GOID builder plugin provides goids capability."""
    assert "goids" in goid_builder_plugin.metadata.provides


def test_goid_builder_plugin_produces_tables() -> None:
    """GOID builder plugin produces expected tables."""
    tables = goid_builder_plugin.metadata.produces_tables
    assert "core.goids" in tables
    assert "core.goid_crosswalk" in tables


def test_goid_builder_plugin_no_dependencies() -> None:
    """GOID builder plugin has no dependencies (foundational plugin)."""
    assert len(goid_builder_plugin.metadata.depends_on) == 0


# ===========================================================================
# Build Entry Construction Tests
# ===========================================================================


def test_build_goid_entries_basic(
    sample_ast_row: pd.Series,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries from basic AST row."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {TEST_PATH: TEST_MODULE}

    goid_row, xwalk_row = build_goid_entries_for_testing(
        sample_ast_row, goid_config, now, module_by_path
    )

    # GOID row assertions
    assert goid_row.rel_path == TEST_PATH
    assert goid_row.qualname == TEST_QUALNAME
    assert goid_row.start_line == EXPECTED_START_LINE
    assert goid_row.end_line == EXPECTED_END_LINE
    assert goid_row.kind == "method"  # FunctionDef with parent_qualname

    # Crosswalk row assertions
    assert xwalk_row.file_path == TEST_PATH
    assert xwalk_row.module_path == TEST_MODULE
    assert xwalk_row.start_line == EXPECTED_START_LINE


def test_build_goid_entries_with_decorator(
    decorated_ast_row: pd.Series,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries uses decorator start line when present."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {TEST_PATH: TEST_MODULE}

    goid_row, xwalk_row = build_goid_entries_for_testing(
        decorated_ast_row, goid_config, now, module_by_path
    )

    # Start line should use decorator start line (earlier)
    assert goid_row.start_line == DECORATOR_START_LINE
    assert xwalk_row.start_line == DECORATOR_START_LINE


def test_build_goid_entries_module(
    module_ast_row: pd.Series,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries for module node."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {TEST_PATH: TEST_MODULE}

    goid_row, xwalk_row = build_goid_entries_for_testing(
        module_ast_row, goid_config, now, module_by_path
    )

    assert goid_row.kind == "module"
    assert goid_row.start_line == 1


def test_build_goid_entries_class(
    class_ast_row: pd.Series,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries for class node."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {TEST_PATH: TEST_MODULE}

    goid_row, xwalk_row = build_goid_entries_for_testing(
        class_ast_row, goid_config, now, module_by_path
    )

    assert goid_row.kind == "class"
    assert goid_row.qualname == "MyClass"


def test_build_goid_entries_async_function(
    async_function_ast_row: pd.Series,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries for async function node."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {TEST_PATH: TEST_MODULE}

    goid_row, xwalk_row = build_goid_entries_for_testing(
        async_function_ast_row, goid_config, now, module_by_path
    )

    # AsyncFunctionDef with parent is a method
    assert goid_row.kind == "method"
    assert goid_row.qualname == "MyClass.async_method"


def test_build_goid_entries_missing_module_in_map(
    sample_ast_row: pd.Series,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries falls back to path-based module when not in map."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {}  # Empty map

    goid_row, xwalk_row = build_goid_entries_for_testing(
        sample_ast_row, goid_config, now, module_by_path
    )

    # Should still work, deriving module from path
    assert goid_row.rel_path == TEST_PATH
    # Module path derived from file path
    assert xwalk_row.module_path is not None


def test_build_goid_entries_windows_path_normalized(
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries normalizes Windows-style paths."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {}

    # Windows-style path
    row = pd.Series({
        "path": "src\\pkg\\module.py",
        "node_type": "FunctionDef",
        "name": "func",
        "qualname": "func",
        "lineno": 1,
        "end_lineno": 10,
        "decorator_start_line": None,
        "decorator_end_line": None,
        "parent_qualname": None,
    })

    goid_row, _ = build_goid_entries_for_testing(row, goid_config, now, module_by_path)

    # Path should be normalized to forward slashes
    assert "/" in goid_row.rel_path
    assert "\\" not in goid_row.rel_path


# ===========================================================================
# Edge Cases Tests
# ===========================================================================


def test_build_goid_entries_none_end_line(
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries handles None end_lineno."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {}

    row = pd.Series({
        "path": TEST_PATH,
        "node_type": "FunctionDef",
        "name": "func",
        "qualname": "func",
        "lineno": 1,
        "end_lineno": None,
        "decorator_start_line": None,
        "decorator_end_line": None,
        "parent_qualname": None,
    })

    goid_row, _ = build_goid_entries_for_testing(row, goid_config, now, module_by_path)

    # Should not raise, end_line should be None
    assert goid_row.end_line is None


def test_build_goid_entries_nan_values(
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries handles NaN values."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {}

    row = pd.Series({
        "path": TEST_PATH,
        "node_type": "FunctionDef",
        "name": "func",
        "qualname": "func",
        "lineno": pd.NA,  # pandas NA
        "end_lineno": float("nan"),  # NaN
        "decorator_start_line": pd.NA,
        "decorator_end_line": None,
        "parent_qualname": None,
    })

    goid_row, _ = build_goid_entries_for_testing(row, goid_config, now, module_by_path)

    # Should handle NaN/NA gracefully
    assert goid_row.start_line is not None  # Falls back to default


def test_build_goid_entries_top_level_function(
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries identifies top-level function."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {}

    row = pd.Series({
        "path": TEST_PATH,
        "node_type": "FunctionDef",
        "name": "top_level_func",
        "qualname": "top_level_func",
        "lineno": 5,
        "end_lineno": 15,
        "decorator_start_line": None,
        "decorator_end_line": None,
        "parent_qualname": None,  # No parent = top-level
    })

    goid_row, _ = build_goid_entries_for_testing(row, goid_config, now, module_by_path)

    # Top-level function (no parent class)
    assert goid_row.kind == "function"


def test_build_goid_entries_init_file(
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries handles __init__.py files."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {"src/pkg/__init__.py": "src.pkg"}

    row = pd.Series({
        "path": "src/pkg/__init__.py",
        "node_type": "Module",
        "name": "__init__",
        "qualname": "__init__",
        "lineno": 1,
        "end_lineno": 10,
        "decorator_start_line": None,
        "decorator_end_line": None,
        "parent_qualname": None,
    })

    goid_row, xwalk_row = build_goid_entries_for_testing(
        row, goid_config, now, module_by_path
    )

    assert goid_row.kind == "module"
    assert xwalk_row.module_path == "src.pkg"


# ===========================================================================
# GOID Hash Consistency Tests
# ===========================================================================


def test_build_goid_entries_consistent_hash(
    sample_ast_row: pd.Series,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries produces consistent hash for same input."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {TEST_PATH: TEST_MODULE}

    goid_row1, _ = build_goid_entries_for_testing(
        sample_ast_row, goid_config, now, module_by_path
    )
    goid_row2, _ = build_goid_entries_for_testing(
        sample_ast_row, goid_config, now, module_by_path
    )

    # Same input should produce same GOID hash
    assert goid_row1.goid_h128 == goid_row2.goid_h128
    assert goid_row1.urn == goid_row2.urn


def test_build_goid_entries_different_hash_for_different_input(
    sample_ast_row: pd.Series,
    class_ast_row: pd.Series,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries produces different hash for different input."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {TEST_PATH: TEST_MODULE}

    goid_row1, _ = build_goid_entries_for_testing(
        sample_ast_row, goid_config, now, module_by_path
    )
    goid_row2, _ = build_goid_entries_for_testing(
        class_ast_row, goid_config, now, module_by_path
    )

    # Different input should produce different GOID hash
    assert goid_row1.goid_h128 != goid_row2.goid_h128


# ===========================================================================
# URN Format Tests
# ===========================================================================


def test_build_goid_entries_urn_format(
    sample_ast_row: pd.Series,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries produces valid URN format."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {TEST_PATH: TEST_MODULE}

    goid_row, _ = build_goid_entries_for_testing(
        sample_ast_row, goid_config, now, module_by_path
    )

    # URN should be well-formed
    assert goid_row.urn is not None
    assert len(goid_row.urn) > 0
    # URN typically contains repo, path, qualname info
    assert ":" in goid_row.urn or "/" in goid_row.urn


# ===========================================================================
# Timestamp Tests
# ===========================================================================


def test_build_goid_entries_uses_provided_timestamp(
    sample_ast_row: pd.Series,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Build GOID entries uses provided timestamp."""
    specific_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
    module_by_path: dict[str, str] = {TEST_PATH: TEST_MODULE}

    goid_row, xwalk_row = build_goid_entries_for_testing(
        sample_ast_row, goid_config, specific_time, module_by_path
    )

    assert goid_row.created_at == specific_time
    assert xwalk_row.updated_at == specific_time


# ===========================================================================
# Parametrized Tests
# ===========================================================================


@pytest.mark.parametrize(
    ("node_type", "parent_qualname", "expected_kind"),
    [
        ("Module", None, "module"),
        ("ClassDef", None, "class"),
        ("FunctionDef", None, "function"),
        ("FunctionDef", "MyClass", "method"),
        ("AsyncFunctionDef", None, "function"),
        ("AsyncFunctionDef", "MyClass", "method"),
    ],
)
def test_goid_kind_determination(
    node_type: str,
    parent_qualname: str | None,
    expected_kind: str,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """GOID kind is correctly determined from node type and parent."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {}

    row = pd.Series({
        "path": TEST_PATH,
        "node_type": node_type,
        "name": "test_entity",
        "qualname": "test_entity",
        "lineno": 1,
        "end_lineno": 10,
        "decorator_start_line": None,
        "decorator_end_line": None,
        "parent_qualname": parent_qualname,
    })

    goid_row, _ = build_goid_entries_for_testing(row, goid_config, now, module_by_path)

    assert goid_row.kind == expected_kind


@pytest.mark.parametrize(
    ("lineno", "decorator_start", "expected_start"),
    [
        (10, None, 10),  # No decorator
        (10, 8, 8),  # Decorator before function
        (10, 10, 10),  # Decorator on same line (edge case)
        (10, 12, 10),  # Decorator after lineno (shouldn't happen, but handle)
        (10, 0, 10),  # Zero decorator line
        (10, -1, 10),  # Negative decorator line
    ],
)
def test_start_line_computation(
    lineno: int,
    decorator_start: int | None,
    expected_start: int,
    goid_config: GoidBuilderStepConfig,
) -> None:
    """Start line computation handles decorator lines correctly."""
    now = datetime.now(UTC)
    module_by_path: dict[str, str] = {}

    row = pd.Series({
        "path": TEST_PATH,
        "node_type": "FunctionDef",
        "name": "func",
        "qualname": "func",
        "lineno": lineno,
        "end_lineno": 20,
        "decorator_start_line": decorator_start,
        "decorator_end_line": None,
        "parent_qualname": None,
    })

    goid_row, _ = build_goid_entries_for_testing(row, goid_config, now, module_by_path)

    assert goid_row.start_line == expected_start

