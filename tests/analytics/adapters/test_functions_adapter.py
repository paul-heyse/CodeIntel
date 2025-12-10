"""Test function adapter classes.

Test the function-specific adapters for loading GOIDs and persisting
function metrics and types using real DuckDB instances.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from codeintel.analytics.adapters.functions import (
    FunctionGoid,
    FunctionGoidLoader,
    FunctionMetricsAdapter,
    FunctionTypesAdapter,
    GoidRow,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.catalog import FunctionCatalog
from tests._helpers import assert_frozen
from tests._helpers.assertions import (
    expect_equal,
    expect_in,
    expect_is_instance,
    expect_length,
    expect_true,
)
from tests._helpers.catalogs import ensure_catalog_with_goids
from tests._helpers.context import TestContext, create_test_context
from tests._helpers.env_options import EnvOptions
from tests._helpers.rows import function_meta

# =============================================================================
# Constants
# =============================================================================

DEMO_REPO = "demo/repo"
DEMO_COMMIT = "abc123"
EXPECTED_GOID_COUNT_4 = 4
EXPECTED_FILE_COUNT_2 = 2
EXPECTED_MODULE_FUNCS_3 = 3
EXPECTED_OTHER_FUNCS_1 = 1
EXPECTED_INPUTS_4 = 4
TEST_GOID_123 = 123
TEST_START_LINE_5 = 5
TEST_START_LINE_7 = 7
TEST_END_LINE_10 = 10


# =============================================================================
# Test Data
# =============================================================================


def _goid_row(
    *,
    goid: int,
    qualname: str,
    rel_path: str = "src/module.py",
    start_line: int = 10,
    end_line: int | None = 20,
) -> GoidRow:
    """Build a GoidRow dictionary for FunctionGoid.from_row tests."""
    return {
        "goid_h128": goid,
        "urn": f"urn:{DEMO_REPO}:{DEMO_COMMIT}:{rel_path}#{qualname}",
        "repo": DEMO_REPO,
        "commit": DEMO_COMMIT,
        "rel_path": rel_path,
        "language": "python",
        "kind": "function",
        "qualname": qualname,
        "start_line": start_line,
        "end_line": end_line,
    }


def _function_catalog(repo: str, commit: str) -> FunctionCatalog:
    """Create a FunctionCatalog aligned with the test constants."""
    functions = [
        function_meta(
            goid=1001,
            rel_path="src/module.py",
            qualname="module.func_a",
            snapshot=(repo, commit),
            line_span=(10, 20),
        ),
        function_meta(
            goid=1002,
            rel_path="src/module.py",
            qualname="module.func_b",
            snapshot=(repo, commit),
            line_span=(25, 35),
        ),
        function_meta(
            goid=1003,
            rel_path="src/module.py",
            qualname="module.Class.method",
            snapshot=(repo, commit),
            line_span=(40, 50),
        ),
        function_meta(
            goid=2001,
            rel_path="src/other.py",
            qualname="other.func",
            snapshot=(repo, commit),
            line_span=(5, 15),
        ),
    ]
    modules = {
        "src/module.py": "module",
        "src/other.py": "other",
    }
    return FunctionCatalog(functions=functions, module_by_path=modules)


def _build_ctx(tmp_path: Path) -> TestContext:
    """Construct a TestContext pinned to the module constants."""
    options = EnvOptions(repo=DEMO_REPO, commit=DEMO_COMMIT)
    return create_test_context(tmp_path, options=options)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def goid_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """
    Create a context with function GOIDs seeded via catalog helper.

    Parameters
    ----------
    tmp_path
        Temporary directory for the test.

    Yields
    ------
    TestContext
        Context with GOIDs seeded.
    """
    ctx = _build_ctx(tmp_path)
    ensure_catalog_with_goids(ctx, _function_catalog(ctx.repo, ctx.commit))
    try:
        yield ctx
    finally:
        ctx.close()


@pytest.fixture
def empty_ctx(tmp_path: Path) -> Iterator[TestContext]:
    """
    Create a context without any GOIDs.

    Parameters
    ----------
    tmp_path
        Temporary directory for the test.

    Yields
    ------
    TestContext
        Empty context.
    """
    ctx = _build_ctx(tmp_path)
    try:
        yield ctx
    finally:
        ctx.close()


# =============================================================================
# FunctionGoid Tests
# =============================================================================


def test_function_goid_from_row() -> None:
    """Create FunctionGoid from database row."""
    row = _goid_row(
        goid=TEST_GOID_123,
        qualname="module.func",
        start_line=TEST_START_LINE_5,
        end_line=TEST_END_LINE_10,
    )
    goid = FunctionGoid.from_row(row)
    expect_equal(goid.goid, TEST_GOID_123)
    expect_equal(goid.qualname, "module.func")
    expect_equal(goid.start_line, TEST_START_LINE_5)
    expect_equal(goid.end_line, TEST_END_LINE_10)


def test_function_goid_from_row_null_end_line() -> None:
    """Handle null end_line by using start_line."""
    row = _goid_row(
        goid=TEST_GOID_123,
        qualname="module.func",
        start_line=TEST_START_LINE_7,
        end_line=None,
    )
    goid = FunctionGoid.from_row(row)
    expect_equal(goid.end_line, TEST_START_LINE_7)


def test_function_goid_from_row_normalizes_path() -> None:
    """Backslashes in rel_path are normalized to forward slashes."""
    row = _goid_row(goid=TEST_GOID_123, qualname="module.func")
    row["rel_path"] = "src\\windows\\module.py"
    goid = FunctionGoid.from_row(row)
    expect_true("\\" not in goid.rel_path)
    expect_equal(goid.rel_path, "src/windows/module.py")


def test_function_goid_is_frozen() -> None:
    """FunctionGoid is immutable."""
    row = _goid_row(goid=TEST_GOID_123, qualname="module.func")
    goid = FunctionGoid.from_row(row)
    assert_frozen(goid, "goid", 456)


# =============================================================================
# FunctionGoidLoader Tests
# =============================================================================


def test_loader_load_all(
    goid_ctx: TestContext,
) -> None:
    """Load all function and method GOIDs."""
    loader = FunctionGoidLoader(goid_ctx.gateway, goid_ctx.snapshot)
    goids = loader.load_all()
    expect_length(goids, EXPECTED_GOID_COUNT_4)


def test_loader_load_all_empty(
    empty_ctx: TestContext,
) -> None:
    """Load from empty table returns empty list."""
    loader = FunctionGoidLoader(empty_ctx.gateway, empty_ctx.snapshot)
    goids = loader.load_all()
    expect_true(not goids)


def test_loader_iter_goids(
    goid_ctx: TestContext,
) -> None:
    """Iterate over GOIDs."""
    loader = FunctionGoidLoader(goid_ctx.gateway, goid_ctx.snapshot)
    goids = list(loader.iter_goids())
    expect_length(goids, EXPECTED_GOID_COUNT_4)


def test_loader_group_by_file(
    goid_ctx: TestContext,
) -> None:
    """Group GOIDs by file path."""
    loader = FunctionGoidLoader(goid_ctx.gateway, goid_ctx.snapshot)
    by_file = loader.group_by_file()
    expect_length(by_file, EXPECTED_FILE_COUNT_2)
    # module.py has 3 functions/methods
    expect_in("src/module.py", by_file)
    expect_length(by_file["src/module.py"], EXPECTED_MODULE_FUNCS_3)
    # other.py has 1 function
    expect_in("src/other.py", by_file)
    expect_length(by_file["src/other.py"], EXPECTED_OTHER_FUNCS_1)


def test_loader_resolve_abs_path(
    goid_ctx: TestContext,
) -> None:
    """Resolve absolute path for a GOID."""
    loader = FunctionGoidLoader(goid_ctx.gateway, goid_ctx.snapshot)
    goids = loader.load_all()
    expect_true(bool(goids))
    abs_path = loader.resolve_abs_path(goids[0])
    expect_true(abs_path.is_absolute())
    # Path should include repo_root
    path_str = str(abs_path)
    expect_in(str(goid_ctx.repo_root), path_str)


def test_loader_filters_by_snapshot(
    goid_ctx: TestContext,
) -> None:
    """Loader filters by repo and commit."""
    # Different snapshot should find nothing
    other_snapshot = SnapshotRef(
        repo="other/repo",
        commit="different",
        repo_root=Path("/workspace"),
    )
    loader = FunctionGoidLoader(goid_ctx.gateway, other_snapshot)
    goids = loader.load_all()
    expect_true(not goids)


# =============================================================================
# FunctionMetricsAdapter Tests
# =============================================================================


def test_metrics_adapter_table_name(
    goid_ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = FunctionMetricsAdapter(goid_ctx.gateway, goid_ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.function_metrics")


def test_metrics_adapter_goid_loader_property(
    goid_ctx: TestContext,
) -> None:
    """Adapter exposes goid_loader property."""
    adapter = FunctionMetricsAdapter(goid_ctx.gateway, goid_ctx.snapshot)
    loader = adapter.goid_loader
    expect_is_instance(loader, FunctionGoidLoader)


def test_metrics_adapter_load_inputs(
    goid_ctx: TestContext,
) -> None:
    """Load inputs returns function GOIDs."""
    adapter = FunctionMetricsAdapter(goid_ctx.gateway, goid_ctx.snapshot)
    inputs = list(adapter.load_inputs())
    expect_length(inputs, EXPECTED_INPUTS_4)
    for inp in inputs:
        expect_is_instance(inp, FunctionGoid)


def test_metrics_adapter_load_outputs_empty() -> None:
    """Load outputs returns empty (metrics are computed)."""
    outputs = list(FunctionMetricsAdapter.load_outputs())
    expect_true(not outputs)


def test_metrics_adapter_load_empty(
    goid_ctx: TestContext,
) -> None:
    """Load returns empty (delegates to load_outputs)."""
    adapter = FunctionMetricsAdapter(goid_ctx.gateway, goid_ctx.snapshot)
    rows = list(adapter.load())
    expect_true(not rows)


def test_metrics_adapter_persist_empty(
    goid_ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = FunctionMetricsAdapter(goid_ctx.gateway, goid_ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, 0)


# =============================================================================
# FunctionTypesAdapter Tests
# =============================================================================


def test_types_adapter_table_name(
    goid_ctx: TestContext,
) -> None:
    """Adapter exposes correct table name."""
    adapter = FunctionTypesAdapter(goid_ctx.gateway, goid_ctx.snapshot)
    expect_equal(adapter.table_name, "analytics.function_types")


def test_types_adapter_load_raises_not_implemented(
    goid_ctx: TestContext,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = FunctionTypesAdapter(goid_ctx.gateway, goid_ctx.snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_types_adapter_persist_empty(
    goid_ctx: TestContext,
) -> None:
    """Persist empty list returns 0."""
    adapter = FunctionTypesAdapter(goid_ctx.gateway, goid_ctx.snapshot)
    count = adapter.persist([])
    expect_equal(count, 0)
