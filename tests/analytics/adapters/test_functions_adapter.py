"""Test function adapter classes.

Test the function-specific adapters for loading GOIDs and persisting
function metrics and types using real DuckDB instances.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
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
from codeintel.storage.gateway import StorageGateway

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


@dataclass(frozen=True)
class GoidSeedParams:
    """Parameters for seeding a GOID."""

    goid: int
    qualname: str
    rel_path: str = "src/module.py"
    kind: str = "function"
    start_line: int = 10
    end_line: int | None = 20


def _make_goid_row(params: GoidSeedParams) -> GoidRow:
    """
    Create a GoidRow for testing.

    Parameters
    ----------
    params
        Parameters for the GOID.

    Returns
    -------
    GoidRow
        A test GoidRow.
    """
    return {
        "goid_h128": params.goid,
        "urn": f"urn:demo:repo::{params.qualname}",
        "repo": DEMO_REPO,
        "commit": DEMO_COMMIT,
        "rel_path": params.rel_path,
        "language": "python",
        "kind": params.kind,
        "qualname": params.qualname,
        "start_line": params.start_line,
        "end_line": params.end_line,
    }


def _seed_goids(gateway: StorageGateway, goids: list[GoidRow]) -> None:
    """
    Seed GOID rows into the database.

    Parameters
    ----------
    gateway
        Storage gateway.
    goids
        List of GOID rows to seed.
    """
    now = datetime.now(tz=UTC)
    for goid in goids:
        gateway.con.execute(
            """
            INSERT INTO core.goids (
                goid_h128, urn, repo, commit, rel_path,
                language, kind, qualname, start_line, end_line, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                goid["goid_h128"],
                goid["urn"],
                goid["repo"],
                goid["commit"],
                goid["rel_path"],
                goid["language"],
                goid["kind"],
                goid["qualname"],
                goid["start_line"],
                goid["end_line"],
                now,
            ],
        )


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def gateway_with_goids(fresh_gateway: StorageGateway) -> StorageGateway:
    """
    Create gateway with GOID table seeded.

    Parameters
    ----------
    fresh_gateway
        Base gateway from main conftest.

    Returns
    -------
    StorageGateway
        Gateway with test GOIDs.
    """
    # Seed some test GOIDs
    _seed_goids(
        fresh_gateway,
        [
            _make_goid_row(
                GoidSeedParams(1001, "module.func_a", "src/module.py", "function", 10, 20)
            ),
            _make_goid_row(
                GoidSeedParams(1002, "module.func_b", "src/module.py", "function", 25, 35)
            ),
            _make_goid_row(
                GoidSeedParams(1003, "module.Class.method", "src/module.py", "method", 40, 50)
            ),
            _make_goid_row(GoidSeedParams(2001, "other.func", "src/other.py", "function", 5, 15)),
            # Class kind should be excluded
            _make_goid_row(
                GoidSeedParams(9001, "module.MyClass", "src/module.py", "class", 1, 100)
            ),
        ],
    )
    return fresh_gateway


@pytest.fixture
def empty_gateway(fresh_gateway: StorageGateway) -> StorageGateway:
    """
    Create gateway without any GOIDs.

    Parameters
    ----------
    fresh_gateway
        Base gateway from main conftest.

    Returns
    -------
    StorageGateway
        Empty gateway.
    """
    return fresh_gateway


@pytest.fixture
def snapshot() -> SnapshotRef:
    """
    Create snapshot reference.

    Returns
    -------
    SnapshotRef
        Snapshot reference for testing.
    """
    return SnapshotRef(
        repo=DEMO_REPO,
        commit=DEMO_COMMIT,
        repo_root=Path("/workspace/demo"),
    )


# =============================================================================
# FunctionGoid Tests
# =============================================================================


def test_function_goid_from_row() -> None:
    """Create FunctionGoid from database row."""
    row = _make_goid_row(
        GoidSeedParams(
            TEST_GOID_123, "module.func", start_line=TEST_START_LINE_5, end_line=TEST_END_LINE_10
        )
    )
    goid = FunctionGoid.from_row(row)
    assert goid.goid == TEST_GOID_123
    assert goid.qualname == "module.func"
    assert goid.start_line == TEST_START_LINE_5
    assert goid.end_line == TEST_END_LINE_10


def test_function_goid_from_row_null_end_line() -> None:
    """Handle null end_line by using start_line."""
    row = _make_goid_row(
        GoidSeedParams(TEST_GOID_123, "module.func", start_line=TEST_START_LINE_7, end_line=None)
    )
    goid = FunctionGoid.from_row(row)
    assert goid.end_line == TEST_START_LINE_7


def test_function_goid_from_row_normalizes_path() -> None:
    """Backslashes in rel_path are normalized to forward slashes."""
    row = _make_goid_row(GoidSeedParams(TEST_GOID_123, "module.func"))
    row["rel_path"] = "src\\windows\\module.py"
    goid = FunctionGoid.from_row(row)
    assert "\\" not in goid.rel_path
    assert goid.rel_path == "src/windows/module.py"


def test_function_goid_is_frozen() -> None:
    """FunctionGoid is immutable."""
    row = _make_goid_row(GoidSeedParams(TEST_GOID_123, "module.func"))
    goid = FunctionGoid.from_row(row)
    with pytest.raises(AttributeError):
        goid.goid = 456  # type: ignore[misc]


# =============================================================================
# FunctionGoidLoader Tests
# =============================================================================


def test_loader_load_all(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load all function and method GOIDs."""
    loader = FunctionGoidLoader(gateway_with_goids, snapshot)
    goids = loader.load_all()
    # Should exclude class (9001), include functions and methods
    assert len(goids) == EXPECTED_GOID_COUNT_4


def test_loader_load_all_empty(
    empty_gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load from empty table returns empty list."""
    loader = FunctionGoidLoader(empty_gateway, snapshot)
    goids = loader.load_all()
    assert not goids


def test_loader_iter_goids(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Iterate over GOIDs."""
    loader = FunctionGoidLoader(gateway_with_goids, snapshot)
    goids = list(loader.iter_goids())
    assert len(goids) == EXPECTED_GOID_COUNT_4


def test_loader_group_by_file(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Group GOIDs by file path."""
    loader = FunctionGoidLoader(gateway_with_goids, snapshot)
    by_file = loader.group_by_file()
    assert len(by_file) == EXPECTED_FILE_COUNT_2
    # module.py has 3 functions/methods
    assert "src/module.py" in by_file
    assert len(by_file["src/module.py"]) == EXPECTED_MODULE_FUNCS_3
    # other.py has 1 function
    assert "src/other.py" in by_file
    assert len(by_file["src/other.py"]) == EXPECTED_OTHER_FUNCS_1


def test_loader_resolve_abs_path(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Resolve absolute path for a GOID."""
    loader = FunctionGoidLoader(gateway_with_goids, snapshot)
    goids = loader.load_all()
    assert goids
    abs_path = loader.resolve_abs_path(goids[0])
    assert abs_path.is_absolute()
    # Path should include repo_root
    path_str = str(abs_path)
    assert "workspace/demo" in path_str


def test_loader_filters_by_snapshot(
    gateway_with_goids: StorageGateway,
) -> None:
    """Loader filters by repo and commit."""
    # Different snapshot should find nothing
    other_snapshot = SnapshotRef(
        repo="other/repo",
        commit="different",
        repo_root=Path("/workspace"),
    )
    loader = FunctionGoidLoader(gateway_with_goids, other_snapshot)
    goids = loader.load_all()
    assert not goids


# =============================================================================
# FunctionMetricsAdapter Tests
# =============================================================================


def test_metrics_adapter_table_name(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = FunctionMetricsAdapter(gateway_with_goids, snapshot)
    assert adapter.table_name == "analytics.function_metrics"


def test_metrics_adapter_goid_loader_property(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes goid_loader property."""
    adapter = FunctionMetricsAdapter(gateway_with_goids, snapshot)
    loader = adapter.goid_loader
    assert isinstance(loader, FunctionGoidLoader)


def test_metrics_adapter_load_inputs(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load inputs returns function GOIDs."""
    adapter = FunctionMetricsAdapter(gateway_with_goids, snapshot)
    inputs = list(adapter.load_inputs())
    assert len(inputs) == EXPECTED_INPUTS_4
    for inp in inputs:
        assert isinstance(inp, FunctionGoid)


def test_metrics_adapter_load_outputs_empty() -> None:
    """Load outputs returns empty (metrics are computed)."""
    outputs = list(FunctionMetricsAdapter.load_outputs())
    assert not outputs


def test_metrics_adapter_load_empty(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load returns empty (delegates to load_outputs)."""
    adapter = FunctionMetricsAdapter(gateway_with_goids, snapshot)
    rows = list(adapter.load())
    assert not rows


def test_metrics_adapter_persist_empty(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = FunctionMetricsAdapter(gateway_with_goids, snapshot)
    count = adapter.persist([])
    assert count == 0


# =============================================================================
# FunctionTypesAdapter Tests
# =============================================================================


def test_types_adapter_table_name(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Adapter exposes correct table name."""
    adapter = FunctionTypesAdapter(gateway_with_goids, snapshot)
    assert adapter.table_name == "analytics.function_types"


def test_types_adapter_load_raises_not_implemented(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Load raises NotImplementedError (write-only adapter)."""
    adapter = FunctionTypesAdapter(gateway_with_goids, snapshot)
    with pytest.raises(NotImplementedError, match="does not support loading"):
        list(adapter.load())


def test_types_adapter_persist_empty(
    gateway_with_goids: StorageGateway,
    snapshot: SnapshotRef,
) -> None:
    """Persist empty list returns 0."""
    adapter = FunctionTypesAdapter(gateway_with_goids, snapshot)
    count = adapter.persist([])
    assert count == 0
