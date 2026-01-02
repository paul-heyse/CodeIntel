"""Unit tests for RunContext and related utilities.

These tests verify the core runtime context types and factory functions
used for unified run identity across engines.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from codeintel.config.primitives import SnapshotRef
from codeintel.core.execution import RunContext, new_run_context, new_run_id
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_in,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.core.execution.context import RunKind, TriggerKind


UNIQUENESS_SAMPLE_SIZE = 100
RUN_ID_HEX_SUFFIX_LENGTH = 32


@pytest.fixture
def snapshot(tmp_path: Path) -> SnapshotRef:
    """Create a test snapshot reference.

    Parameters
    ----------
    tmp_path
        Pytest temp path fixture.

    Returns
    -------
    SnapshotRef
        Test snapshot reference.
    """
    return SnapshotRef(
        repo="test-org/test-repo",
        commit="abc123def456",
        repo_root=tmp_path,
    )


class TestNewRunId:
    """Tests for new_run_id function."""

    @staticmethod
    def test_default_prefix() -> None:
        """Default prefix should be 'ci'."""
        run_id = new_run_id()
        expect_true(run_id.startswith("ci-"))

    @staticmethod
    def test_custom_prefix() -> None:
        """Custom prefix should be used."""
        run_id = new_run_id(prefix="ingest")
        expect_true(run_id.startswith("ingest-"))

    @staticmethod
    def test_uniqueness() -> None:
        """Each call should generate a unique ID."""
        ids = {new_run_id() for _ in range(UNIQUENESS_SAMPLE_SIZE)}
        expect_equal(len(ids), UNIQUENESS_SAMPLE_SIZE)

    @staticmethod
    def test_hex_suffix_length() -> None:
        """Suffix should be a 32-character hex string."""
        run_id = new_run_id(prefix="test")
        suffix = run_id.split("-", 1)[1]
        expect_equal(len(suffix), RUN_ID_HEX_SUFFIX_LENGTH)

        int(suffix, 16)


class TestRunContext:
    """Tests for RunContext dataclass."""

    @staticmethod
    def test_construction(snapshot: SnapshotRef) -> None:
        """RunContext should be constructed correctly."""
        ctx = RunContext(
            run_id="test-123",
            kind="full",
            snapshot=snapshot,
            trigger="cli",
        )
        expect_equal(ctx.run_id, "test-123")
        expect_equal(ctx.kind, "full")
        expect_equal(ctx.trigger, "cli")
        expect_true(ctx.requested_operation is None)
        expect_equal(ctx.requested_datasets, ())

    @staticmethod
    def test_with_operation(snapshot: SnapshotRef) -> None:
        """RunContext should support requested_operation."""
        ctx = RunContext(
            run_id="test-123",
            kind="op_prereqs",
            snapshot=snapshot,
            trigger="http",
            requested_operation="functions.summary",
        )
        expect_equal(ctx.requested_operation, "functions.summary")

    @staticmethod
    def test_with_datasets(snapshot: SnapshotRef) -> None:
        """RunContext should support requested_datasets."""
        ctx = RunContext(
            run_id="test-123",
            kind="analytics",
            snapshot=snapshot,
            trigger="api",
            requested_datasets=("analytics.function_types", "analytics.static_diagnostics"),
        )
        expect_equal(
            ctx.requested_datasets,
            ("analytics.function_types", "analytics.static_diagnostics"),
        )

    @staticmethod
    def test_repo_property(snapshot: SnapshotRef) -> None:
        """Repo property should delegate to snapshot."""
        ctx = RunContext(
            run_id="test-123",
            kind="full",
            snapshot=snapshot,
            trigger="cli",
        )
        expect_equal(ctx.repo, "test-org/test-repo")

    @staticmethod
    def test_commit_property(snapshot: SnapshotRef) -> None:
        """Commit property should delegate to snapshot."""
        ctx = RunContext(
            run_id="test-123",
            kind="full",
            snapshot=snapshot,
            trigger="cli",
        )
        expect_equal(ctx.commit, "abc123def456")

    @staticmethod
    def test_frozen(snapshot: SnapshotRef) -> None:
        """RunContext should be immutable."""
        ctx = RunContext(
            run_id="test-123",
            kind="full",
            snapshot=snapshot,
            trigger="cli",
        )
        assert_cannot_setattr(ctx, "run_id", "different")


class TestNewRunContext:
    """Tests for new_run_context factory function."""

    @staticmethod
    def test_generates_run_id(snapshot: SnapshotRef) -> None:
        """new_run_context should generate a unique run_id."""
        ctx = new_run_context(snapshot=snapshot, kind="full", trigger="cli")
        expect_true(ctx.run_id.startswith("full-"))
        expect_equal(len(ctx.run_id.split("-", 1)[1]), RUN_ID_HEX_SUFFIX_LENGTH)

    @staticmethod
    def test_ingest_kind_prefix(snapshot: SnapshotRef) -> None:
        """Ingest kind should use 'ingest-' prefix."""
        ctx = new_run_context(snapshot=snapshot, kind="ingest", trigger="cli")
        expect_true(ctx.run_id.startswith("ingest-"))

    @staticmethod
    def test_graphs_kind_prefix(snapshot: SnapshotRef) -> None:
        """Graphs kind should use 'graphs-' prefix."""
        ctx = new_run_context(snapshot=snapshot, kind="graphs", trigger="cli")
        expect_true(ctx.run_id.startswith("graphs-"))

    @staticmethod
    def test_analytics_kind_prefix(snapshot: SnapshotRef) -> None:
        """Analytics kind should use 'analytics-' prefix."""
        ctx = new_run_context(snapshot=snapshot, kind="analytics", trigger="cli")
        expect_true(ctx.run_id.startswith("analytics-"))

    @staticmethod
    def test_preserves_snapshot(snapshot: SnapshotRef) -> None:
        """new_run_context should preserve the snapshot reference."""
        ctx = new_run_context(snapshot=snapshot, kind="full", trigger="cli")
        expect_true(ctx.snapshot is snapshot)

    @staticmethod
    def test_requested_operation(snapshot: SnapshotRef) -> None:
        """new_run_context should accept requested_operation."""
        ctx = new_run_context(
            snapshot=snapshot,
            kind="op_prereqs",
            trigger="http",
            requested_operation="functions.summary",
        )
        expect_equal(ctx.requested_operation, "functions.summary")

    @staticmethod
    def test_requested_datasets_from_list(snapshot: SnapshotRef) -> None:
        """new_run_context should convert datasets list to tuple."""
        datasets = ["analytics.function_types", "analytics.static_diagnostics"]
        ctx = new_run_context(
            snapshot=snapshot,
            kind="analytics",
            trigger="api",
            requested_datasets=datasets,
        )

        expect_equal(ctx.requested_datasets, tuple(datasets))


class TestRunKindValues:
    """Tests for RunKind type values."""

    @staticmethod
    def test_all_run_kinds() -> None:
        """Verify all expected run kinds are available."""
        expected: set[RunKind] = {"ingest", "graphs", "analytics", "full", "op_prereqs"}
        valid_kinds = {"ingest", "graphs", "analytics", "full", "op_prereqs"}

        for kind in expected:
            expect_in(kind, valid_kinds)


class TestTriggerKindValues:
    """Tests for TriggerKind type values."""

    @staticmethod
    def test_all_trigger_kinds() -> None:
        """Verify all expected trigger kinds are available."""
        expected: set[TriggerKind] = {"cli", "http", "mcp", "api"}
        valid_triggers = {"cli", "http", "mcp", "api"}
        for kind in expected:
            expect_in(kind, valid_triggers)
