"""MCP seed pack for MCP backend tests.

This module provides the McpPack which seeds minimal data needed for
MCP (Model Context Protocol) backend tests including function types,
validation issues, call graph edges, and test catalog rows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.fixtures.rows import (
    CallGraphEdgeRow,
    FunctionTypesRow,
    FunctionValidationRow,
    TestCatalogRow,
    dataclass_row,
    insert_rows,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


DEFAULT_GOID: int = 1
CALLEE_GOID: int = 2
CALLER_GOID: int = 3
DEFAULT_URN: str = "urn:foo"
DEFAULT_PATH: str = "foo.py"
CALLER_PATH: str = "bar.py"


@dataclass
class McpPack:
    """Seed pack for MCP backend tests.

    Seeds minimal data needed for MCP backend tests:
    - Function types
    - Function validation issues
    - Call graph edges
    - Test catalog

    Attributes
    ----------
    name : str
        Unique pack identifier.
    repo : str
        Repository identifier.
    commit : str
        Commit hash.
    """

    name: str = "mcp"
    repo: str = "demo/repo"
    commit: str = "deadbeef"
    _dependencies: tuple[SeedPack, ...] = field(default_factory=tuple)

    @property
    def dependencies(self) -> tuple[SeedPack, ...]:
        """Return seed packs that must be applied before this one.

        Returns
        -------
        tuple[SeedPack, ...]
            No dependencies - this pack is self-contained.
        """
        return ()

    def apply(self, ctx: TestContext) -> None:
        """Apply MCP backend seeds to the test context.

        Seeds all tables required for MCP backend tests.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)
        repo = self.repo
        commit = self.commit

        self._cleanup_existing_data(ctx, repo, commit)

        self._seed_function_types(ctx, repo, commit, now)
        self._seed_function_validation(ctx, repo, commit, now)
        self._seed_call_graph_edges(ctx, repo, commit)
        self._seed_test_catalog(ctx, repo, commit, now)

    @staticmethod
    def _cleanup_existing_data(ctx: TestContext, repo: str, commit: str) -> None:
        """Remove existing data to ensure clean state."""
        con = ctx.gateway.con
        con.execute(
            "DELETE FROM analytics.function_types WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        con.execute(
            "DELETE FROM analytics.function_validation WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        con.execute(
            "DELETE FROM graph.call_graph_edges WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        con.execute(
            "DELETE FROM analytics.test_catalog WHERE repo = ? AND commit = ?",
            [repo, commit],
        )

    @staticmethod
    def _seed_function_types(ctx: TestContext, repo: str, commit: str, now: datetime) -> None:
        """Seed the analytics.function_types table."""
        rows = [
            dataclass_row(
                FunctionTypesRow,
                function_goid_h128=DEFAULT_GOID,
                urn=DEFAULT_URN,
                repo=repo,
                commit=commit,
                rel_path=DEFAULT_PATH,
                language="python",
                kind="function",
                qualname="foo",
                start_line=1,
                end_line=1,
                total_params=0,
                return_type=None,
                type_comment=None,
                param_types=None,
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_function_validation(ctx: TestContext, repo: str, commit: str, now: datetime) -> None:
        """Seed the analytics.function_validation table."""
        rows = [
            dataclass_row(
                FunctionValidationRow,
                repo=repo,
                commit=commit,
                function_goid_h128=DEFAULT_GOID,
                rel_path=DEFAULT_PATH,
                qualname="foo",
                issue="span_not_found",
                detail="Span 1-2",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_call_graph_edges(ctx: TestContext, repo: str, commit: str) -> None:
        """Seed the graph.call_graph_edges table."""
        rows = [
            dataclass_row(
                CallGraphEdgeRow,
                repo=repo,
                commit=commit,
                caller_goid_h128=DEFAULT_GOID,
                callee_goid_h128=CALLEE_GOID,
                callsite_path=DEFAULT_PATH,
                callsite_line=1,
                callsite_col=0,
                language="python",
                kind="direct",
                resolved_via="local_name",
                confidence=1.0,
            ),
            dataclass_row(
                CallGraphEdgeRow,
                repo=repo,
                commit=commit,
                caller_goid_h128=CALLER_GOID,
                callee_goid_h128=DEFAULT_GOID,
                callsite_path=CALLER_PATH,
                callsite_line=1,
                callsite_col=0,
                language="python",
                kind="direct",
                resolved_via="local_name",
                confidence=1.0,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_test_catalog(ctx: TestContext, repo: str, commit: str, now: datetime) -> None:
        """Seed the analytics.test_catalog table."""
        rows = [
            dataclass_row(
                TestCatalogRow,
                test_id="t1",
                repo=repo,
                commit=commit,
                rel_path="tests/t.py",
                qualname="tests.t",
                status="passed",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)


MCP_PACK = McpPack()

__all__ = ["MCP_PACK", "McpPack"]
