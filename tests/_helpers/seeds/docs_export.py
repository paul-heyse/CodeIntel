"""Docs export seed pack for documentation export validation tests.

This module provides the DocsExportPack which seeds comprehensive data needed
for docs export smoke tests and validation, including modules, GOIDs, graphs,
docstrings, metrics, types, coverage, and risk factors.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    CoverageFunctionRow,
    DocstringRow,
    FunctionMetricsRow,
    FunctionTypesRow,
    GoidCrosswalkRow,
    GoidRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RepoMapRow,
    RiskFactorRow,
    SymbolUseEdgeRow,
    TestCatalogRow,
    TestCoverageEdgeRow,
    insert_rows,
)

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Docs Export Data Constants
# =============================================================================

DEFAULT_GOID: int = 1
DEFAULT_MODULE: str = "pkg.foo"
DEFAULT_PATH: str = "foo.py"
DEFAULT_QUALNAME: str = "pkg.foo:func"
DEFAULT_URN: str = "urn:foo"


# =============================================================================
# Docs Export Pack Implementation
# =============================================================================


@dataclass
class DocsExportPack:
    """Seed pack for docs export validation data.

    Seeds comprehensive data needed for docs export smoke tests including:
    - Repository map
    - Modules and GOIDs
    - Call graph nodes and edges
    - CFG blocks
    - Import graph edges
    - Symbol use edges
    - Docstrings
    - Function metrics and types
    - Coverage functions
    - Risk factors
    - Test catalog and coverage edges
    - GOID crosswalk

    Attributes
    ----------
    name : str
        Unique pack identifier.
    repo : str
        Repository identifier.
    commit : str
        Commit hash.
    """

    name: str = "docs_export"
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
        """Apply docs export seeds to the test context.

        Seeds all tables required for docs export validation tests.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)
        goid = DEFAULT_GOID
        repo = self.repo
        commit = self.commit

        # Clean up any existing data first
        self._cleanup_existing_data(ctx, repo, commit, goid)

        # Seed all tables
        self._seed_repo_map(ctx, repo, commit)
        self._seed_modules(ctx, repo, commit)
        self._seed_goids(ctx, repo, commit, goid, now)
        self._seed_goid_crosswalk(ctx, repo, commit, now)
        self._seed_call_graph_nodes(ctx, goid)
        self._seed_call_graph_edges(ctx, repo, commit, goid)
        self._seed_cfg_blocks(ctx, goid)
        self._seed_import_graph_edges(ctx, repo, commit)
        self._seed_symbol_use_edges(ctx, goid)
        self._seed_docstrings(ctx, repo, commit, now)
        self._seed_function_metrics(ctx, repo, commit, goid, now)
        self._seed_function_types(ctx, repo, commit, goid, now)
        self._seed_coverage_functions(ctx, repo, commit, goid, now)
        self._seed_risk_factors(ctx, repo, commit, goid, now)
        self._seed_test_catalog(ctx, repo, commit, now)
        self._seed_test_coverage_edges(ctx, repo, commit, goid, now)

    @staticmethod
    def _cleanup_existing_data(
        ctx: TestContext,
        repo: str,
        commit: str,
        goid: int,
    ) -> None:
        """Remove existing data to ensure clean state."""
        con = ctx.gateway.con
        con.execute("DELETE FROM core.repo_map WHERE repo = ? AND commit = ?", [repo, commit])
        con.execute("DELETE FROM core.modules WHERE repo = ? AND commit = ?", [repo, commit])
        con.execute("DELETE FROM core.goids WHERE repo = ? AND commit = ?", [repo, commit])
        con.execute("DELETE FROM core.goid_crosswalk WHERE repo = ? AND commit = ?", [repo, commit])
        con.execute("DELETE FROM graph.call_graph_nodes WHERE goid_h128 = ?", [goid])
        con.execute(
            "DELETE FROM graph.call_graph_edges WHERE repo = ? AND commit = ?", [repo, commit]
        )
        con.execute("DELETE FROM graph.cfg_blocks WHERE function_goid_h128 = ?", [goid])
        con.execute(
            "DELETE FROM graph.import_graph_edges WHERE repo = ? AND commit = ?", [repo, commit]
        )
        con.execute("DELETE FROM graph.symbol_use_edges WHERE symbol = 'sym'")
        con.execute(
            "DELETE FROM analytics.test_catalog WHERE repo = ? AND commit = ?", [repo, commit]
        )
        con.execute(
            "DELETE FROM analytics.test_coverage_edges WHERE repo = ? AND commit = ?",
            [repo, commit],
        )

    @staticmethod
    def _seed_repo_map(ctx: TestContext, repo: str, commit: str) -> None:
        """Seed the core.repo_map table."""
        rows = [
            RepoMapRow(
                repo=repo,
                commit=commit,
                modules={DEFAULT_MODULE: DEFAULT_PATH},
                overlays={},
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_modules(ctx: TestContext, repo: str, commit: str) -> None:
        """Seed the core.modules table."""
        rows = [
            ModuleRow(
                module=DEFAULT_MODULE,
                path=DEFAULT_PATH,
                repo=repo,
                commit=commit,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_goids(ctx: TestContext, repo: str, commit: str, goid: int, now: datetime) -> None:
        """Seed the core.goids table."""
        rows = [
            GoidRow(
                goid_h128=goid,
                urn=DEFAULT_URN,
                repo=repo,
                commit=commit,
                rel_path=DEFAULT_PATH,
                kind="function",
                qualname=DEFAULT_QUALNAME,
                start_line=1,
                end_line=10,
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_goid_crosswalk(ctx: TestContext, repo: str, commit: str, now: datetime) -> None:
        """Seed the core.goid_crosswalk table."""
        rows = [
            GoidCrosswalkRow(
                repo=repo,
                commit=commit,
                goid=DEFAULT_URN,
                lang="python",
                module_path=DEFAULT_MODULE,
                file_path=DEFAULT_PATH,
                start_line=1,
                end_line=10,
                scip_symbol="scip-python foo",
                ast_qualname=DEFAULT_QUALNAME,
                cst_node_id=None,
                chunk_id=None,
                symbol_id=None,
                updated_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_call_graph_nodes(ctx: TestContext, goid: int) -> None:
        """Seed the graph.call_graph_nodes table."""
        rows = [
            CallGraphNodeRow(
                goid_h128=goid,
                language="python",
                kind="function",
                arity=0,
                is_public=True,
                rel_path=DEFAULT_PATH,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_call_graph_edges(ctx: TestContext, repo: str, commit: str, goid: int) -> None:
        """Seed the graph.call_graph_edges table."""
        rows = [
            CallGraphEdgeRow(
                repo=repo,
                commit=commit,
                caller_goid_h128=goid,
                callee_goid_h128=goid,
                callsite_path=DEFAULT_PATH,
                callsite_line=1,
                callsite_col=0,
                language="python",
                kind="direct",
                resolved_via="local_name",
                confidence=1.0,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_cfg_blocks(ctx: TestContext, goid: int) -> None:
        """Seed the graph.cfg_blocks table."""
        rows = [
            CFGBlockRow(
                function_goid_h128=goid,
                block_idx=0,
                block_id=f"{goid}:block0",
                label="entry",
                file_path=DEFAULT_PATH,
                start_line=1,
                end_line=1,
                kind="entry",
                stmts_json="[]",
                in_degree=0,
                out_degree=0,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_import_graph_edges(ctx: TestContext, repo: str, commit: str) -> None:
        """Seed the graph.import_graph_edges table."""
        rows = [
            ImportGraphEdgeRow(
                repo=repo,
                commit=commit,
                src_module=DEFAULT_MODULE,
                dst_module="pkg.bar",
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=1,
                module_layer=0,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_symbol_use_edges(ctx: TestContext, goid: int) -> None:
        """Seed the graph.symbol_use_edges table."""
        rows = [
            SymbolUseEdgeRow(
                symbol="sym",
                def_path=DEFAULT_PATH,
                use_path=DEFAULT_PATH,
                same_file=True,
                same_module=True,
                def_goid_h128=goid,
                use_goid_h128=goid,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_docstrings(ctx: TestContext, repo: str, commit: str, now: datetime) -> None:
        """Seed the analytics.docstrings table."""
        rows = [
            DocstringRow(
                repo=repo,
                commit=commit,
                rel_path=DEFAULT_PATH,
                module=DEFAULT_MODULE,
                qualname=DEFAULT_QUALNAME,
                kind="function",
                lineno=1,
                end_lineno=1,
                raw_docstring="demo",
                style="auto",
                short_desc="demo",
                long_desc="",
                params_json="[]",
                returns_json='{"type": "str"}',
                raises_json="[]",
                examples_json="[]",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_function_metrics(
        ctx: TestContext, repo: str, commit: str, goid: int, now: datetime
    ) -> None:
        """Seed the analytics.function_metrics table."""
        rows = [
            FunctionMetricsRow(
                function_goid_h128=goid,
                urn=DEFAULT_URN,
                repo=repo,
                commit=commit,
                rel_path=DEFAULT_PATH,
                language="python",
                kind="function",
                qualname=DEFAULT_QUALNAME,
                start_line=1,
                end_line=10,
                loc=10,
                logical_loc=10,
                param_count=1,
                positional_params=1,
                keyword_only_params=0,
                has_varargs=False,
                has_varkw=False,
                is_async=False,
                is_generator=False,
                return_count=1,
                yield_count=0,
                raise_count=0,
                cyclomatic_complexity=1,
                max_nesting_depth=1,
                stmt_count=1,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="low",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_function_types(
        ctx: TestContext, repo: str, commit: str, goid: int, now: datetime
    ) -> None:
        """Seed the analytics.function_types table."""
        rows = [
            FunctionTypesRow(
                function_goid_h128=goid,
                urn=DEFAULT_URN,
                repo=repo,
                commit=commit,
                rel_path=DEFAULT_PATH,
                language="python",
                kind="function",
                qualname=DEFAULT_QUALNAME,
                start_line=1,
                end_line=10,
                total_params=1,
                annotated_params=1,
                unannotated_params=0,
                param_typed_ratio=1.0,
                has_return_annotation=True,
                return_type="str",
                return_type_source="annotation",
                type_comment=None,
                param_types_json="{}",
                fully_typed=True,
                partial_typed=False,
                untyped=False,
                typedness_bucket="typed",
                typedness_source="pyright",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_coverage_functions(
        ctx: TestContext, repo: str, commit: str, goid: int, now: datetime
    ) -> None:
        """Seed the coverage.functions table."""
        rows = [
            CoverageFunctionRow(
                function_goid_h128=goid,
                urn=DEFAULT_URN,
                repo=repo,
                commit=commit,
                rel_path=DEFAULT_PATH,
                language="python",
                kind="function",
                qualname=DEFAULT_QUALNAME,
                start_line=1,
                end_line=10,
                executable_lines=1,
                covered_lines=1,
                coverage_ratio=1.0,
                tested=True,
                untested_reason=None,
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_risk_factors(
        ctx: TestContext, repo: str, commit: str, goid: int, now: datetime
    ) -> None:
        """Seed the analytics.goid_risk_factors table."""
        rows = [
            RiskFactorRow(
                function_goid_h128=goid,
                urn=DEFAULT_URN,
                repo=repo,
                commit=commit,
                rel_path=DEFAULT_PATH,
                language="python",
                kind="function",
                qualname=DEFAULT_QUALNAME,
                loc=10,
                logical_loc=10,
                cyclomatic_complexity=1,
                complexity_bucket="low",
                typedness_bucket="typed",
                typedness_source="pyright",
                hotspot_score=0.0,
                file_typed_ratio=1.0,
                static_error_count=0,
                has_static_errors=False,
                executable_lines=1,
                covered_lines=1,
                coverage_ratio=1.0,
                tested=True,
                test_count=1,
                failing_test_count=0,
                last_test_status="passed",
                risk_score=0.1,
                risk_level="low",
                tags="[]",
                owners="[]",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_test_catalog(ctx: TestContext, repo: str, commit: str, now: datetime) -> None:
        """Seed the analytics.test_catalog table."""
        rows = [
            TestCatalogRow(
                test_id="t1",
                repo=repo,
                commit=commit,
                rel_path=DEFAULT_PATH,
                qualname="pkg.foo::test_func",
                status="passed",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_test_coverage_edges(
        ctx: TestContext, repo: str, commit: str, goid: int, now: datetime
    ) -> None:
        """Seed the analytics.test_coverage_edges table."""
        rows = [
            TestCoverageEdgeRow(
                test_id="t1",
                function_goid_h128=goid,
                urn=DEFAULT_URN,
                repo=repo,
                commit=commit,
                rel_path=DEFAULT_PATH,
                qualname=DEFAULT_QUALNAME,
                covered_lines=1,
                executable_lines=1,
                coverage_ratio=1.0,
                last_status="passed",
                created_at=now,
                test_goid_h128=None,
            )
        ]
        insert_rows(ctx.gateway, rows)


DOCS_EXPORT_PACK = DocsExportPack()

__all__ = ["DOCS_EXPORT_PACK", "DocsExportPack"]
