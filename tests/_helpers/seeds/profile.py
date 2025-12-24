"""Profile seed pack for analytics profile tests.

This module provides the ProfilePack which seeds comprehensive profile-related
data needed for analytics tests including modules, metrics, types, coverage,
docstrings, risk factors, hotspots, typedness, and static diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.assertions import ModulesAssertions
from tests._helpers.fixtures.rows import (
    AstMetricsRow,
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CoverageFunctionRow,
    DocstringRow,
    FunctionMetricsRow,
    FunctionTypesRow,
    HotspotRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RepoMapRow,
    RiskFactorRow,
    StaticDiagnosticsRow,
    TestCatalogRow,
    TestCoverageEdgeRow,
    TypednessRow,
    dataclass_row,
    insert_rows,
)
from tests._helpers.modules_expectations import modules_expected_from_repo_tree

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


# =============================================================================
# Profile Data Constants
# =============================================================================

DEFAULT_GOID: int = 1
TEST_GOID: int = 2
CALLER_GOID: int = 3
DEFAULT_QUALNAME: str = "pkg.mod.func"
DEFAULT_URN_PREFIX: str = "goid:demo/repo#python:function:"


# =============================================================================
# Profile Pack Implementation
# =============================================================================


@dataclass
class ProfilePack:
    """Seed pack for analytics profile data.

    Seeds comprehensive profile-related data needed for analytics tests:
    - Modules with tags and owners
    - AST metrics
    - Hotspots
    - Typedness
    - Static diagnostics
    - Docstrings
    - Risk factors
    - Function metrics and types
    - Coverage functions
    - Test catalog and coverage edges
    - Call graph nodes and edges
    - Import graph edges

    Attributes
    ----------
    name : str
        Unique pack identifier.
    repo : str
        Repository identifier.
    commit : str
        Commit hash.
    rel_path : str
        Relative file path.
    module : str
        Module name.
    """

    name: str = "profile"
    repo: str = "demo/repo"
    commit: str = "deadbeef"
    rel_path: str = "pkg/mod.py"
    module: str = "pkg.mod"
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
        """Apply profile seeds to the test context.

        Seeds all tables required for analytics profile tests.

        Parameters
        ----------
        ctx
            Test context to seed.
        """
        now = datetime.now(UTC)

        # Clean up existing data
        self._cleanup_existing_data(ctx)

        # Seed all tables
        module_map = self._resolve_module_map(ctx)
        self._seed_repo_map(ctx, module_map)
        self._seed_modules(ctx, module_map)
        ModulesAssertions(ctx.gateway, ctx.snapshot).inventory_consistent()
        self._seed_ast_metrics(ctx, now)
        self._seed_hotspots(ctx)
        self._seed_typedness(ctx)
        self._seed_static_diagnostics(ctx)
        self._seed_docstrings(ctx, now)
        self._seed_risk_factors(ctx)
        self._seed_function_metrics(ctx, now)
        self._seed_function_types(ctx, now)
        self._seed_coverage_functions(ctx, now)
        self._seed_test_catalog(ctx, now)
        self._seed_test_coverage_edges(ctx, now)
        self._seed_call_graph_nodes(ctx)
        self._seed_call_graph_edges(ctx)
        self._seed_import_graph_edges(ctx)

    def _cleanup_existing_data(self, ctx: TestContext) -> None:
        """Remove existing data to ensure clean state."""
        con = ctx.gateway.con
        con.execute(
            "DELETE FROM analytics.typedness WHERE path = ? AND repo = ? AND commit = ?",
            [self.rel_path, self.repo, self.commit],
        )
        con.execute(
            "DELETE FROM analytics.static_diagnostics WHERE rel_path = ? AND repo = ? "
            "AND commit = ?",
            [self.rel_path, self.repo, self.commit],
        )
        con.execute(
            "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
            [self.repo, self.commit],
        )
        con.execute(
            "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
            [self.repo, self.commit],
        )

    def _seed_repo_map(self, ctx: TestContext, module_map: dict[str, str]) -> None:
        """Seed the core.repo_map table."""
        rows = [
            dataclass_row(
                RepoMapRow,
                repo=self.repo,
                commit=self.commit,
                modules=module_map,
                overlays={},
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_modules(self, ctx: TestContext, module_map: dict[str, str]) -> None:
        """Seed the core.modules table."""
        rows = [
            dataclass_row(
                ModuleRow,
                module=module,
                path=path,
                repo=self.repo,
                commit=self.commit,
                tags='["server"]',
                owners='["team@example.com"]',
            )
            for module, path in sorted(module_map.items())
        ]
        insert_rows(ctx.gateway, rows)

    def _resolve_module_map(self, ctx: TestContext) -> dict[str, str]:
        path_map = modules_expected_from_repo_tree(ctx.repo_root)
        module_map = {module: path for path, module in path_map.items()}
        if not module_map:
            module_map = {self.module: self.rel_path}
        return module_map

    def _seed_ast_metrics(self, ctx: TestContext, now: datetime) -> None:
        """Seed the analytics.ast_metrics table."""
        rows = [
            dataclass_row(
                AstMetricsRow,
                rel_path=self.rel_path,
                node_count=10,
                function_count=1,
                class_count=0,
                avg_depth=1.0,
                max_depth=1,
                complexity=2.0,
                generated_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_hotspots(self, ctx: TestContext) -> None:
        """Seed the analytics.hotspots table."""
        rows = [
            dataclass_row(
                HotspotRow,
                rel_path=self.rel_path,
                commit_count=1,
                author_count=1,
                lines_added=5,
                lines_deleted=1,
                complexity=2.0,
                score=0.5,
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_typedness(self, ctx: TestContext) -> None:
        """Seed the analytics.typedness table."""
        rows = [
            dataclass_row(
                TypednessRow,
                repo=self.repo,
                commit=self.commit,
                path=self.rel_path,
                type_error_count=1,
                annotation_ratio='{"params": 0.5}',
                untyped_defs=0,
                overlay_needed=False,
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_static_diagnostics(self, ctx: TestContext) -> None:
        """Seed the analytics.static_diagnostics table."""
        rows = [
            dataclass_row(
                StaticDiagnosticsRow,
                repo=self.repo,
                commit=self.commit,
                rel_path=self.rel_path,
                pyrefly_errors=1,
                pyright_errors=0,
                ruff_errors=0,
                total_errors=1,
                has_errors=True,
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_docstrings(self, ctx: TestContext, now: datetime) -> None:
        """Seed the analytics.docstrings table."""
        rows = [
            dataclass_row(
                DocstringRow,
                repo=self.repo,
                commit=self.commit,
                rel_path=self.rel_path,
                module=self.module,
                qualname=DEFAULT_QUALNAME,
                kind="function",
                lineno=1,
                end_lineno=2,
                raw_docstring="Doc",
                style="auto",
                short_desc="Short doc",
                long_desc="Longer doc",
                params_json="[]",
                returns_json='{"return": "int"}',
                raises_json="[]",
                examples_json="[]",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_risk_factors(self, ctx: TestContext) -> None:
        """Seed the analytics.goid_risk_factors table."""
        rows = [
            dataclass_row(
                RiskFactorRow,
                function_goid_h128=DEFAULT_GOID,
                repo=self.repo,
                commit=self.commit,
                risk_score=9,
                risk_level="high",
                cyclomatic_complexity=2,
                fan_in_count=1,
                fan_out_count=1,
                has_tests=True,
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_function_metrics(self, ctx: TestContext, now: datetime) -> None:
        """Seed the analytics.function_metrics table."""
        rows = [
            dataclass_row(
                FunctionMetricsRow,
                function_goid_h128=DEFAULT_GOID,
                urn=f"{DEFAULT_URN_PREFIX}{DEFAULT_QUALNAME}",
                repo=self.repo,
                commit=self.commit,
                rel_path=self.rel_path,
                language="python",
                kind="function",
                qualname=DEFAULT_QUALNAME,
                start_line=1,
                end_line=2,
                loc=4,
                logical_loc=3,
                param_count=2,
                positional_params=1,
                keyword_only_params=1,
                has_varargs=True,
                has_varkw=False,
                is_async=False,
                is_generator=False,
                return_count=1,
                yield_count=0,
                raise_count=0,
                cyclomatic_complexity=2,
                max_nesting_depth=1,
                stmt_count=2,
                decorator_count=0,
                has_docstring=True,
                complexity_bucket="medium",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_function_types(self, ctx: TestContext, now: datetime) -> None:
        """Seed the analytics.function_types table."""
        rows = [
            dataclass_row(
                FunctionTypesRow,
                function_goid_h128=DEFAULT_GOID,
                urn=f"{DEFAULT_URN_PREFIX}{DEFAULT_QUALNAME}",
                repo=self.repo,
                commit=self.commit,
                rel_path=self.rel_path,
                language="python",
                kind="function",
                qualname=DEFAULT_QUALNAME,
                start_line=1,
                end_line=2,
                total_params=2,
                annotated_params=2,
                unannotated_params=0,
                param_typed_ratio=1.0,
                has_return_annotation=True,
                return_type="int",
                return_type_source="annotation",
                type_comment=None,
                param_types_json="[]",
                fully_typed=True,
                partial_typed=False,
                untyped=False,
                typedness_bucket="typed",
                typedness_source="analysis",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_coverage_functions(self, ctx: TestContext, now: datetime) -> None:
        """Seed the coverage.functions table."""
        rows = [
            dataclass_row(
                CoverageFunctionRow,
                function_goid_h128=DEFAULT_GOID,
                urn=f"{DEFAULT_URN_PREFIX}{DEFAULT_QUALNAME}",
                repo=self.repo,
                commit=self.commit,
                rel_path=self.rel_path,
                language="python",
                kind="function",
                qualname=DEFAULT_QUALNAME,
                start_line=1,
                end_line=2,
                executable_lines=4,
                covered_lines=2,
                coverage_ratio=0.5,
                tested=True,
                untested_reason="",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_test_catalog(self, ctx: TestContext, now: datetime) -> None:
        """Seed the analytics.test_catalog table."""
        rows = [
            dataclass_row(
                TestCatalogRow,
                test_id="pkg/mod.py::test_func",
                test_goid_h128=TEST_GOID,
                urn=f"{DEFAULT_URN_PREFIX}pkg.mod.test_func",
                repo=self.repo,
                commit=self.commit,
                rel_path=self.rel_path,
                qualname="pkg.mod.test_func",
                kind="function",
                status="failed",
                duration_ms=1500,
                markers="[]",
                parametrized=False,
                flaky=True,
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_test_coverage_edges(self, ctx: TestContext, now: datetime) -> None:
        """Seed the analytics.test_coverage_edges table."""
        rows = [
            dataclass_row(
                TestCoverageEdgeRow,
                test_id="pkg/mod.py::test_func",
                test_goid_h128=TEST_GOID,
                function_goid_h128=DEFAULT_GOID,
                urn=f"{DEFAULT_URN_PREFIX}{DEFAULT_QUALNAME}",
                repo=self.repo,
                commit=self.commit,
                rel_path=self.rel_path,
                qualname=DEFAULT_QUALNAME,
                covered_lines=2,
                executable_lines=4,
                coverage_ratio=0.5,
                last_status="failed",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_call_graph_nodes(self, ctx: TestContext) -> None:
        """Seed the graph.call_graph_nodes table."""
        rows = [
            dataclass_row(
                CallGraphNodeRow,
                goid_h128=DEFAULT_GOID,
                language="python",
                kind="function",
                arity=0,
                is_public=True,
                rel_path=self.rel_path,
            ),
            dataclass_row(
                CallGraphNodeRow,
                goid_h128=TEST_GOID,
                language="python",
                kind="function",
                arity=0,
                is_public=False,
                rel_path=self.rel_path,
            ),
            dataclass_row(
                CallGraphNodeRow,
                goid_h128=CALLER_GOID,
                language="python",
                kind="function",
                arity=0,
                is_public=False,
                rel_path=self.rel_path,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_call_graph_edges(self, ctx: TestContext) -> None:
        """Seed the graph.call_graph_edges table."""
        rows = [
            dataclass_row(
                CallGraphEdgeRow,
                repo=self.repo,
                commit=self.commit,
                caller_goid_h128=DEFAULT_GOID,
                callee_goid_h128=TEST_GOID,
                callsite_path=self.rel_path,
                callsite_line=1,
                callsite_col=1,
                language="python",
                kind="direct",
                resolved_via="local_name",
                confidence=1.0,
            ),
            dataclass_row(
                CallGraphEdgeRow,
                repo=self.repo,
                commit=self.commit,
                caller_goid_h128=CALLER_GOID,
                callee_goid_h128=DEFAULT_GOID,
                callsite_path=self.rel_path,
                callsite_line=2,
                callsite_col=2,
                language="python",
                kind="direct",
                resolved_via="global_name",
                confidence=1.0,
            ),
        ]
        insert_rows(ctx.gateway, rows)

    def _seed_import_graph_edges(self, ctx: TestContext) -> None:
        """Seed the graph.import_graph_edges table."""
        rows = [
            dataclass_row(
                ImportGraphEdgeRow,
                repo=self.repo,
                commit=self.commit,
                src_module=self.module,
                dst_module=self.module,
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=1,
            )
        ]
        insert_rows(ctx.gateway, rows)


PROFILE_PACK = ProfilePack()

__all__ = ["PROFILE_PACK", "ProfilePack"]
