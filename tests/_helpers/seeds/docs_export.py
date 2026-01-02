"""Docs export seed pack for documentation export validation tests.

This module provides the DocsExportPack which seeds comprehensive data needed
for docs export smoke tests and validation, including modules, GOIDs, graphs,
docstrings, and function types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from tests._helpers.assertions import ModulesAssertions
from tests._helpers.fixtures.rows import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    DocstringRow,
    FunctionTypesRow,
    GoidCrosswalkRow,
    GoidRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RepoMapRow,
    SymbolEdgeOptions,
    TestCatalogRow,
    dataclass_row,
    insert_rows,
    insert_symbol_use_edges,
    make_symbol_use_edge_row,
)
from tests._helpers.modules_expectations import modules_expected_from_repo_tree

if TYPE_CHECKING:
    from tests._helpers.context import SeedPack, TestContext


DEFAULT_GOID: int = 1
DEFAULT_MODULE: str = "pkg.foo"
DEFAULT_PATH: str = "foo.py"
DEFAULT_QUALNAME: str = "pkg.foo:func"
DEFAULT_URN: str = "urn:foo"


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
    - Function types
    - Test catalog
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

        self._cleanup_existing_data(ctx, repo, commit, goid)

        module_map = self._resolve_module_map(ctx)

        self._seed_repo_map(ctx, repo, commit, module_map)
        self._seed_modules(ctx, repo, commit, module_map)
        ModulesAssertions(ctx.gateway, ctx.snapshot).inventory_consistent()
        self._seed_goids(ctx, repo, commit, goid, now)
        self._seed_goid_crosswalk(ctx, repo, commit, now)
        self._seed_call_graph_nodes(ctx, goid)
        self._seed_call_graph_edges(ctx, repo, commit, goid)
        self._seed_cfg_blocks(ctx, goid)
        self._seed_import_graph_edges(ctx, repo, commit)
        self._seed_symbol_use_edges(ctx, goid)
        self._seed_docstrings(ctx, repo, commit, now)
        self._seed_function_types(ctx, repo, commit, goid, now)
        self._seed_test_catalog(ctx, repo, commit, now)

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
            "DELETE FROM analytics.function_types WHERE repo = ? AND commit = ?", [repo, commit]
        )
        con.execute(
            "DELETE FROM analytics.test_catalog WHERE repo = ? AND commit = ?", [repo, commit]
        )

    @staticmethod
    def _seed_repo_map(
        ctx: TestContext,
        repo: str,
        commit: str,
        module_map: dict[str, str],
    ) -> None:
        """Seed the core.repo_map table."""
        rows = [
            dataclass_row(
                RepoMapRow,
                repo=repo,
                commit=commit,
                modules=module_map,
                overlays={},
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_modules(
        ctx: TestContext,
        repo: str,
        commit: str,
        module_map: dict[str, str],
    ) -> None:
        """Seed the core.modules table."""
        rows = [
            dataclass_row(ModuleRow, module=module, path=path, repo=repo, commit=commit)
            for module, path in sorted(module_map.items())
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_goids(ctx: TestContext, repo: str, commit: str, goid: int, now: datetime) -> None:
        """Seed the core.goids table."""
        rows = [
            dataclass_row(
                GoidRow,
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
            dataclass_row(
                GoidCrosswalkRow,
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
    def _resolve_module_map(ctx: TestContext) -> dict[str, str]:
        path_map = modules_expected_from_repo_tree(ctx.repo_root)
        module_map = {module: path for path, module in path_map.items()}
        if not module_map:
            module_map = {DEFAULT_MODULE: DEFAULT_PATH}
        elif DEFAULT_MODULE not in module_map:
            module_map[DEFAULT_MODULE] = DEFAULT_PATH
        return module_map

    @staticmethod
    def _seed_call_graph_nodes(ctx: TestContext, goid: int) -> None:
        """Seed the graph.call_graph_nodes table."""
        rows = [
            dataclass_row(
                CallGraphNodeRow,
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
            dataclass_row(
                CallGraphEdgeRow,
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
            dataclass_row(
                CFGBlockRow,
                function_goid_h128=goid,
                block_idx=0,
                block_id=f"{goid}:block0",
                label="entry",
                file_path=DEFAULT_PATH,
                start_line=1,
                end_line=1,
                kind="entry",
                stmts_json=[],
                in_degree=0,
                out_degree=0,
            )
        ]
        insert_rows(ctx.gateway, rows)

    @staticmethod
    def _seed_import_graph_edges(ctx: TestContext, repo: str, commit: str) -> None:
        """Seed the graph.import_graph_edges table."""
        rows = [
            dataclass_row(
                ImportGraphEdgeRow,
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
            make_symbol_use_edge_row(
                "sym",
                DEFAULT_PATH,
                DEFAULT_PATH,
                options=SymbolEdgeOptions(
                    same_file=True,
                    same_module=True,
                    def_goid_h128=goid,
                    use_goid_h128=goid,
                ),
            )
        ]
        insert_symbol_use_edges(ctx.gateway, rows)

    @staticmethod
    def _seed_docstrings(ctx: TestContext, repo: str, commit: str, now: datetime) -> None:
        """Seed the analytics.docstrings table."""
        rows = [
            dataclass_row(
                DocstringRow,
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
                params_json=[],
                returns_json={"type": "str"},
                raises_json=[],
                examples_json=[],
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
            dataclass_row(
                FunctionTypesRow,
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
                return_type="str",
                type_comment=None,
                param_types={},
                created_at=now,
            )
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
                rel_path=DEFAULT_PATH,
                qualname="pkg.foo::test_func",
                status="passed",
                created_at=now,
            )
        ]
        insert_rows(ctx.gateway, rows)


DOCS_EXPORT_PACK = DocsExportPack()

__all__ = ["DOCS_EXPORT_PACK", "DocsExportPack"]
