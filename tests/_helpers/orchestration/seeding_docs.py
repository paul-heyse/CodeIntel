"""Seed functions for docs export and profile data.

This module provides standalone seed functions for docs export tests,
profile analytics tests, and MCP backend tests. These complement the
SeedPack system for cases where direct function calls are needed.
"""

from __future__ import annotations

from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.primitives import SnapshotRef
from codeintel.storage.schema import apply_all_schemas
from codeintel.storage.warehouse import Warehouse
from tests._helpers.assertions import ModulesAssertions
from tests._helpers.fakes import utcnow
from tests._helpers.fixtures.rows import (
    AstMetricsRow,
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    DocstringRow,
    FunctionMetricsRow,
    FunctionTypesRow,
    FunctionValidationRow,
    GoidCrosswalkRow,
    GoidRow,
    HotspotRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RepoMapRow,
    RiskFactorRow,
    StaticDiagnosticsRow,
    SymbolUseEdgeRow,
    TestCatalogRow,
    TypednessRow,
    function_profile_row,
    insert_rows,
)
from tests._helpers.modules_expectations import modules_expected_from_repo_tree

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


def seed_docs_export_minimal(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    repo_root: Path | None = None,
    module_map: dict[str, str] | None = None,
) -> None:
    """Seed the minimal rows needed for docs export smoke tests.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    repo
        Repository identifier.
    commit
        Commit hash.
    repo_root
        Optional repo root to derive module inventory from the filesystem.
    module_map
        Optional module->path mapping to seed core tables.
    """
    con = gateway.con
    now = utcnow()
    goid = 1
    apply_all_schemas(con)

    con.execute("DELETE FROM core.repo_map WHERE repo = ? AND commit = ?", [repo, commit])
    con.execute("DELETE FROM core.modules WHERE repo = ? AND commit = ?", [repo, commit])
    con.execute("DELETE FROM core.goids WHERE repo = ? AND commit = ?", [repo, commit])
    con.execute("DELETE FROM core.goid_crosswalk WHERE repo = ? AND commit = ?", [repo, commit])
    con.execute("DELETE FROM graph.call_graph_nodes WHERE goid_h128 = ?", [goid])
    con.execute("DELETE FROM graph.call_graph_edges WHERE repo = ? AND commit = ?", [repo, commit])
    con.execute("DELETE FROM graph.cfg_blocks WHERE function_goid_h128 = ?", [goid])
    con.execute(
        "DELETE FROM graph.import_graph_edges WHERE repo = ? AND commit = ?", [repo, commit]
    )
    con.execute("DELETE FROM graph.symbol_use_edges WHERE symbol = 'sym'")
    con.execute("DELETE FROM analytics.test_catalog WHERE repo = ? AND commit = ?", [repo, commit])

    resolved_module_map = dict(module_map) if module_map is not None else None
    if resolved_module_map is None and repo_root is not None:
        path_map = modules_expected_from_repo_tree(repo_root)
        resolved_module_map = {module: path for path, module in path_map.items()}
    if not resolved_module_map:
        resolved_module_map = {"pkg.foo": "foo.py"}
    elif "pkg.foo" not in resolved_module_map:
        resolved_module_map["pkg.foo"] = "foo.py"
    insert_rows(
        gateway,
        [RepoMapRow(repo=repo, commit=commit, modules=resolved_module_map, overlays={})],
    )
    insert_rows(
        gateway,
        [
            ModuleRow(module=module, path=path, repo=repo, commit=commit)
            for module, path in sorted(resolved_module_map.items())
        ],
    )
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=repo_root or Path.cwd())
    ModulesAssertions(gateway, snapshot).inventory_consistent()
    insert_rows(
        gateway,
        [
            GoidRow(
                goid_h128=goid,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                kind="function",
                qualname="pkg.foo:func",
                start_line=1,
                end_line=10,
                created_at=now,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            GoidCrosswalkRow(
                repo=repo,
                commit=commit,
                goid="urn:foo",
                lang="python",
                module_path="pkg.foo",
                file_path="foo.py",
                start_line=1,
                end_line=10,
                scip_symbol="scip-python foo",
                ast_qualname="pkg.foo:func",
                cst_node_id=None,
                chunk_id=None,
                symbol_id=None,
                updated_at=now,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            CallGraphNodeRow(
                goid,
                "python",
                "function",
                0,
                is_public=True,
                rel_path="foo.py",
            )
        ],
    )
    insert_rows(
        gateway,
        [
            CallGraphEdgeRow(
                repo,
                commit,
                goid,
                goid,
                "foo.py",
                1,
                0,
                "python",
                "direct",
                "local_name",
                1.0,
            )
        ],
    )
    insert_rows(
        gateway,
        [CFGBlockRow(goid, 0, f"{goid}:block0", "entry", "foo.py", 1, 1, "entry", [], 0, 0)],
    )
    insert_rows(
        gateway,
        [
            ImportGraphEdgeRow(
                repo=repo,
                commit=commit,
                src_module="pkg.foo",
                dst_module="pkg.bar",
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=1,
                module_layer=0,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            SymbolUseEdgeRow(
                symbol="sym",
                def_path="foo.py",
                use_path="foo.py",
                same_file=True,
                same_module=True,
                def_goid_h128=goid,
                use_goid_h128=goid,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            DocstringRow(
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                module="pkg.foo",
                qualname="pkg.foo:func",
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
        ],
    )
    insert_rows(
        gateway,
        [
            FunctionMetricsRow(
                function_goid_h128=goid,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                language="python",
                kind="function",
                qualname="pkg.foo:func",
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
        ],
    )
    insert_rows(
        gateway,
        [
            FunctionTypesRow(
                function_goid_h128=goid,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                language="python",
                kind="function",
                qualname="pkg.foo:func",
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
                param_types_json={},
                fully_typed=True,
                partial_typed=False,
                untyped=False,
                typedness_bucket="typed",
                typedness_source="pyright",
                created_at=now,
            )
        ],
    )
    warehouse = Warehouse(gateway)
    warehouse.materialize_mappings(
        "analytics.function_profile",
        [
            function_profile_row(
                goid=Decimal(goid),
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                qualname="pkg.foo:func",
                urn="urn:foo",
                module="pkg.foo",
            )
        ],
    )
    insert_rows(
        gateway,
        [
            RiskFactorRow(
                function_goid_h128=goid,
                repo=repo,
                commit=commit,
                risk_score=1,
                risk_level="low",
                cyclomatic_complexity=1,
                fan_in_count=0,
                fan_out_count=0,
                has_tests=True,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            TestCatalogRow(
                test_id="t1",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                qualname="pkg.foo::test_func",
                status="passed",
                created_at=now,
            )
        ],
    )


def seed_profile_data(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    rel_path: str,
    module: str,
) -> None:
    """Seed profile-related tables with realistic rows for analytics tests.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    repo
        Repository identifier.
    commit
        Commit hash.
    rel_path
        Relative path for the seeded data.
    module
        Module name for the seeded data.
    """
    con = gateway.con
    now = utcnow()

    con.execute(
        "DELETE FROM analytics.typedness WHERE path = ? AND repo = ? AND commit = ?",
        [rel_path, repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.static_diagnostics WHERE rel_path = ? AND repo = ? AND commit = ?",
        [rel_path, repo, commit],
    )
    con.execute("DELETE FROM core.modules WHERE repo = ? AND commit = ?", [repo, commit])
    insert_rows(
        gateway,
        [
            ModuleRow(
                module=module,
                path=rel_path,
                repo=repo,
                commit=commit,
                tags=["server"],
                owners=["team@example.com"],
            )
        ],
    )

    insert_rows(
        gateway,
        [
            AstMetricsRow(
                rel_path=rel_path,
                node_count=10,
                function_count=1,
                class_count=0,
                avg_depth=1.0,
                max_depth=1,
                complexity=2.0,
                generated_at=now,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            HotspotRow(
                rel_path=rel_path,
                commit_count=1,
                author_count=1,
                lines_added=5,
                lines_deleted=1,
                complexity=2.0,
                score=0.5,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            TypednessRow(
                repo=repo,
                commit=commit,
                path=rel_path,
                type_error_count=1,
                annotation_ratio='{"params": 0.5}',
                untyped_defs=0,
                overlay_needed=False,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            StaticDiagnosticsRow(
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                pyrefly_errors=1,
                pyright_errors=0,
                ruff_errors=0,
                total_errors=1,
                has_errors=True,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            DocstringRow(
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                module=module,
                qualname="pkg.mod.func",
                kind="function",
                lineno=1,
                end_lineno=2,
                raw_docstring="Doc",
                style="auto",
                short_desc="Short doc",
                long_desc="Longer doc",
                params_json=[],
                returns_json={"return": "int"},
                raises_json=[],
                examples_json=[],
                created_at=now,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            RiskFactorRow(
                function_goid_h128=1,
                repo=repo,
                commit=commit,
                risk_score=9,
                risk_level="high",
                cyclomatic_complexity=2,
                fan_in_count=1,
                fan_out_count=1,
                has_tests=True,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            FunctionMetricsRow(
                function_goid_h128=1,
                urn="goid:demo/repo#python:function:pkg.mod.func",
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                language="python",
                kind="function",
                qualname="pkg.mod.func",
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
        ],
    )
    insert_rows(
        gateway,
        [
            FunctionTypesRow(
                function_goid_h128=1,
                urn="goid:demo/repo#python:function:pkg.mod.func",
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                language="python",
                kind="function",
                qualname="pkg.mod.func",
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
                param_types_json=[],
                fully_typed=True,
                partial_typed=False,
                untyped=False,
                typedness_bucket="typed",
                typedness_source="analysis",
                created_at=now,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            TestCatalogRow(
                test_id="pkg/mod.py::test_func",
                test_goid_h128=2,
                urn="goid:demo/repo#python:function:pkg.mod.test_func",
                repo=repo,
                commit=commit,
                rel_path=rel_path,
                qualname="pkg.mod.test_func",
                kind="function",
                status="failed",
                duration_ms=1500,
                markers=[],
                parametrized=False,
                flaky=True,
                created_at=now,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            CallGraphEdgeRow(
                repo,
                commit,
                1,
                2,
                rel_path,
                1,
                1,
                "python",
                "direct",
                "local_name",
                1.0,
            ),
            CallGraphEdgeRow(
                repo,
                commit,
                3,
                1,
                rel_path,
                2,
                2,
                "python",
                "direct",
                "global_name",
                1.0,
            ),
        ],
    )
    insert_rows(
        gateway,
        [
            CallGraphNodeRow(
                1,
                "python",
                "function",
                0,
                is_public=True,
                rel_path=rel_path,
            ),
            CallGraphNodeRow(
                2,
                "python",
                "function",
                0,
                is_public=False,
                rel_path=rel_path,
            ),
            CallGraphNodeRow(
                3,
                "python",
                "function",
                0,
                is_public=False,
                rel_path=rel_path,
            ),
        ],
    )
    insert_rows(
        gateway,
        [
            ImportGraphEdgeRow(
                repo=repo,
                commit=commit,
                src_module=module,
                dst_module=module,
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=1,
            )
        ],
    )


def seed_mcp_backend(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> None:
    """Seed minimal data for MCP backend tests.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    repo
        Repository identifier.
    commit
        Commit hash.
    """
    con = gateway.con
    now = utcnow()
    con.execute(
        "DELETE FROM analytics.goid_risk_factors WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    con.execute(
        "DELETE FROM analytics.function_metrics WHERE repo = ? AND commit = ?",
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

    insert_rows(
        gateway,
        [
            RiskFactorRow(
                function_goid_h128=1,
                repo=repo,
                commit=commit,
                risk_score=1,
                risk_level="low",
                cyclomatic_complexity=1,
                fan_in_count=0,
                fan_out_count=0,
                has_tests=True,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            FunctionMetricsRow(
                function_goid_h128=1,
                urn="urn:foo",
                repo=repo,
                commit=commit,
                rel_path="foo.py",
                language="python",
                kind="function",
                qualname="foo",
                start_line=1,
                end_line=1,
                loc=1,
                logical_loc=1,
                param_count=0,
                positional_params=0,
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
        ],
    )
    insert_rows(
        gateway,
        [
            FunctionValidationRow(
                repo=repo,
                commit=commit,
                function_goid_h128=1,
                rel_path="foo.py",
                qualname="foo",
                issue="span_not_found",
                detail="Span 1-2",
                created_at=now,
            )
        ],
    )
    insert_rows(
        gateway,
        [
            CallGraphEdgeRow(
                repo,
                commit,
                1,
                2,
                "foo.py",
                1,
                0,
                "python",
                "direct",
                "local_name",
                1.0,
            ),
            CallGraphEdgeRow(
                repo,
                commit,
                3,
                1,
                "bar.py",
                1,
                0,
                "python",
                "direct",
                "local_name",
                1.0,
            ),
        ],
    )
    insert_rows(
        gateway,
        [
            TestCatalogRow(
                test_id="t1",
                repo=repo,
                commit=commit,
                rel_path="tests/t.py",
                qualname="tests.t",
                status="passed",
                created_at=now,
            )
        ],
    )


__all__ = [
    "seed_docs_export_minimal",
    "seed_mcp_backend",
    "seed_profile_data",
]
