"""Standalone seed functions for test data setup.

This module provides seed functions that are not covered by SeedPacks,
including functions for specific test scenarios like graph validation,
call graph scoping, and invalid profile setups.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from tests._helpers.builders import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    GoidRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RiskFactorRow,
    SymbolUseEdgeRow,
    insert_rows,
)
from tests._helpers.fakes import utcnow

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

# Import for seed_docs_export_invalid_profile - not circular because seeding_docs
# doesn't import from this module
from tests._helpers.orchestration.seeding_docs import (
    seed_docs_export_minimal as _seed_docs_export_minimal,
)


def seed_cfg_dfg_for_metrics(
    gateway: StorageGateway,
    *,
    rel_path: str,
) -> None:
    """Seed minimal CFG/DFG rows so compute_cfg/dfg_metrics can run.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    rel_path
        Relative path for the seeded blocks.
    """
    cfg_blocks = [
        CFGBlockRow(1, 0, "1:block0", "entry", rel_path, 1, 1, "entry", "[]", 0, 1),
        CFGBlockRow(1, 1, "1:block1", "body", rel_path, 2, 3, "body", "[]", 1, 1),
        CFGBlockRow(1, 2, "1:block2", "loop_head", rel_path, 4, 4, "loop_head", "[]", 1, 2),
        CFGBlockRow(1, 3, "1:block3", "unreachable", rel_path, 10, 10, "body", "[]", 0, 0),
        CFGBlockRow(1, 4, "1:block4", "exit", rel_path, 11, 11, "exit", "[]", 1, 0),
    ]
    insert_rows(gateway, cfg_blocks)
    cfg_edges = [
        CFGEdgeRow(1, "1:block0", "1:block1", "fallthrough"),
        CFGEdgeRow(1, "1:block1", "1:block2", "loop"),
        CFGEdgeRow(1, "1:block2", "1:block1", "back"),
        CFGEdgeRow(1, "1:block2", "1:block4", "fallthrough"),
    ]
    insert_rows(gateway, cfg_edges)

    dfg_edges = [
        DFGEdgeRow(
            1,
            "1:block0",
            "1:block1",
            "a",
            "a",
            "data-flow",
            via_phi=False,
            use_kind="data-flow",
        ),
        DFGEdgeRow(
            1,
            "1:block1",
            "1:block2",
            "a",
            "a",
            "phi",
            via_phi=True,
            use_kind="phi",
        ),
        DFGEdgeRow(
            1,
            "1:block1",
            "1:block1",
            "a",
            "a",
            "intra-block",
            via_phi=False,
            use_kind="intra-block",
        ),
    ]
    insert_rows(gateway, dfg_edges)


def seed_callgraph_goids(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    entries: list[tuple[int, str, str, int, int, str]],
) -> None:
    """Insert GOIDs for callgraph tests using gateway helpers.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    repo
        Repository identifier.
    commit
        Commit hash.
    entries
        List of tuples containing (goid_h128, urn, rel_path, start_line, end_line, kind).
    """
    now = utcnow()
    rows = [
        GoidRow(
            goid_h128=goid,
            urn=urn,
            repo=repo,
            commit=commit,
            rel_path=rel_path,
            kind=kind_value,
            qualname=urn.split(":", maxsplit=1)[-1],
            start_line=start_line,
            end_line=end_line,
            created_at=now,
        )
        for goid, urn, rel_path, start_line, end_line, kind_value in entries
    ]
    insert_rows(gateway, rows)


def seed_function_graph_cycle(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    rel_path: str,
) -> None:
    """Seed minimal callgraph nodes/edges to exercise cycle detection.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    repo
        Repository identifier.
    commit
        Commit hash.
    rel_path
        Relative path for the seeded nodes/edges.
    """
    gateway.con.execute(
        """
        DELETE FROM graph.call_graph_edges
        WHERE repo = ? AND commit = ? AND caller_goid_h128 IN (1,2)
        """,
        [repo, commit],
    )
    gateway.con.execute("DELETE FROM graph.call_graph_nodes WHERE goid_h128 IN (1, 2)")
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
                1,
                2,
                rel_path,
                2,
                2,
                "python",
                "direct",
                "local_name",
                1.0,
            ),
            CallGraphEdgeRow(
                repo,
                commit,
                2,
                1,
                rel_path,
                3,
                1,
                "python",
                "direct",
                "local_name",
                1.0,
            ),
        ],
    )


def seed_module_graph_inputs(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    module_a: str,
    module_b: str,
) -> None:
    """Seed import/symbol edges for module graph metrics calculations.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    repo
        Repository identifier.
    commit
        Commit hash.
    module_a
        First module name.
    module_b
        Second module name.
    """
    gateway.con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ? AND module IN (?, ?)",
        [repo, commit, module_a, module_b],
    )
    insert_rows(
        gateway,
        [
            ModuleRow(module=module_a, path="pkg/mod_a.py", repo=repo, commit=commit),
            ModuleRow(module=module_b, path="pkg/mod_b.py", repo=repo, commit=commit),
        ],
    )
    gateway.con.execute(
        """
        DELETE FROM graph.import_graph_edges
        WHERE repo = ? AND commit = ? AND src_module = ? AND dst_module = ?
        """,
        [repo, commit, module_a, module_b],
    )
    insert_rows(
        gateway,
        [
            ImportGraphEdgeRow(
                repo=repo,
                commit=commit,
                src_module=module_a,
                dst_module=module_b,
                src_fan_out=1,
                dst_fan_in=1,
                cycle_group=0,
            )
        ],
    )
    gateway.con.execute(
        """
        DELETE FROM graph.symbol_use_edges
        WHERE symbol = 'sym' AND def_path = ? AND use_path = ?
        """,
        ["pkg/mod_b.py", "pkg/mod_a.py"],
    )
    insert_rows(
        gateway,
        [
            SymbolUseEdgeRow(
                symbol="sym",
                def_path="pkg/mod_b.py",
                use_path="pkg/mod_a.py",
                same_file=False,
                same_module=False,
            )
        ],
    )


def seed_graph_validation_gaps(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
) -> None:
    """Seed rows that trigger graph validation warnings.

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
        """
        INSERT INTO core.ast_nodes (
            path, node_type, name, qualname, lineno, end_lineno, col_offset, end_col_offset,
            parent_qualname, decorators, docstring, hash
        ) VALUES ('pkg/a.py', 'FunctionDef', 'foo', 'pkg.a.foo', 1, 2, 0, 0, 'pkg.a',
                  '[]', NULL, 'h1')
        """
    )
    con.execute(
        """
        INSERT INTO core.modules (module, path, repo, commit, language, tags, owners)
        VALUES ('pkg.a', 'pkg/a.py', ?, ?, 'python', '[]', '[]')
        """,
        [repo, commit],
    )
    con.execute(
        """
        INSERT INTO core.goids (
            goid_h128, urn, repo, commit, rel_path, language, kind, qualname,
            start_line, end_line, created_at
        ) VALUES (1, 'urn:pkg.b.caller', ?, ?, 'pkg/b.py', 'python', 'function',
                  'pkg.b.caller', 1, 5, ?)
        """,
        [repo, commit, now],
    )
    con.execute(
        """
        INSERT INTO graph.call_graph_edges (
            repo, commit, caller_goid_h128, callee_goid_h128, callsite_path, callsite_line,
            callsite_col, language, kind, resolved_via, confidence, evidence_json
        ) VALUES (?, ?, 1, NULL, 'pkg/b.py', 50, 0, 'python', 'unresolved', 'unresolved',
                  0.0, '{}')
        """,
        [repo, commit],
    )


def seed_call_graph_scoping(
    gateway: StorageGateway,
    *,
    now_iso: str,
) -> None:
    """Seed call graph edges across repos/commits for scoping tests.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    now_iso
        ISO-formatted datetime string for created_at timestamps.
    """
    now = datetime.fromisoformat(now_iso)
    con = gateway.con
    con.execute("DELETE FROM graph.call_graph_edges WHERE repo IN ('r1', 'r2')")
    con.execute("DELETE FROM graph.call_graph_nodes WHERE goid_h128 IN (1, 2)")
    con.execute(
        "DELETE FROM analytics.goid_risk_factors WHERE (repo, commit) IN (('r1','c1'),('r2','c2'))"
    )
    con.execute("DELETE FROM core.goids WHERE goid_h128 IN (1, 2)")
    insert_rows(
        gateway,
        [
            GoidRow(
                goid_h128=1,
                urn="urn:1",
                repo="r1",
                commit="c1",
                rel_path="a.py",
                kind="function",
                qualname="a.f",
                start_line=1,
                end_line=2,
                created_at=now,
            ),
            GoidRow(
                goid_h128=2,
                urn="urn:2",
                repo="r2",
                commit="c2",
                rel_path="b.py",
                kind="function",
                qualname="b.f",
                start_line=1,
                end_line=2,
                created_at=now,
            ),
        ],
    )
    insert_rows(
        gateway,
        [
            RiskFactorRow(
                function_goid_h128=1,
                urn="urn:1",
                repo="r1",
                commit="c1",
                rel_path="a.py",
                language="python",
                kind="function",
                qualname="a.f",
                loc=0,
                logical_loc=0,
                cyclomatic_complexity=0,
                complexity_bucket="low",
                typedness_bucket="typed",
                typedness_source="analysis",
                hotspot_score=0.0,
                file_typed_ratio=0.0,
                static_error_count=0,
                has_static_errors=False,
                executable_lines=0,
                covered_lines=0,
                coverage_ratio=0.0,
                tested=False,
                test_count=0,
                failing_test_count=0,
                last_test_status="",
                risk_score=0.1,
                risk_level="low",
                tags="[]",
                owners="[]",
                created_at=now,
            ),
            RiskFactorRow(
                function_goid_h128=2,
                urn="urn:2",
                repo="r2",
                commit="c2",
                rel_path="b.py",
                language="python",
                kind="function",
                qualname="b.f",
                loc=0,
                logical_loc=0,
                cyclomatic_complexity=0,
                complexity_bucket="low",
                typedness_bucket="typed",
                typedness_source="analysis",
                hotspot_score=0.0,
                file_typed_ratio=0.0,
                static_error_count=0,
                has_static_errors=False,
                executable_lines=0,
                covered_lines=0,
                coverage_ratio=0.0,
                tested=False,
                test_count=0,
                failing_test_count=0,
                last_test_status="",
                risk_score=0.9,
                risk_level="high",
                tags="[]",
                owners="[]",
                created_at=now,
            ),
        ],
    )
    insert_rows(
        gateway,
        [
            CallGraphEdgeRow(
                "r1",
                "c1",
                1,
                None,
                "a.py",
                1,
                0,
                "python",
                "direct",
                "local",
                1.0,
            ),
            CallGraphEdgeRow(
                "r2",
                "c2",
                2,
                None,
                "b.py",
                2,
                0,
                "python",
                "direct",
                "local",
                1.0,
            ),
        ],
    )


def seed_docs_export_invalid_profile(
    gateway: StorageGateway,
    *,
    repo: str,
    commit: str,
    null_commit: bool = True,
    drop_commit_column: bool = False,
) -> None:
    """Seed minimal docs export data and flip required fields to trigger validation failures.

    Parameters
    ----------
    gateway
        Gateway to mutate.
    repo
        Repository identifier.
    commit
        Commit hash.
    null_commit
        When True, sets commit column in function_profile to NULL.
    drop_commit_column
        When True, removes the commit column from function_profile to induce
        schema validation failures.

    Raises
    ------
    ValueError
        If invoked against a strict gateway; use ``loose_gateway`` instead.
    """
    if getattr(gateway, "config", None) is not None and gateway.config.validate_schema:
        message = (
            "seed_docs_export_invalid_profile requires a non-strict gateway (use loose_gateway)."
        )
        raise ValueError(message)
    _seed_docs_export_minimal(gateway, repo=repo, commit=commit)
    con = gateway.con
    con.execute("DROP TABLE IF EXISTS analytics.function_profile")
    commit_value = None if null_commit else commit
    if drop_commit_column:
        con.execute(
            """
            CREATE TABLE analytics.function_profile (
                function_goid_h128 BIGINT,
                urn TEXT,
                repo TEXT,
                rel_path TEXT,
                module TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO analytics.function_profile (function_goid_h128, urn, repo, rel_path, module)
            VALUES (1, 'urn:foo', ?, 'foo.py', 'pkg.foo')
            """,
            [repo],
        )
    else:
        con.execute(
            """
            CREATE TABLE analytics.function_profile (
                function_goid_h128 BIGINT,
                urn TEXT,
                repo TEXT,
                commit TEXT,
                rel_path TEXT,
                module TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO analytics.function_profile (
                function_goid_h128, urn, repo, commit, rel_path, module
            )
            VALUES (1, 'urn:foo', ?, ?, 'foo.py', 'pkg.foo')
            """,
            [repo, commit_value],
        )
    con.execute(
        """
        CREATE TABLE IF NOT EXISTS analytics.graph_validation (
            repo TEXT NOT NULL,
            commit TEXT NOT NULL,
            graph_name TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            issue TEXT NOT NULL,
            severity TEXT,
            rel_path TEXT,
            detail TEXT NOT NULL,
            metadata JSON,
            created_at TIMESTAMP NOT NULL
        )
        """
    )
    con.execute(
        "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
        [repo, commit],
    )
    gateway.con.execute(
        """
        INSERT INTO core.repo_map (repo, commit, modules, overlays, generated_at)
        VALUES (?, ?, '{}', '{}', CURRENT_TIMESTAMP)
        """,
        [repo, commit],
    )


__all__ = [
    "seed_call_graph_scoping",
    "seed_callgraph_goids",
    "seed_cfg_dfg_for_metrics",
    "seed_docs_export_invalid_profile",
    "seed_function_graph_cycle",
    "seed_graph_validation_gaps",
    "seed_module_graph_inputs",
]
