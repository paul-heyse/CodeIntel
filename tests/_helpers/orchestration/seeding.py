"""Standalone seed functions for test data setup.

This module provides seed functions that are not covered by SeedPacks,
including functions for specific test scenarios like graph validation,
call graph scoping, and invalid profile setups.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from codeintel.config.primitives import SnapshotRef
from tests._helpers.assertions import ModulesAssertions
from tests._helpers.fakes import utcnow
from tests._helpers.fixtures.rows import (
    CallGraphEdgeRow,
    CallGraphNodeRow,
    CFGBlockRow,
    CFGEdgeRow,
    DFGEdgeRow,
    GoidRow,
    ImportGraphEdgeRow,
    ModuleRow,
    RepoMapRow,
    SymbolUseEdgeRow,
    insert_rows,
)
from tests._helpers.modules_expectations import modules_expected_from_repo_tree

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway

from tests._helpers.orchestration.seeding_docs import (
    seed_docs_export_minimal as _seed_docs_export_minimal,
)


@dataclass(frozen=True)
class ModuleGraphInputSeed:
    """Configuration for seeding module graph inputs."""

    repo: str
    commit: str
    module_a: str
    module_b: str
    repo_root: Path | None = None
    module_map: dict[str, str] | None = None


@dataclass(frozen=True)
class GraphValidationGapSeed:
    """Configuration for seeding graph validation gaps."""

    repo: str
    commit: str
    include_modules: bool = True
    repo_root: Path | None = None
    module_map: dict[str, str] | None = None


@dataclass(frozen=True)
class DocsExportInvalidProfileOptions:
    """Options for seeding invalid docs export profiles."""

    repo_root: Path | None = None
    null_commit: bool = True
    drop_commit_column: bool = False


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
        CFGBlockRow(1, 0, "1:block0", "entry", rel_path, 1, 1, "entry", [], 0, 1),
        CFGBlockRow(1, 1, "1:block1", "body", rel_path, 2, 3, "body", [], 1, 1),
        CFGBlockRow(1, 2, "1:block2", "loop_head", rel_path, 4, 4, "loop_head", [], 1, 2),
        CFGBlockRow(1, 3, "1:block3", "unreachable", rel_path, 10, 10, "body", [], 0, 0),
        CFGBlockRow(1, 4, "1:block4", "exit", rel_path, 11, 11, "exit", [], 1, 0),
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
    spec: ModuleGraphInputSeed,
) -> None:
    """Seed import/symbol edges for module graph metrics calculations.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    spec
        Module graph seed configuration.

    Raises
    ------
    ValueError
        If module paths cannot be resolved for the requested modules.
    """
    resolved_module_map = spec.module_map
    if resolved_module_map is None:
        if spec.repo_root is not None:
            path_map = modules_expected_from_repo_tree(spec.repo_root)
            resolved_module_map = {module: path for path, module in path_map.items()}
        else:
            resolved_module_map = {
                spec.module_a: "pkg/mod_a.py",
                spec.module_b: "pkg/mod_b.py",
            }
    path_a = resolved_module_map.get(spec.module_a)
    path_b = resolved_module_map.get(spec.module_b)
    if path_a is None or path_b is None:
        message = f"Missing module paths for {spec.module_a!r} or {spec.module_b!r}"
        raise ValueError(message)
    gateway.con.execute(
        "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
        [spec.repo, spec.commit],
    )
    gateway.con.execute(
        "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
        [spec.repo, spec.commit],
    )
    insert_rows(
        gateway,
        [
            ModuleRow(module=module, path=path, repo=spec.repo, commit=spec.commit)
            for module, path in sorted(resolved_module_map.items())
        ],
    )
    insert_rows(
        gateway,
        [
            RepoMapRow(
                repo=spec.repo,
                commit=spec.commit,
                modules=resolved_module_map,
            )
        ],
    )
    snapshot = SnapshotRef(
        repo=spec.repo,
        commit=spec.commit,
        repo_root=spec.repo_root or Path.cwd(),
    )
    ModulesAssertions(gateway, snapshot).inventory_consistent()
    gateway.con.execute(
        """
        DELETE FROM graph.import_graph_edges
        WHERE repo = ? AND commit = ? AND src_module = ? AND dst_module = ?
        """,
        [spec.repo, spec.commit, spec.module_a, spec.module_b],
    )
    insert_rows(
        gateway,
        [
            ImportGraphEdgeRow(
                repo=spec.repo,
                commit=spec.commit,
                src_module=spec.module_a,
                dst_module=spec.module_b,
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
    spec: GraphValidationGapSeed,
) -> None:
    """Seed rows that trigger graph validation warnings.

    Parameters
    ----------
    gateway
        Storage gateway to seed.
    spec
        Graph validation gap seed configuration.
    """
    repo = spec.repo
    commit = spec.commit
    include_modules = spec.include_modules
    repo_root = spec.repo_root
    module_map = spec.module_map
    con = gateway.con
    now = utcnow()
    con.execute(
        """
        INSERT INTO core.ast_nodes (
            path, node_type, name, qualname, lineno, end_lineno, col_offset, end_col_offset,
            parent_qualname, decorators, docstring, hash
        ) VALUES ('pkg/a.py', 'FunctionDef', 'foo', 'pkg.a.foo', 1, 2, 0, 0, 'pkg.a',
                  ?, NULL, 'h1')
        """,
        [[]],
    )
    if include_modules:
        con.execute(
            "DELETE FROM core.modules WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        con.execute(
            "DELETE FROM core.repo_map WHERE repo = ? AND commit = ?",
            [repo, commit],
        )
        resolved_module_map = module_map
        if resolved_module_map is None and repo_root is not None:
            path_map = modules_expected_from_repo_tree(repo_root)
            module_name = path_map.get("pkg/a.py", "pkg.a")
            resolved_module_map = {module_name: "pkg/a.py"}
        if resolved_module_map is None:
            resolved_module_map = {"pkg.a": "pkg/a.py"}
        insert_rows(
            gateway,
            [
                ModuleRow(module=module, path=path, repo=repo, commit=commit)
                for module, path in sorted(resolved_module_map.items())
            ],
        )
        insert_rows(
            gateway,
            [
                RepoMapRow(
                    repo=repo,
                    commit=commit,
                    modules=resolved_module_map,
                )
            ],
        )
        snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=repo_root or Path.cwd())
        ModulesAssertions(gateway, snapshot).inventory_consistent()
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
                  0.0, ?)
        """,
        [repo, commit, {}],
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
    options: DocsExportInvalidProfileOptions | None = None,
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
    options
        Optional invalid profile seed options.

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
    resolved_options = options or DocsExportInvalidProfileOptions()
    _seed_docs_export_minimal(
        gateway,
        repo=repo,
        commit=commit,
        repo_root=resolved_options.repo_root,
    )
    con = gateway.con
    con.execute("DROP TABLE IF EXISTS analytics.function_types")
    commit_value = None if resolved_options.null_commit else commit
    if resolved_options.drop_commit_column:
        con.execute(
            """
            CREATE TABLE analytics.function_types (
                function_goid_h128 BIGINT,
                urn TEXT,
                repo TEXT,
                rel_path TEXT,
                qualname TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO analytics.function_types (
                function_goid_h128, urn, repo, rel_path, qualname
            )
            VALUES (1, 'urn:foo', ?, 'foo.py', 'pkg.foo')
            """,
            [repo],
        )
    else:
        con.execute(
            """
            CREATE TABLE analytics.function_types (
                function_goid_h128 BIGINT,
                urn TEXT,
                repo TEXT,
                commit TEXT,
                rel_path TEXT,
                qualname TEXT
            )
            """
        )
        con.execute(
            """
            INSERT INTO analytics.function_types (
                function_goid_h128, urn, repo, commit, rel_path, qualname
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
        VALUES (?, ?, ?, ?, CURRENT_TIMESTAMP)
        """,
        [repo, commit, {}, {}],
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
