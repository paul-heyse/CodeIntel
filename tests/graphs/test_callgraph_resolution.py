"""Unit tests for call resolution heuristics and SCIP upgrades."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from codeintel.build.graphs.compute import callgraph as call_resolution
from codeintel.build.graphs.compute.callgraph import persistence as call_persist
from codeintel.core.helpers.payload import encode_payload
from tests._helpers.assertions import expect_equal, expect_true

if TYPE_CHECKING:
    from codeintel.core.schemas.generated_rows.graph import (
        GraphCallGraphEdgesRow as CallGraphEdgeRow,
    )

ALIAS_GOID = 10
SCIP_GOID = 1234
UNIQUE_EDGE_COUNT = 4


def test_resolve_callee_precedence_local_then_alias() -> None:
    """Local name resolution should win before alias/global fallbacks."""
    local = {"foo": 1}
    global_map = {"foo": 2, "pkg.bar": 3}
    aliases = {"alias": "pkg"}

    result = call_resolution.resolve_callee(
        callee_name="foo",
        attr_chain=["foo"],
        local_callees=local,
        global_callees=global_map,
        import_aliases=aliases,
    )

    expect_equal(result.callee_goid, 1)
    expect_equal(result.resolved_via, "local_name")
    expect_true(
        result.confidence > 0.0, message="Confidence should be positive for local resolution"
    )


def test_resolve_callee_import_alias_global_attr_chain() -> None:
    """Attribute chains resolved via import aliases should map to global GOIDs."""
    local: dict[str, int] = {}
    global_map = {"pkg.bar.baz": ALIAS_GOID, "baz": 11}
    aliases = {"alias": "pkg.bar"}

    result = call_resolution.resolve_callee(
        callee_name="alias",
        attr_chain=["alias", "baz"],
        local_callees=local,
        global_callees=global_map,
        import_aliases=aliases,
    )

    expect_equal(result.callee_goid, ALIAS_GOID)
    expect_equal(result.resolved_via, "import_alias")
    expect_true(
        result.confidence > 0.0, message="Confidence should be positive for alias resolution"
    )


def test_resolve_callee_unresolved_returns_none() -> None:
    """Unknown callees remain unresolved with zero confidence."""
    result = call_resolution.resolve_callee(
        callee_name="missing",
        attr_chain=["missing"],
        local_callees={},
        global_callees={},
        import_aliases={},
    )

    expect_true(result.callee_goid is None, message="Expected unresolved GOID")
    expect_equal(result.resolved_via, "unresolved")
    expect_equal(result.confidence, 0.0)


def test_resolve_via_scip_uses_crosswalk_mapping() -> None:
    """SCIP candidate paths should upgrade resolution when present in the crosswalk map."""
    candidates = ("src/pkg/module.py", "src/other.py")
    crosswalk = {"src/pkg/module.py": 1234}

    result = call_resolution.resolve_via_scip(candidates, crosswalk)

    expect_equal(result.callee_goid, SCIP_GOID)
    expect_equal(result.resolved_via, "scip_def_path")
    expect_true(result.confidence > 0.0, message="SCIP resolution should set positive confidence")


def test_dedupe_includes_repo_commit_scope() -> None:
    """Deduplication should treat repo/commit as part of the identity."""
    edge_base: CallGraphEdgeRow = {
        "repo": "r1",
        "commit": "c1",
        "caller_goid_h128": 1,
        "callee_goid_h128": 2,
        "callsite_path": "a.py",
        "callsite_line": 1,
        "callsite_col": 0,
        "language": "python",
        "kind": "direct",
        "resolved_via": "local_name",
        "confidence": 1.0,
        "evidence_json": encode_payload({}),
    }
    edges: list[CallGraphEdgeRow] = [
        edge_base,
        cast("CallGraphEdgeRow", {**edge_base, "commit": "c2"}),
        cast("CallGraphEdgeRow", {**edge_base, "repo": "r2"}),
        cast("CallGraphEdgeRow", {**edge_base, "callsite_line": 2}),
    ]

    unique = call_persist.dedupe_edge_rows(edges)

    expect_equal(len(unique), UNIQUE_EDGE_COUNT)
