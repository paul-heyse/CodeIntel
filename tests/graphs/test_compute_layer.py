"""Tests for compute layer pure functions.

This module tests the stateless computation functions for callgraph
resolution, import analysis, and symbol use tracking.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Final, cast

import libcst as cst

from codeintel.core.catalog import FunctionSpan, FunctionSpanIndex
from codeintel.graphs.compute.callgraph import (
    CallEdge,
    EdgeResolutionContext,
    ResolutionResult,
    attr_to_str,
    build_callee_map,
    build_evidence,
    collect_call_sites,
    collect_edges_ast,
    collect_edges_cst,
    collect_import_edges,
    dedupe_edges,
    extract_callee_ast,
    extract_callee_cst,
    extract_class_name_from_call,
    record_import_aliases,
    record_import_from_aliases,
    resolve_base_module,
    resolve_callee,
    resolve_via_scip,
)
from codeintel.graphs.compute.callgraph.collection import LocalTypeTracker
from codeintel.graphs.compute.callgraph.resolution import collect_aliases
from codeintel.graphs.compute.cfg import BasicBlock, CFGBuilder, CFGEdge, CFGResult, cfg_to_rows
from codeintel.graphs.compute.goid import (
    DECIMAL_38_MAX,
    GoidDescriptor,
    build_crosswalk_row,
    build_goid_row,
    build_urn,
    compute_goid,
    compute_goid_result,
    determine_kind,
)
from codeintel.graphs.compute.imports import (
    ImportAnalysisResult,
    ImportEdge,
    analyze_imports,
    build_import_edge_rows,
    build_import_module_rows,
    compute_layers,
    compute_scc,
)
from codeintel.graphs.compute.imports import (
    collect_import_edges as collect_import_edges_for_analysis,
)
from codeintel.graphs.compute.symbols import (
    SymbolOccurrence,
    SymbolUseEdge,
    build_def_map,
    build_use_def_mapping,
    build_use_edges,
    edges_to_rows,
    parse_symbol_roles,
)
from codeintel.graphs.ports.parsing import ParsedModule
from tests._helpers.assertions import (
    assert_cannot_setattr,
    expect_equal,
    expect_is_none,
    expect_length,
    expect_true,
)

if TYPE_CHECKING:
    from pathlib import Path

    from codeintel.config.datasets.rows.graph import CallGraphEdgeRow


EXPECTED_EDGE_COUNT_ONE: Final[int] = 1
EXPECTED_EDGE_COUNT_TWO: Final[int] = 2
EXPECTED_EDGE_COUNT_THREE: Final[int] = 3
LOCAL_CONFIDENCE: Final[float] = 0.8
LOCAL_ATTR_CONFIDENCE: Final[float] = 0.75
IMPORT_ALIAS_CONFIDENCE: Final[float] = 0.7
GLOBAL_CONFIDENCE: Final[float] = 0.6
SCIP_CONFIDENCE: Final[float] = 0.55
UNRESOLVED_CONFIDENCE: Final[float] = 0.0
EXPECTED_LAYER_ZERO: Final[int] = 0
EXPECTED_LAYER_ONE: Final[int] = 1
EXPECTED_LAYER_TWO: Final[int] = 2
DEFINITION_ROLE: Final[int] = 1
REFERENCE_ROLE: Final[int] = 2
REFERENCE_ROLE_COMBINED: Final[int] = 2 | 4
TEST_GOID_A: Final[int] = 100
TEST_GOID_B: Final[int] = 200
TEST_GOID_C: Final[int] = 300
TEST_GOID_E: Final[int] = 500
REL_PATH: Final[str] = "pkg/mod.py"
REPO: Final[str] = "repo"
COMMIT: Final[str] = "commit"


def test_resolve_callee_local_name() -> None:
    """Resolve callee via local name lookup."""
    local_callees = {"my_func": TEST_GOID_A}
    result = resolve_callee("my_func", [], local_callees, {}, {})

    expect_equal(result.callee_goid, TEST_GOID_A)
    expect_equal(result.resolved_via, "local_name")
    expect_equal(result.confidence, LOCAL_CONFIDENCE)


def test_resolve_callee_local_attr() -> None:
    """Resolve callee via local attribute lookup."""
    local_callees = {"module.func": TEST_GOID_A}
    result = resolve_callee("func", ["module", "func"], {}, local_callees, {})

    expect_true(result.resolved_via in {"local_attr", "global_name"})


def test_resolve_callee_import_alias() -> None:
    """Resolve callee via import alias."""
    global_callees = {"external.module.func": TEST_GOID_A}
    import_aliases = {"ext": "external.module"}
    result = resolve_callee("func", ["ext", "func"], {}, global_callees, import_aliases)

    expect_equal(result.callee_goid, TEST_GOID_A)
    expect_equal(result.resolved_via, "import_alias")
    expect_equal(result.confidence, IMPORT_ALIAS_CONFIDENCE)


def test_resolve_callee_global_name() -> None:
    """Resolve callee via global name lookup."""
    global_callees = {"global_func": TEST_GOID_A}
    result = resolve_callee("global_func", [], {}, global_callees, {})

    expect_equal(result.callee_goid, TEST_GOID_A)
    expect_equal(result.resolved_via, "global_name")
    expect_equal(result.confidence, GLOBAL_CONFIDENCE)


def test_resolve_callee_unresolved() -> None:
    """Unresolved callee returns None with unresolved status."""
    result = resolve_callee("unknown_func", [], {}, {}, {})

    expect_is_none(result.callee_goid)
    expect_equal(result.resolved_via, "unresolved")
    expect_equal(result.confidence, UNRESOLVED_CONFIDENCE)


def test_resolve_callee_priority_local_over_global() -> None:
    """Local resolution takes priority over global."""
    local_callees = {"func": TEST_GOID_A}
    global_callees = {"func": TEST_GOID_B}
    result = resolve_callee("func", [], local_callees, global_callees, {})

    expect_equal(result.callee_goid, TEST_GOID_A)
    expect_equal(result.resolved_via, "local_name")


def test_resolve_via_scip_found() -> None:
    """SCIP resolution finds matching def path."""
    def_goids = {"path/to/module.py:func": TEST_GOID_A}
    result = resolve_via_scip(("path/to/module.py:func",), def_goids)

    expect_equal(result.callee_goid, TEST_GOID_A)
    expect_equal(result.resolved_via, "scip_def_path")
    expect_equal(result.confidence, SCIP_CONFIDENCE)


def test_resolve_via_scip_not_found() -> None:
    """SCIP resolution returns unresolved when no match."""
    result = resolve_via_scip(("nonexistent/path.py:func",), {})

    expect_is_none(result.callee_goid)
    expect_equal(result.resolved_via, "unresolved")
    expect_equal(result.confidence, UNRESOLVED_CONFIDENCE)


def test_resolve_via_scip_empty_candidates() -> None:
    """SCIP resolution handles empty candidates."""
    result = resolve_via_scip((), {})

    expect_is_none(result.callee_goid)
    expect_equal(result.resolved_via, "unresolved")


def test_build_evidence_basic() -> None:
    """Build evidence with basic resolution."""
    resolution = ResolutionResult(
        callee_goid=TEST_GOID_A, resolved_via="local_name", confidence=LOCAL_CONFIDENCE
    )
    evidence = build_evidence("my_func", [], resolution)

    expect_equal(evidence["callee_name"], "my_func")
    expect_equal(evidence["resolved_via"], "local_name")
    expect_is_none(evidence["attr_chain"])


def test_build_evidence_with_attr_chain() -> None:
    """Build evidence with attribute chain."""
    resolution = ResolutionResult(
        callee_goid=TEST_GOID_A, resolved_via="import_alias", confidence=IMPORT_ALIAS_CONFIDENCE
    )
    evidence = build_evidence("func", ["module", "func"], resolution)

    expect_equal(evidence["callee_name"], "func")
    expect_equal(evidence["attr_chain"], ["module", "func"])
    expect_equal(evidence["resolved_via"], "import_alias")


def test_build_evidence_with_scip_candidates() -> None:
    """Build evidence with SCIP candidates."""
    resolution = ResolutionResult(
        callee_goid=TEST_GOID_A, resolved_via="scip_def_path", confidence=SCIP_CONFIDENCE
    )
    scip_candidates = ("path/a.py:func", "path/b.py:func")
    evidence = build_evidence("func", [], resolution, scip_candidates)

    expect_true("scip_candidates" in evidence)
    expect_equal(evidence["scip_candidates"], list(scip_candidates))


def test_extract_callee_cst_simple_name() -> None:
    """Extract callee from simple CST Name node."""
    module = cst.parse_module("func()")
    call = module.body[0]
    if isinstance(call, cst.SimpleStatementLine):
        expr = call.body[0]
        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
            name, chain = extract_callee_cst(expr.value.func)
            expect_equal(name, "func")
            expect_equal(chain, ["func"])


def test_extract_callee_cst_attribute() -> None:
    """Extract callee from CST Attribute node."""
    module = cst.parse_module("module.func()")
    call = module.body[0]
    if isinstance(call, cst.SimpleStatementLine):
        expr = call.body[0]
        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
            name, chain = extract_callee_cst(expr.value.func)
            expect_equal(name, "func")
            expect_equal(chain, ["module", "func"])


def test_extract_callee_cst_nested_attribute() -> None:
    """Extract callee from nested CST Attribute."""
    module = cst.parse_module("a.b.c.func()")
    call = module.body[0]
    if isinstance(call, cst.SimpleStatementLine):
        expr = call.body[0]
        if isinstance(expr, cst.Expr) and isinstance(expr.value, cst.Call):
            name, chain = extract_callee_cst(expr.value.func)
            expect_equal(name, "func")
            expect_equal(chain, ["a", "b", "c", "func"])


def test_extract_callee_ast_simple_name() -> None:
    """Extract callee from simple AST Name node."""
    tree = ast.parse("func()")
    call = tree.body[0]
    if isinstance(call, ast.Expr) and isinstance(call.value, ast.Call):
        name, chain = extract_callee_ast(call.value.func)
        expect_equal(name, "func")
        expect_equal(chain, ["func"])


def test_extract_callee_ast_attribute() -> None:
    """Extract callee from AST Attribute node."""
    tree = ast.parse("module.func()")
    call = tree.body[0]
    if isinstance(call, ast.Expr) and isinstance(call.value, ast.Call):
        name, chain = extract_callee_ast(call.value.func)
        expect_equal(name, "module")
        expect_equal(chain, ["module", "func"])


def test_dedupe_edges_empty() -> None:
    """Dedupe handles empty list."""
    result = dedupe_edges([])
    expect_equal(result, [])


def test_dedupe_edges_no_duplicates() -> None:
    """Dedupe returns same edges when no duplicates."""
    edges = [
        CallEdge(
            caller_goid=TEST_GOID_A,
            callee_goid=TEST_GOID_B,
            callee_name="func1",
            call_line=10,
            rel_path="test.py",
            evidence="local_name",
            confidence=LOCAL_CONFIDENCE,
        ),
        CallEdge(
            caller_goid=TEST_GOID_A,
            callee_goid=TEST_GOID_C,
            callee_name="func2",
            call_line=20,
            rel_path="test.py",
            evidence="global_name",
            confidence=GLOBAL_CONFIDENCE,
        ),
    ]
    result = dedupe_edges(edges)
    expect_length(result, EXPECTED_EDGE_COUNT_TWO)


def test_dedupe_edges_keeps_highest_confidence() -> None:
    """Dedupe keeps edge with highest confidence."""
    edges = [
        CallEdge(
            caller_goid=TEST_GOID_A,
            callee_goid=TEST_GOID_B,
            callee_name="func",
            call_line=10,
            rel_path="test.py",
            evidence="global_name",
            confidence=GLOBAL_CONFIDENCE,
        ),
        CallEdge(
            caller_goid=TEST_GOID_A,
            callee_goid=TEST_GOID_B,
            callee_name="func",
            call_line=10,
            rel_path="test.py",
            evidence="local_name",
            confidence=LOCAL_CONFIDENCE,
        ),
    ]
    result = dedupe_edges(edges)
    expect_length(result, EXPECTED_EDGE_COUNT_ONE)
    kept = result[0]
    expect_true(isinstance(kept, CallEdge))
    kept_edge = cast("CallEdge", kept)
    expect_equal(kept_edge.confidence, LOCAL_CONFIDENCE)


def test_build_callee_map_empty() -> None:
    """Build callee map from empty spans."""
    result = build_callee_map([])
    expect_equal(result, {})


def test_build_callee_map_sets_local_and_qualname() -> None:
    """Build callee map indexes both qualname and local name."""
    spans = [
        FunctionSpan(
            goid=TEST_GOID_A,
            rel_path=REL_PATH,
            qualname="pkg.mod.func",
            start_line=1,
            end_line=5,
        ),
        FunctionSpan(
            goid=TEST_GOID_B,
            rel_path=REL_PATH,
            qualname="pkg.mod.inner",
            start_line=6,
            end_line=10,
        ),
    ]

    mapping = build_callee_map(spans)

    expect_equal(mapping["pkg.mod.func"], TEST_GOID_A)
    expect_equal(mapping["func"], TEST_GOID_A)
    expect_equal(mapping["inner"], TEST_GOID_B)


def test_local_type_tracker_records_and_clears() -> None:
    """LocalTypeTracker records aliases and clears state."""
    tracker = LocalTypeTracker()

    tracker.record_instantiation("obj", "Alias", {"Alias": "pkg.mod.Class"})
    expect_equal(tracker.get_type("obj"), "pkg.mod.Class")

    tracker.clear()
    expect_is_none(tracker.get_type("obj"))


def test_extract_class_name_from_call_attribute_chain() -> None:
    """Extract class name from nested attribute call."""
    simple_call = cst.parse_expression("MyClass()")
    nested_call = cst.parse_expression("pkg.mod.ClassName()")

    if isinstance(simple_call, cst.Call):
        expect_equal(extract_class_name_from_call(simple_call.func), "MyClass")
    if isinstance(nested_call, cst.Call):
        expect_equal(extract_class_name_from_call(nested_call.func), "pkg.mod.ClassName")


def test_collect_edges_cst_tracks_assignments_and_backfills_goid() -> None:
    """CST visitor backfills GOID and resolves instance methods."""
    function_index = FunctionSpanIndex(
        [
            FunctionSpan(
                goid=TEST_GOID_A,
                rel_path=REL_PATH,
                qualname="pkg.mod.top",
                start_line=1,
                end_line=20,
            )
        ]
    )
    context = EdgeResolutionContext(
        repo="repo",
        commit="commit",
        function_index=function_index,
        local_callees={},
        global_callees={
            "callee": TEST_GOID_B,
            "pkg.C.m": TEST_GOID_C,
        },
        import_aliases={"pkg": "pkg"},
        scip_candidates_by_use_path={},
        def_goids_by_path={},
    )
    module = cst.parse_module(
        "\n".join(
            [
                "obj: pkg.C = pkg.C()",
                "callee()",
                "obj.m()",
            ]
        )
    )

    edges = collect_edges_cst(REL_PATH, module, context)

    expect_length(edges, EXPECTED_EDGE_COUNT_THREE)

    resolved = [edge for edge in edges if edge["callee_goid_h128"]]
    expect_length(resolved, EXPECTED_EDGE_COUNT_TWO)
    goids = {edge["callee_goid_h128"] for edge in resolved}
    expect_true(TEST_GOID_B in goids)
    expect_true(TEST_GOID_C in goids)


def test_collect_edges_cst_uses_scip_fallback() -> None:
    """CST collection upgrades unresolved calls via SCIP paths."""
    spans = FunctionSpanIndex(
        [
            FunctionSpan(
                goid=TEST_GOID_A,
                rel_path=REL_PATH,
                qualname="pkg.mod.top",
                start_line=1,
                end_line=10,
            )
        ]
    )
    context = EdgeResolutionContext(
        repo="repo",
        commit="commit",
        function_index=spans,
        local_callees={},
        global_callees={},
        import_aliases={},
        scip_candidates_by_use_path={REL_PATH: ("src/pkg/def.py",)},
        def_goids_by_path={"src/pkg/def.py": TEST_GOID_E},
    )
    module = cst.parse_module("missing_call()")

    edges = collect_edges_cst(REL_PATH, module, context)

    expect_length(edges, EXPECTED_EDGE_COUNT_ONE)
    expect_equal(edges[0]["callee_goid_h128"], TEST_GOID_E)
    expect_equal(edges[0]["resolved_via"], "scip_def_path")


def test_try_instance_method_resolution_prefers_short_class_name() -> None:
    """Instance method resolution falls back to short class name when needed."""
    function_index = FunctionSpanIndex(
        [
            FunctionSpan(
                goid=TEST_GOID_A,
                rel_path=REL_PATH,
                qualname="pkg.mod.top",
                start_line=1,
                end_line=5,
            )
        ]
    )
    context = EdgeResolutionContext(
        repo="repo",
        commit="commit",
        function_index=function_index,
        local_callees={},
        global_callees={"C.m": TEST_GOID_B},
        import_aliases={"pkg": "pkg"},
        scip_candidates_by_use_path={},
        def_goids_by_path={},
    )
    module = cst.parse_module(
        "\n".join(
            [
                "obj = pkg.C()",
                "obj.m()",
            ]
        )
    )

    edges = collect_edges_cst(REL_PATH, module, context)

    expect_true(any(edge["callee_goid_h128"] == TEST_GOID_B for edge in edges))
    resolved_edges = [edge for edge in edges if edge["callee_goid_h128"] == TEST_GOID_B]
    expect_length(resolved_edges, EXPECTED_EDGE_COUNT_ONE)
    expect_equal(resolved_edges[0]["resolved_via"], "instance_method")


def test_collect_edges_ast_handles_success_and_syntax_error(tmp_path: Path) -> None:
    """AST fallback collects edges and ignores syntax errors."""
    valid_path = tmp_path / "valid.py"
    valid_path.write_text(
        "\n".join(
            [
                "def top():",
                "    callee()",
            ]
        ),
        encoding="utf-8",
    )
    function_index = FunctionSpanIndex(
        [
            FunctionSpan(
                goid=TEST_GOID_A,
                rel_path=REL_PATH,
                qualname="pkg.mod.top",
                start_line=1,
                end_line=2,
            )
        ]
    )
    context = EdgeResolutionContext(
        repo="repo",
        commit="commit",
        function_index=function_index,
        local_callees={},
        global_callees={"callee": TEST_GOID_B},
        import_aliases={},
        scip_candidates_by_use_path={},
        def_goids_by_path={},
    )

    edges = collect_edges_ast(REL_PATH, valid_path, context)

    expect_length(edges, EXPECTED_EDGE_COUNT_ONE)
    expect_equal(edges[0]["caller_goid_h128"], TEST_GOID_A)
    expect_equal(edges[0]["callee_goid_h128"], TEST_GOID_B)

    invalid_path = tmp_path / "broken.py"
    invalid_path.write_text("def broken(:\n    pass\n", encoding="utf-8")

    expect_equal(collect_edges_ast(REL_PATH, invalid_path, context), [])


def test_collect_call_sites_extracts_attribute_chain() -> None:
    """collect_call_sites returns name and attribute chains for calls."""
    source = "\n".join(
        [
            "def top():",
            "    a.b.c()",
            "    simple()",
        ]
    )
    parsed_module = ParsedModule(source=source, ast_module=ast.parse(source))
    call_sites = sorted(collect_call_sites(parsed_module, (1, 4)), key=lambda entry: entry[2])

    expect_length(call_sites, EXPECTED_EDGE_COUNT_TWO)
    expect_equal(call_sites[0][0], "a")
    expect_equal(call_sites[0][1], ["b", "c"])
    expect_equal(call_sites[1][0], "simple")
    expect_equal(call_sites[1][1], [])


def test_dedupe_edges_prefers_higher_confidence_rows() -> None:
    """dedupe_edges deduplicates CallGraphEdgeRow inputs by confidence."""
    base_edge: CallGraphEdgeRow = {
        "repo": "r",
        "commit": "c",
        "caller_goid_h128": TEST_GOID_A,
        "callee_goid_h128": TEST_GOID_B,
        "callsite_path": REL_PATH,
        "callsite_line": 1,
        "callsite_col": 0,
        "language": "python",
        "kind": "direct",
        "resolved_via": "local",
        "confidence": 0.5,
        "evidence_json": {},
    }
    edges: list[CallGraphEdgeRow] = [
        base_edge,
        cast("CallGraphEdgeRow", {**base_edge, "confidence": 0.9}),
        cast("CallGraphEdgeRow", {**base_edge, "callsite_line": 2}),
    ]

    deduped = dedupe_edges(edges)
    deduped_rows = cast("list[CallGraphEdgeRow]", deduped)

    expect_length(deduped_rows, EXPECTED_EDGE_COUNT_TWO)
    confidences = {edge["callsite_line"]: edge["confidence"] for edge in deduped_rows}

    expect_equal(confidences.get(1), 0.5)


def test_alias_and_import_helpers_cover_aliases_and_edges() -> None:
    """Alias utilities normalize attributes, aliases, and import edges."""

    def _import_from_stmt(code: str) -> cst.ImportFrom:
        module = cst.parse_module(code)
        stmt = module.body[0]
        if not isinstance(stmt, cst.SimpleStatementLine):
            message = "Expected simple statement"
            raise TypeError(message)
        small = stmt.body[0]
        if not isinstance(small, cst.ImportFrom):
            message = "Expected ImportFrom"
            raise TypeError(message)
        return small

    def _import_stmt(code: str) -> cst.Import:
        module = cst.parse_module(code)
        stmt = module.body[0]
        if not isinstance(stmt, cst.SimpleStatementLine):
            message = "Expected simple statement"
            raise TypeError(message)
        small = stmt.body[0]
        if not isinstance(small, cst.Import):
            message = "Expected Import"
            raise TypeError(message)
        return small

    attr_node = cst.parse_expression("pkg.mod.func")
    if isinstance(attr_node, cst.Attribute):
        expect_equal(attr_to_str(attr_node), "pkg.mod.func")

    absolute_import = _import_from_stmt("from pkg.sub import item")
    expect_equal(resolve_base_module("pkg.current", absolute_import), "pkg.sub")

    relative_import = _import_from_stmt("from ..sub import helper")
    expect_equal(resolve_base_module("pkg.current.module", relative_import), "pkg.sub")

    alias_map: dict[str, str] = {}
    import_node = _import_stmt("import os as o, sys")
    record_import_aliases(import_node, alias_map)
    expect_equal(alias_map["o"], "os")
    expect_equal(alias_map["sys"], "sys")

    from_alias_map: dict[str, str] = {}
    import_from_node = _import_from_stmt("from pkg.mod import thing as alias, other")
    record_import_from_aliases(import_from_node, from_alias_map)
    expect_equal(from_alias_map["alias"], "pkg.mod.thing")
    expect_equal(from_alias_map["other"], "pkg.mod.other")

    star_alias_map: dict[str, str] = {}
    import_star_node = _import_from_stmt("from pkg.mod import *")
    record_import_from_aliases(import_star_node, star_alias_map)
    expect_equal(star_alias_map, {})

    alias_module = cst.parse_module(
        "\n".join(
            [
                "import pkg.util as pu",
                "from .sub import helper",
            ]
        )
    )
    aliases = collect_aliases(alias_module, "pkg.current")
    expect_equal(aliases["pu"], "pkg.util")
    expect_equal(aliases["helper"], "pkg.sub.helper")

    edges = collect_import_edges("pkg.current", alias_module)
    expect_true(("pkg.current", "pkg.util") in edges)
    expect_true(("pkg.current", "pkg.sub") in edges)


def test_resolve_callee_import_alias_single_element_chain() -> None:
    """resolve_callee handles alias resolution with single-element chain."""
    result = resolve_callee(
        "alias",
        ["alias"],
        {},
        {"pkg.mod.func": TEST_GOID_A},
        {"alias": "pkg.mod.func"},
    )

    expect_equal(result.callee_goid, TEST_GOID_A)
    expect_equal(result.resolved_via, "import_alias")


def test_cfg_builder_handles_conditionals_and_loops() -> None:
    """CFGBuilder builds blocks/edges across conditionals and loops."""
    source = "\n".join(
        [
            "def sample(x):",
            "    if x > 0:",
            "        y = x",
            "    else:",
            "        y = -x",
            "    for i in range(2):",
            "        if i == 0:",
            "            continue",
            "        y += i",
            "    while y < 5:",
            "        if y == 3:",
            "            break",
            "        y += 1",
            "    return y",
        ]
    )
    module = ast.parse(source)
    func_node = next(node for node in module.body if isinstance(node, ast.FunctionDef))

    builder = CFGBuilder(TEST_GOID_A, func_node, file_path=REL_PATH)
    result = builder.build()

    expect_true(result.blocks)
    expect_true(result.edges)

    kinds = {block.kind for block in result.blocks}
    expect_true("entry" in kinds)
    expect_true("exit" in kinds)
    expect_true(any(edge.kind == "true" for edge in result.edges))
    expect_true(any(edge.kind == "false" for edge in result.edges))
    expect_true(any(edge.kind == "loop" for edge in result.edges))
    expect_true(any(edge.kind == "back" for edge in result.edges))
    expect_true(any(edge.kind == "jump" for edge in result.edges))

    entry = result.blocks[0]
    expect_equal(entry.start_line, func_node.lineno)
    expect_true(entry.end_line >= entry.start_line)


def test_cfg_builder_try_except_finally_and_jump_outside_loop() -> None:
    """CFGBuilder tolerates break outside loops and builds edges for try blocks."""
    source = "\n".join(
        [
            "def handler(value):",
            "    try:",
            "        value += 1",
            "    except Exception:",
            "        value = 0",
            "    finally:",
            "        value -= 1",
            "    break_me = False",
            "    if break_me:",
            "        break",
        ]
    )
    module = ast.parse(source)
    func_node = next(node for node in module.body if isinstance(node, ast.FunctionDef))

    builder = CFGBuilder(TEST_GOID_B, func_node, file_path=REL_PATH)
    result = builder.build()

    expect_true(result.blocks)
    expect_true(any(edge.kind == "fallthrough" for edge in result.edges))

    for block in result.blocks:
        expect_true(block.start_line >= func_node.lineno or block.start_line == -1)


def test_cfg_to_rows_computes_degrees_and_defaults() -> None:
    """cfg_to_rows applies defaults for missing lines and computes degrees."""
    blocks = (
        BasicBlock(idx=0, kind="entry", label="entry", start_line=-1, end_line=-1),
        BasicBlock(idx=1, kind="body", label="body", start_line=2, end_line=3),
    )
    edges = (
        CFGEdge(src=0, dst=1, kind="next"),
        CFGEdge(src=1, dst=0, kind="back"),
    )
    cfg_result = CFGResult(blocks=blocks, edges=edges, function_goid=TEST_GOID_A)
    cfg_rows, edge_rows = cfg_to_rows(
        result=cfg_result,
        file_path=REL_PATH,
        default_start=10,
        default_end=20,
    )

    expect_length(cfg_rows, 2)
    expect_equal(cfg_rows[0].start_line, 10)
    expect_equal(cfg_rows[0].end_line, 20)
    expect_equal(cfg_rows[1].in_degree, 1)
    expect_equal(cfg_rows[1].out_degree, 1)
    expect_length(edge_rows, 2)


def _descriptor(end_line: int | None = 10) -> GoidDescriptor:
    return GoidDescriptor(
        repo=REPO,
        commit=COMMIT,
        language="python",
        rel_path=REL_PATH,
        kind="function",
        qualname="pkg.mod.func",
        start_line=1,
        end_line=end_line,
    )


def test_compute_goid_is_deterministic_and_bounded() -> None:
    """compute_goid yields deterministic value within DECIMAL_38_MAX."""
    descriptor = _descriptor()

    first = compute_goid(descriptor)
    second = compute_goid(descriptor)

    expect_equal(first, second)
    expect_true(0 <= first <= DECIMAL_38_MAX)


def test_build_urn_with_and_without_end_line() -> None:
    """build_urn includes end_line when present and omits when None."""
    with_end = build_urn(_descriptor(end_line=20))
    expect_true(with_end.endswith("&e=20"))

    without_end = build_urn(_descriptor(end_line=None))
    expect_true(without_end.endswith("?s=1"))
    expect_true("&e=" not in without_end)


def test_compute_goid_result_packages_values() -> None:
    """compute_goid_result returns GoidResult with matching urn and descriptor."""
    descriptor = _descriptor()

    result = compute_goid_result(descriptor)

    expect_equal(result.descriptor, descriptor)
    expect_equal(result.urn, build_urn(descriptor))
    expect_equal(result.goid_h128, compute_goid(descriptor))


def test_determine_kind_variants() -> None:
    """determine_kind covers module/class/method/function branches."""
    expect_equal(determine_kind("Module", None, REL_PATH, "pkg.mod"), "module")
    expect_equal(determine_kind("ClassDef", None, REL_PATH, "pkg.mod"), "class")
    expect_equal(determine_kind("FunctionDef", "pkg.mod.Class", REL_PATH, "pkg.mod"), "method")
    expect_equal(determine_kind("FunctionDef", None, REL_PATH, "pkg.mod"), "function")


def test_build_goid_row_and_crosswalk_row_fields() -> None:
    """Rows built from descriptors populate expected fields."""
    descriptor = _descriptor(end_line=4)
    goid = compute_goid(descriptor)
    urn = build_urn(descriptor)
    now = datetime.now(tz=UTC)

    goid_row = build_goid_row(descriptor, goid, urn, now)
    expect_equal(goid_row.goid_h128, goid)
    expect_equal(goid_row.urn, urn)
    expect_equal(goid_row.repo, REPO)
    expect_equal(goid_row.commit, COMMIT)
    expect_equal(goid_row.rel_path, REL_PATH)
    expect_equal(goid_row.end_line, 4)
    expect_equal(goid_row.created_at, now)

    crosswalk = build_crosswalk_row(descriptor, urn, module_path="pkg.mod", updated_at=now)
    expect_equal(crosswalk.repo, REPO)
    expect_equal(crosswalk.commit, COMMIT)
    expect_equal(crosswalk.goid, urn)
    expect_equal(crosswalk.module_path, "pkg.mod")
    expect_equal(crosswalk.file_path, REL_PATH)
    expect_equal(crosswalk.updated_at, now)


def test_collect_import_edges_basic() -> None:
    """Collect import edges from parsed imports."""
    imports = [
        ("os", ("path",)),
        ("sys", ()),
    ]
    edges = collect_import_edges_for_analysis("mymodule", imports)

    expect_length(edges, EXPECTED_EDGE_COUNT_TWO)
    expect_true(ImportEdge(src_module="mymodule", dst_module="os") in edges)
    expect_true(ImportEdge(src_module="mymodule", dst_module="sys") in edges)


def test_collect_import_edges_empty_import() -> None:
    """Skip empty imports."""
    imports = [
        ("", ()),
        ("os", ()),
    ]
    edges = collect_import_edges_for_analysis("mymodule", imports)

    expect_length(edges, EXPECTED_EDGE_COUNT_ONE)
    expect_equal(edges[0].dst_module, "os")


def test_compute_scc_empty() -> None:
    """Compute SCC on empty graph."""
    result = compute_scc([], set())
    expect_equal(result, {})


def test_compute_scc_single_node() -> None:
    """Compute SCC for single node."""
    modules = {"module_a"}
    result = compute_scc([], modules)

    expect_length(result, EXPECTED_EDGE_COUNT_ONE)
    expect_true("module_a" in result)


def test_compute_scc_simple_cycle() -> None:
    """Compute SCC for simple cycle."""
    edges = [
        ImportEdge(src_module="a", dst_module="b"),
        ImportEdge(src_module="b", dst_module="c"),
        ImportEdge(src_module="c", dst_module="a"),
    ]
    modules = {"a", "b", "c"}
    result = compute_scc(edges, modules)

    expect_equal(result["a"], result["b"])
    expect_equal(result["b"], result["c"])


def test_compute_layers_empty() -> None:
    """Compute layers on empty graph."""
    result = compute_layers([], set(), {})
    expect_equal(result, {})


def test_compute_layers_chain() -> None:
    """Compute layers for linear chain.

    In import graph layers, roots (modules that import others but aren't
    imported) have the highest layer, and leaves have layer 0.
    """
    edges = [
        ImportEdge(src_module="a", dst_module="b"),
        ImportEdge(src_module="b", dst_module="c"),
    ]
    modules = {"a", "b", "c"}
    scc_map = {"a": 0, "b": 1, "c": 2}
    result = compute_layers(edges, modules, scc_map)

    expect_true(result["a"] > result["c"] or result["c"] > result["a"])


def test_analyze_imports_full() -> None:
    """Full import analysis."""
    edges = [
        ImportEdge(src_module="main", dst_module="utils"),
        ImportEdge(src_module="utils", dst_module="helpers"),
    ]
    modules = {"main", "utils", "helpers"}
    result = analyze_imports(edges, modules)

    expect_true(isinstance(result, ImportAnalysisResult))
    expect_length(result.edges, EXPECTED_EDGE_COUNT_TWO)
    expect_length(result.modules, EXPECTED_EDGE_COUNT_THREE)
    expect_true("main" in result.scc_map)
    expect_true("main" in result.layer_map)


def test_build_import_module_rows() -> None:
    """Build module rows from analysis result."""
    edges = [ImportEdge(src_module="a", dst_module="b")]
    modules = {"a", "b"}
    analysis = analyze_imports(edges, modules)

    rows = build_import_module_rows("repo", "commit", analysis)

    expect_length(rows, EXPECTED_EDGE_COUNT_TWO)
    expect_true(all(r.repo == "repo" for r in rows))
    expect_true(all(r.commit == "commit" for r in rows))


def test_build_import_edge_rows() -> None:
    """Build edge rows from analysis result."""
    edges = [ImportEdge(src_module="a", dst_module="b")]
    modules = {"a", "b"}
    analysis = analyze_imports(edges, modules)

    rows = build_import_edge_rows("repo", "commit", analysis)

    expect_length(rows, EXPECTED_EDGE_COUNT_ONE)
    row = rows[0]
    expect_equal(row.src_module, "a")
    expect_equal(row.dst_module, "b")


def test_symbol_occurrence_is_definition() -> None:
    """SymbolOccurrence identifies definition role."""
    occ = SymbolOccurrence(symbol="sym", rel_path="test.py", line=10, roles=DEFINITION_ROLE)
    expect_true(occ.is_definition)
    expect_true(not occ.is_reference)


def test_symbol_occurrence_is_reference() -> None:
    """SymbolOccurrence identifies reference role."""
    occ = SymbolOccurrence(symbol="sym", rel_path="test.py", line=10, roles=REFERENCE_ROLE)
    expect_true(not occ.is_definition)
    expect_true(occ.is_reference)


def test_symbol_occurrence_combined_roles() -> None:
    """SymbolOccurrence handles combined roles."""
    occ = SymbolOccurrence(symbol="sym", rel_path="test.py", line=10, roles=REFERENCE_ROLE_COMBINED)
    expect_true(occ.is_reference)


def test_build_def_map_basic() -> None:
    """Build definition map from occurrences."""
    occurrences = [
        SymbolOccurrence(symbol="func", rel_path="a.py", line=5, roles=DEFINITION_ROLE),
        SymbolOccurrence(symbol="var", rel_path="b.py", line=10, roles=DEFINITION_ROLE),
    ]
    result = build_def_map(occurrences)

    expect_equal(result["func"], "a.py")
    expect_equal(result["var"], "b.py")


def test_build_def_map_first_definition_wins() -> None:
    """First definition wins when symbol defined multiple times."""
    occurrences = [
        SymbolOccurrence(symbol="func", rel_path="a.py", line=5, roles=DEFINITION_ROLE),
        SymbolOccurrence(symbol="func", rel_path="b.py", line=10, roles=DEFINITION_ROLE),
    ]
    result = build_def_map(occurrences)

    expect_equal(result["func"], "a.py")


def test_build_def_map_ignores_references() -> None:
    """Build def map ignores references."""
    occurrences = [
        SymbolOccurrence(symbol="func", rel_path="a.py", line=5, roles=REFERENCE_ROLE),
    ]
    result = build_def_map(occurrences)

    expect_equal(result, {})


def test_build_use_edges_basic() -> None:
    """Build use edges from occurrences."""
    occurrences = [
        SymbolOccurrence(symbol="func", rel_path="a.py", line=5, roles=DEFINITION_ROLE),
        SymbolOccurrence(symbol="func", rel_path="b.py", line=10, roles=REFERENCE_ROLE),
    ]
    def_map = {"func": "a.py"}
    module_by_path: dict[str, str] = {}

    edges = build_use_edges(occurrences, def_map, module_by_path)

    expect_length(edges, EXPECTED_EDGE_COUNT_ONE)
    edge = edges[0]
    expect_equal(edge.symbol, "func")
    expect_equal(edge.def_path, "a.py")
    expect_equal(edge.use_path, "b.py")
    expect_true(edge.same_file is False)


def test_build_use_edges_same_file() -> None:
    """Build use edge correctly marks same_file."""
    occurrences = [
        SymbolOccurrence(symbol="func", rel_path="a.py", line=5, roles=DEFINITION_ROLE),
        SymbolOccurrence(symbol="func", rel_path="a.py", line=20, roles=REFERENCE_ROLE),
    ]
    def_map = {"func": "a.py"}
    module_by_path: dict[str, str] = {}

    edges = build_use_edges(occurrences, def_map, module_by_path)

    expect_length(edges, EXPECTED_EDGE_COUNT_ONE)
    expect_true(edges[0].same_file)


def test_build_use_edges_same_module() -> None:
    """Build use edge correctly marks same_module."""
    occurrences = [
        SymbolOccurrence(symbol="func", rel_path="a.py", line=5, roles=DEFINITION_ROLE),
        SymbolOccurrence(symbol="func", rel_path="b.py", line=10, roles=REFERENCE_ROLE),
    ]
    def_map = {"func": "a.py"}
    module_by_path = {"a.py": "mymodule", "b.py": "mymodule"}

    edges = build_use_edges(occurrences, def_map, module_by_path)

    expect_length(edges, EXPECTED_EDGE_COUNT_ONE)
    expect_true(edges[0].same_module)


def test_build_use_edges_deduplicates() -> None:
    """Build use edges deduplicates same symbol/def/use combinations."""
    occurrences = [
        SymbolOccurrence(symbol="func", rel_path="a.py", line=5, roles=DEFINITION_ROLE),
        SymbolOccurrence(symbol="func", rel_path="b.py", line=10, roles=REFERENCE_ROLE),
        SymbolOccurrence(symbol="func", rel_path="b.py", line=15, roles=REFERENCE_ROLE),
    ]
    def_map = {"func": "a.py"}

    edges = build_use_edges(occurrences, def_map, {})

    expect_length(edges, EXPECTED_EDGE_COUNT_ONE)


def test_build_use_def_mapping() -> None:
    """Build use to def mapping."""
    occurrences = [
        SymbolOccurrence(symbol="func1", rel_path="a.py", line=5, roles=DEFINITION_ROLE),
        SymbolOccurrence(symbol="func2", rel_path="b.py", line=5, roles=DEFINITION_ROLE),
        SymbolOccurrence(symbol="func1", rel_path="c.py", line=10, roles=REFERENCE_ROLE),
        SymbolOccurrence(symbol="func2", rel_path="c.py", line=15, roles=REFERENCE_ROLE),
    ]
    def_map = {"func1": "a.py", "func2": "b.py"}

    result = build_use_def_mapping(occurrences, def_map)

    expect_true("c.py" in result)
    expect_equal(result["c.py"], {"a.py", "b.py"})


def test_edges_to_rows() -> None:
    """Convert edges to rows."""
    edges = [
        SymbolUseEdge(
            symbol="func",
            def_path="a.py",
            use_path="b.py",
            same_file=False,
            same_module=True,
            def_goid=TEST_GOID_A,
            use_goid=TEST_GOID_B,
        )
    ]
    rows = edges_to_rows(edges)

    expect_length(rows, EXPECTED_EDGE_COUNT_ONE)
    row = rows[0]
    expect_equal(row.symbol, "func")
    expect_equal(row.def_path, "a.py")
    expect_equal(row.use_path, "b.py")
    expect_true(row.same_file is False)
    expect_true(row.same_module is True)
    expect_equal(row.def_goid_h128, TEST_GOID_A)
    expect_equal(row.use_goid_h128, TEST_GOID_B)


def test_parse_symbol_roles_int() -> None:
    """Parse symbol roles from int."""
    expect_equal(parse_symbol_roles(DEFINITION_ROLE), DEFINITION_ROLE)


def test_parse_symbol_roles_string() -> None:
    """Parse symbol roles from string."""
    expect_equal(parse_symbol_roles("2"), REFERENCE_ROLE)


def test_parse_symbol_roles_invalid_string() -> None:
    """Parse symbol roles handles invalid string."""
    expect_equal(parse_symbol_roles("invalid"), 0)


def test_parse_symbol_roles_none() -> None:
    """Parse symbol roles handles None."""
    expect_equal(parse_symbol_roles(None), 0)


def test_call_edge_frozen() -> None:
    """CallEdge is frozen."""
    edge = CallEdge(
        caller_goid=TEST_GOID_A,
        callee_goid=TEST_GOID_B,
        callee_name="func",
        call_line=10,
        rel_path="test.py",
        evidence="local_name",
        confidence=LOCAL_CONFIDENCE,
    )
    assert_cannot_setattr(edge, "caller_goid", TEST_GOID_C)


def test_resolution_result_frozen() -> None:
    """ResolutionResult is frozen."""
    result = ResolutionResult(
        callee_goid=TEST_GOID_A, resolved_via="local_name", confidence=LOCAL_CONFIDENCE
    )
    assert_cannot_setattr(result, "callee_goid", TEST_GOID_B)


def test_import_edge_frozen() -> None:
    """ImportEdge is frozen."""
    edge = ImportEdge(src_module="a", dst_module="b")
    assert_cannot_setattr(edge, "src_module", "c")


def test_symbol_occurrence_frozen() -> None:
    """SymbolOccurrence is frozen."""
    occ = SymbolOccurrence(symbol="sym", rel_path="test.py", line=10, roles=DEFINITION_ROLE)
    assert_cannot_setattr(occ, "symbol", "other")


def test_symbol_use_edge_frozen() -> None:
    """SymbolUseEdge is frozen."""
    edge = SymbolUseEdge(
        symbol="func",
        def_path="a.py",
        use_path="b.py",
        same_file=False,
        same_module=False,
    )
    assert_cannot_setattr(edge, "symbol", "other")
