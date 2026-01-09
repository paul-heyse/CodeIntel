"""Bytecode overlay CPG edges."""

from __future__ import annotations

import opcode
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

import pyarrow as pa

from codeintel.build.graphs.assembly import table_rows
from codeintel.build.hamilton.native.graphs.cpg2.edge_helpers import (
    finalize_cpg_edge_rows,
)
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_edge_ordinal, cpg_node_id
from codeintel.build.tabular.extras_ops import extras_kv_from_mapping
from codeintel.core.columnar.rows import empty_table_for_table

CPG_EDGES_TABLE_KEY = "graph.cpg_edges"
AST_NODES_TABLE_KEY = "core.ast_nodes"
SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
SYNTAX_CALLS_TABLE_KEY = "core.syntax_calls"
SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbol_information"
PY_SYM_SCOPES_TABLE_KEY = "core.py_sym_scopes"
PY_SYM_BINDINGS_TABLE_KEY = "core.py_sym_bindings"
PY_SYM_RESOLUTION_EDGES_TABLE_KEY = "core.py_sym_resolution_edges"
PY_BC_CODE_UNITS_TABLE_KEY = "core.py_bc_code_units"
PY_BC_INSTRUCTIONS_TABLE_KEY = "core.py_bc_instructions"
PY_BC_BLOCKS_TABLE_KEY = "core.py_bc_blocks"
PY_BC_CFG_EDGES_TABLE_KEY = "core.py_bc_cfg_edges"
PY_BC_DEFUSE_EVENTS_TABLE_KEY = "core.py_bc_defuse_events"


@dataclass(frozen=True)
class OverlayEdgeDiagnostics:
    """Diagnostics for overlay edge resolution."""

    expected_edges: int
    produced_edges: int
    dropped_edges: int


@dataclass(frozen=True)
class _AstAnchorMatch:
    node_hash: str
    node_type: str | None
    match_kind: str


@dataclass(frozen=True)
class _StackValue:
    instr_id: str
    push_index: int
    opname: str | None
    emit_edge: bool


@dataclass(frozen=True)
class _StackEdgeContext:
    instr: Mapping[str, object]
    value: _StackValue
    block_id: str
    pop_index: int
    depth_before: int
    depth_after: int


@dataclass(frozen=True, slots=True)
class _CallsiteSymbolInputs:
    instr_rows: list[dict[str, object]]
    calls_by_path: dict[str, list[dict[str, object]]]
    exact_index: dict[tuple[str, str], dict[str, list[Mapping[str, object]]]]
    leaf_index: dict[tuple[str, str], dict[str, list[Mapping[str, object]]]]


@dataclass(frozen=True)
class _ReachesContext:
    scope_map: Mapping[str, str]
    blocks_by_unit: Mapping[str, list[_PyBcBlock]]
    cfg_by_unit: Mapping[str, list[tuple[str, str]]]
    bindings_by_scope: Mapping[tuple[str, str, str], Mapping[str, object]]
    resolution_map: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class PyBcReachesInputs:
    """Inputs required for bytecode reachability overlays."""

    defuse_events: pa.Table
    code_units: pa.Table
    scopes: pa.Table
    bindings: pa.Table
    resolution_edges: pa.Table
    blocks: pa.Table
    cfg_edges: pa.Table


@dataclass(frozen=True)
class _PyBcReachesRows:
    event_rows: list[dict[str, object]]
    code_unit_rows: list[dict[str, object]]
    scope_rows: list[dict[str, object]]
    binding_rows: list[dict[str, object]]
    resolution_rows: list[dict[str, object]]
    block_rows: list[dict[str, object]]
    cfg_rows: list[dict[str, object]]


class _DefUseEvent(TypedDict):
    repo: str
    commit: str
    rel_path: str
    code_unit_id: str
    instr_id: str
    instr_index: int
    event_kind: str
    space: str | None
    name: str | None


class _ResolvedDefUseEvent(_DefUseEvent):
    binding_id: str | None
    var_key: tuple[str, str, str]


class _PyBcBlock(TypedDict):
    code_unit_id: str
    block_id: str
    first_instr_index: int
    last_instr_index: int


def cpg2_edges__py_bc_instruction_ast(
    instructions: pa.Table,
    ast_nodes: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build bytecode instruction -> AST anchor edges.

    Returns
    -------
    pyarrow.Table
        CPG edges linking bytecode instructions to AST nodes.
    """
    edges = _py_bc_instruction_ast_edges_to_rows(instructions, ast_nodes)
    table = finalize_cpg_edge_rows(edges)
    _record_diagnostics(
        diagnostics,
        "overlay_bytecode_instruction_ast",
        expected_edges=instructions.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__py_bc_callsite(
    instructions: pa.Table,
    syntax_calls: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build bytecode callsite -> syntax call edges.

    Returns
    -------
    pyarrow.Table
        CPG edges linking bytecode callsites to syntax call nodes.
    """
    edges = _py_bc_callsite_edges_to_rows(instructions, syntax_calls)
    table = finalize_cpg_edge_rows(edges)
    _record_diagnostics(
        diagnostics,
        "overlay_bytecode_callsite",
        expected_edges=instructions.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__py_bc_callsite_symbol(
    instructions: pa.Table,
    syntax_calls: pa.Table,
    scip_symbols: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build bytecode callsite -> SCIP symbol edges.

    Returns
    -------
    pyarrow.Table
        CPG edges linking bytecode callsites to resolved symbols.
    """
    edges = _py_bc_callsite_symbol_edges_to_rows(instructions, syntax_calls, scip_symbols)
    table = finalize_cpg_edge_rows(edges)
    _record_diagnostics(
        diagnostics,
        "overlay_bytecode_callsite_symbol",
        expected_edges=instructions.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__py_bc_cfg(
    cfg_edges: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build bytecode CFG edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for bytecode CFG.
    """
    edges = _py_bc_cfg_edges_to_rows(cfg_edges)
    table = finalize_cpg_edge_rows(edges)
    _record_diagnostics(
        diagnostics,
        "overlay_bytecode_cfg",
        expected_edges=cfg_edges.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__py_bc_defuse_binding(
    inputs: PyBcReachesInputs,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build bytecode def/use -> binding edges.

    Returns
    -------
    pyarrow.Table
        CPG edges linking def/use events to symtable bindings.
    """
    edges = _py_bc_defuse_binding_edges_to_rows(
        inputs.defuse_events,
        inputs.code_units,
        inputs.scopes,
        inputs.bindings,
        inputs.resolution_edges,
    )
    table = finalize_cpg_edge_rows(edges)
    _record_diagnostics(
        diagnostics,
        "overlay_bytecode_defuse_binding",
        expected_edges=inputs.defuse_events.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__py_bc_memory(
    defuse_events: pa.Table,
    instructions: pa.Table,
    ast_nodes: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build bytecode memory access edges.

    Returns
    -------
    pyarrow.Table
        CPG edges linking memory operations to AST anchors.
    """
    edges = _py_bc_memory_edges_to_rows(defuse_events, instructions, ast_nodes)
    table = finalize_cpg_edge_rows(edges)
    _record_diagnostics(
        diagnostics,
        "overlay_bytecode_memory",
        expected_edges=defuse_events.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__py_bc_stack(
    instructions: pa.Table,
    blocks: pa.Table,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build bytecode stack edges.

    Returns
    -------
    pyarrow.Table
        CPG edges that capture stack flow.
    """
    edges = _py_bc_stack_edges_to_rows(instructions, blocks)
    table = finalize_cpg_edge_rows(edges)
    _record_diagnostics(
        diagnostics,
        "overlay_bytecode_stack",
        expected_edges=instructions.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def cpg2_edges__py_bc_reaches(
    inputs: PyBcReachesInputs,
    *,
    diagnostics: dict[str, object] | None = None,
) -> pa.Table:
    """Build bytecode reaches edges.

    Returns
    -------
    pyarrow.Table
        CPG edges for reaching-defs analysis.
    """
    edges = _py_bc_reaches_edges_to_rows(inputs)
    table = finalize_cpg_edge_rows(edges)
    _record_diagnostics(
        diagnostics,
        "overlay_bytecode_reaches",
        expected_edges=inputs.defuse_events.num_rows,
        produced_edges=table.num_rows,
    )
    return table


def _py_bc_instruction_ast_edges_to_rows(
    instructions: pa.Table,
    ast_nodes: pa.Table,
) -> list[dict[str, object]]:
    instr_rows = _collect_rows(
        instructions,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "code_unit_id",
            "instr_id",
            "span_start_byte",
            "span_end_byte",
            "line_number",
        ),
    )
    ast_rows = _collect_rows(
        ast_nodes,
        columns=("path", "hash", "node_type", "start_byte", "end_byte", "lineno", "end_lineno"),
    )
    if not instr_rows or not ast_rows:
        return []
    ast_by_path = _ast_nodes_by_path(ast_rows)
    edges: list[dict[str, object]] = []
    for instr in instr_rows:
        rel_path = _coerce_str(instr.get("rel_path"))
        if rel_path is None:
            continue
        nodes = ast_by_path.get(rel_path)
        if not nodes:
            continue
        anchor = _select_ast_anchor(
            nodes,
            instr_start=_coerce_int(instr.get("span_start_byte")),
            instr_end=_coerce_int(instr.get("span_end_byte")),
            line_number=_coerce_int(instr.get("line_number")),
        )
        if anchor is None:
            continue
        edge = _bytecode_ast_anchor_edge_row(instr, anchor)
        if edge:
            edges.append(edge)
    return edges


def _bytecode_ast_anchor_edge_row(
    instr: Mapping[str, object],
    anchor: _AstAnchorMatch,
) -> dict[str, object]:
    repo = _coerce_str(instr.get("repo"))
    commit = _coerce_str(instr.get("commit"))
    rel_path = _coerce_str(instr.get("rel_path"))
    code_unit_id = _coerce_str(instr.get("code_unit_id"))
    instr_id = _coerce_str(instr.get("instr_id"))
    if _has_missing(repo, commit, rel_path, code_unit_id, instr_id):
        return {}
    src_cpg_node_id = _instruction_cpg_id(
        repo=cast("str", repo),
        commit=cast("str", commit),
        rel_path=cast("str", rel_path),
        code_unit_id=cast("str", code_unit_id),
        instr_id=cast("str", instr_id),
    )
    extras = {
        "match_kind": anchor.match_kind,
        "ast_node_type": anchor.node_type,
    }
    extras_kv = extras_kv_from_mapping(extras)
    ordinal = cpg_edge_ordinal(
        "graph.cpg_edges_bc_instr_ast",
        {"code_unit_id": code_unit_id, "instr_id": instr_id, "ast_hash": anchor.node_hash},
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": cpg_node_id(AST_NODES_TABLE_KEY, {"hash": anchor.node_hash}),
        "edge_kind": "BYTECODE_ANCHOR",
        "edge_layer": "SYNTAX",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _is_call_op(opname: str | None) -> bool:
    if opname is None:
        return False
    return opname in {"CALL", "CALL_FUNCTION", "CALL_FUNCTION_EX", "CALL_METHOD", "CALL_KW"}


def _select_syntax_call(
    calls: list[dict[str, object]],
    *,
    instr_start: int | None,
    instr_end: int | None,
    line_number: int | None,
) -> dict[str, object] | None:
    candidates = _ast_anchor_candidates_by_span(calls, instr_start=instr_start, instr_end=instr_end)
    match_kind = "SPAN_CONTAINS"
    if not candidates:
        candidates = []
        if line_number is not None:
            for call in calls:
                start_line = _coerce_int(call.get("start_line"))
                end_line = _coerce_int(call.get("end_line")) or start_line
                if start_line is None or end_line is None:
                    continue
                if start_line <= line_number <= end_line:
                    span_len = _span_length(
                        _coerce_int(call.get("start_byte")),
                        _coerce_int(call.get("end_byte")),
                    )
                    sort_key = span_len if span_len is not None else 2**63
                    candidates.append((sort_key, call))
        match_kind = "LINE_CONTAINS"
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    candidate = dict(candidates[0][1])
    candidate["match_kind"] = match_kind
    return candidate


def _bytecode_callsite_edge_row(
    instr: Mapping[str, object],
    call_row: Mapping[str, object],
) -> dict[str, object]:
    repo = _coerce_str(instr.get("repo"))
    commit = _coerce_str(instr.get("commit"))
    rel_path = _coerce_str(instr.get("rel_path"))
    code_unit_id = _coerce_str(instr.get("code_unit_id"))
    instr_id = _coerce_str(instr.get("instr_id"))
    producer = _coerce_str(call_row.get("producer"))
    call_node_id = _coerce_str(call_row.get("call_node_id"))
    call_id = _coerce_str(call_row.get("call_id"))
    match_kind = _coerce_str(call_row.get("match_kind"))
    if _has_missing(
        repo,
        commit,
        rel_path,
        code_unit_id,
        instr_id,
        producer,
        call_node_id,
        call_id,
    ):
        return {}
    src_cpg_node_id = _instruction_cpg_id(
        repo=cast("str", repo),
        commit=cast("str", commit),
        rel_path=cast("str", rel_path),
        code_unit_id=cast("str", code_unit_id),
        instr_id=cast("str", instr_id),
    )
    dst_cpg_node_id = _syntax_node_cpg_id(
        repo=cast("str", repo),
        commit=cast("str", commit),
        rel_path=cast("str", rel_path),
        producer=cast("str", producer),
        node_id=cast("str", call_node_id),
    )
    extras = {
        "call_id": call_id,
        "callee_text": call_row.get("callee_text"),
        "match_kind": match_kind,
    }
    extras_kv = extras_kv_from_mapping(extras)
    ordinal = cpg_edge_ordinal(
        "graph.cpg_edges_bc_callsite",
        {"code_unit_id": code_unit_id, "instr_id": instr_id, "call_id": call_id},
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": "BYTECODE_CALLSITE",
        "edge_layer": "CALL",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _py_bc_callsite_edges_to_rows(
    instructions: pa.Table,
    syntax_calls: pa.Table,
) -> list[dict[str, object]]:
    instr_rows = _collect_rows(
        instructions,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "code_unit_id",
            "instr_id",
            "baseopname",
            "opname",
            "span_start_byte",
            "span_end_byte",
            "line_number",
        ),
    )
    call_rows = _collect_rows(
        syntax_calls,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "producer",
            "call_id",
            "call_node_id",
            "start_byte",
            "end_byte",
            "start_line",
            "end_line",
            "callee_text",
        ),
    )
    if not instr_rows or not call_rows:
        return []
    calls_by_path = _ast_nodes_by_path(call_rows)
    edges: list[dict[str, object]] = []
    for instr in instr_rows:
        opname = _coerce_str(instr.get("baseopname")) or _coerce_str(instr.get("opname"))
        if not _is_call_op(opname):
            continue
        rel_path = _coerce_str(instr.get("rel_path"))
        if rel_path is None:
            continue
        calls = calls_by_path.get(rel_path)
        if not calls:
            continue
        call_match = _select_syntax_call(
            calls,
            instr_start=_coerce_int(instr.get("span_start_byte")),
            instr_end=_coerce_int(instr.get("span_end_byte")),
            line_number=_coerce_int(instr.get("line_number")),
        )
        if call_match is None:
            continue
        edge = _bytecode_callsite_edge_row(instr, call_match)
        if edge:
            edges.append(edge)
    return edges


def _coerce_callee_text(value: object) -> str | None:
    text = _coerce_str(value)
    if text is None:
        return None
    stripped = text.strip()
    return stripped or None


def _leaf_name(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _display_name_variants(display_name: str) -> list[str]:
    normalized = display_name.replace("::", ".")
    if "#" in normalized:
        normalized = normalized.split("#", 1)[0]
    if normalized and normalized != display_name:
        return [display_name, normalized]
    return [display_name] if display_name else []


def _index_symbol_rows(
    rows: Sequence[Mapping[str, object]],
) -> tuple[
    dict[tuple[str, str], dict[str, list[Mapping[str, object]]]],
    dict[tuple[str, str], dict[str, list[Mapping[str, object]]]],
]:
    exact: dict[tuple[str, str], dict[str, list[Mapping[str, object]]]] = {}
    leaf: dict[tuple[str, str], dict[str, list[Mapping[str, object]]]] = {}
    for row in rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        display_name = _coerce_str(row.get("display_name"))
        if _has_missing(repo, commit, display_name):
            continue
        key = (cast("str", repo), cast("str", commit))
        for variant in _display_name_variants(cast("str", display_name)):
            if not variant:
                continue
            exact.setdefault(key, {}).setdefault(variant, []).append(row)
            leaf_name = _leaf_name(variant)
            leaf.setdefault(key, {}).setdefault(leaf_name, []).append(row)
    return exact, leaf


def _symbol_matches_from_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    match_kind: str,
    confidence: float,
) -> list[tuple[Mapping[str, object], str, float]]:
    matches: list[tuple[Mapping[str, object], str, float]] = []
    seen_symbols: set[str] = set()
    for row in rows:
        symbol = _coerce_str(row.get("symbol"))
        if symbol is None or symbol in seen_symbols:
            continue
        seen_symbols.add(symbol)
        matches.append((row, match_kind, confidence))
    return matches


def _callsite_symbol_matches(
    *,
    repo: str,
    commit: str,
    callee_text: str,
    exact_index: Mapping[tuple[str, str], Mapping[str, Sequence[Mapping[str, object]]]],
    leaf_index: Mapping[tuple[str, str], Mapping[str, Sequence[Mapping[str, object]]]],
) -> list[tuple[Mapping[str, object], str, float]]:
    key = (repo, commit)
    exact_matches = exact_index.get(key, {}).get(callee_text)
    if exact_matches:
        return _symbol_matches_from_rows(
            exact_matches,
            match_kind="display_name",
            confidence=0.7,
        )
    leaf_matches = leaf_index.get(key, {}).get(_leaf_name(callee_text), [])
    return _symbol_matches_from_rows(
        leaf_matches,
        match_kind="leaf_name",
        confidence=0.35,
    )


def _bytecode_callsite_symbol_edge_row(
    instr: Mapping[str, object],
    call_row: Mapping[str, object],
    symbol_row: Mapping[str, object],
    *,
    match_kind: str,
    confidence: float,
) -> dict[str, object]:
    repo = _coerce_str(instr.get("repo"))
    commit = _coerce_str(instr.get("commit"))
    rel_path = _coerce_str(instr.get("rel_path"))
    code_unit_id = _coerce_str(instr.get("code_unit_id"))
    instr_id = _coerce_str(instr.get("instr_id"))
    symbol = _coerce_str(symbol_row.get("symbol"))
    if _has_missing(repo, commit, rel_path, code_unit_id, instr_id, symbol):
        return {}
    src_cpg_node_id = _instruction_cpg_id(
        repo=cast("str", repo),
        commit=cast("str", commit),
        rel_path=cast("str", rel_path),
        code_unit_id=cast("str", code_unit_id),
        instr_id=cast("str", instr_id),
    )
    dst_cpg_node_id = cpg_node_id(
        SCIP_SYMBOLS_TABLE_KEY,
        {"repo": repo, "commit": commit, "symbol": symbol},
    )
    extras = {
        "call_id": call_row.get("call_id"),
        "callee_text": call_row.get("callee_text"),
        "match_kind": match_kind,
        "confidence": confidence,
        "symbol_display_name": symbol_row.get("display_name"),
    }
    extras_kv = extras_kv_from_mapping(extras)
    ordinal = cpg_edge_ordinal(
        "graph.cpg_edges_bc_callsite_symbol",
        {
            "code_unit_id": code_unit_id,
            "instr_id": instr_id,
            "symbol": symbol,
            "match_kind": match_kind,
        },
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": "BYTECODE_CALLS_SYMBOL",
        "edge_layer": "CALL",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _callsite_symbol_edges_for_instr(
    instr: Mapping[str, object],
    *,
    calls_by_path: Mapping[str, list[dict[str, object]]],
    exact_index: Mapping[tuple[str, str], Mapping[str, Sequence[Mapping[str, object]]]],
    leaf_index: Mapping[tuple[str, str], Mapping[str, Sequence[Mapping[str, object]]]],
) -> list[dict[str, object]]:
    opname = _coerce_str(instr.get("baseopname")) or _coerce_str(instr.get("opname"))
    if not _is_call_op(opname):
        return []
    rel_path = _coerce_str(instr.get("rel_path"))
    calls = calls_by_path.get(rel_path) if rel_path else None
    if not rel_path or not calls:
        return []
    call_match = _select_syntax_call(
        calls,
        instr_start=_coerce_int(instr.get("span_start_byte")),
        instr_end=_coerce_int(instr.get("span_end_byte")),
        line_number=_coerce_int(instr.get("line_number")),
    )
    callee_text = _coerce_callee_text(call_match.get("callee_text")) if call_match else None
    if call_match is None or callee_text is None:
        return []
    repo = _coerce_str(instr.get("repo"))
    commit = _coerce_str(instr.get("commit"))
    if _has_missing(repo, commit):
        return []
    matches = _callsite_symbol_matches(
        repo=cast("str", repo),
        commit=cast("str", commit),
        callee_text=callee_text,
        exact_index=exact_index,
        leaf_index=leaf_index,
    )
    edges: list[dict[str, object]] = []
    for symbol_row, match_kind, confidence in matches:
        edge = _bytecode_callsite_symbol_edge_row(
            instr,
            call_match,
            symbol_row,
            match_kind=match_kind,
            confidence=confidence,
        )
        if edge:
            edges.append(edge)
    return edges


def _callsite_symbol_inputs(
    instructions: pa.Table,
    syntax_calls: pa.Table,
    scip_symbols: pa.Table,
) -> _CallsiteSymbolInputs | None:
    instr_rows = _collect_rows(
        instructions,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "code_unit_id",
            "instr_id",
            "baseopname",
            "opname",
            "span_start_byte",
            "span_end_byte",
            "line_number",
        ),
    )
    call_rows = _collect_rows(
        syntax_calls,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "producer",
            "call_id",
            "call_node_id",
            "start_byte",
            "end_byte",
            "start_line",
            "end_line",
            "callee_text",
        ),
    )
    symbol_rows = _collect_rows(
        scip_symbols,
        columns=("repo", "commit", "symbol", "display_name"),
    )
    if not instr_rows or not call_rows or not symbol_rows:
        return None
    calls_by_path = _ast_nodes_by_path(call_rows)
    exact_index, leaf_index = _index_symbol_rows(symbol_rows)
    return _CallsiteSymbolInputs(
        instr_rows=instr_rows,
        calls_by_path=calls_by_path,
        exact_index=exact_index,
        leaf_index=leaf_index,
    )


def _py_bc_callsite_symbol_edges_to_rows(
    instructions: pa.Table,
    syntax_calls: pa.Table,
    scip_symbols: pa.Table,
) -> list[dict[str, object]]:
    inputs = _callsite_symbol_inputs(instructions, syntax_calls, scip_symbols)
    if inputs is None:
        return []
    edges: list[dict[str, object]] = []
    for instr in inputs.instr_rows:
        edges.extend(
            _callsite_symbol_edges_for_instr(
                instr,
                calls_by_path=inputs.calls_by_path,
                exact_index=inputs.exact_index,
                leaf_index=inputs.leaf_index,
            )
        )
    return edges


def _py_bc_cfg_edges_to_rows(cfg_edges: pa.Table) -> list[dict[str, object]]:
    required = {
        "repo",
        "commit",
        "rel_path",
        "edge_id",
        "src_block_id",
        "dst_block_id",
        "kind",
    }
    if not required.issubset(cfg_edges.column_names):
        return []
    rows: list[dict[str, object]] = []
    for row in table_rows(cfg_edges):
        src_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "block_id": row.get("src_block_id"),
        }
        dst_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "block_id": row.get("dst_block_id"),
        }
        extras_values = {
            "kind": row.get("kind"),
            "cond_instr_id": row.get("cond_instr_id"),
            "exc_entry_index": row.get("exc_entry_index"),
        }
        extras_kv = extras_kv_from_mapping(extras_values)
        ordinal = cpg_edge_ordinal(
            PY_BC_CFG_EDGES_TABLE_KEY,
            {"edge_id": row.get("edge_id")},
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": cpg_node_id(PY_BC_BLOCKS_TABLE_KEY, src_pk),
                "dst_cpg_node_id": cpg_node_id(PY_BC_BLOCKS_TABLE_KEY, dst_pk),
                "edge_kind": "CFG",
                "edge_layer": "FLOW",
                "rel_path": row.get("rel_path"),
                "ordinal": ordinal,
                "extras": None,
                "extras_kv": extras_kv,
            }
        )
    return rows


def _py_bc_defuse_binding_edges_to_rows(
    defuse_events: pa.Table,
    code_units: pa.Table,
    scopes: pa.Table,
    bindings: pa.Table,
    resolution_edges: pa.Table,
) -> list[dict[str, object]]:
    event_rows = _collect_rows(
        defuse_events,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "code_unit_id",
            "instr_id",
            "instr_index",
            "event_kind",
            "space",
            "name",
        ),
    )
    if not event_rows:
        return []
    code_unit_rows = _collect_rows(
        code_units,
        columns=(
            "code_unit_id",
            "rel_path",
            "kind",
            "co_firstlineno",
            "span_start_byte",
            "span_end_byte",
        ),
    )
    scope_rows = _collect_rows(
        scopes,
        columns=(
            "rel_path",
            "scope_id",
            "scope_type",
            "lineno",
            "span_start_byte",
            "span_end_byte",
        ),
    )
    binding_rows = _collect_rows(
        bindings,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "binding_id",
            "scope_id",
            "name",
            "binding_kind",
        ),
    )
    resolution_rows = _collect_rows(
        resolution_edges,
        columns=("src_binding_id", "dst_binding_id", "kind", "confidence", "reason"),
    )
    scope_map = _build_code_unit_scope_map(code_unit_rows, scope_rows)
    bindings_by_scope, binding_meta = _build_binding_index(binding_rows)
    resolution_map = _build_resolution_map(resolution_rows)
    edges: list[dict[str, object]] = []
    for event in event_rows:
        code_unit_id = _coerce_str(event.get("code_unit_id"))
        if code_unit_id is None:
            continue
        scope_id = scope_map.get(code_unit_id)
        if scope_id is None:
            continue
        edge = _binding_edge_for_event(
            event,
            scope_id=scope_id,
            bindings_by_scope=bindings_by_scope,
            binding_meta=binding_meta,
            resolution_map=resolution_map,
        )
        if edge is not None:
            edges.append(edge)
    return edges


def _py_bc_reaches_edges_to_rows(inputs: PyBcReachesInputs) -> list[dict[str, object]]:
    rows = _collect_py_bc_reaches_rows(inputs)
    if rows is None:
        return []
    events_by_unit = _group_defuse_events(rows.event_rows)
    context = _reaches_context_from_rows(rows)
    edges: list[dict[str, object]] = []
    for code_unit_id, events in events_by_unit.items():
        edges.extend(
            _build_reaches_edges_for_unit(
                code_unit_id=code_unit_id,
                events=events,
                context=context,
            )
        )
    return edges


def _memory_edge_kind(space: str | None, event_kind: str | None) -> str | None:
    if space is None or event_kind is None:
        return None
    return _MEMORY_EDGE_KIND_MAP.get((space, event_kind))


def _bytecode_memory_edge_row(
    event: Mapping[str, object],
    *,
    anchor: _AstAnchorMatch,
    edge_kind: str,
) -> dict[str, object]:
    repo = _coerce_str(event.get("repo"))
    commit = _coerce_str(event.get("commit"))
    rel_path = _coerce_str(event.get("rel_path"))
    code_unit_id = _coerce_str(event.get("code_unit_id"))
    instr_id = _coerce_str(event.get("instr_id"))
    if _has_missing(repo, commit, rel_path, code_unit_id, instr_id):
        return {}
    src_cpg_node_id = _instruction_cpg_id(
        repo=cast("str", repo),
        commit=cast("str", commit),
        rel_path=cast("str", rel_path),
        code_unit_id=cast("str", code_unit_id),
        instr_id=cast("str", instr_id),
    )
    extras = {
        "space": event.get("space"),
        "name": event.get("name"),
        "event_kind": event.get("event_kind"),
        "confidence": event.get("confidence"),
        "match_kind": anchor.match_kind,
        "ast_node_type": anchor.node_type,
    }
    extras_kv = extras_kv_from_mapping(extras)
    ordinal = cpg_edge_ordinal(
        "graph.cpg_edges_bc_memory",
        {"code_unit_id": code_unit_id, "instr_id": instr_id, "edge_kind": edge_kind},
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": cpg_node_id(AST_NODES_TABLE_KEY, {"hash": anchor.node_hash}),
        "edge_kind": edge_kind,
        "edge_layer": "FLOW",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _instr_index(
    instr_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str], dict[str, object]]:
    instr_by_key: dict[tuple[str, str], dict[str, object]] = {}
    for instr in instr_rows:
        rel_path = _coerce_str(instr.get("rel_path"))
        instr_id = _coerce_str(instr.get("instr_id"))
        if rel_path is None or instr_id is None:
            continue
        instr_by_key[rel_path, instr_id] = dict(instr)
    return instr_by_key


def _anchor_for_key(
    key: tuple[str, str],
    *,
    rel_path: str,
    instr: Mapping[str, object],
    ast_by_path: Mapping[str, list[dict[str, object]]],
    cache: dict[tuple[str, str], _AstAnchorMatch | None],
) -> _AstAnchorMatch | None:
    anchor = cache.get(key)
    if anchor is None and key not in cache:
        nodes = ast_by_path.get(rel_path, [])
        anchor = _select_ast_anchor(
            nodes,
            instr_start=_coerce_int(instr.get("span_start_byte")),
            instr_end=_coerce_int(instr.get("span_end_byte")),
            line_number=_coerce_int(instr.get("line_number")),
        )
        cache[key] = anchor
    return anchor


def _memory_edge_for_event(
    event: Mapping[str, object],
    *,
    instr_by_key: Mapping[tuple[str, str], dict[str, object]],
    ast_by_path: Mapping[str, list[dict[str, object]]],
    anchor_cache: dict[tuple[str, str], _AstAnchorMatch | None],
) -> dict[str, object]:
    edge_kind = _memory_edge_kind(
        _coerce_str(event.get("space")),
        _coerce_str(event.get("event_kind")),
    )
    if edge_kind is None:
        return {}
    rel_path = _coerce_str(event.get("rel_path"))
    instr_id = _coerce_str(event.get("instr_id"))
    if rel_path is None or instr_id is None:
        return {}
    key = (rel_path, instr_id)
    instr = instr_by_key.get(key)
    if instr is None:
        return {}
    anchor = _anchor_for_key(
        key,
        rel_path=rel_path,
        instr=instr,
        ast_by_path=ast_by_path,
        cache=anchor_cache,
    )
    if anchor is None:
        return {}
    return _bytecode_memory_edge_row(event, anchor=anchor, edge_kind=edge_kind)


def _py_bc_memory_edges_to_rows(
    defuse_events: pa.Table,
    instructions: pa.Table,
    ast_nodes: pa.Table,
) -> list[dict[str, object]]:
    event_rows = _collect_rows(
        defuse_events,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "code_unit_id",
            "instr_id",
            "event_kind",
            "space",
            "name",
            "confidence",
        ),
    )
    instr_rows = _collect_rows(
        instructions,
        columns=("rel_path", "instr_id", "span_start_byte", "span_end_byte", "line_number"),
    )
    ast_rows = _collect_rows(
        ast_nodes,
        columns=("path", "hash", "node_type", "start_byte", "end_byte", "lineno", "end_lineno"),
    )
    if not event_rows or not instr_rows or not ast_rows:
        return []
    ast_by_path = _ast_nodes_by_path(ast_rows)
    instr_by_key = _instr_index(instr_rows)
    anchor_cache: dict[tuple[str, str], _AstAnchorMatch | None] = {}
    edges: list[dict[str, object]] = []
    for event in event_rows:
        edge = _memory_edge_for_event(
            event,
            instr_by_key=instr_by_key,
            ast_by_path=ast_by_path,
            anchor_cache=anchor_cache,
        )
        if edge:
            edges.append(edge)
    return edges


def _stack_effect_net(opname: str, arg: int | None) -> int | None:
    opcode_value = opcode.opmap.get(opname)
    if opcode_value is None:
        return None
    arg_value = arg if arg is not None else 0
    try:
        return opcode.stack_effect(opcode_value, arg_value)
    except (ValueError, TypeError):
        return None


def _load_push_count(opname: str) -> int | None:
    if not opname.startswith("LOAD_"):
        return None
    tokens = opname.split("_")
    load_tokens = sum(1 for token in tokens if token == "LOAD")
    return load_tokens if load_tokens > 0 else None


def _effect_from_push(
    *,
    opname: str,
    arg: int | None,
    push_count: int,
    emit_edge: bool,
) -> tuple[int, int, bool] | None:
    net = _stack_effect_net(opname, arg)
    if net is None:
        return None
    pop_count = push_count - net
    if pop_count < 0:
        return None
    return pop_count, push_count, emit_edge


def _effect_pop_only(*, opname: str, arg: int | None) -> tuple[int, int, bool] | None:
    net = _stack_effect_net(opname, arg)
    if net is None:
        return None
    if net > 0:
        return None
    return -net, 0, True


_STACK_SKIP_OPS = {"CACHE", "NOP", "RESUME"}
_STACK_POP_ONLY_OPS = {"POP_TOP", "RETURN_VALUE"}
_STACK_LOAD_WITH_POP = {"LOAD_ATTR", "LOAD_METHOD", "LOAD_SUPER_ATTR", "LOAD_SUPER_METHOD"}
_STACK_BINARY_OPS = {"BINARY_OP", "BINARY_SUBSCR", "COMPARE_OP", "IS_OP", "CONTAINS_OP"}
_STACK_ITER_OPS = {"GET_ITER", "FOR_ITER"}
_STACK_POP_PREFIXES = ("STORE_", "DELETE_")
_STACK_PUSH_EXACT: dict[str, tuple[int, int, bool]] = {"PUSH_NULL": (0, 1, False)}


def _stack_push_spec(opname: str, arg: int | None) -> tuple[int, int, bool] | None:
    for handler in _STACK_PUSH_HANDLERS:
        spec = handler(opname, arg)
        if spec is not None:
            return spec
    if opname in _STACK_PUSH_EXACT:
        return _STACK_PUSH_EXACT[opname]
    return None


def _stack_spec_simple(opname: str, arg: int | None) -> tuple[int, int, bool] | None:
    spec: tuple[int, int, bool] | None = None
    if opname in _STACK_SKIP_OPS:
        spec = (0, 0, False)
    elif opname in _STACK_POP_ONLY_OPS:
        spec = _effect_pop_only(opname=opname, arg=arg)
    elif opname in _STACK_LOAD_WITH_POP or opname in _STACK_BINARY_OPS:
        spec = _effect_from_push(opname=opname, arg=arg, push_count=1, emit_edge=True)
    elif opname in _STACK_ITER_OPS:
        spec = _effect_from_push(opname=opname, arg=arg, push_count=1, emit_edge=False)
    elif opname.startswith(_STACK_POP_PREFIXES):
        spec = _effect_pop_only(opname=opname, arg=arg)
    elif opname.startswith("LOAD_"):
        push_count = _load_push_count(opname)
        if push_count is not None:
            spec = _effect_from_push(opname=opname, arg=arg, push_count=push_count, emit_edge=False)
    elif opname.startswith("UNARY_"):
        spec = (1, 1, True)
    return spec


def _stack_spec_call(opname: str, arg: int | None) -> tuple[int, int, bool] | None:
    if not opname.startswith("CALL"):
        return None
    net = _stack_effect_net(opname, arg)
    if net is None:
        return None
    pop_count = (0 if net == 0 else 1) if net >= 0 else -net + 1
    return pop_count, 1, True


_STACK_PUSH_HANDLERS = (_stack_spec_simple, _stack_spec_call)


def _stack_edges_for_instruction(
    instr: Mapping[str, object],
    *,
    block_id: str,
    stack: list[_StackValue],
) -> list[dict[str, object]]:
    instr_id = _coerce_str(instr.get("instr_id"))
    opname = _coerce_str(instr.get("baseopname")) or _coerce_str(instr.get("opname"))
    if instr_id is None or opname is None:
        return []
    arg = _coerce_int(instr.get("arg"))
    spec = _stack_push_spec(opname, arg)
    if spec is None:
        return []
    pop_count, push_count, emit_edge = spec
    stack_depth = len(stack)
    pop_total = min(pop_count, len(stack))
    popped = [stack.pop() for _ in range(pop_total)]
    edges: list[dict[str, object]] = []
    if emit_edge:
        edges = [
            _stack_edge_row(
                _StackEdgeContext(
                    instr=instr,
                    value=value,
                    block_id=block_id,
                    pop_index=pop_index,
                    depth_before=stack_depth,
                    depth_after=len(stack),
                )
            )
            for pop_index, value in enumerate(reversed(popped))
        ]
    stack.extend(
        _StackValue(
            instr_id=instr_id,
            push_index=push_index,
            opname=opname,
            emit_edge=emit_edge,
        )
        for push_index in range(push_count)
    )
    return edges


def _stack_edge_row(context: _StackEdgeContext) -> dict[str, object]:
    instr = context.instr
    repo = _coerce_str(instr.get("repo"))
    commit = _coerce_str(instr.get("commit"))
    rel_path = _coerce_str(instr.get("rel_path"))
    code_unit_id = _coerce_str(instr.get("code_unit_id"))
    instr_id = _coerce_str(instr.get("instr_id"))
    if _has_missing(repo, commit, rel_path, code_unit_id, instr_id):
        return {}
    src_cpg_node_id = _instruction_cpg_id(
        repo=cast("str", repo),
        commit=cast("str", commit),
        rel_path=cast("str", rel_path),
        code_unit_id=cast("str", code_unit_id),
        instr_id=context.value.instr_id,
    )
    dst_cpg_node_id = _instruction_cpg_id(
        repo=cast("str", repo),
        commit=cast("str", commit),
        rel_path=cast("str", rel_path),
        code_unit_id=cast("str", code_unit_id),
        instr_id=cast("str", instr_id),
    )
    extras = {
        "block_id": context.block_id,
        "push_instr_id": context.value.instr_id,
        "push_index": context.value.push_index,
        "pop_index": context.pop_index,
        "opname": context.value.opname,
        "depth_before": context.depth_before,
        "depth_after": context.depth_after,
    }
    extras_kv = extras_kv_from_mapping(extras)
    ordinal = cpg_edge_ordinal(
        "graph.cpg_edges_bc_stack",
        {
            "code_unit_id": code_unit_id,
            "instr_id": instr_id,
            "push_instr_id": context.value.instr_id,
            "push_index": context.value.push_index,
            "pop_index": context.pop_index,
        },
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": "STACK_FLOW",
        "edge_layer": "FLOW",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _group_stack_instructions(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        code_unit_id = _coerce_str(row.get("code_unit_id"))
        instr_index = _coerce_int(row.get("instr_index"))
        if code_unit_id is None or instr_index is None:
            continue
        grouped[code_unit_id].append(dict(row))
    for code_unit_id, instrs in grouped.items():
        instrs.sort(key=lambda item: _coerce_int(item.get("instr_index")) or 0)
        grouped[code_unit_id] = instrs
    return grouped


def _stack_edges_for_block(
    block: _PyBcBlock,
    instrs: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    stack: list[_StackValue] = []
    for instr in instrs:
        instr_index = _coerce_int(instr.get("instr_index"))
        if instr_index is None:
            continue
        if instr_index < block["first_instr_index"]:
            continue
        if instr_index > block["last_instr_index"]:
            break
        edges.extend(
            _stack_edges_for_instruction(
                instr,
                block_id=block["block_id"],
                stack=stack,
            )
        )
    return edges


def _py_bc_stack_edges_to_rows(
    instructions: pa.Table,
    blocks: pa.Table,
) -> list[dict[str, object]]:
    instr_rows = _collect_rows(
        instructions,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "code_unit_id",
            "instr_id",
            "instr_index",
            "opname",
            "baseopname",
            "arg",
        ),
    )
    block_rows = _collect_rows(
        blocks,
        columns=("code_unit_id", "block_id", "first_instr_index", "last_instr_index"),
    )
    if not instr_rows or not block_rows:
        return []
    instrs_by_unit = _group_stack_instructions(instr_rows)
    blocks_by_unit = _group_blocks(block_rows)
    edges: list[dict[str, object]] = []
    for code_unit_id, unit_blocks in blocks_by_unit.items():
        unit_instrs = instrs_by_unit.get(code_unit_id)
        if not unit_instrs:
            continue
        for block in sorted(unit_blocks, key=lambda item: item["first_instr_index"]):
            edges.extend(_stack_edges_for_block(block, unit_instrs))
    return edges


def _collect_py_bc_reaches_rows(inputs: PyBcReachesInputs) -> _PyBcReachesRows | None:
    event_rows = _collect_rows(
        inputs.defuse_events,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "code_unit_id",
            "instr_id",
            "instr_index",
            "event_kind",
            "space",
            "name",
        ),
    )
    if not event_rows:
        return None
    return _PyBcReachesRows(
        event_rows=event_rows,
        code_unit_rows=_collect_rows(
            inputs.code_units,
            columns=(
                "code_unit_id",
                "rel_path",
                "kind",
                "co_firstlineno",
                "span_start_byte",
                "span_end_byte",
            ),
        ),
        scope_rows=_collect_rows(
            inputs.scopes,
            columns=(
                "rel_path",
                "scope_id",
                "scope_type",
                "lineno",
                "span_start_byte",
                "span_end_byte",
            ),
        ),
        binding_rows=_collect_rows(
            inputs.bindings,
            columns=(
                "repo",
                "commit",
                "rel_path",
                "binding_id",
                "scope_id",
                "name",
                "binding_kind",
            ),
        ),
        resolution_rows=_collect_rows(
            inputs.resolution_edges,
            columns=("src_binding_id", "dst_binding_id", "kind", "confidence", "reason"),
        ),
        block_rows=_collect_rows(
            inputs.blocks,
            columns=("code_unit_id", "block_id", "first_instr_index", "last_instr_index"),
        ),
        cfg_rows=_collect_rows(
            inputs.cfg_edges,
            columns=("code_unit_id", "src_block_id", "dst_block_id"),
        ),
    )


def _reaches_context_from_rows(rows: _PyBcReachesRows) -> _ReachesContext:
    scope_map = _build_code_unit_scope_map(rows.code_unit_rows, rows.scope_rows)
    bindings_by_scope, _binding_meta = _build_binding_index(rows.binding_rows)
    resolution_map = _build_resolution_map(rows.resolution_rows)
    return _ReachesContext(
        scope_map=scope_map,
        blocks_by_unit=_group_blocks(rows.block_rows),
        cfg_by_unit=_group_cfg_edges(rows.cfg_rows),
        bindings_by_scope=bindings_by_scope,
        resolution_map=resolution_map,
    )


def _build_reaches_edges_for_unit(
    *,
    code_unit_id: str,
    events: list[_DefUseEvent],
    context: _ReachesContext,
) -> list[dict[str, object]]:
    scope_id = context.scope_map.get(code_unit_id)
    if scope_id is None:
        return []
    blocks_for_unit = context.blocks_by_unit.get(code_unit_id, [])
    if not blocks_for_unit:
        return []
    rel_path = events[0]["rel_path"]
    resolved_events = _resolve_events_for_unit(
        events,
        rel_path=rel_path,
        scope_id=scope_id,
        bindings_by_scope=context.bindings_by_scope,
        resolution_map=context.resolution_map,
    )
    if not resolved_events:
        return []
    block_events = _assign_events_to_blocks(resolved_events, blocks_for_unit)
    if not block_events:
        return []
    gen_by_block, kill_by_block = _block_gen_kill(block_events)
    block_ids = [block["block_id"] for block in blocks_for_unit]
    predecessors = _build_predecessor_map(block_ids, context.cfg_by_unit.get(code_unit_id, []))
    in_defs = _compute_reaching_defs(block_ids, predecessors, gen_by_block, kill_by_block)
    return _emit_reaches_edges(block_events, in_defs)


def _resolve_events_for_unit(
    events: list[_DefUseEvent],
    *,
    rel_path: str,
    scope_id: str,
    bindings_by_scope: Mapping[tuple[str, str, str], Mapping[str, object]],
    resolution_map: Mapping[str, Mapping[str, object]],
) -> list[_ResolvedDefUseEvent]:
    resolved: list[_ResolvedDefUseEvent] = []
    for event in events:
        name = event["name"]
        if name is None:
            continue
        space = event["space"]
        binding_info = _resolve_binding_for_event(
            rel_path=rel_path,
            scope_id=scope_id,
            name=name,
            bindings_by_scope=bindings_by_scope,
            resolution_map=resolution_map,
        )
        binding_id = _coerce_str(binding_info.get("resolved_binding_id")) if binding_info else None
        var_key = _event_var_key(binding_id=binding_id, space=space, name=name)
        if var_key == ("", "", ""):
            continue
        resolved_event: _ResolvedDefUseEvent = {
            **event,
            "binding_id": binding_id,
            "var_key": var_key,
        }
        resolved.append(resolved_event)
    return resolved


def _build_predecessor_map(
    block_ids: Sequence[str],
    cfg_edges: Sequence[tuple[str, str]],
) -> dict[str, list[str]]:
    predecessors: dict[str, list[str]] = {block_id: [] for block_id in block_ids}
    for src_block_id, dst_block_id in cfg_edges:
        if dst_block_id in predecessors:
            predecessors[dst_block_id].append(src_block_id)
        elif src_block_id not in predecessors:
            predecessors[src_block_id] = []
    return predecessors


def _emit_reaches_edges(
    block_events: Mapping[str, list[_ResolvedDefUseEvent]],
    in_defs: Mapping[str, Mapping[tuple[str, str, str], set[str]]],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for block_id, events in block_events.items():
        current_defs = {key: set(defs) for key, defs in in_defs.get(block_id, {}).items()}
        for event in events:
            event_kind = event["event_kind"]
            var_key = event["var_key"]
            if event_kind == "USE":
                edges.extend(
                    _reaches_edge_row(event=event, def_instr_id=def_instr_id)
                    for def_instr_id in current_defs.get(var_key, set())
                )
            if event_kind == "DEF":
                current_defs[var_key] = {event["instr_id"]}
            elif event_kind == "KILL":
                current_defs.pop(var_key, None)
    return edges


def _parse_defuse_event_row(row: Mapping[str, object]) -> _DefUseEvent | None:
    event_kind = _coerce_str(row.get("event_kind"))
    if event_kind not in {"DEF", "USE", "KILL"}:
        return None
    code_unit_id = _coerce_str(row.get("code_unit_id"))
    instr_id = _coerce_str(row.get("instr_id"))
    instr_index = _coerce_int(row.get("instr_index"))
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    rel_path = _coerce_str(row.get("rel_path"))
    if _has_missing(code_unit_id, instr_id, instr_index, repo, commit, rel_path):
        return None
    code_unit_id_value = cast("str", code_unit_id)
    instr_id_value = cast("str", instr_id)
    instr_index_value = cast("int", instr_index)
    repo_value = cast("str", repo)
    commit_value = cast("str", commit)
    rel_path_value = cast("str", rel_path)
    space = _coerce_str(row.get("space"))
    name = _coerce_str(row.get("name"))
    return {
        "repo": repo_value,
        "commit": commit_value,
        "rel_path": rel_path_value,
        "code_unit_id": code_unit_id_value,
        "instr_id": instr_id_value,
        "instr_index": instr_index_value,
        "event_kind": event_kind,
        "space": space,
        "name": name,
    }


def _group_defuse_events(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[_DefUseEvent]]:
    grouped: dict[str, list[_DefUseEvent]] = defaultdict(list)
    for row in rows:
        event = _parse_defuse_event_row(row)
        if event is None:
            continue
        grouped[event["code_unit_id"]].append(event)
    return grouped


def _parse_block_row(row: Mapping[str, object]) -> _PyBcBlock | None:
    code_unit_id = _coerce_str(row.get("code_unit_id"))
    block_id = _coerce_str(row.get("block_id"))
    first_idx = _coerce_int(row.get("first_instr_index"))
    last_idx = _coerce_int(row.get("last_instr_index"))
    if code_unit_id is None or block_id is None or first_idx is None or last_idx is None:
        return None
    return {
        "code_unit_id": code_unit_id,
        "block_id": block_id,
        "first_instr_index": first_idx,
        "last_instr_index": last_idx,
    }


def _group_blocks(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[_PyBcBlock]]:
    grouped: dict[str, list[_PyBcBlock]] = defaultdict(list)
    for row in rows:
        block = _parse_block_row(row)
        if block is None:
            continue
        grouped[block["code_unit_id"]].append(block)
    return grouped


def _parse_cfg_edge_row(row: Mapping[str, object]) -> tuple[str, str, str] | None:
    code_unit_id = _coerce_str(row.get("code_unit_id"))
    src_block_id = _coerce_str(row.get("src_block_id"))
    dst_block_id = _coerce_str(row.get("dst_block_id"))
    if code_unit_id is None or src_block_id is None or dst_block_id is None:
        return None
    return code_unit_id, src_block_id, dst_block_id


def _group_cfg_edges(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[tuple[str, str]]]:
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        parsed = _parse_cfg_edge_row(row)
        if parsed is None:
            continue
        code_unit_id, src_block_id, dst_block_id = parsed
        grouped[code_unit_id].append((src_block_id, dst_block_id))
    return grouped


def _assign_events_to_blocks(
    events: list[_ResolvedDefUseEvent],
    blocks: list[_PyBcBlock],
) -> dict[str, list[_ResolvedDefUseEvent]]:
    block_events: dict[str, list[_ResolvedDefUseEvent]] = defaultdict(list)
    for event in events:
        instr_index = event["instr_index"]
        block_id = _block_for_instr(blocks, instr_index)
        if block_id is None:
            continue
        block_events[block_id].append(event)
    for block_id, event_list in block_events.items():
        block_events[block_id] = sorted(event_list, key=lambda item: item["instr_index"])
    return block_events


def _block_for_instr(blocks: list[_PyBcBlock], instr_index: int) -> str | None:
    for block in blocks:
        if block["first_instr_index"] <= instr_index <= block["last_instr_index"]:
            return block["block_id"]
    return None


def _block_gen_kill(
    block_events: Mapping[str, list[_ResolvedDefUseEvent]],
) -> tuple[
    dict[str, dict[tuple[str, str, str], str]],
    dict[str, set[tuple[str, str, str]]],
]:
    gen_by_block: dict[str, dict[tuple[str, str, str], str]] = {}
    kill_by_block: dict[str, set[tuple[str, str, str]]] = {}
    for block_id, events in block_events.items():
        gen_map: dict[tuple[str, str, str], str] = {}
        kill_set: set[tuple[str, str, str]] = set()
        for event in events:
            var_key = event["var_key"]
            event_kind = event["event_kind"]
            if event_kind == "DEF":
                gen_map[var_key] = event["instr_id"]
                kill_set.add(var_key)
            elif event_kind == "KILL":
                kill_set.add(var_key)
        gen_by_block[block_id] = gen_map
        kill_by_block[block_id] = kill_set
    return gen_by_block, kill_by_block


def _merge_def_maps(
    maps: Iterable[Mapping[tuple[str, str, str], set[str]]],
) -> dict[tuple[str, str, str], set[str]]:
    merged: dict[tuple[str, str, str], set[str]] = {}
    for mapping in maps:
        for key, defs in mapping.items():
            existing = merged.get(key)
            if existing is None:
                merged[key] = set(defs)
            else:
                existing.update(defs)
    return merged


def _apply_gen_kill(
    in_defs: Mapping[tuple[str, str, str], set[str]],
    gen_map: Mapping[tuple[str, str, str], str],
    kill_set: Iterable[tuple[str, str, str]],
) -> dict[tuple[str, str, str], set[str]]:
    kill_keys = set(kill_set)
    out_defs: dict[tuple[str, str, str], set[str]] = {
        key: set(defs) for key, defs in in_defs.items() if key not in kill_keys
    }
    for key, instr_id in gen_map.items():
        out_defs[key] = {instr_id}
    return out_defs


def _compute_reaching_defs(
    block_ids: Sequence[str],
    predecessors: Mapping[str, Sequence[str]],
    gen_by_block: Mapping[str, Mapping[tuple[str, str, str], str]],
    kill_by_block: Mapping[str, Iterable[tuple[str, str, str]]],
) -> dict[str, dict[tuple[str, str, str], set[str]]]:
    in_defs: dict[str, dict[tuple[str, str, str], set[str]]] = {
        block_id: {} for block_id in block_ids
    }
    out_defs: dict[str, dict[tuple[str, str, str], set[str]]] = {
        block_id: {} for block_id in block_ids
    }
    changed = True
    while changed:
        changed = False
        for block_id in block_ids:
            pred_maps = [out_defs[pred] for pred in predecessors.get(block_id, [])]
            new_in = _merge_def_maps(pred_maps)
            if new_in != in_defs[block_id]:
                in_defs[block_id] = new_in
                changed = True
            new_out = _apply_gen_kill(
                new_in,
                gen_by_block.get(block_id, {}),
                kill_by_block.get(block_id, set()),
            )
            if new_out != out_defs[block_id]:
                out_defs[block_id] = new_out
                changed = True
    return in_defs


def _binding_edge_for_event(
    event: Mapping[str, object],
    *,
    scope_id: str,
    bindings_by_scope: Mapping[tuple[str, str, str], Mapping[str, object]],
    binding_meta: Mapping[str, Mapping[str, object]],
    resolution_map: Mapping[str, Mapping[str, object]],
) -> dict[str, object] | None:
    event_kind = _coerce_str(event.get("event_kind"))
    name = _coerce_str(event.get("name"))
    rel_path = _coerce_str(event.get("rel_path"))
    if event_kind not in {"DEF", "USE"} or name is None or rel_path is None:
        return None
    binding_info = _resolve_binding_for_event(
        rel_path=rel_path,
        scope_id=scope_id,
        name=name,
        bindings_by_scope=bindings_by_scope,
        resolution_map=resolution_map,
    )
    if binding_info is None:
        return None
    binding_kind = _coerce_str(binding_info.get("binding_kind"))
    resolved_id = _coerce_str(binding_info.get("resolved_binding_id"))
    if binding_kind is None or resolved_id is None:
        return None
    binding_meta_entry = binding_meta.get(resolved_id)
    if binding_meta_entry is None:
        return None
    edge_kind = "DEFINES_BINDING" if event_kind == "DEF" else "USES_BINDING"
    resolution_payload = binding_info.get("resolution")
    resolution = resolution_payload if isinstance(resolution_payload, Mapping) else None
    edge = _defuse_binding_edge_row(
        event=event,
        binding_meta=binding_meta_entry,
        edge_kind=edge_kind,
        binding_kind=binding_kind,
        resolution=resolution,
    )
    return edge or None


def _binding_payload_from_row(
    row: Mapping[str, object],
) -> tuple[tuple[str, str, str], str, dict[str, object]] | None:
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    rel_path = _coerce_str(row.get("rel_path"))
    binding_id = _coerce_str(row.get("binding_id"))
    scope_id = _coerce_str(row.get("scope_id"))
    name = _coerce_str(row.get("name"))
    binding_kind = _coerce_str(row.get("binding_kind"))
    if _has_missing(repo, commit, rel_path, binding_id, scope_id, name, binding_kind):
        return None
    rel_path_value = cast("str", rel_path)
    scope_id_value = cast("str", scope_id)
    name_value = cast("str", name)
    scope_key = (rel_path_value, scope_id_value, name_value)
    binding_id_value = cast("str", binding_id)
    payload: dict[str, object] = {
        "repo": cast("str", repo),
        "commit": cast("str", commit),
        "rel_path": rel_path_value,
        "binding_id": binding_id_value,
        "scope_id": scope_id_value,
        "name": name_value,
        "binding_kind": cast("str", binding_kind),
    }
    return scope_key, binding_id_value, payload


def _build_binding_index(
    bindings: list[dict[str, object]],
) -> tuple[dict[tuple[str, str, str], dict[str, object]], dict[str, dict[str, object]]]:
    by_scope_name: dict[tuple[str, str, str], dict[str, object]] = {}
    by_id: dict[str, dict[str, object]] = {}
    for row in bindings:
        parsed = _binding_payload_from_row(row)
        if parsed is None:
            continue
        scope_key, binding_id_value, payload = parsed
        by_scope_name[scope_key] = payload
        by_id[binding_id_value] = payload
    return by_scope_name, by_id


def _build_resolution_map(
    resolutions: list[dict[str, object]],
) -> dict[str, dict[str, object]]:
    by_src: dict[str, dict[str, object]] = {}
    for row in resolutions:
        src_binding_id = _coerce_str(row.get("src_binding_id"))
        dst_binding_id = _coerce_str(row.get("dst_binding_id"))
        kind = _coerce_str(row.get("kind"))
        if src_binding_id is None or dst_binding_id is None or kind is None:
            continue
        confidence = _coerce_float(row.get("confidence"))
        reason = _coerce_str(row.get("reason"))
        existing = by_src.get(src_binding_id)
        existing_conf = _coerce_float(existing.get("confidence")) if existing else None
        prefer = existing is None or (
            confidence is not None and (existing_conf is None or confidence > existing_conf)
        )
        if prefer:
            by_src[src_binding_id] = {
                "dst_binding_id": dst_binding_id,
                "kind": kind,
                "confidence": confidence,
                "reason": reason,
            }
    return by_src


def _resolve_binding_for_event(
    *,
    rel_path: str,
    scope_id: str,
    name: str,
    bindings_by_scope: Mapping[tuple[str, str, str], Mapping[str, object]],
    resolution_map: Mapping[str, Mapping[str, object]],
) -> dict[str, object] | None:
    binding = bindings_by_scope.get((rel_path, scope_id, name))
    if binding is None:
        return None
    binding_id = _coerce_str(binding.get("binding_id"))
    binding_kind = _coerce_str(binding.get("binding_kind"))
    if binding_id is None or binding_kind is None:
        return None
    resolved_id = binding_id
    resolution = resolution_map.get(binding_id)
    if resolution is not None:
        resolved_id = _coerce_str(resolution.get("dst_binding_id")) or resolved_id
    return {
        "binding_id": binding_id,
        "binding_kind": binding_kind,
        "resolved_binding_id": resolved_id,
        "resolution": resolution,
    }


def _build_code_unit_scope_map(
    code_units: list[dict[str, object]],
    scopes: list[dict[str, object]],
) -> dict[str, str]:
    scopes_by_path: dict[str, list[dict[str, object]]] = defaultdict(list)
    for scope in scopes:
        rel_path = _coerce_str(scope.get("rel_path"))
        scope_id = _coerce_str(scope.get("scope_id"))
        if rel_path is None or scope_id is None:
            continue
        scopes_by_path[rel_path].append(scope)
    mapping: dict[str, str] = {}
    for unit in code_units:
        code_unit_id = _coerce_str(unit.get("code_unit_id"))
        rel_path = _coerce_str(unit.get("rel_path"))
        if code_unit_id is None or rel_path is None:
            continue
        scopes_for_path = scopes_by_path.get(rel_path, [])
        scope_id = _select_scope_for_unit(
            scopes_for_path,
            unit_kind=_coerce_str(unit.get("kind")),
            unit_lineno=_coerce_int(unit.get("co_firstlineno")),
            unit_start=_coerce_int(unit.get("span_start_byte")),
            unit_end=_coerce_int(unit.get("span_end_byte")),
        )
        if scope_id is not None:
            mapping[code_unit_id] = scope_id
    return mapping


def _expected_scope_type(kind: str | None) -> str | None:
    if kind == "MODULE":
        return "MODULE"
    if kind == "CLASS":
        return "CLASS"
    if kind is None:
        return None
    return "FUNCTION"


def _scope_candidates(
    scopes: list[dict[str, object]],
    *,
    scope_type: str | None,
) -> list[dict[str, object]]:
    if scope_type is None:
        return scopes
    typed = [scope for scope in scopes if scope.get("scope_type") == scope_type]
    return typed if typed else scopes


def _span_length(start: int | None, end: int | None) -> int | None:
    if start is None or end is None:
        return None
    return max(end - start, 0)


def _span_contains(
    span_start: int | None,
    span_end: int | None,
    unit_start: int | None,
    unit_end: int | None,
) -> bool:
    if span_start is None or span_end is None or unit_start is None or unit_end is None:
        return False
    return span_start <= unit_start and span_end >= unit_end


def _select_scope_by_span(
    scopes: list[dict[str, object]],
    *,
    unit_start: int | None,
    unit_end: int | None,
) -> str | None:
    candidates: list[tuple[int, str]] = []
    for scope in scopes:
        scope_id = _coerce_str(scope.get("scope_id"))
        span_start = _coerce_int(scope.get("span_start_byte"))
        span_end = _coerce_int(scope.get("span_end_byte"))
        if scope_id is None or not _span_contains(span_start, span_end, unit_start, unit_end):
            continue
        span_len = _span_length(span_start, span_end)
        sort_key = span_len if span_len is not None else 2**63
        candidates.append((sort_key, scope_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _select_scope_by_lineno(
    scopes: list[dict[str, object]],
    *,
    lineno: int | None,
) -> str | None:
    if lineno is None:
        return None
    candidates: list[tuple[int, str]] = []
    for scope in scopes:
        scope_id = _coerce_str(scope.get("scope_id"))
        scope_line = _coerce_int(scope.get("lineno"))
        if scope_id is None or scope_line is None:
            continue
        candidates.append((abs(scope_line - lineno), scope_id))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _select_scope_for_unit(
    scopes: list[dict[str, object]],
    *,
    unit_kind: str | None,
    unit_lineno: int | None,
    unit_start: int | None,
    unit_end: int | None,
) -> str | None:
    target_type = _expected_scope_type(unit_kind)
    candidates = _scope_candidates(scopes, scope_type=target_type)
    scope_id = _select_scope_by_span(candidates, unit_start=unit_start, unit_end=unit_end)
    if scope_id is not None:
        return scope_id
    scope_id = _select_scope_by_lineno(candidates, lineno=unit_lineno)
    if scope_id is not None:
        return scope_id
    module_scopes = _scope_candidates(scopes, scope_type="MODULE")
    return _select_scope_by_lineno(module_scopes, lineno=unit_lineno)


def _event_var_key(
    *,
    binding_id: str | None,
    space: str | None,
    name: str | None,
) -> tuple[str, str, str]:
    binding = binding_id or ""
    space_value = space or ""
    name_value = name or ""
    return binding, space_value, name_value


def _defuse_binding_edge_row(
    *,
    event: Mapping[str, object],
    binding_meta: Mapping[str, object],
    edge_kind: str,
    binding_kind: str,
    resolution: Mapping[str, object] | None,
) -> dict[str, object]:
    repo = _coerce_str(event.get("repo")) or ""
    commit = _coerce_str(event.get("commit")) or ""
    rel_path = _coerce_str(event.get("rel_path")) or ""
    code_unit_id = _coerce_str(event.get("code_unit_id")) or ""
    instr_id = _coerce_str(event.get("instr_id")) or ""
    src_cpg_node_id = _instruction_cpg_id(
        repo=repo,
        commit=commit,
        rel_path=rel_path,
        code_unit_id=code_unit_id,
        instr_id=instr_id,
    )
    dst_cpg_node_id = _binding_cpg_id(binding_meta)
    if dst_cpg_node_id is None:
        return {}
    extras = {
        "space": event.get("space"),
        "name": event.get("name"),
        "binding_kind": binding_kind,
        "resolution_kind": _coerce_str(resolution.get("kind")) if resolution else None,
        "resolution_reason": _coerce_str(resolution.get("reason")) if resolution else None,
    }
    extras_kv = extras_kv_from_mapping(extras)
    ordinal = cpg_edge_ordinal(
        "graph.cpg_edges_defuse_binding",
        {
            "code_unit_id": code_unit_id,
            "instr_id": instr_id,
            "binding_id": binding_meta.get("binding_id"),
            "edge_kind": edge_kind,
        },
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": edge_kind,
        "edge_layer": "SYMBOL",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _reaches_edge_row(
    *,
    event: Mapping[str, object],
    def_instr_id: str,
) -> dict[str, object]:
    repo = _coerce_str(event.get("repo")) or ""
    commit = _coerce_str(event.get("commit")) or ""
    rel_path = _coerce_str(event.get("rel_path")) or ""
    code_unit_id = _coerce_str(event.get("code_unit_id")) or ""
    use_instr_id = _coerce_str(event.get("instr_id")) or ""
    src_cpg_node_id = _instruction_cpg_id(
        repo=repo,
        commit=commit,
        rel_path=rel_path,
        code_unit_id=code_unit_id,
        instr_id=def_instr_id,
    )
    dst_cpg_node_id = _instruction_cpg_id(
        repo=repo,
        commit=commit,
        rel_path=rel_path,
        code_unit_id=code_unit_id,
        instr_id=use_instr_id,
    )
    extras = {
        "space": event.get("space"),
        "name": event.get("name"),
        "binding_id": event.get("binding_id"),
    }
    extras_kv = extras_kv_from_mapping(extras)
    ordinal = cpg_edge_ordinal(
        "graph.cpg_edges_reaches",
        {
            "code_unit_id": code_unit_id,
            "def_instr_id": def_instr_id,
            "use_instr_id": use_instr_id,
            "var_key": str(event.get("var_key")),
        },
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": "REACHES",
        "edge_layer": "FLOW",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras": None,
        "extras_kv": extras_kv,
    }


def _instruction_cpg_id(
    *,
    repo: str,
    commit: str,
    rel_path: str,
    code_unit_id: str,
    instr_id: str,
) -> int:
    pk_values = {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "code_unit_id": code_unit_id,
        "instr_id": instr_id,
    }
    return cpg_node_id(PY_BC_INSTRUCTIONS_TABLE_KEY, pk_values)


def _binding_cpg_id(binding_meta: Mapping[str, object]) -> int | None:
    repo = _coerce_str(binding_meta.get("repo"))
    commit = _coerce_str(binding_meta.get("commit"))
    rel_path = _coerce_str(binding_meta.get("rel_path"))
    binding_id = _coerce_str(binding_meta.get("binding_id"))
    if repo is None or commit is None or rel_path is None or binding_id is None:
        return None
    pk_values = {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "binding_id": binding_id,
    }
    return cpg_node_id(PY_SYM_BINDINGS_TABLE_KEY, pk_values)


def _syntax_node_cpg_id(
    *,
    repo: str,
    commit: str,
    rel_path: str,
    producer: str,
    node_id: str,
) -> int:
    pk_values = {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "producer": producer,
        "node_id": node_id,
    }
    return cpg_node_id(SYNTAX_NODES_TABLE_KEY, pk_values)


def _ast_nodes_by_path(
    ast_rows: Sequence[dict[str, object]],
) -> dict[str, list[dict[str, object]]]:
    by_path: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ast_rows:
        rel_path = _coerce_str(row.get("path"))
        if rel_path is None:
            continue
        by_path[rel_path].append(row)
    return by_path


def _ast_anchor_candidates_by_span(
    nodes: list[dict[str, object]],
    *,
    instr_start: int | None,
    instr_end: int | None,
) -> list[tuple[int, dict[str, object]]]:
    if instr_start is None or instr_end is None:
        return []
    candidates: list[tuple[int, dict[str, object]]] = []
    for node in nodes:
        start_byte = _coerce_int(node.get("start_byte"))
        end_byte = _coerce_int(node.get("end_byte"))
        if not _span_contains(start_byte, end_byte, instr_start, instr_end):
            continue
        span_len = _span_length(start_byte, end_byte)
        sort_key = span_len if span_len is not None else 2**63
        candidates.append((sort_key, node))
    return candidates


def _ast_anchor_candidates_by_line(
    nodes: list[dict[str, object]],
    *,
    line_number: int | None,
) -> list[tuple[int, dict[str, object]]]:
    if line_number is None:
        return []
    candidates: list[tuple[int, dict[str, object]]] = []
    for node in nodes:
        start_line = _coerce_int(node.get("lineno"))
        end_line = _coerce_int(node.get("end_lineno")) or start_line
        if start_line is None or end_line is None:
            continue
        if start_line <= line_number <= end_line:
            span_len = _span_length(
                _coerce_int(node.get("start_byte")),
                _coerce_int(node.get("end_byte")),
            )
            sort_key = span_len if span_len is not None else 2**63
            candidates.append((sort_key, node))
    return candidates


def _select_ast_anchor(
    nodes: list[dict[str, object]],
    *,
    instr_start: int | None,
    instr_end: int | None,
    line_number: int | None,
) -> _AstAnchorMatch | None:
    candidates = _ast_anchor_candidates_by_span(nodes, instr_start=instr_start, instr_end=instr_end)
    match_kind = "SPAN_CONTAINS"
    if not candidates:
        candidates = _ast_anchor_candidates_by_line(nodes, line_number=line_number)
        match_kind = "LINE_CONTAINS"
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    node = candidates[0][1]
    node_hash = _coerce_str(node.get("hash"))
    if node_hash is None:
        return None
    return _AstAnchorMatch(
        node_hash=node_hash,
        node_type=_coerce_str(node.get("node_type")),
        match_kind=match_kind,
    )


def _record_diagnostics(
    diagnostics: dict[str, object] | None,
    key: str,
    *,
    expected_edges: int,
    produced_edges: int,
) -> None:
    if diagnostics is None:
        return
    dropped = max(expected_edges - produced_edges, 0)
    diagnostics[key] = OverlayEdgeDiagnostics(
        expected_edges=expected_edges,
        produced_edges=produced_edges,
        dropped_edges=dropped,
    )


def _collect_rows(frame: pa.Table, *, columns: Sequence[str]) -> list[dict[str, object]]:
    if not set(columns).issubset(frame.column_names):
        return []
    return [{column: row.get(column) for column in columns} for row in table_rows(frame)]


def _empty_edges() -> pa.Table:
    return empty_table_for_table(CPG_EDGES_TABLE_KEY)


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _has_missing(*values: object) -> bool:
    return any(value is None for value in values)


_MEMORY_EDGE_KIND_MAP = {
    ("attribute", "USE"): "READS_ATTR",
    ("attribute", "DEF"): "WRITES_ATTR",
    ("subscript", "USE"): "READS_SUBSCR",
    ("subscript", "DEF"): "WRITES_SUBSCR",
    ("global", "USE"): "READS_GLOBAL",
    ("global", "DEF"): "WRITES_GLOBAL",
}


__all__ = [
    "OverlayEdgeDiagnostics",
    "PyBcReachesInputs",
    "cpg2_edges__py_bc_callsite",
    "cpg2_edges__py_bc_callsite_symbol",
    "cpg2_edges__py_bc_cfg",
    "cpg2_edges__py_bc_defuse_binding",
    "cpg2_edges__py_bc_instruction_ast",
    "cpg2_edges__py_bc_memory",
    "cpg2_edges__py_bc_reaches",
    "cpg2_edges__py_bc_stack",
]
