"""Unified CPG node and edge assembly for property graph exports."""

from __future__ import annotations

import inspect
import opcode
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import TypedDict, cast

import pyarrow as pa
import pyarrow.compute as pc

from codeintel.build.graphs.assembly import (
    ensure_table_columns as _ensure_table_columns,
)
from codeintel.build.graphs.assembly import (
    rename_table_columns as _rename_table_columns,
)
from codeintel.build.graphs.assembly import (
    select_table_columns as _select_table_columns,
)
from codeintel.build.graphs.assembly import (
    stable_decimal_id as _stable_decimal_id,
)
from codeintel.build.graphs.assembly import (
    stable_int_hash as _stable_int_hash_core,
)
from codeintel.build.graphs.assembly import (
    table_rows as _table_rows,
)
from codeintel.build.graphs.assembly import (
    tabular_to_table,
)
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.options.graphs import CpgOptions
from codeintel.build.hamilton.native.options.ingestion import (
    BytecodeExtractOptions,
    InspectExtractOptions,
    SymtableExtractOptions,
)
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.arrow_ops import (
    ArrowJoinSpec,
    align_table_to_contract,
    arrow_join_tables,
    concat_tables_unified,
    dedupe_table_for_table,
)
from codeintel.build.tabular.compute_columns import (
    append_constant_columns as _append_constant_columns,
)
from codeintel.build.tabular.compute_columns import empty_table as _empty_table
from codeintel.build.tabular.frames import JoinStrategy, JoinValidation
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table
from codeintel.core.intervals.span_resolver import SpanResolver
from codeintel.core.schemas.row_models import columns_for_table_key
from codeintel.core.serialization.payload import decode_payload, encode_payload

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

OccurrenceSpanKey = tuple[object, object, object, object, object, object, object, object]

CPG_TARGET_NAME = "cpg"
CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"

SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
SYNTAX_CALLS_TABLE_KEY = "core.syntax_calls"
SYNTAX_CALL_ARGS_TABLE_KEY = "core.syntax_call_args"
SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbol_information"
GOIDS_TABLE_KEY = "core.goids"
CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
IMPORT_MODULES_TABLE_KEY = "graph.import_modules"
TS_TOKENS_TABLE_KEY = "core.ts_tokens"
TS_TRIVIA_TABLE_KEY = "core.ts_trivia"
AST_NODES_TABLE_KEY = "core.ast_nodes"
PY_SYM_SCOPES_TABLE_KEY = "core.py_sym_scopes"
PY_SYM_BINDINGS_TABLE_KEY = "core.py_sym_bindings"
PY_SYM_SCOPE_EDGES_TABLE_KEY = "core.py_sym_scope_edges"
PY_SYM_NAMESPACE_EDGES_TABLE_KEY = "core.py_sym_namespace_edges"
PY_SYM_RESOLUTION_EDGES_TABLE_KEY = "core.py_sym_resolution_edges"
PY_BC_CODE_UNITS_TABLE_KEY = "core.py_bc_code_units"
PY_BC_INSTRUCTIONS_TABLE_KEY = "core.py_bc_instructions"
PY_BC_BLOCKS_TABLE_KEY = "core.py_bc_blocks"
PY_BC_CFG_EDGES_TABLE_KEY = "core.py_bc_cfg_edges"
PY_BC_DEFUSE_EVENTS_TABLE_KEY = "core.py_bc_defuse_events"
PY_INSPECT_OBJECTS_TABLE_KEY = "core.py_inspect_objects"
PY_INSPECT_CLASS_MRO_TABLE_KEY = "core.py_inspect_class_mro"
PY_INSPECT_CLASS_ATTRS_TABLE_KEY = "core.py_inspect_class_attrs"
PY_INSPECT_UNWRAP_TABLE_KEY = "core.py_inspect_unwrap_hops"
PY_INSPECT_SIGNATURES_TABLE_KEY = "core.py_inspect_signatures"
PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY = "core.py_inspect_signature_params"
PY_INSPECT_SOURCE_TABLE_KEY = "core.py_inspect_source"
PY_INSPECT_RUNTIME_STATE_TABLE_KEY = "core.py_inspect_runtime_state"

ORDINAL_MOD = 2**31 - 1

_CPG_NODE_COLUMNS = columns_for_table_key(CPG_NODES_TABLE_KEY) or (
    "repo",
    "commit",
    "cpg_node_id",
    "node_kind",
    "source_table_key",
    "source_pk_json",
    "rel_path",
    "start_byte",
    "end_byte",
    "extras_json",
)

_CPG_EDGE_COLUMNS = columns_for_table_key(CPG_EDGES_TABLE_KEY) or (
    "repo",
    "commit",
    "src_cpg_node_id",
    "dst_cpg_node_id",
    "edge_kind",
    "edge_layer",
    "rel_path",
    "ordinal",
    "extras_json",
)


def cpg__options(env: BuildEnv) -> CpgOptions:
    """Load CPG options from the build environment.

    Returns
    -------
    CpgOptions
        Options controlling CPG assembly behavior.
    """
    return load_target_options(
        env,
        target_name=CPG_TARGET_NAME,
        options_type=CpgOptions,
    )


@dataclass(frozen=True, slots=True)
class CpgOverlayOptions:
    """Enablement flags for optional CPG overlays."""

    enable_symtable: bool
    enable_bytecode: bool
    enable_inspect: bool
    inspect_allowlist: tuple[str, ...]


def cpg__overlay_options(env: BuildEnv) -> CpgOverlayOptions:
    """Load overlay enablement flags from ingestion options.

    Returns
    -------
    CpgOverlayOptions
        Overlay gating options for CPG edge assembly.
    """
    symtable_options = load_target_options(
        env,
        target_name="symtable",
        options_type=SymtableExtractOptions,
    )
    bytecode_options = load_target_options(
        env,
        target_name="bytecode",
        options_type=BytecodeExtractOptions,
    )
    inspect_options = load_target_options(
        env,
        target_name="inspect",
        options_type=InspectExtractOptions,
    )
    allowlist = tuple(inspect_options.module_allowlist)
    enable_inspect = inspect_options.enable and bool(allowlist)
    return CpgOverlayOptions(
        enable_symtable=symtable_options.enable,
        enable_bytecode=bytecode_options.enable,
        enable_inspect=enable_inspect,
        inspect_allowlist=allowlist,
    )


@dataclass(frozen=True)
class _CpgSymbolInputs:
    syntax_edges: pa.Table
    occ_syntax: pa.Table
    occ_span: pa.Table
    symbol_rels: pa.Table
    symbol_goid: pa.Table


@dataclass(frozen=True)
class _CpgFlowInputs:
    goids: pa.Table
    cfg_edges: pa.Table
    dfg_edges: pa.Table
    cfg_blocks: pa.Table
    cdg_edges: pa.Table


@dataclass(frozen=True)
class _OccurrenceRolePayload:
    scip_roles: int | None
    is_definition: bool | None
    is_reference: bool | None
    is_import: bool | None
    is_write: bool | None
    is_read: bool | None


@dataclass(frozen=True)
class _CpgNodeCoreInputs:
    syntax_nodes: InferableTabularInput
    ast_nodes: InferableTabularInput
    scip_symbol_information: InferableTabularInput
    goids: InferableTabularInput
    py_sym_scopes: InferableTabularInput
    py_sym_bindings: InferableTabularInput
    py_bc_code_units: InferableTabularInput
    py_bc_instructions: InferableTabularInput
    py_bc_blocks: InferableTabularInput
    py_inspect_objects: InferableTabularInput
    py_inspect_signatures: InferableTabularInput
    py_inspect_signature_params: InferableTabularInput
    ts_tokens: InferableTabularInput
    ts_trivia: InferableTabularInput


@dataclass(frozen=True)
class _CpgNodeSyntaxInputs:
    syntax_nodes: InferableTabularInput
    ast_nodes: InferableTabularInput
    scip_symbol_information: InferableTabularInput
    goids: InferableTabularInput


@dataclass(frozen=True)
class _CpgNodePyInputs:
    py_sym_scopes: InferableTabularInput
    py_sym_bindings: InferableTabularInput
    py_bc_code_units: InferableTabularInput
    py_bc_instructions: InferableTabularInput
    py_bc_blocks: InferableTabularInput


@dataclass(frozen=True)
class _CpgNodeInspectInputs:
    py_inspect_objects: InferableTabularInput
    py_inspect_signatures: InferableTabularInput
    py_inspect_signature_params: InferableTabularInput
    ts_tokens: InferableTabularInput
    ts_trivia: InferableTabularInput


@dataclass(frozen=True)
class _CpgNodeGraphInputs:
    cfg_blocks: InferableTabularInput
    import_modules: InferableTabularInput


@dataclass(frozen=True)
class _CpgNodeInputs:
    core: _CpgNodeCoreInputs
    graph: _CpgNodeGraphInputs


@dataclass(frozen=True)
class _CpgNodeCoreLazyFrames:
    syntax_nodes: pa.Table
    ast_nodes: pa.Table
    scip_symbol_information: pa.Table
    goids: pa.Table
    py_sym_scopes: pa.Table
    py_sym_bindings: pa.Table
    py_bc_code_units: pa.Table
    py_bc_instructions: pa.Table
    py_bc_blocks: pa.Table
    py_inspect_objects: pa.Table
    py_inspect_signatures: pa.Table
    py_inspect_signature_params: pa.Table
    ts_tokens: pa.Table
    ts_trivia: pa.Table


@dataclass(frozen=True)
class _CpgNodeGraphLazyFrames:
    cfg_blocks: pa.Table
    import_modules: pa.Table


@dataclass(frozen=True)
class _CpgLinkInputs:
    call_edges: pa.Table
    import_edges: pa.Table


@dataclass(frozen=True)
class _CpgCallWiringInputs:
    call_edges: pa.Table
    arg_to_param_edges: pa.Table
    ret_to_call_edges: pa.Table


@dataclass(frozen=True)
class _CpgSyntaxNodeInputs:
    syntax_nodes: pa.Table


@dataclass(frozen=True)
class _CpgOverlayEdgeInputs:
    ast_nodes: pa.Table
    syntax_calls: pa.Table
    syntax_call_args: pa.Table
    scip_symbols: pa.Table
    py_sym_scopes: pa.Table
    py_sym_bindings: pa.Table
    py_sym_scope_edges: pa.Table
    py_sym_namespace_edges: pa.Table
    py_sym_resolution_edges: pa.Table
    py_bc_code_units: pa.Table
    py_bc_instructions: pa.Table
    py_bc_blocks: pa.Table
    py_bc_cfg_edges: pa.Table
    py_bc_defuse_events: pa.Table
    py_inspect_objects: pa.Table
    py_inspect_class_mro: pa.Table
    py_inspect_class_attrs: pa.Table
    py_inspect_unwrap_hops: pa.Table
    py_inspect_signatures: pa.Table
    py_inspect_signature_params: pa.Table
    py_inspect_source: pa.Table
    py_inspect_runtime_state: pa.Table


@dataclass(frozen=True)
class _CpgOverlayScopeInputs:
    py_sym_scopes: pa.Table
    py_sym_bindings: pa.Table
    py_sym_scope_edges: pa.Table
    py_sym_namespace_edges: pa.Table
    py_sym_resolution_edges: pa.Table


@dataclass(frozen=True)
class _CpgOverlaySymbolInputs:
    ast_nodes: pa.Table
    scip_symbols: pa.Table
    scope_inputs: _CpgOverlayScopeInputs


@dataclass(frozen=True)
class _CpgOverlayBytecodeInputs:
    py_bc_code_units: pa.Table
    py_bc_instructions: pa.Table
    py_bc_blocks: pa.Table
    py_bc_cfg_edges: pa.Table
    py_bc_defuse_events: pa.Table


@dataclass(frozen=True)
class _CpgOverlaySyntaxCallInputs:
    syntax_calls: pa.Table
    syntax_call_args: pa.Table


@dataclass(frozen=True)
class _CpgOverlayInspectCoreInputs:
    py_inspect_objects: pa.Table
    py_inspect_class_mro: pa.Table
    py_inspect_class_attrs: pa.Table
    py_inspect_unwrap_hops: pa.Table
    py_inspect_source: pa.Table


@dataclass(frozen=True)
class _CpgOverlayInspectRuntimeInputs:
    py_inspect_signatures: pa.Table
    py_inspect_signature_params: pa.Table
    py_inspect_runtime_state: pa.Table


@dataclass(frozen=True)
class _CpgOverlayInspectInputs:
    py_inspect_objects: pa.Table
    py_inspect_class_mro: pa.Table
    py_inspect_class_attrs: pa.Table
    py_inspect_unwrap_hops: pa.Table
    py_inspect_signatures: pa.Table
    py_inspect_signature_params: pa.Table
    py_inspect_source: pa.Table
    py_inspect_runtime_state: pa.Table
    syntax_calls: pa.Table
    syntax_call_args: pa.Table


@dataclass(frozen=True)
class _CpgEdgeCoreInputs:
    symbol: _CpgSymbolInputs
    flow: _CpgFlowInputs
    link: _CpgLinkInputs
    call_wiring: _CpgCallWiringInputs
    syntax_nodes: _CpgSyntaxNodeInputs


@dataclass(frozen=True, slots=True)
class _CpgOverlayRegistryEntry:
    name: str
    enabled: bool
    builder: Callable[[], Sequence[pa.Table]]


def _stable_int_hash(
    payload: object,
    *,
    digest_size: int,
    modulus: int,
) -> int:
    return _stable_int_hash_core(payload, digest_size=digest_size, modulus=modulus)


def _stable_cpg_id(table_key: str, pk: Mapping[str, object]) -> int:
    payload = {"table_key": table_key, "pk": dict(pk)}
    return _stable_decimal_id(payload, digest_size=16)


def _stable_ordinal(table_key: str, payload: Mapping[str, object]) -> int:
    wrapped = {"table_key": table_key, "payload": dict(payload)}
    return _stable_int_hash(wrapped, digest_size=8, modulus=ORDINAL_MOD)


def _stable_cpg_id_from_row(table_key: str, row: Mapping[str, object]) -> int:
    return _stable_cpg_id(table_key, row)


def _stable_ordinal_from_row(table_key: str, row: Mapping[str, object]) -> int:
    return _stable_ordinal(table_key, row)


def _row_to_payload(row: Mapping[str, object]) -> bytes:
    payload = dict(row)
    encoded = encode_payload(payload)
    if encoded is None:
        msg = "Expected payload encoding to return bytes"
        raise ValueError(msg)
    return encoded


def _encode_optional_payload(value: object) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, (bytes, bytearray, memoryview)):
        return encode_payload(value)
    if isinstance(value, Mapping):
        return encode_payload(dict(value))
    if isinstance(value, (str, int, float, bool)):
        return encode_payload(value)
    return None


_CPG_DECIMAL_TYPE = pa.decimal128(38, 0)
_CPG_NODE_DECIMAL_COLUMNS = frozenset({"cpg_node_id"})
_CPG_EDGE_DECIMAL_COLUMNS = frozenset({"src_cpg_node_id", "dst_cpg_node_id"})


def _rows_to_table(
    rows: Sequence[Mapping[str, object]],
    *,
    columns: Sequence[str],
    decimal_columns: frozenset[str],
) -> pa.Table:
    if not rows:
        return _empty_table(columns)
    arrays: list[pa.Array] = []
    for column in columns:
        values = [row.get(column) for row in rows]
        if column in decimal_columns:
            arrays.append(pa.array(values, type=_CPG_DECIMAL_TYPE))
        else:
            arrays.append(pa.array(values))
    return pa.Table.from_arrays(arrays, names=list(columns))


def _node_rows_to_table(rows: Sequence[Mapping[str, object]]) -> pa.Table:
    return _rows_to_table(
        rows,
        columns=_CPG_NODE_COLUMNS,
        decimal_columns=_CPG_NODE_DECIMAL_COLUMNS,
    )


def _edge_rows_to_table(rows: Sequence[Mapping[str, object]]) -> pa.Table:
    return _rows_to_table(
        rows,
        columns=_CPG_EDGE_COLUMNS,
        decimal_columns=_CPG_EDGE_DECIMAL_COLUMNS,
    )


def _filter_valid_values(table: pa.Table, columns: Sequence[str]) -> pa.Table:
    if table.num_rows == 0 or not columns:
        return table
    names = set(table.column_names)
    if not all(column in names for column in columns):
        return table
    try:
        mask = pc.call_function("is_valid", [table.column(columns[0])])
        for column in columns[1:]:
            col_mask = pc.call_function("is_valid", [table.column(column)])
            mask = pc.call_function("and_kleene", [mask, col_mask])
        return table.filter(mask)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, pa.ArrowTypeError, TypeError, ValueError):
        return table


def _pk_from_row(table_key: str, values: Mapping[str, object]) -> int:
    return _stable_cpg_id_from_row(table_key, values)


def _ordinal_from_row(table_key: str, values: Mapping[str, object]) -> int:
    return _stable_ordinal_from_row(table_key, values)


def _pk_json_from_row(values: Mapping[str, object]) -> bytes:
    return _row_to_payload(values)


def _payload_json_from_row(values: Mapping[str, object]) -> bytes:
    return _row_to_payload(values)


@dataclass(frozen=True, slots=True)
class JoinSpec:
    on: Sequence[str] | None = None
    left_on: Sequence[str] | None = None
    right_on: Sequence[str] | None = None
    how: JoinStrategy = "inner"
    validate: JoinValidation | None = None
    suffix: str = ""


def _select_node_columns(table: pa.Table) -> pa.Table:
    return _ensure_table_columns(table, _CPG_NODE_COLUMNS)


def _select_edge_columns(table: pa.Table) -> pa.Table:
    return _ensure_table_columns(table, _CPG_EDGE_COLUMNS)


def _syntax_node_keys(syntax_nodes: pa.Table) -> pa.Table:
    return _select_table_columns(
        syntax_nodes,
        ["repo", "commit", "rel_path", "producer", "node_id"],
    )


def _syntax_nodes_to_cpg(syntax_nodes: pa.Table) -> pa.Table:
    rows: list[dict[str, object]] = []
    for row in _table_rows(syntax_nodes):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "producer": row.get("producer"),
            "node_id": row.get("node_id"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(SYNTAX_NODES_TABLE_KEY, pk_values),
                "node_kind": "SYNTAX_NODE",
                "source_table_key": SYNTAX_NODES_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("start_byte"),
                "end_byte": row.get("end_byte"),
                "extras_json": _encode_optional_payload(row.get("extras_json")),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _scip_symbols_to_cpg(symbols: pa.Table) -> pa.Table:
    rows: list[dict[str, object]] = []
    for row in _table_rows(symbols):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "symbol": row.get("symbol"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(SCIP_SYMBOLS_TABLE_KEY, pk_values),
                "node_kind": "SCIP_SYMBOL",
                "source_table_key": SCIP_SYMBOLS_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": None,
                "start_byte": None,
                "end_byte": None,
                "extras_json": None,
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _goids_to_cpg(goids: pa.Table) -> pa.Table:
    rows: list[dict[str, object]] = []
    for row in _table_rows(goids):
        pk_values = {"goid_h128": row.get("goid_h128")}
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(GOIDS_TABLE_KEY, pk_values),
                "node_kind": "GOID",
                "source_table_key": GOIDS_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": None,
                "end_byte": None,
                "extras_json": None,
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _cfg_block_index(
    cfg_blocks: pa.Table,
    goids: pa.Table,
) -> dict[tuple[object, object], dict[str, object]]:
    goid_context: dict[object, dict[str, object]] = {}
    filtered_goids = _filter_valid_values(goids, ["goid_h128"])
    for row in _table_rows(filtered_goids):
        goid = row.get("goid_h128")
        if goid is None:
            continue
        goid_context[goid] = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
        }
    block_index: dict[tuple[object, object], dict[str, object]] = {}
    filtered_blocks = _filter_valid_values(cfg_blocks, ["function_goid_h128", "block_id"])
    for row in _table_rows(filtered_blocks):
        function_goid = row.get("function_goid_h128")
        block_id = row.get("block_id")
        if function_goid is None or block_id is None:
            continue
        context = goid_context.get(function_goid, {})
        block_index[function_goid, block_id] = {
            "block_idx": row.get("block_idx"),
            "rel_path": row.get("file_path"),
            "repo": context.get("repo"),
            "commit": context.get("commit"),
        }
    return block_index


def _block_id_index(cfg_blocks: pa.Table) -> dict[object, dict[str, object]]:
    index: dict[object, dict[str, object]] = {}
    filtered_blocks = _filter_valid_values(cfg_blocks, ["block_id"])
    for row in _table_rows(filtered_blocks):
        block_id = row.get("block_id")
        if block_id is None:
            continue
        index[block_id] = {
            "function_goid_h128": row.get("function_goid_h128"),
            "block_idx": row.get("block_idx"),
        }
    return index


def _syntax_node_index(
    syntax_nodes: pa.Table,
) -> dict[tuple[object, object, object], dict[str, object]]:
    index: dict[tuple[object, object, object], dict[str, object]] = {}
    for row in _table_rows(syntax_nodes):
        key = (row.get("repo"), row.get("commit"), row.get("node_id"))
        index[key] = {
            "rel_path": row.get("rel_path"),
            "producer": row.get("producer"),
        }
    return index


def _cfg_blocks_to_cpg(cfg_blocks: pa.Table, goids: pa.Table) -> pa.Table:
    goid_ctx = _select_table_columns(goids, ["goid_h128", "repo", "commit"])
    goid_ctx = _rename_table_columns(goid_ctx, {"goid_h128": "function_goid_h128"})
    joined_table = arrow_join_tables(
        cfg_blocks,
        goid_ctx,
        spec=ArrowJoinSpec(on=["function_goid_h128"], how="left"),
    )
    rows: list[dict[str, object]] = []
    for row in _table_rows(joined_table):
        pk_values = {
            "function_goid_h128": row.get("function_goid_h128"),
            "block_idx": row.get("block_idx"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(CFG_BLOCKS_TABLE_KEY, pk_values),
                "node_kind": "CFG_BLOCK",
                "source_table_key": CFG_BLOCKS_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": row.get("file_path"),
                "start_byte": None,
                "end_byte": None,
                "extras_json": None,
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _import_modules_to_cpg(import_modules: pa.Table) -> pa.Table:
    rows: list[dict[str, object]] = []
    for row in _table_rows(import_modules):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "module": row.get("module"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(IMPORT_MODULES_TABLE_KEY, pk_values),
                "node_kind": "MODULE",
                "source_table_key": IMPORT_MODULES_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": None,
                "start_byte": None,
                "end_byte": None,
                "extras_json": None,
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _ts_tokens_to_cpg(tokens: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "rel_path",
        "language",
        "token_id",
        "token_kind",
        "node_type",
        "start_byte",
        "end_byte",
        "text_preview",
        "extras_json",
    }
    if not required.issubset(set(tokens.column_names)):
        return _empty_table(_CPG_NODE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(tokens):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "language": row.get("language"),
            "token_id": row.get("token_id"),
        }
        extras_values = {
            "token_kind": row.get("token_kind"),
            "node_type": row.get("node_type"),
            "text_preview": row.get("text_preview"),
            "token_extras": row.get("extras_json"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(TS_TOKENS_TABLE_KEY, pk_values),
                "node_kind": "TS_TOKEN",
                "source_table_key": TS_TOKENS_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("start_byte"),
                "end_byte": row.get("end_byte"),
                "extras_json": _payload_json_from_row(extras_values),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _ts_trivia_to_cpg(trivia: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "rel_path",
        "language",
        "trivia_id",
        "trivia_kind",
        "node_type",
        "start_byte",
        "end_byte",
        "text_preview",
        "extras_json",
    }
    if not required.issubset(set(trivia.column_names)):
        return _empty_table(_CPG_NODE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(trivia):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "language": row.get("language"),
            "trivia_id": row.get("trivia_id"),
        }
        extras_values = {
            "trivia_kind": row.get("trivia_kind"),
            "node_type": row.get("node_type"),
            "text_preview": row.get("text_preview"),
            "trivia_extras": row.get("extras_json"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(TS_TRIVIA_TABLE_KEY, pk_values),
                "node_kind": "TS_TRIVIA",
                "source_table_key": TS_TRIVIA_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("start_byte"),
                "end_byte": row.get("end_byte"),
                "extras_json": _payload_json_from_row(extras_values),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _ast_nodes_to_cpg(ast_nodes: pa.Table, env: BuildEnv) -> pa.Table:
    required = {
        "path",
        "node_type",
        "hash",
        "start_byte",
        "end_byte",
    }
    if not required.issubset(set(ast_nodes.column_names)):
        return _empty_table(_CPG_NODE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(ast_nodes):
        pk_values = {"hash": row.get("hash")}
        extras_values = {
            "node_type": row.get("node_type"),
            "name": row.get("name"),
            "qualname": row.get("qualname"),
            "parent_qualname": row.get("parent_qualname"),
            "lineno": row.get("lineno"),
            "end_lineno": row.get("end_lineno"),
            "col_offset": row.get("col_offset"),
            "end_col_offset": row.get("end_col_offset"),
            "decorator_start_line": row.get("decorator_start_line"),
            "decorator_end_line": row.get("decorator_end_line"),
            "decorators": row.get("decorators"),
            "docstring": row.get("docstring"),
            "ctx": row.get("ctx"),
            "type_comment": row.get("type_comment"),
            "type_ignores": row.get("type_ignores"),
            "identifier": row.get("identifier"),
            "attribute": row.get("attribute"),
            "imported": row.get("imported"),
            "asname": row.get("asname"),
            "module": row.get("module"),
            "level": row.get("level"),
            "constant_kind": row.get("constant_kind"),
        }
        rows.append(
            {
                "repo": env.repo,
                "commit": env.commit,
                "cpg_node_id": _pk_from_row(AST_NODES_TABLE_KEY, pk_values),
                "node_kind": "AST_NODE",
                "source_table_key": AST_NODES_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": row.get("path"),
                "start_byte": row.get("start_byte"),
                "end_byte": row.get("end_byte"),
                "extras_json": _payload_json_from_row(extras_values),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _py_sym_scopes_to_cpg(scopes: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "rel_path",
        "scope_id",
        "scope_type",
    }
    if not required.issubset(set(scopes.column_names)):
        return _empty_table(_CPG_NODE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(scopes):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "scope_id": row.get("scope_id"),
        }
        extras_values = {
            "scope_type": row.get("scope_type"),
            "scope_name": row.get("scope_name"),
            "qualpath": row.get("qualpath"),
            "lineno": row.get("lineno"),
            "is_nested": row.get("is_nested"),
            "is_optimized": row.get("is_optimized"),
            "has_children": row.get("has_children"),
            "parent_scope_id": row.get("parent_scope_id"),
            "anchor_ast_node_id": row.get("anchor_ast_node_id"),
            "anchor_confidence": row.get("anchor_confidence"),
            "anchor_reason": row.get("anchor_reason"),
            "scope_local_id": row.get("scope_local_id"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(PY_SYM_SCOPES_TABLE_KEY, pk_values),
                "node_kind": "SCOPE",
                "source_table_key": PY_SYM_SCOPES_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("span_start_byte"),
                "end_byte": row.get("span_end_byte"),
                "extras_json": _payload_json_from_row(extras_values),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _py_sym_bindings_to_cpg(bindings: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "rel_path",
        "binding_id",
        "scope_id",
        "name",
        "binding_kind",
    }
    if not required.issubset(set(bindings.column_names)):
        return _empty_table(_CPG_NODE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(bindings):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "binding_id": row.get("binding_id"),
        }
        extras_values = {
            "scope_id": row.get("scope_id"),
            "name": row.get("name"),
            "binding_kind": row.get("binding_kind"),
            "declared_here": row.get("declared_here"),
            "referenced_here": row.get("referenced_here"),
            "assigned_here": row.get("assigned_here"),
            "annotated_here": row.get("annotated_here"),
            "scoping_class": row.get("scoping_class"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(PY_SYM_BINDINGS_TABLE_KEY, pk_values),
                "node_kind": "BINDING",
                "source_table_key": PY_SYM_BINDINGS_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": None,
                "end_byte": None,
                "extras_json": _payload_json_from_row(extras_values),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _py_bc_code_units_to_cpg(code_units: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "rel_path",
        "code_unit_id",
    }
    if not required.issubset(set(code_units.column_names)):
        return _empty_table(_CPG_NODE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(code_units):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "code_unit_id": row.get("code_unit_id"),
        }
        extras_values = {
            "qualpath": row.get("qualpath"),
            "co_name": row.get("co_name"),
            "co_qualname": row.get("co_qualname"),
            "kind": row.get("kind"),
            "co_firstlineno": row.get("co_firstlineno"),
            "flags": row.get("flags"),
            "argcount": row.get("argcount"),
            "posonlyargcount": row.get("posonlyargcount"),
            "kwonlyargcount": row.get("kwonlyargcount"),
            "nlocals": row.get("nlocals"),
            "stacksize": row.get("stacksize"),
            "varnames": row.get("varnames"),
            "names": row.get("names"),
            "freevars": row.get("freevars"),
            "cellvars": row.get("cellvars"),
            "bytecode_len": row.get("bytecode_len"),
            "exceptiontable_len": row.get("exceptiontable_len"),
            "python_version": row.get("python_version"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(PY_BC_CODE_UNITS_TABLE_KEY, pk_values),
                "node_kind": "BC_CODE_UNIT",
                "source_table_key": PY_BC_CODE_UNITS_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("span_start_byte"),
                "end_byte": row.get("span_end_byte"),
                "extras_json": _payload_json_from_row(extras_values),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _py_bc_instructions_to_cpg(instructions: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "rel_path",
        "code_unit_id",
        "instr_id",
    }
    if not required.issubset(set(instructions.column_names)):
        return _empty_table(_CPG_NODE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(instructions):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "code_unit_id": row.get("code_unit_id"),
            "instr_id": row.get("instr_id"),
        }
        extras_values = {
            "instr_index": row.get("instr_index"),
            "start_offset": row.get("start_offset"),
            "offset": row.get("offset"),
            "end_offset": row.get("end_offset"),
            "opcode": row.get("opcode"),
            "opname": row.get("opname"),
            "baseopname": row.get("baseopname"),
            "arg": row.get("arg"),
            "argrepr": row.get("argrepr"),
            "argval_kind": row.get("argval_kind"),
            "argval_str": row.get("argval_str"),
            "argval_int": row.get("argval_int"),
            "argval_repr": row.get("argval_repr"),
            "is_jump_target": row.get("is_jump_target"),
            "jump_target_offset": row.get("jump_target_offset"),
            "jump_target_label": row.get("jump_target_label"),
            "label": row.get("label"),
            "starts_line": row.get("starts_line"),
            "line_number": row.get("line_number"),
            "pos": row.get("pos"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(PY_BC_INSTRUCTIONS_TABLE_KEY, pk_values),
                "node_kind": "BC_INSTR",
                "source_table_key": PY_BC_INSTRUCTIONS_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("span_start_byte"),
                "end_byte": row.get("span_end_byte"),
                "extras_json": _payload_json_from_row(extras_values),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _py_bc_blocks_to_cpg(blocks: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "rel_path",
        "block_id",
        "code_unit_id",
    }
    if not required.issubset(set(blocks.column_names)):
        return _empty_table(_CPG_NODE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(blocks):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "block_id": row.get("block_id"),
        }
        extras_values = {
            "code_unit_id": row.get("code_unit_id"),
            "start_offset": row.get("start_offset"),
            "end_offset": row.get("end_offset"),
            "start_label": row.get("start_label"),
            "kind": row.get("kind"),
            "first_instr_index": row.get("first_instr_index"),
            "last_instr_index": row.get("last_instr_index"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(PY_BC_BLOCKS_TABLE_KEY, pk_values),
                "node_kind": "BC_BLOCK",
                "source_table_key": PY_BC_BLOCKS_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": row.get("rel_path"),
                "start_byte": row.get("anchor_span_start_byte"),
                "end_byte": row.get("anchor_span_end_byte"),
                "extras_json": _payload_json_from_row(extras_values),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _py_inspect_objects_to_cpg(objects: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "object_id",
        "kind",
    }
    if not required.issubset(set(objects.column_names)):
        return _empty_table(_CPG_NODE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(objects):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "object_id": row.get("object_id"),
        }
        extras_values = {
            "kind": row.get("kind"),
            "module_name": row.get("module_name"),
            "qualname": row.get("qualname"),
            "name": row.get("name"),
            "type_qualname": row.get("type_qualname"),
            "object_addr": row.get("object_addr"),
            "is_builtin": row.get("is_builtin"),
            "is_callable": row.get("is_callable"),
            "is_descriptor": row.get("is_descriptor"),
            "has_wrapped": row.get("has_wrapped"),
            "has_signature_override": row.get("has_signature_override"),
            "has_annotations": row.get("has_annotations"),
            "status": row.get("status"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(PY_INSPECT_OBJECTS_TABLE_KEY, pk_values),
                "node_kind": "INSPECT_OBJECT",
                "source_table_key": PY_INSPECT_OBJECTS_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": None,
                "start_byte": None,
                "end_byte": None,
                "extras_json": _payload_json_from_row(extras_values),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _py_inspect_signatures_to_cpg(signatures: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "signature_id",
        "object_id",
    }
    if not required.issubset(set(signatures.column_names)):
        return _empty_table(_CPG_NODE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(signatures):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "signature_id": row.get("signature_id"),
        }
        extras_values = {
            "object_id": row.get("object_id"),
            "mode": row.get("mode"),
            "variant": row.get("variant"),
            "follow_wrapped": row.get("follow_wrapped"),
            "eval_str": row.get("eval_str"),
            "effective_object_id": row.get("effective_object_id"),
            "sig_text": row.get("sig_text"),
            "sig_format": row.get("sig_format"),
            "has_varargs": row.get("has_varargs"),
            "has_varkw": row.get("has_varkw"),
            "status": row.get("status"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(PY_INSPECT_SIGNATURES_TABLE_KEY, pk_values),
                "node_kind": "INSPECT_SIGNATURE",
                "source_table_key": PY_INSPECT_SIGNATURES_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": None,
                "start_byte": None,
                "end_byte": None,
                "extras_json": _payload_json_from_row(extras_values),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def _py_inspect_signature_params_to_cpg(params: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "signature_id",
        "param_index",
    }
    if not required.issubset(set(params.column_names)):
        return _empty_table(_CPG_NODE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(params):
        pk_values = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "signature_id": row.get("signature_id"),
            "param_index": row.get("param_index"),
        }
        extras_values = {
            "mode": row.get("mode"),
            "name": row.get("name"),
            "kind": row.get("kind"),
            "default_present": row.get("default_present"),
            "default_value": row.get("default_value"),
            "annotation_present": row.get("annotation_present"),
            "annotation_value": row.get("annotation_value"),
            "status": row.get("status"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "cpg_node_id": _pk_from_row(PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY, pk_values),
                "node_kind": "INSPECT_SIGNATURE_PARAM",
                "source_table_key": PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY,
                "source_pk_json": _pk_json_from_row(pk_values),
                "rel_path": None,
                "start_byte": None,
                "end_byte": None,
                "extras_json": _payload_json_from_row(extras_values),
            }
        )
    table = _node_rows_to_table(rows)
    return _select_node_columns(table)


def cpg_nodes__syntax_inputs(
    q__core__syntax_nodes: InferableTabularInput,
    q__core__ast_nodes: InferableTabularInput,
    q__core__scip_symbol_information: InferableTabularInput,
    q__core__goids: InferableTabularInput,
) -> _CpgNodeSyntaxInputs:
    """Bundle syntax-driven inputs for CPG node assembly.

    Returns
    -------
    _CpgNodeSyntaxInputs
        Core syntax inputs for CPG node assembly.
    """
    return _CpgNodeSyntaxInputs(
        syntax_nodes=q__core__syntax_nodes,
        ast_nodes=q__core__ast_nodes,
        scip_symbol_information=q__core__scip_symbol_information,
        goids=q__core__goids,
    )


def cpg_nodes__py_inputs(
    q__core__py_sym_scopes: InferableTabularInput,
    q__core__py_sym_bindings: InferableTabularInput,
    q__core__py_bc_code_units: InferableTabularInput,
    q__core__py_bc_instructions: InferableTabularInput,
    q__core__py_bc_blocks: InferableTabularInput,
) -> _CpgNodePyInputs:
    """Bundle Python symbol and bytecode inputs for CPG node assembly.

    Returns
    -------
    _CpgNodePyInputs
        Python symbol and bytecode inputs for CPG node assembly.
    """
    return _CpgNodePyInputs(
        py_sym_scopes=q__core__py_sym_scopes,
        py_sym_bindings=q__core__py_sym_bindings,
        py_bc_code_units=q__core__py_bc_code_units,
        py_bc_instructions=q__core__py_bc_instructions,
        py_bc_blocks=q__core__py_bc_blocks,
    )


def cpg_nodes__inspect_inputs(
    q__core__py_inspect_objects: InferableTabularInput,
    q__core__py_inspect_signatures: InferableTabularInput,
    q__core__py_inspect_signature_params: InferableTabularInput,
    q__core__ts_tokens: InferableTabularInput,
    q__core__ts_trivia: InferableTabularInput,
) -> _CpgNodeInspectInputs:
    """Bundle inspect + tree-sitter inputs for CPG node assembly.

    Returns
    -------
    _CpgNodeInspectInputs
        Inspect and token inputs for CPG node assembly.
    """
    return _CpgNodeInspectInputs(
        py_inspect_objects=q__core__py_inspect_objects,
        py_inspect_signatures=q__core__py_inspect_signatures,
        py_inspect_signature_params=q__core__py_inspect_signature_params,
        ts_tokens=q__core__ts_tokens,
        ts_trivia=q__core__ts_trivia,
    )


def cpg_nodes__core_inputs(
    cpg_nodes__syntax_inputs: _CpgNodeSyntaxInputs,
    cpg_nodes__py_inputs: _CpgNodePyInputs,
    cpg_nodes__inspect_inputs: _CpgNodeInspectInputs,
) -> _CpgNodeCoreInputs:
    """Bundle core tables for CPG node assembly.

    Returns
    -------
    _CpgNodeCoreInputs
        Core inputs for CPG node assembly.
    """
    return _CpgNodeCoreInputs(
        syntax_nodes=cpg_nodes__syntax_inputs.syntax_nodes,
        ast_nodes=cpg_nodes__syntax_inputs.ast_nodes,
        scip_symbol_information=cpg_nodes__syntax_inputs.scip_symbol_information,
        goids=cpg_nodes__syntax_inputs.goids,
        py_sym_scopes=cpg_nodes__py_inputs.py_sym_scopes,
        py_sym_bindings=cpg_nodes__py_inputs.py_sym_bindings,
        py_bc_code_units=cpg_nodes__py_inputs.py_bc_code_units,
        py_bc_instructions=cpg_nodes__py_inputs.py_bc_instructions,
        py_bc_blocks=cpg_nodes__py_inputs.py_bc_blocks,
        py_inspect_objects=cpg_nodes__inspect_inputs.py_inspect_objects,
        py_inspect_signatures=cpg_nodes__inspect_inputs.py_inspect_signatures,
        py_inspect_signature_params=cpg_nodes__inspect_inputs.py_inspect_signature_params,
        ts_tokens=cpg_nodes__inspect_inputs.ts_tokens,
        ts_trivia=cpg_nodes__inspect_inputs.ts_trivia,
    )


def cpg_nodes__graph_inputs(
    q__graph__cfg_blocks: InferableTabularInput,
    q__graph__import_modules: InferableTabularInput,
) -> _CpgNodeGraphInputs:
    """Bundle graph tables for CPG node assembly.

    Returns
    -------
    _CpgNodeGraphInputs
        Graph inputs for CPG node assembly.
    """
    return _CpgNodeGraphInputs(
        cfg_blocks=q__graph__cfg_blocks,
        import_modules=q__graph__import_modules,
    )


def cpg_nodes__inputs(
    cpg_nodes__core_inputs: _CpgNodeCoreInputs,
    cpg_nodes__graph_inputs: _CpgNodeGraphInputs,
) -> _CpgNodeInputs:
    """Bundle inputs for CPG node assembly.

    Returns
    -------
    _CpgNodeInputs
        Combined inputs for CPG node assembly.
    """
    return _CpgNodeInputs(
        core=cpg_nodes__core_inputs,
        graph=cpg_nodes__graph_inputs,
    )


def _core_lazyframes(core_inputs: _CpgNodeCoreInputs) -> _CpgNodeCoreLazyFrames:
    return _CpgNodeCoreLazyFrames(
        syntax_nodes=tabular_to_table(core_inputs.syntax_nodes),
        ast_nodes=tabular_to_table(core_inputs.ast_nodes),
        scip_symbol_information=tabular_to_table(core_inputs.scip_symbol_information),
        goids=tabular_to_table(core_inputs.goids),
        py_sym_scopes=tabular_to_table(core_inputs.py_sym_scopes),
        py_sym_bindings=tabular_to_table(core_inputs.py_sym_bindings),
        py_bc_code_units=tabular_to_table(core_inputs.py_bc_code_units),
        py_bc_instructions=tabular_to_table(core_inputs.py_bc_instructions),
        py_bc_blocks=tabular_to_table(core_inputs.py_bc_blocks),
        py_inspect_objects=tabular_to_table(core_inputs.py_inspect_objects),
        py_inspect_signatures=tabular_to_table(core_inputs.py_inspect_signatures),
        py_inspect_signature_params=tabular_to_table(core_inputs.py_inspect_signature_params),
        ts_tokens=tabular_to_table(core_inputs.ts_tokens),
        ts_trivia=tabular_to_table(core_inputs.ts_trivia),
    )


def _graph_lazyframes(graph_inputs: _CpgNodeGraphInputs) -> _CpgNodeGraphLazyFrames:
    return _CpgNodeGraphLazyFrames(
        cfg_blocks=tabular_to_table(graph_inputs.cfg_blocks),
        import_modules=tabular_to_table(graph_inputs.import_modules),
    )


def _frame_to_reader(table_key: str, frame: pa.Table) -> pa.Table:
    return align_table_to_contract(table_key, frame, extras_policy=None)


def _arrow_join_frames(
    left: pa.Table,
    right: pa.Table,
    *,
    spec: JoinSpec | ArrowJoinSpec,
) -> pa.Table:
    if isinstance(spec, ArrowJoinSpec):
        join_spec = spec
    else:
        join_spec = ArrowJoinSpec(
            on=spec.on,
            left_on=spec.left_on,
            right_on=spec.right_on,
            how=spec.how,
            validate=spec.validate,
            suffix=spec.suffix,
        )
    return arrow_join_tables(left, right, spec=join_spec)


def cpg_nodes(
    env: BuildEnv,
    cpg_nodes__inputs: _CpgNodeInputs,
) -> InferableTabularInput:
    """Build CPG nodes from syntax, symbol, and flow inventories.

    Returns
    -------
    InferableTabularInput
        Arrow reader for graph.cpg_nodes.
    """
    core = _core_lazyframes(cpg_nodes__inputs.core)
    graph = _graph_lazyframes(cpg_nodes__inputs.graph)

    frames = [
        _syntax_nodes_to_cpg(core.syntax_nodes),
        _ast_nodes_to_cpg(core.ast_nodes, env),
        _scip_symbols_to_cpg(core.scip_symbol_information),
        _goids_to_cpg(core.goids),
        _py_sym_scopes_to_cpg(core.py_sym_scopes),
        _py_sym_bindings_to_cpg(core.py_sym_bindings),
        _py_bc_code_units_to_cpg(core.py_bc_code_units),
        _py_bc_instructions_to_cpg(core.py_bc_instructions),
        _py_bc_blocks_to_cpg(core.py_bc_blocks),
        _py_inspect_objects_to_cpg(core.py_inspect_objects),
        _py_inspect_signatures_to_cpg(core.py_inspect_signatures),
        _py_inspect_signature_params_to_cpg(core.py_inspect_signature_params),
        _ts_tokens_to_cpg(core.ts_tokens),
        _ts_trivia_to_cpg(core.ts_trivia),
        _cfg_blocks_to_cpg(graph.cfg_blocks, core.goids),
        _import_modules_to_cpg(graph.import_modules),
    ]
    tables = [frame for frame in frames if frame.num_rows > 0]
    if tables:
        combined = concat_tables_unified(tables)
        combined = _select_node_columns(combined)
        combined = dedupe_table_for_table(CPG_NODES_TABLE_KEY, combined)
        return _frame_to_reader(CPG_NODES_TABLE_KEY, combined)
    return empty_table_for_table(CPG_NODES_TABLE_KEY)


def _syntax_edges_to_cpg(syntax_edges: pa.Table) -> pa.Table:
    rows: list[dict[str, object]] = []
    for row in _table_rows(syntax_edges):
        parent_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "producer": row.get("producer"),
            "node_id": row.get("parent_node_id"),
        }
        child_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "producer": row.get("producer"),
            "node_id": row.get("child_node_id"),
        }
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(SYNTAX_NODES_TABLE_KEY, parent_pk),
                "dst_cpg_node_id": _pk_from_row(SYNTAX_NODES_TABLE_KEY, child_pk),
                "edge_kind": "AST",
                "edge_layer": "SYNTAX",
                "rel_path": row.get("rel_path"),
                "ordinal": row.get("child_ordinal"),
                "extras_json": None,
            }
        )
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _occurrence_role_resolvers(
    span_frame: pa.Table,
) -> dict[tuple[str, str], SpanResolver[_OccurrenceRolePayload]]:
    resolvers: dict[tuple[str, str], SpanResolver[_OccurrenceRolePayload]] = {}
    for row in _table_rows(span_frame):
        rel_path = row.get("rel_path")
        scip_symbol = row.get("scip_symbol")
        if not isinstance(rel_path, str) or not isinstance(scip_symbol, str):
            continue
        start_line = row.get("occ_start_line", row.get("start_line"))
        end_line = row.get("occ_end_line", row.get("end_line"))
        if not isinstance(start_line, int):
            continue
        end_line_value = end_line if isinstance(end_line, int) else start_line
        resolver = resolvers.get((rel_path, scip_symbol))
        if resolver is None:
            resolver = SpanResolver.for_lines(path_normalizer=lambda value: value)
            resolvers[rel_path, scip_symbol] = resolver
        resolver.add_span(
            rel_path,
            start_line,
            end_line_value,
            _OccurrenceRolePayload(
                scip_roles=_coerce_int(row.get("scip_roles", row.get("roles"))),
                is_definition=_coerce_bool(row.get("is_definition")),
                is_reference=_coerce_bool(row.get("is_reference")),
                is_import=_coerce_bool(row.get("is_import")),
                is_write=_coerce_bool(row.get("is_write")),
                is_read=_coerce_bool(row.get("is_read")),
            ),
        )
    return resolvers


def _occurrence_fallback_rows(
    joined_frame: pa.Table,
    span_frame: pa.Table,
) -> list[dict[str, object]]:
    if "scip_roles" not in joined_frame.column_names:
        return []
    resolvers = _occurrence_role_resolvers(span_frame)
    rows: list[dict[str, object]] = []
    for row in _table_rows(joined_frame):
        if row.get("scip_roles") is not None:
            continue
        row_id = row.get("__row_id")
        rel_path = row.get("rel_path")
        scip_symbol = row.get("scip_symbol")
        if (
            not isinstance(row_id, int)
            or not isinstance(rel_path, str)
            or not isinstance(scip_symbol, str)
        ):
            continue
        start_line = row.get("occ_start_line")
        end_line = row.get("occ_end_line")
        if not isinstance(start_line, int):
            continue
        end_line_value = end_line if isinstance(end_line, int) else start_line
        resolver = resolvers.get((rel_path, scip_symbol))
        if resolver is None:
            continue
        match = resolver.resolve(rel_path, start_line, end_line_value)
        if match.match_kind == "NONE" or match.payload is None:
            continue
        payload = match.payload
        rows.append(
            {
                "__row_id": row_id,
                "scip_roles_fallback": payload.scip_roles,
                "is_definition_fallback": payload.is_definition,
                "is_reference_fallback": payload.is_reference,
                "is_import_fallback": payload.is_import,
                "is_write_fallback": payload.is_write,
                "is_read_fallback": payload.is_read,
                "span_match_kind_fallback": match.match_kind,
                "span_candidate_count_fallback": match.candidate_count,
            }
        )
    return rows


def _occurrence_roles(
    occ_syntax: pa.Table,
    occ_span: pa.Table,
) -> pa.Table:
    syntax_rows = _table_rows(occ_syntax)
    if not syntax_rows:
        return _empty_table([])
    span_index = _occurrence_span_index(_table_rows(occ_span))
    resolvers = _occurrence_role_resolvers(occ_span)
    joined_rows = _occurrence_joined_rows(syntax_rows, span_index)
    _apply_occurrence_resolvers(joined_rows, resolvers)
    return pa.Table.from_pylist(joined_rows)


def _occurrence_span_index(
    span_rows: Sequence[Mapping[str, object]],
) -> dict[OccurrenceSpanKey, Mapping[str, object]]:
    index: dict[OccurrenceSpanKey, Mapping[str, object]] = {}
    for row in span_rows:
        key = (
            row.get("repo"),
            row.get("commit"),
            row.get("rel_path"),
            row.get("scip_symbol"),
            row.get("start_line"),
            row.get("start_col"),
            row.get("end_line"),
            row.get("end_col"),
        )
        index[key] = row
    return index


def _occurrence_joined_rows(
    syntax_rows: Sequence[Mapping[str, object]],
    span_index: Mapping[OccurrenceSpanKey, Mapping[str, object]],
) -> list[dict[str, object]]:
    joined_rows: list[dict[str, object]] = []
    for row in syntax_rows:
        key = (
            row.get("repo"),
            row.get("commit"),
            row.get("rel_path"),
            row.get("scip_symbol"),
            row.get("occ_start_line"),
            row.get("occ_start_col"),
            row.get("occ_end_line"),
            row.get("occ_end_col"),
        )
        span_row = span_index.get(key)
        joined = dict(row)
        if span_row is not None:
            joined["scip_roles"] = span_row.get("roles")
            joined["is_definition"] = span_row.get("is_definition")
            joined["is_reference"] = span_row.get("is_reference")
            joined["is_import"] = span_row.get("is_import")
            joined["is_write"] = span_row.get("is_write")
            joined["is_read"] = span_row.get("is_read")
        else:
            joined["scip_roles"] = None
            joined["is_definition"] = None
            joined["is_reference"] = None
            joined["is_import"] = None
            joined["is_write"] = None
            joined["is_read"] = None
        joined["span_match_kind"] = None
        joined["span_candidate_count"] = None
        joined_rows.append(joined)
    return joined_rows


def _apply_occurrence_resolvers(
    joined_rows: list[dict[str, object]],
    resolvers: Mapping[tuple[str, str], SpanResolver[_OccurrenceRolePayload]],
) -> None:
    for joined in joined_rows:
        if joined.get("scip_roles") is not None:
            continue
        rel_path = joined.get("rel_path")
        scip_symbol = joined.get("scip_symbol")
        if not isinstance(rel_path, str) or not isinstance(scip_symbol, str):
            continue
        resolver = resolvers.get((rel_path, scip_symbol))
        start_line = joined.get("occ_start_line")
        if resolver is None or not isinstance(start_line, int):
            continue
        end_line = joined.get("occ_end_line")
        end_line_value = end_line if isinstance(end_line, int) else start_line
        match = resolver.resolve(rel_path, start_line, end_line_value)
        if match.match_kind == "NONE" or match.payload is None:
            continue
        payload = match.payload
        joined["scip_roles"] = payload.scip_roles
        joined["is_definition"] = payload.is_definition
        joined["is_reference"] = payload.is_reference
        joined["is_import"] = payload.is_import
        joined["is_write"] = payload.is_write
        joined["is_read"] = payload.is_read
        joined["span_match_kind"] = match.match_kind
        joined["span_candidate_count"] = match.candidate_count


def _scip_occurrence_edges_to_cpg(
    occ_syntax: pa.Table,
    occ_span: pa.Table,
) -> pa.Table:
    joined = _occurrence_roles(occ_syntax, occ_span)
    rows: list[dict[str, object]] = []
    for row in _table_rows(joined):
        if row.get("syntax_node_id") is None:
            continue
        syntax_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "producer": row.get("producer"),
            "node_id": row.get("syntax_node_id"),
        }
        symbol_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "symbol": row.get("scip_symbol"),
        }
        is_def = bool(row.get("is_definition")) if row.get("is_definition") is not None else False
        is_import = bool(row.get("is_import")) if row.get("is_import") is not None else False
        is_write = bool(row.get("is_write")) if row.get("is_write") is not None else False
        is_read = bool(row.get("is_read")) if row.get("is_read") is not None else False
        edge_kind = "REFERS_TO"
        if is_def:
            edge_kind = "DEFINES"
        elif is_import:
            edge_kind = "IMPORTS"
        elif is_write:
            edge_kind = "WRITES"
        elif is_read:
            edge_kind = "REFERS_TO"
        extras_values = {
            "scip_occurrence_id": row.get("scip_occurrence_id"),
            "match_kind": row.get("match_kind"),
            "candidate_count": row.get("candidate_count"),
            "scip_roles": row.get("scip_roles"),
            "span_match_kind": row.get("span_match_kind"),
            "span_candidate_count": row.get("span_candidate_count"),
        }
        ordinal = _ordinal_from_row(
            "core.scip_occurrence_syntax_xref",
            {"scip_occurrence_id": row.get("scip_occurrence_id")},
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(SYNTAX_NODES_TABLE_KEY, syntax_pk),
                "dst_cpg_node_id": _pk_from_row(SCIP_SYMBOLS_TABLE_KEY, symbol_pk),
                "edge_kind": edge_kind,
                "edge_layer": "SYMBOL",
                "rel_path": row.get("rel_path"),
                "ordinal": ordinal,
                "extras_json": _pk_json_from_row(extras_values),
            }
        )
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _scip_symbol_relationships_to_cpg(symbol_rels: pa.Table) -> pa.Table:
    rows: list[dict[str, object]] = []
    for row in _table_rows(symbol_rels):
        src_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "symbol": row.get("symbol"),
        }
        dst_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "symbol": row.get("related_symbol"),
        }
        ordinal = _ordinal_from_row(
            "core.scip_symbol_relationships",
            {
                "symbol": row.get("symbol"),
                "related_symbol": row.get("related_symbol"),
                "relationship_kind": row.get("relationship_kind"),
            },
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(SCIP_SYMBOLS_TABLE_KEY, src_pk),
                "dst_cpg_node_id": _pk_from_row(SCIP_SYMBOLS_TABLE_KEY, dst_pk),
                "edge_kind": row.get("relationship_kind"),
                "edge_layer": "SYMBOL",
                "rel_path": None,
                "ordinal": ordinal,
                "extras_json": None,
            }
        )
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _scip_symbol_goid_edges_to_cpg(symbol_goid: pa.Table) -> pa.Table:
    rows: list[dict[str, object]] = []
    for row in _table_rows(symbol_goid):
        if row.get("goid_h128") is None:
            continue
        src_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "symbol": row.get("scip_symbol"),
        }
        dst_pk = {"goid_h128": row.get("goid_h128")}
        extras_values = {
            "def_rel_path": row.get("def_rel_path"),
            "def_start_line": row.get("def_start_line"),
            "def_start_col": row.get("def_start_col"),
            "def_end_line": row.get("def_end_line"),
            "def_end_col": row.get("def_end_col"),
        }
        ordinal = _ordinal_from_row(
            "core.scip_symbol_goid_xref",
            {"scip_symbol": row.get("scip_symbol"), "goid_h128": row.get("goid_h128")},
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(SCIP_SYMBOLS_TABLE_KEY, src_pk),
                "dst_cpg_node_id": _pk_from_row(GOIDS_TABLE_KEY, dst_pk),
                "edge_kind": "RESOLVES_TO",
                "edge_layer": "SYMBOL",
                "rel_path": row.get("def_rel_path"),
                "ordinal": ordinal,
                "extras_json": _pk_json_from_row(extras_values),
            }
        )
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _call_graph_edges_to_cpg(call_edges: pa.Table) -> pa.Table:
    rows: list[dict[str, object]] = []
    for row in _table_rows(call_edges):
        if row.get("caller_goid_h128") is None or row.get("callee_goid_h128") is None:
            continue
        src_pk = {"goid_h128": row.get("caller_goid_h128")}
        dst_pk = {"goid_h128": row.get("callee_goid_h128")}
        extras_values = {
            "resolved_via": row.get("resolved_via"),
            "confidence": row.get("confidence"),
            "kind": row.get("kind"),
        }
        ordinal = _ordinal_from_row(
            "graph.call_graph_edges",
            {
                "caller_goid_h128": row.get("caller_goid_h128"),
                "callee_goid_h128": row.get("callee_goid_h128"),
                "callsite_path": row.get("callsite_path"),
                "callsite_line": row.get("callsite_line"),
                "callsite_col": row.get("callsite_col"),
            },
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(GOIDS_TABLE_KEY, src_pk),
                "dst_cpg_node_id": _pk_from_row(GOIDS_TABLE_KEY, dst_pk),
                "rel_path": row.get("callsite_path"),
                "ordinal": ordinal,
                "extras_json": _pk_json_from_row(extras_values),
            }
        )
    table = _edge_rows_to_table(rows)
    table = _append_constant_columns(table, {"edge_kind": "CALLS", "edge_layer": "FLOW"})
    return _select_edge_columns(table)


def _import_graph_edges_to_cpg(import_edges: pa.Table) -> pa.Table:
    rows: list[dict[str, object]] = []
    for row in _table_rows(import_edges):
        src_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "module": row.get("src_module"),
        }
        dst_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "module": row.get("dst_module"),
        }
        extras_values = {
            "src_fan_out": row.get("src_fan_out"),
            "dst_fan_in": row.get("dst_fan_in"),
            "cycle_group": row.get("cycle_group"),
            "module_layer": row.get("module_layer"),
        }
        ordinal = _ordinal_from_row(
            "graph.import_graph_edges",
            {
                "src_module": row.get("src_module"),
                "dst_module": row.get("dst_module"),
                "cycle_group": row.get("cycle_group"),
            },
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(IMPORT_MODULES_TABLE_KEY, src_pk),
                "dst_cpg_node_id": _pk_from_row(IMPORT_MODULES_TABLE_KEY, dst_pk),
                "rel_path": None,
                "ordinal": ordinal,
                "extras_json": _pk_json_from_row(extras_values),
            }
        )
    table = _edge_rows_to_table(rows)
    table = _append_constant_columns(table, {"edge_kind": "IMPORTS", "edge_layer": "SYMBOL"})
    return _select_edge_columns(table)


def _cfg_edges_to_cpg(
    cfg_edges: pa.Table,
    cfg_blocks: pa.Table,
    goids: pa.Table,
) -> pa.Table:
    block_index = _cfg_block_index(cfg_blocks, goids)
    rows: list[dict[str, object]] = []
    for row in _table_rows(cfg_edges):
        function_goid = row.get("function_goid_h128")
        src_block_id = row.get("src_block_id")
        dst_block_id = row.get("dst_block_id")
        if function_goid is None or src_block_id is None or dst_block_id is None:
            continue
        src_info = block_index.get((function_goid, src_block_id))
        dst_info = block_index.get((function_goid, dst_block_id))
        if src_info is None or dst_info is None:
            continue
        src_idx = src_info.get("block_idx")
        dst_idx = dst_info.get("block_idx")
        if src_idx is None or dst_idx is None:
            continue
        src_pk = {"function_goid_h128": function_goid, "block_idx": src_idx}
        dst_pk = {"function_goid_h128": function_goid, "block_idx": dst_idx}
        extras_values = {"cfg_edge_kind": row.get("edge_kind")}
        ordinal = _ordinal_from_row(
            "graph.cfg_edges",
            {
                "function_goid_h128": function_goid,
                "src_block_id": src_block_id,
                "dst_block_id": dst_block_id,
                "edge_kind": row.get("edge_kind"),
            },
        )
        rel_path = src_info.get("rel_path") or dst_info.get("rel_path")
        rows.append(
            {
                "repo": src_info.get("repo") or dst_info.get("repo"),
                "commit": src_info.get("commit") or dst_info.get("commit"),
                "src_cpg_node_id": _pk_from_row(CFG_BLOCKS_TABLE_KEY, src_pk),
                "dst_cpg_node_id": _pk_from_row(CFG_BLOCKS_TABLE_KEY, dst_pk),
                "edge_kind": "CFG",
                "edge_layer": "FLOW",
                "rel_path": rel_path,
                "ordinal": ordinal,
                "extras_json": _pk_json_from_row(extras_values),
            }
        )
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _dfg_edges_to_cpg(
    dfg_edges: pa.Table,
    cfg_blocks: pa.Table,
    goids: pa.Table,
) -> pa.Table:
    block_index = _cfg_block_index(cfg_blocks, goids)
    rows: list[dict[str, object]] = []
    for row in _table_rows(dfg_edges):
        edge_row = _dfg_edge_row(row, block_index)
        if edge_row is not None:
            rows.append(edge_row)
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _dfg_edge_row(
    row: Mapping[str, object],
    block_index: Mapping[tuple[object, object], Mapping[str, object]],
) -> dict[str, object] | None:
    function_goid = row.get("function_goid_h128")
    src_block_id = row.get("src_block_id")
    dst_block_id = row.get("dst_block_id")
    if function_goid is None or src_block_id is None or dst_block_id is None:
        return None
    src_info = block_index.get((function_goid, src_block_id))
    dst_info = block_index.get((function_goid, dst_block_id))
    if src_info is None or dst_info is None:
        return None
    src_idx = src_info.get("block_idx")
    dst_idx = dst_info.get("block_idx")
    if src_idx is None or dst_idx is None:
        return None
    src_pk = {"function_goid_h128": function_goid, "block_idx": src_idx}
    dst_pk = {"function_goid_h128": function_goid, "block_idx": dst_idx}
    extras_values = {
        "src_var": row.get("src_var"),
        "dst_var": row.get("dst_var"),
        "edge_kind": row.get("edge_kind"),
        "via_phi": row.get("via_phi"),
        "use_kind": row.get("use_kind"),
    }
    ordinal = _ordinal_from_row(
        "graph.dfg_edges",
        {
            "function_goid_h128": function_goid,
            "src_block_id": src_block_id,
            "dst_block_id": dst_block_id,
            "src_var": row.get("src_var"),
            "dst_var": row.get("dst_var"),
        },
    )
    rel_path = src_info.get("rel_path") or dst_info.get("rel_path")
    return {
        "repo": src_info.get("repo") or dst_info.get("repo"),
        "commit": src_info.get("commit") or dst_info.get("commit"),
        "src_cpg_node_id": _pk_from_row(CFG_BLOCKS_TABLE_KEY, src_pk),
        "dst_cpg_node_id": _pk_from_row(CFG_BLOCKS_TABLE_KEY, dst_pk),
        "edge_kind": "DFG",
        "edge_layer": "FLOW",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras_json": _pk_json_from_row(extras_values),
    }


def _cdg_edges_to_cpg(
    cdg_edges: pa.Table,
    cfg_blocks: pa.Table,
    goids: pa.Table,
) -> pa.Table:
    block_index = _cfg_block_index(cfg_blocks, goids)
    rows: list[dict[str, object]] = []
    for row in _table_rows(cdg_edges):
        edge_row = _cdg_edge_row(row, block_index)
        if edge_row is not None:
            rows.append(edge_row)
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _cdg_edge_row(
    row: Mapping[str, object],
    block_index: Mapping[tuple[object, object], Mapping[str, object]],
) -> dict[str, object] | None:
    function_goid = row.get("function_goid_h128")
    src_block_id = row.get("src_block_id")
    dst_block_id = row.get("dst_block_id")
    if function_goid is None or src_block_id is None or dst_block_id is None:
        return None
    src_info = block_index.get((function_goid, src_block_id))
    dst_info = block_index.get((function_goid, dst_block_id))
    if src_info is None or dst_info is None:
        return None
    src_idx = src_info.get("block_idx")
    dst_idx = dst_info.get("block_idx")
    if src_idx is None or dst_idx is None:
        return None
    src_pk = {"function_goid_h128": function_goid, "block_idx": src_idx}
    dst_pk = {"function_goid_h128": function_goid, "block_idx": dst_idx}
    extras_values = {
        "via_succ_block_id": row.get("via_succ_block_id"),
        "via_edge_kind": row.get("via_edge_kind"),
    }
    ordinal = _ordinal_from_row(
        "graph.cdg_edges",
        {
            "function_goid_h128": function_goid,
            "src_block_id": src_block_id,
            "dst_block_id": dst_block_id,
            "via_succ_block_id": row.get("via_succ_block_id"),
        },
    )
    rel_path = src_info.get("rel_path") or dst_info.get("rel_path")
    edge_kind = row.get("edge_kind") or "CDG"
    return {
        "repo": src_info.get("repo") or dst_info.get("repo"),
        "commit": src_info.get("commit") or dst_info.get("commit"),
        "src_cpg_node_id": _pk_from_row(CFG_BLOCKS_TABLE_KEY, src_pk),
        "dst_cpg_node_id": _pk_from_row(CFG_BLOCKS_TABLE_KEY, dst_pk),
        "edge_kind": edge_kind,
        "edge_layer": "FLOW",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras_json": _pk_json_from_row(extras_values),
    }


def _call_wiring_calls_to_cpg(
    call_edges: pa.Table,
    cfg_blocks: pa.Table,
    syntax_nodes: pa.Table,
) -> pa.Table:
    syntax_index = _syntax_node_index(syntax_nodes)
    block_index = _block_id_index(cfg_blocks)
    rows: list[dict[str, object]] = []
    for row in _table_rows(call_edges):
        call_node_id = row.get("call_node_id")
        if call_node_id is None:
            continue
        syntax_info = syntax_index.get((row.get("repo"), row.get("commit"), call_node_id))
        if syntax_info is None:
            continue
        block_info = block_index.get(row.get("callee_entry_block_id"))
        if block_info is None:
            continue
        block_idx = block_info.get("block_idx")
        function_goid = block_info.get("function_goid_h128")
        if block_idx is None or function_goid is None:
            continue
        src_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": syntax_info.get("rel_path"),
            "producer": syntax_info.get("producer"),
            "node_id": call_node_id,
        }
        dst_pk = {
            "function_goid_h128": function_goid,
            "block_idx": block_idx,
        }
        extras_values = {
            "call_id": row.get("call_id"),
            "confidence": row.get("confidence"),
            "call_extras": decode_payload(row.get("extras_json")),
        }
        ordinal = _ordinal_from_row(
            "graph.cpg_edges_calls",
            {
                "call_id": row.get("call_id"),
                "callee_entry_block_id": row.get("callee_entry_block_id"),
            },
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(SYNTAX_NODES_TABLE_KEY, src_pk),
                "dst_cpg_node_id": _pk_from_row(CFG_BLOCKS_TABLE_KEY, dst_pk),
                "edge_kind": row.get("edge_kind") or "CALLS",
                "edge_layer": "FLOW",
                "rel_path": syntax_info.get("rel_path"),
                "ordinal": ordinal,
                "extras_json": _pk_json_from_row(extras_values),
            }
        )
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _call_wiring_arg_to_param_to_cpg(
    arg_edges: pa.Table,
    syntax_nodes: pa.Table,
) -> pa.Table:
    syntax_index = _syntax_node_index(syntax_nodes)
    rows: list[dict[str, object]] = []
    for row in _table_rows(arg_edges):
        src_arg_node_id = row.get("src_arg_node_id")
        dst_param_node_id = row.get("dst_param_node_id")
        if src_arg_node_id is None or dst_param_node_id is None:
            continue
        src_info = syntax_index.get((row.get("repo"), row.get("commit"), src_arg_node_id))
        dst_info = syntax_index.get((row.get("repo"), row.get("commit"), dst_param_node_id))
        if src_info is None or dst_info is None:
            continue
        src_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": src_info.get("rel_path"),
            "producer": src_info.get("producer"),
            "node_id": src_arg_node_id,
        }
        dst_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": dst_info.get("rel_path"),
            "producer": dst_info.get("producer"),
            "node_id": dst_param_node_id,
        }
        extras_values = {
            "call_id": row.get("call_id"),
            "arg_ordinal": row.get("arg_ordinal"),
            "param_ordinal": row.get("param_ordinal"),
            "arg_name": row.get("arg_name"),
            "param_name": row.get("param_name"),
            "arg_slot": row.get("arg_slot"),
            "arg_role": row.get("arg_role"),
            "arg_is_implicit": row.get("arg_is_implicit"),
            "call_kind": row.get("call_kind"),
            "augop": row.get("augop"),
            "confidence": row.get("confidence"),
        }
        ordinal = _ordinal_from_row(
            "graph.cpg_edges_arg_to_param",
            {
                "call_id": row.get("call_id"),
                "arg_ordinal": row.get("arg_ordinal"),
                "param_ordinal": row.get("param_ordinal"),
                "src_arg_node_id": src_arg_node_id,
                "dst_param_node_id": dst_param_node_id,
            },
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(SYNTAX_NODES_TABLE_KEY, src_pk),
                "dst_cpg_node_id": _pk_from_row(SYNTAX_NODES_TABLE_KEY, dst_pk),
                "edge_kind": row.get("edge_kind") or "ARG_TO_PARAM",
                "edge_layer": "FLOW",
                "rel_path": src_info.get("rel_path"),
                "ordinal": ordinal,
                "extras_json": _pk_json_from_row(extras_values),
            }
        )
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _call_wiring_ret_to_call_to_cpg(
    ret_edges: pa.Table,
    cfg_blocks: pa.Table,
    syntax_nodes: pa.Table,
) -> pa.Table:
    syntax_index = _syntax_node_index(syntax_nodes)
    block_index = _block_id_index(cfg_blocks)
    rows: list[dict[str, object]] = []
    for row in _table_rows(ret_edges):
        call_node_id = row.get("call_node_id")
        if call_node_id is None:
            continue
        syntax_info = syntax_index.get((row.get("repo"), row.get("commit"), call_node_id))
        if syntax_info is None:
            continue
        block_info = block_index.get(row.get("exit_block_id"))
        if block_info is None:
            continue
        function_goid = block_info.get("function_goid_h128")
        block_idx = block_info.get("block_idx")
        if function_goid is None or block_idx is None:
            continue
        src_pk = {
            "function_goid_h128": function_goid,
            "block_idx": block_idx,
        }
        dst_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": syntax_info.get("rel_path"),
            "producer": syntax_info.get("producer"),
            "node_id": call_node_id,
        }
        extras_values = {
            "call_id": row.get("call_id"),
            "confidence": row.get("confidence"),
            "target_role": row.get("target_role"),
            "call_kind": row.get("call_kind"),
            "origin": row.get("origin"),
            "summary": row.get("extras_json"),
        }
        ordinal = _ordinal_from_row(
            "graph.cpg_edges_ret_to_call",
            {"call_id": row.get("call_id"), "exit_block_id": row.get("exit_block_id")},
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(CFG_BLOCKS_TABLE_KEY, src_pk),
                "dst_cpg_node_id": _pk_from_row(SYNTAX_NODES_TABLE_KEY, dst_pk),
                "edge_kind": row.get("edge_kind") or "RET_TO_CALL",
                "edge_layer": "FLOW",
                "rel_path": syntax_info.get("rel_path"),
                "ordinal": ordinal,
                "extras_json": _pk_json_from_row(extras_values),
            }
        )
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _collect_rows(
    frame: pa.Table,
    *,
    columns: Sequence[str],
) -> list[dict[str, object]]:
    if not set(columns).issubset(frame.column_names):
        return []
    return [{column: row.get(column) for column in columns} for row in _table_rows(frame)]


def _coerce_str(value: object) -> str | None:
    if isinstance(value, str):
        return value
    return None


def _has_missing(*values: object) -> bool:
    return any(value is None for value in values)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, int):
        return value
    return None


def _coerce_bool(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _coerce_float(value: object) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


_VAR_KEY_LEN = 3
_QUALPATH_SUFFIX_RE = re.compile(r"#\d+")


def _coerce_var_key(value: object) -> tuple[str, str, str] | None:
    if not isinstance(value, tuple) or len(value) != _VAR_KEY_LEN:
        return None
    first, second, third = value
    if isinstance(first, str) and isinstance(second, str) and isinstance(third, str):
        return first, second, third
    return None


def _scope_qualname_from_qualpath(value: object) -> str | None:
    if not isinstance(value, str) or not value:
        return None
    cleaned = value.replace("::", ".")
    cleaned = _QUALPATH_SUFFIX_RE.sub("", cleaned)
    return cleaned or None


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
    if resolution is not None and binding_kind in {"global_ref", "nonlocal_ref", "free_ref"}:
        dst_binding_id = _coerce_str(resolution.get("dst_binding_id"))
        if dst_binding_id is not None:
            resolved_id = dst_binding_id
    return {
        "binding_id": binding_id,
        "binding_kind": binding_kind,
        "resolved_binding_id": resolved_id,
        "resolution": resolution,
    }


def _event_var_key(
    *,
    binding_id: str | None,
    space: str | None,
    name: str | None,
) -> tuple[str, str, str]:
    return (binding_id or "", space or "", name or "")


def _assign_events_to_blocks(
    events: list[_ResolvedDefUseEvent],
    blocks: list[_PyBcBlock],
) -> dict[str, list[_ResolvedDefUseEvent]]:
    if not events or not blocks:
        return {}
    events_sorted = sorted(events, key=lambda event: event["instr_index"])
    blocks_sorted = sorted(blocks, key=lambda block: block["first_instr_index"])
    block_events: dict[str, list[_ResolvedDefUseEvent]] = defaultdict(list)
    block_idx = 0
    for event in events_sorted:
        instr_index = event["instr_index"]
        while (
            block_idx < len(blocks_sorted)
            and instr_index > blocks_sorted[block_idx]["last_instr_index"]
        ):
            block_idx += 1
        if block_idx >= len(blocks_sorted):
            break
        block = blocks_sorted[block_idx]
        if instr_index < block["first_instr_index"]:
            continue
        block_events[block["block_id"]].append(event)
    return block_events


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
    return _stable_cpg_id(PY_BC_INSTRUCTIONS_TABLE_KEY, pk_values)


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
    return _stable_cpg_id(PY_SYM_BINDINGS_TABLE_KEY, pk_values)


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
    ordinal = _stable_ordinal(
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
        "extras_json": _row_to_payload(extras),
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
    ordinal = _stable_ordinal(
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
        "extras_json": _row_to_payload(extras),
    }


def _edge_rows_to_lazyframe(rows: Sequence[Mapping[str, object]]) -> pa.Table:
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _ast_cpg_id(node_hash: str) -> int:
    return _stable_cpg_id(AST_NODES_TABLE_KEY, {"hash": node_hash})


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
    return _stable_cpg_id(SYNTAX_NODES_TABLE_KEY, pk_values)


def _inspect_signature_param_cpg_id(
    *,
    repo: str,
    commit: str,
    signature_id: str,
    param_index: int,
) -> int:
    pk_values = {
        "repo": repo,
        "commit": commit,
        "signature_id": signature_id,
        "param_index": param_index,
    }
    return _stable_cpg_id(PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY, pk_values)


def _ast_event_kind(node_type: str, ctx: str | None) -> str | None:
    if node_type == "Name":
        if ctx == "store":
            return "DEF"
        if ctx == "load":
            return "USE"
        return None
    if node_type in {"arg", "FunctionDef", "AsyncFunctionDef", "ClassDef", "alias"}:
        return "DEF"
    return None


def _ast_binding_name(node_type: str, row: Mapping[str, object]) -> str | None:
    if node_type == "Name":
        return _coerce_str(row.get("identifier"))
    if node_type == "arg":
        return _coerce_str(row.get("name"))
    if node_type in {"FunctionDef", "AsyncFunctionDef", "ClassDef"}:
        return _coerce_str(row.get("name"))
    if node_type == "alias":
        alias_name = _coerce_str(row.get("asname"))
        if alias_name:
            return alias_name
        imported = _coerce_str(row.get("imported"))
        if imported:
            return imported.rsplit(".", maxsplit=1)[-1]
    return None


def _ast_event_row(row: Mapping[str, object]) -> dict[str, object] | None:
    node_type = _coerce_str(row.get("node_type"))
    if node_type is None:
        return None
    ctx = _coerce_str(row.get("ctx"))
    event_kind = _ast_event_kind(node_type, ctx)
    if event_kind is None:
        return None
    name = _ast_binding_name(node_type, row)
    if name is None:
        return None
    node_hash = _coerce_str(row.get("hash"))
    rel_path = _coerce_str(row.get("path"))
    if node_hash is None or rel_path is None:
        return None
    return {
        "rel_path": rel_path,
        "node_hash": node_hash,
        "event_kind": event_kind,
        "name": name,
        "ctx": ctx,
        "node_type": node_type,
        "start_byte": _coerce_int(row.get("start_byte")),
        "end_byte": _coerce_int(row.get("end_byte")),
        "lineno": _coerce_int(row.get("lineno")),
    }


def _scope_for_ast_event(
    scopes: list[dict[str, object]],
    event: Mapping[str, object],
) -> str | None:
    scope_id = _select_scope_by_span(
        scopes,
        unit_start=_coerce_int(event.get("start_byte")),
        unit_end=_coerce_int(event.get("end_byte")),
    )
    if scope_id is not None:
        return scope_id
    return _select_scope_by_lineno(scopes, lineno=_coerce_int(event.get("lineno")))


@dataclass(frozen=True)
class _AstBindingContext:
    repo: str
    commit: str
    rel_path: str
    binding_meta: Mapping[str, object]
    binding_kind: str
    resolution: Mapping[str, object] | None


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


def _ast_binding_context_for_event(
    event: Mapping[str, object],
    *,
    scopes_by_path: Mapping[str, list[dict[str, object]]],
    bindings_by_scope: Mapping[tuple[str, str, str], Mapping[str, object]],
    binding_meta: Mapping[str, Mapping[str, object]],
    resolution_map: Mapping[str, Mapping[str, object]],
) -> _AstBindingContext | None:
    rel_path = _coerce_str(event.get("rel_path"))
    scopes_for_path = scopes_by_path.get(rel_path) if rel_path else None
    scope_id = _scope_for_ast_event(scopes_for_path, event) if scopes_for_path else None
    name = _coerce_str(event.get("name")) if scope_id else None
    if _has_missing(rel_path, scope_id, name):
        return None
    rel_path_value = cast("str", rel_path)
    scope_id_value = cast("str", scope_id)
    name_value = cast("str", name)
    binding_info = _resolve_binding_for_event(
        rel_path=rel_path_value,
        scope_id=scope_id_value,
        name=name_value,
        bindings_by_scope=bindings_by_scope,
        resolution_map=resolution_map,
    )
    if binding_info is None:
        return None
    return _binding_context_from_info(
        rel_path=rel_path_value,
        binding_info=binding_info,
        binding_meta=binding_meta,
    )


def _binding_context_from_info(
    *,
    rel_path: str,
    binding_info: Mapping[str, object],
    binding_meta: Mapping[str, Mapping[str, object]],
) -> _AstBindingContext | None:
    binding_kind = _coerce_str(binding_info.get("binding_kind"))
    resolved_id = _coerce_str(binding_info.get("resolved_binding_id"))
    meta = binding_meta.get(resolved_id) if resolved_id else None
    repo = _coerce_str(meta.get("repo")) if meta else None
    commit = _coerce_str(meta.get("commit")) if meta else None
    if _has_missing(binding_kind, resolved_id, meta, repo, commit):
        return None
    resolution_payload = binding_info.get("resolution")
    resolution = resolution_payload if isinstance(resolution_payload, Mapping) else None
    return _AstBindingContext(
        repo=cast("str", repo),
        commit=cast("str", commit),
        rel_path=rel_path,
        binding_meta=cast("Mapping[str, object]", meta),
        binding_kind=cast("str", binding_kind),
        resolution=resolution,
    )


def _ast_binding_edge_row(
    *,
    event: Mapping[str, object],
    context: _AstBindingContext,
) -> dict[str, object]:
    node_hash = _coerce_str(event.get("node_hash"))
    if node_hash is None:
        return {}
    src_cpg_node_id = _ast_cpg_id(node_hash)
    dst_cpg_node_id = _binding_cpg_id(context.binding_meta)
    if dst_cpg_node_id is None:
        return {}
    event_kind = _coerce_str(event.get("event_kind")) or ""
    edge_kind = "BINDS_DEF" if event_kind == "DEF" else "BINDS_USE"
    extras = {
        "name": event.get("name"),
        "ctx": event.get("ctx"),
        "ast_node_type": event.get("node_type"),
        "binding_kind": context.binding_kind,
        "resolution_kind": _coerce_str(context.resolution.get("kind"))
        if context.resolution
        else None,
        "resolution_reason": _coerce_str(context.resolution.get("reason"))
        if context.resolution
        else None,
    }
    ordinal = _stable_ordinal(
        "graph.cpg_edges_ast_binding",
        {
            "node_hash": node_hash,
            "binding_id": context.binding_meta.get("binding_id"),
            "edge_kind": edge_kind,
        },
    )
    return {
        "repo": context.repo,
        "commit": context.commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": edge_kind,
        "edge_layer": "SYMBOL",
        "rel_path": context.rel_path,
        "ordinal": ordinal,
        "extras_json": _row_to_payload(extras),
    }


def _scopes_by_path(scope_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
    by_path: dict[str, list[dict[str, object]]] = defaultdict(list)
    for scope in scope_rows:
        rel_path = _coerce_str(scope.get("rel_path"))
        if rel_path is None:
            continue
        by_path[rel_path].append(scope)
    return by_path


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


def _normalize_path(value: str) -> str:
    return value.replace("\\", "/")


def _best_source_path(file_name: str, ast_paths: Sequence[str]) -> str | None:
    normalized = _normalize_path(file_name)
    best_match: str | None = None
    for path in ast_paths:
        if normalized.endswith(path) and (best_match is None or len(path) > len(best_match)):
            best_match = path
    return best_match


def _ast_span_for_source(node: Mapping[str, object]) -> tuple[int | None, int | None]:
    decorator_start = _coerce_int(node.get("decorator_start_line"))
    start_line = decorator_start if decorator_start is not None else _coerce_int(node.get("lineno"))
    end_line = _coerce_int(node.get("end_lineno")) or start_line
    return start_line, end_line


def _select_ast_anchor_for_source(
    nodes: list[dict[str, object]],
    *,
    source_start: int | None,
    source_end: int | None,
) -> tuple[dict[str, object], float, str] | None:
    if source_start is None or source_end is None:
        return None
    candidates: list[tuple[int, dict[str, object]]] = []
    for node in nodes:
        start_line, end_line = _ast_span_for_source(node)
        if start_line is None or end_line is None:
            continue
        if source_start < start_line or source_end > end_line:
            continue
        span_len = end_line - start_line
        candidates.append((span_len, node))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    node = candidates[0][1]
    start_line, end_line = _ast_span_for_source(node)
    if start_line is None or end_line is None:
        return None
    confidence = 0.9
    if source_start == start_line and source_end == end_line:
        confidence = 0.95
    return node, confidence, "SOURCE_SPAN"


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


def _ast_binding_edges_to_cpg(
    ast_nodes: pa.Table,
    scopes: pa.Table,
    bindings: pa.Table,
    resolution_edges: pa.Table,
) -> pa.Table:
    ast_rows = _collect_rows(
        ast_nodes,
        columns=(
            "path",
            "node_type",
            "hash",
            "ctx",
            "identifier",
            "name",
            "imported",
            "asname",
            "start_byte",
            "end_byte",
            "lineno",
        ),
    )
    if not ast_rows:
        return _empty_table(_CPG_EDGE_COLUMNS)
    scope_rows = _collect_rows(
        scopes,
        columns=(
            "rel_path",
            "scope_id",
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
    scopes_by_path = _scopes_by_path(scope_rows)
    bindings_by_scope, binding_meta = _build_binding_index(binding_rows)
    resolution_map = _build_resolution_map(resolution_rows)
    edges: list[dict[str, object]] = []
    for row in ast_rows:
        event = _ast_event_row(row)
        if event is None:
            continue
        context = _ast_binding_context_for_event(
            event,
            scopes_by_path=scopes_by_path,
            bindings_by_scope=bindings_by_scope,
            binding_meta=binding_meta,
            resolution_map=resolution_map,
        )
        if context is None:
            continue
        edge = _ast_binding_edge_row(event=event, context=context)
        if edge:
            edges.append(edge)
    return _edge_rows_to_lazyframe(edges)


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
    ordinal = _stable_ordinal(
        "graph.cpg_edges_bc_instr_ast",
        {"code_unit_id": code_unit_id, "instr_id": instr_id, "ast_hash": anchor.node_hash},
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": _ast_cpg_id(anchor.node_hash),
        "edge_kind": "BYTECODE_ANCHOR",
        "edge_layer": "SYNTAX",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras_json": _row_to_payload(extras),
    }


def _py_bc_instruction_ast_edges_to_cpg(
    instructions: pa.Table,
    ast_nodes: pa.Table,
) -> pa.Table:
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
        return _empty_table(_CPG_EDGE_COLUMNS)
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
    return _edge_rows_to_lazyframe(edges)


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
    ordinal = _stable_ordinal(
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
        "extras_json": _row_to_payload(extras),
    }


def _py_bc_callsite_edges_to_cpg(
    instructions: pa.Table,
    syntax_calls: pa.Table,
) -> pa.Table:
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
        return _empty_table(_CPG_EDGE_COLUMNS)
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
    return _edge_rows_to_lazyframe(edges)


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
    dst_cpg_node_id = _stable_cpg_id(
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
    ordinal = _stable_ordinal(
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
        "extras_json": _row_to_payload(extras),
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


@dataclass(frozen=True, slots=True)
class _CallsiteSymbolInputs:
    instr_rows: list[dict[str, object]]
    calls_by_path: dict[str, list[dict[str, object]]]
    exact_index: dict[tuple[str, str], dict[str, list[Mapping[str, object]]]]
    leaf_index: dict[tuple[str, str], dict[str, list[Mapping[str, object]]]]


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


def _py_bc_callsite_symbol_edges_to_cpg(
    instructions: pa.Table,
    syntax_calls: pa.Table,
    scip_symbols: pa.Table,
) -> pa.Table:
    inputs = _callsite_symbol_inputs(instructions, syntax_calls, scip_symbols)
    if inputs is None:
        return _empty_table(_CPG_EDGE_COLUMNS)
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
    return _edge_rows_to_lazyframe(edges)


_MEMORY_EDGE_KIND_MAP = {
    ("attribute", "USE"): "READS_ATTR",
    ("attribute", "DEF"): "WRITES_ATTR",
    ("subscript", "USE"): "READS_SUBSCR",
    ("subscript", "DEF"): "WRITES_SUBSCR",
    ("global", "USE"): "READS_GLOBAL",
    ("global", "DEF"): "WRITES_GLOBAL",
}


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
    ordinal = _stable_ordinal(
        "graph.cpg_edges_bc_memory",
        {"code_unit_id": code_unit_id, "instr_id": instr_id, "edge_kind": edge_kind},
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": _ast_cpg_id(anchor.node_hash),
        "edge_kind": edge_kind,
        "edge_layer": "FLOW",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras_json": _row_to_payload(extras),
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


def _py_bc_memory_edges_to_cpg(
    defuse_events: pa.Table,
    instructions: pa.Table,
    ast_nodes: pa.Table,
) -> pa.Table:
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
        return _empty_table(_CPG_EDGE_COLUMNS)
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
    return _edge_rows_to_lazyframe(edges)


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
_STACK_PUSH_EXACT: dict[str, tuple[int, bool]] = {"PUSH_NULL": (1, False)}


def _stack_push_spec(opname: str, arg: int | None) -> tuple[int, bool] | None:
    for handler in _STACK_PUSH_HANDLERS:
        spec = handler(opname, arg)
        if spec is not None:
            return spec
    return None


def _stack_push_from_exact(opname: str, arg: int | None) -> tuple[int, bool] | None:
    _ = arg
    return _STACK_PUSH_EXACT.get(opname)


def _stack_push_from_load(opname: str, arg: int | None) -> tuple[int, bool] | None:
    _ = arg
    if opname in _STACK_LOAD_WITH_POP:
        return 1, True
    if not opname.startswith("LOAD_"):
        return None
    load_count = _load_push_count(opname)
    if load_count is None:
        return None
    return load_count, True


def _stack_push_from_unary(opname: str, arg: int | None) -> tuple[int, bool] | None:
    _ = arg
    if opname.startswith("UNARY_"):
        return 1, True
    return None


def _stack_push_from_binary(opname: str, arg: int | None) -> tuple[int, bool] | None:
    _ = arg
    if opname in _STACK_BINARY_OPS:
        return 1, True
    return None


def _stack_push_from_call(opname: str, arg: int | None) -> tuple[int, bool] | None:
    _ = arg
    if opname.startswith("CALL"):
        return 1, True
    return None


def _stack_push_from_iter(opname: str, arg: int | None) -> tuple[int, bool] | None:
    _ = arg
    if opname in _STACK_ITER_OPS:
        return 1, True
    return None


def _stack_push_from_unpack(opname: str, arg: int | None) -> tuple[int, bool] | None:
    if opname != "UNPACK_SEQUENCE":
        return None
    if arg is None or arg < 0:
        return None
    return arg, True


def _stack_push_from_build(opname: str, arg: int | None) -> tuple[int, bool] | None:
    _ = arg
    if opname.startswith("BUILD_"):
        return 1, True
    return None


_STACK_PUSH_HANDLERS: tuple[
    Callable[[str, int | None], tuple[int, bool] | None],
    ...,
] = (
    _stack_push_from_exact,
    _stack_push_from_load,
    _stack_push_from_unary,
    _stack_push_from_binary,
    _stack_push_from_call,
    _stack_push_from_iter,
    _stack_push_from_unpack,
    _stack_push_from_build,
)


def _stack_effect_counts(
    opname: str | None,
    arg: int | None,
) -> tuple[int, int, bool] | None:
    if opname is None or opname in _STACK_SKIP_OPS:
        return None
    if opname in _STACK_POP_ONLY_OPS:
        return _effect_pop_only(opname=opname, arg=arg)
    if opname.startswith(_STACK_POP_PREFIXES):
        return _effect_pop_only(opname=opname, arg=arg)
    push_spec = _stack_push_spec(opname, arg)
    if push_spec is None:
        return None
    push_count, emit_edge = push_spec
    return _effect_from_push(
        opname=opname,
        arg=arg,
        push_count=push_count,
        emit_edge=emit_edge,
    )


def _parse_stack_instr_row(row: Mapping[str, object]) -> dict[str, object] | None:
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    rel_path = _coerce_str(row.get("rel_path"))
    code_unit_id = _coerce_str(row.get("code_unit_id"))
    instr_id = _coerce_str(row.get("instr_id"))
    instr_index = _coerce_int(row.get("instr_index"))
    if _has_missing(repo, commit, rel_path, code_unit_id, instr_id, instr_index):
        return None
    opname = _coerce_str(row.get("baseopname")) or _coerce_str(row.get("opname"))
    return {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "code_unit_id": code_unit_id,
        "instr_id": instr_id,
        "instr_index": instr_index,
        "opname": opname,
        "arg": _coerce_int(row.get("arg")),
    }


def _group_stack_instructions(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        parsed = _parse_stack_instr_row(row)
        if parsed is None:
            continue
        code_unit_id = cast("str", parsed["code_unit_id"])
        grouped[code_unit_id].append(parsed)
    for instrs in grouped.values():
        instrs.sort(key=lambda item: cast("int", item["instr_index"]))
    return grouped


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
        "stack_pop_index": context.pop_index,
        "stack_push_index": context.value.push_index,
        "stack_depth_before": context.depth_before,
        "stack_depth_after": context.depth_after,
        "src_opname": context.value.opname,
        "dst_opname": instr.get("opname"),
        "confidence": 0.4,
    }
    ordinal = _stable_ordinal(
        "graph.cpg_edges_bc_stack",
        {
            "code_unit_id": code_unit_id,
            "src_instr_id": context.value.instr_id,
            "dst_instr_id": instr_id,
            "block_id": context.block_id,
            "stack_pop_index": context.pop_index,
            "stack_push_index": context.value.push_index,
        },
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": "STACK_REACHES",
        "edge_layer": "FLOW",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras_json": _row_to_payload(extras),
    }


def _block_instruction_rows(
    block: _PyBcBlock,
    instructions: Sequence[Mapping[str, object]],
) -> list[Mapping[str, object]]:
    filtered: list[Mapping[str, object]] = []
    first_index = block["first_instr_index"]
    last_index = block["last_instr_index"]
    for instr in instructions:
        instr_index = _coerce_int(instr.get("instr_index"))
        if instr_index is None:
            continue
        if instr_index < first_index:
            continue
        if instr_index > last_index:
            break
        filtered.append(instr)
    return filtered


def _stack_pop_edges(
    instr: Mapping[str, object],
    *,
    block_id: str,
    pop_count: int,
    depth_before: int,
    stack: list[_StackValue],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for pop_index in range(pop_count):
        if not stack:
            break
        value = stack.pop()
        if not value.emit_edge:
            continue
        context = _StackEdgeContext(
            instr=instr,
            value=value,
            block_id=block_id,
            pop_index=pop_index,
            depth_before=depth_before,
            depth_after=len(stack),
        )
        edge = _stack_edge_row(context)
        if edge:
            edges.append(edge)
    return edges


def _stack_push_values(
    instr: Mapping[str, object],
    *,
    push_count: int,
    emit_edge: bool,
    stack: list[_StackValue],
) -> None:
    instr_id = _coerce_str(instr.get("instr_id"))
    if instr_id is None:
        return
    opname = _coerce_str(instr.get("opname"))
    stack.extend(
        [
            _StackValue(
                instr_id=instr_id,
                push_index=push_index,
                opname=opname,
                emit_edge=emit_edge,
            )
            for push_index in range(push_count)
        ]
    )


def _stack_edges_for_instruction(
    instr: Mapping[str, object],
    *,
    block_id: str,
    stack: list[_StackValue],
) -> list[dict[str, object]]:
    effect = _stack_effect_counts(
        _coerce_str(instr.get("opname")),
        _coerce_int(instr.get("arg")),
    )
    if effect is None:
        return []
    pop_count, push_count, emit_edge = effect
    depth_before = len(stack)
    edges = _stack_pop_edges(
        instr,
        block_id=block_id,
        pop_count=pop_count,
        depth_before=depth_before,
        stack=stack,
    )
    _stack_push_values(instr, push_count=push_count, emit_edge=emit_edge, stack=stack)
    return edges


def _stack_edges_for_block(
    block: _PyBcBlock,
    instructions: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    stack: list[_StackValue] = []
    for instr in _block_instruction_rows(block, instructions):
        edges.extend(
            _stack_edges_for_instruction(
                instr,
                block_id=block["block_id"],
                stack=stack,
            )
        )
    return edges


def _py_bc_stack_edges_to_cpg(
    instructions: pa.Table,
    blocks: pa.Table,
) -> pa.Table:
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
        return _empty_table(_CPG_EDGE_COLUMNS)
    instrs_by_unit = _group_stack_instructions(instr_rows)
    blocks_by_unit = _group_blocks(block_rows)
    edges: list[dict[str, object]] = []
    for code_unit_id, unit_blocks in blocks_by_unit.items():
        unit_instrs = instrs_by_unit.get(code_unit_id)
        if not unit_instrs:
            continue
        for block in sorted(unit_blocks, key=lambda item: item["first_instr_index"]):
            edges.extend(_stack_edges_for_block(block, unit_instrs))
    return _edge_rows_to_lazyframe(edges)


def _py_sym_scope_edges_to_cpg(scope_edges: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "rel_path",
        "parent_scope_id",
        "child_scope_id",
        "edge_kind",
    }
    if not required.issubset(scope_edges.column_names):
        return _empty_table(_CPG_EDGE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(scope_edges):
        parent_scope_id = row.get("parent_scope_id")
        child_scope_id = row.get("child_scope_id")
        if parent_scope_id is None or child_scope_id is None:
            continue
        parent_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "scope_id": parent_scope_id,
        }
        child_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "scope_id": child_scope_id,
        }
        extras = _pk_json_from_row({"edge_kind": row.get("edge_kind")})
        owns_ordinal = _ordinal_from_row(
            "graph.cpg_edges_scope",
            {
                "parent_scope_id": parent_scope_id,
                "child_scope_id": child_scope_id,
                "edge_kind": "OWNS_SCOPE",
            },
        )
        parent_ordinal = _ordinal_from_row(
            "graph.cpg_edges_scope",
            {
                "parent_scope_id": parent_scope_id,
                "child_scope_id": child_scope_id,
                "edge_kind": "PARENT_SCOPE",
            },
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(PY_SYM_SCOPES_TABLE_KEY, parent_pk),
                "dst_cpg_node_id": _pk_from_row(PY_SYM_SCOPES_TABLE_KEY, child_pk),
                "edge_kind": "OWNS_SCOPE",
                "edge_layer": "SYMBOL",
                "rel_path": row.get("rel_path"),
                "ordinal": owns_ordinal,
                "extras_json": extras,
            }
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(PY_SYM_SCOPES_TABLE_KEY, child_pk),
                "dst_cpg_node_id": _pk_from_row(PY_SYM_SCOPES_TABLE_KEY, parent_pk),
                "edge_kind": "PARENT_SCOPE",
                "edge_layer": "SYMBOL",
                "rel_path": row.get("rel_path"),
                "ordinal": parent_ordinal,
                "extras_json": extras,
            }
        )
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _py_sym_namespace_edges_to_cpg(
    namespace_edges: pa.Table,
    bindings: pa.Table,
) -> pa.Table:
    edge_rows = _collect_rows(
        namespace_edges,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "scope_id",
            "name",
            "symbol_row_id",
            "child_scope_id",
            "edge_kind",
            "is_ambiguous",
        ),
    )
    binding_rows = _collect_rows(
        bindings,
        columns=("repo", "commit", "rel_path", "binding_id", "scope_id", "name"),
    )
    if not edge_rows or not binding_rows:
        return _empty_table(_CPG_EDGE_COLUMNS)
    bindings_by_scope, _ = _build_binding_index(binding_rows)
    edges: list[dict[str, object]] = []
    for row in edge_rows:
        edge = _namespace_edge_row(row, bindings_by_scope)
        if edge:
            edges.append(edge)
    return _edge_rows_to_lazyframe(edges)


def _namespace_edge_row(
    row: Mapping[str, object],
    bindings_by_scope: Mapping[tuple[str, str, str], dict[str, object]],
) -> dict[str, object]:
    rel_path = _coerce_str(row.get("rel_path"))
    scope_id = _coerce_str(row.get("scope_id"))
    name = _coerce_str(row.get("name"))
    child_scope_id = _coerce_str(row.get("child_scope_id"))
    if rel_path is None or scope_id is None or name is None or child_scope_id is None:
        return {}
    binding = bindings_by_scope.get((rel_path, scope_id, name))
    if binding is None:
        return {}
    repo = _coerce_str(binding.get("repo"))
    commit = _coerce_str(binding.get("commit"))
    binding_id = _coerce_str(binding.get("binding_id"))
    if _has_missing(repo, commit, binding_id):
        return {}
    src_cpg_node_id = _stable_cpg_id(
        PY_SYM_BINDINGS_TABLE_KEY,
        {
            "repo": repo,
            "commit": commit,
            "rel_path": rel_path,
            "binding_id": binding_id,
        },
    )
    dst_cpg_node_id = _stable_cpg_id(
        PY_SYM_SCOPES_TABLE_KEY,
        {
            "repo": repo,
            "commit": commit,
            "rel_path": rel_path,
            "scope_id": child_scope_id,
        },
    )
    extras = {
        "name": name,
        "symbol_row_id": row.get("symbol_row_id"),
        "is_ambiguous": row.get("is_ambiguous"),
    }
    ordinal = _stable_ordinal(
        "graph.cpg_edges_namespace",
        {
            "binding_id": binding_id,
            "child_scope_id": child_scope_id,
        },
    )
    return {
        "repo": repo,
        "commit": commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": _coerce_str(row.get("edge_kind")) or "BINDS_NAMESPACE",
        "edge_layer": "SYMBOL",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras_json": _row_to_payload(extras),
    }


def _py_sym_binding_edges_to_cpg(bindings: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "rel_path",
        "scope_id",
        "binding_id",
        "binding_kind",
        "name",
    }
    if not required.issubset(bindings.column_names):
        return _empty_table(_CPG_EDGE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(bindings):
        scope_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "scope_id": row.get("scope_id"),
        }
        binding_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "binding_id": row.get("binding_id"),
        }
        extras_values = {
            "binding_kind": row.get("binding_kind"),
            "declared_here": row.get("declared_here"),
            "referenced_here": row.get("referenced_here"),
            "assigned_here": row.get("assigned_here"),
            "annotated_here": row.get("annotated_here"),
            "scoping_class": row.get("scoping_class"),
        }
        ordinal = _ordinal_from_row(
            "graph.cpg_edges_binding",
            {
                "scope_id": row.get("scope_id"),
                "binding_id": row.get("binding_id"),
            },
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(PY_SYM_SCOPES_TABLE_KEY, scope_pk),
                "dst_cpg_node_id": _pk_from_row(PY_SYM_BINDINGS_TABLE_KEY, binding_pk),
                "edge_kind": "DECLARES",
                "edge_layer": "SYMBOL",
                "rel_path": row.get("rel_path"),
                "ordinal": ordinal,
                "extras_json": _pk_json_from_row(extras_values),
            }
        )
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _py_sym_resolution_edges_to_cpg(resolution_edges: pa.Table) -> pa.Table:
    required = {
        "repo",
        "commit",
        "rel_path",
        "edge_id",
        "src_binding_id",
        "dst_binding_id",
        "kind",
    }
    if not required.issubset(resolution_edges.column_names):
        return _empty_table(_CPG_EDGE_COLUMNS)
    rows = [_py_sym_resolution_edge_row(row) for row in _table_rows(resolution_edges)]
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _py_sym_resolution_edge_row(row: Mapping[str, object]) -> dict[str, object]:
    src_pk = {
        "repo": row.get("repo"),
        "commit": row.get("commit"),
        "rel_path": row.get("rel_path"),
        "binding_id": row.get("src_binding_id"),
    }
    dst_pk = {
        "repo": row.get("repo"),
        "commit": row.get("commit"),
        "rel_path": row.get("rel_path"),
        "binding_id": row.get("dst_binding_id"),
    }
    extras_values = {
        "kind": row.get("kind"),
        "confidence": row.get("confidence"),
        "reason": row.get("reason"),
    }
    ordinal = _ordinal_from_row(
        PY_SYM_RESOLUTION_EDGES_TABLE_KEY,
        {"edge_id": row.get("edge_id")},
    )
    return {
        "repo": row.get("repo"),
        "commit": row.get("commit"),
        "src_cpg_node_id": _pk_from_row(PY_SYM_BINDINGS_TABLE_KEY, src_pk),
        "dst_cpg_node_id": _pk_from_row(PY_SYM_BINDINGS_TABLE_KEY, dst_pk),
        "edge_kind": "RESOLVES_TO",
        "edge_layer": "SYMBOL",
        "rel_path": row.get("rel_path"),
        "ordinal": ordinal,
        "extras_json": _pk_json_from_row(extras_values),
    }


def _py_sym_binding_symbol_edges_to_cpg(
    bindings: pa.Table,
    scopes: pa.Table,
    scip_symbols: pa.Table,
) -> pa.Table:
    required_bindings = {
        "repo",
        "commit",
        "rel_path",
        "scope_id",
        "binding_id",
        "binding_kind",
        "name",
    }
    required_scopes = {"repo", "commit", "rel_path", "scope_id", "qualpath"}
    required_symbols = {"repo", "commit", "symbol", "display_name"}
    if (
        not required_bindings.issubset(bindings.column_names)
        or not required_scopes.issubset(scopes.column_names)
        or not required_symbols.issubset(scip_symbols.column_names)
    ):
        return _empty_table(_CPG_EDGE_COLUMNS)
    scope_index = _scope_qualname_index(scopes)
    symbol_index = _symbol_display_index(scip_symbols)
    rows = _binding_symbol_edge_rows(bindings, scope_index, symbol_index)
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


def _scope_qualname_index(scopes: pa.Table) -> dict[tuple[object, object, object, object], str]:
    index: dict[tuple[object, object, object, object], str] = {}
    for row in _table_rows(scopes):
        scope_key = (row.get("repo"), row.get("commit"), row.get("rel_path"), row.get("scope_id"))
        qualname = _scope_qualname_from_qualpath(row.get("qualpath"))
        if qualname:
            index[scope_key] = qualname
    return index


def _symbol_display_index(
    scip_symbols: pa.Table,
) -> dict[tuple[object, object, object], list[Mapping[str, object]]]:
    index: dict[tuple[object, object, object], list[Mapping[str, object]]] = {}
    for row in _table_rows(scip_symbols):
        key = (row.get("repo"), row.get("commit"), row.get("display_name"))
        index.setdefault(key, []).append(row)
    return index


def _binding_symbol_edge_rows(
    bindings: pa.Table,
    scope_index: Mapping[tuple[object, object, object, object], str],
    symbol_index: Mapping[tuple[object, object, object], Sequence[Mapping[str, object]]],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in _table_rows(bindings):
        scope_key = (row.get("repo"), row.get("commit"), row.get("rel_path"), row.get("scope_id"))
        scope_qualname = scope_index.get(scope_key)
        name = _coerce_str(row.get("name"))
        if scope_qualname is None or name is None:
            continue
        binding_qualname = f"{scope_qualname}.{name}"
        symbols = symbol_index.get((row.get("repo"), row.get("commit"), binding_qualname))
        if not symbols:
            continue
        binding_pk = {
            "repo": row.get("repo"),
            "commit": row.get("commit"),
            "rel_path": row.get("rel_path"),
            "binding_id": row.get("binding_id"),
        }
        for symbol in symbols:
            symbol_pk = {
                "repo": symbol.get("repo"),
                "commit": symbol.get("commit"),
                "symbol": symbol.get("symbol"),
            }
            extras_values = {
                "binding_kind": row.get("binding_kind"),
                "match_kind": "qualpath",
            }
            ordinal = _ordinal_from_row(
                "graph.cpg_edges_binding_symbol",
                {"binding_id": row.get("binding_id"), "symbol": symbol.get("symbol")},
            )
            rows.append(
                {
                    "repo": row.get("repo"),
                    "commit": row.get("commit"),
                    "src_cpg_node_id": _pk_from_row(PY_SYM_BINDINGS_TABLE_KEY, binding_pk),
                    "dst_cpg_node_id": _pk_from_row(SCIP_SYMBOLS_TABLE_KEY, symbol_pk),
                    "edge_kind": "BINDS_SYMBOL",
                    "edge_layer": "SYMBOL",
                    "rel_path": row.get("rel_path"),
                    "ordinal": ordinal,
                    "extras_json": _pk_json_from_row(extras_values),
                }
            )
    return rows


def _py_bc_cfg_edges_to_cpg(cfg_edges: pa.Table) -> pa.Table:
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
        return _empty_table(_CPG_EDGE_COLUMNS)
    rows: list[dict[str, object]] = []
    for row in _table_rows(cfg_edges):
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
        ordinal = _ordinal_from_row(
            PY_BC_CFG_EDGES_TABLE_KEY,
            {"edge_id": row.get("edge_id")},
        )
        rows.append(
            {
                "repo": row.get("repo"),
                "commit": row.get("commit"),
                "src_cpg_node_id": _pk_from_row(PY_BC_BLOCKS_TABLE_KEY, src_pk),
                "dst_cpg_node_id": _pk_from_row(PY_BC_BLOCKS_TABLE_KEY, dst_pk),
                "edge_kind": "CFG",
                "edge_layer": "FLOW",
                "rel_path": row.get("rel_path"),
                "ordinal": ordinal,
                "extras_json": _pk_json_from_row(extras_values),
            }
        )
    table = _edge_rows_to_table(rows)
    return _select_edge_columns(table)


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


def _binding_edge_for_event(
    event: Mapping[str, object],
    *,
    scope_id: str,
    bindings_by_scope: Mapping[tuple[str, str, str], Mapping[str, object]],
    binding_meta: Mapping[str, Mapping[str, object]],
    resolution_map: Mapping[str, Mapping[str, object]],
) -> dict[str, object] | None:
    event_kind = _coerce_str(event.get("event_kind"))
    if event_kind not in {"DEF", "USE"}:
        return None
    rel_path = _coerce_str(event.get("rel_path"))
    name = _coerce_str(event.get("name"))
    if rel_path is None or name is None:
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


@dataclass(frozen=True)
class _ReachesContext:
    scope_map: Mapping[str, str]
    blocks_by_unit: Mapping[str, list[_PyBcBlock]]
    cfg_by_unit: Mapping[str, list[tuple[str, str]]]
    bindings_by_scope: Mapping[tuple[str, str, str], Mapping[str, object]]
    resolution_map: Mapping[str, Mapping[str, object]]


@dataclass(frozen=True)
class _PyBcReachesInputs:
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


def _collect_py_bc_reaches_rows(inputs: _PyBcReachesInputs) -> _PyBcReachesRows | None:
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


def _py_bc_defuse_binding_edges_to_cpg(
    defuse_events: pa.Table,
    code_units: pa.Table,
    scopes: pa.Table,
    bindings: pa.Table,
    resolution_edges: pa.Table,
) -> pa.Table:
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
        return _empty_table(_CPG_EDGE_COLUMNS)
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
    return _edge_rows_to_lazyframe(edges)


def _py_bc_reaches_edges_to_cpg(inputs: _PyBcReachesInputs) -> pa.Table:
    rows = _collect_py_bc_reaches_rows(inputs)
    if rows is None:
        return _empty_table(_CPG_EDGE_COLUMNS)
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
    return _edge_rows_to_lazyframe(edges)


def _inspect_full_qualname(module_name: str | None, qualname: str | None) -> str | None:
    if module_name is None:
        return None
    if qualname is None:
        return module_name
    if qualname == module_name:
        return module_name
    if qualname.startswith(f"{module_name}."):
        return qualname
    return f"{module_name}.{qualname}"


def _inspect_status_ok(status: object) -> bool:
    if isinstance(status, dict):
        return status.get("ok") is True
    return False


def _callee_qname_priority(source: object) -> int:
    if isinstance(source, str):
        normalized = source.upper()
        if normalized == "IMPORT":
            return 0
        if normalized == "LOCAL":
            return 1
        if normalized == "BUILTIN":
            return 2
    return 3


def _call_callee_candidates(extras: object, callee_text: str | None) -> list[str]:
    decoded = decode_payload(extras)
    candidates: list[tuple[int, str]] = []
    if isinstance(decoded, dict):
        qnames = decoded.get("callee_qnames")
        if isinstance(qnames, list):
            for item in qnames:
                if not isinstance(item, dict):
                    continue
                name = item.get("name")
                if not isinstance(name, str):
                    continue
                priority = _callee_qname_priority(item.get("source"))
                candidates.append((priority, name))
    if not candidates and isinstance(callee_text, str) and "." in callee_text:
        candidates.append((4, callee_text))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [name for _, name in candidates]


@dataclass(frozen=True)
class _ParamGroups:
    positional: list[dict[str, object]]
    keyword: dict[str, dict[str, object]]
    var_positional: dict[str, object] | None
    var_keyword: dict[str, object] | None


def _sorted_rows(
    rows: Sequence[Mapping[str, object]],
    *,
    key_field: str,
) -> list[dict[str, object]]:
    return sorted(
        (dict(row) for row in rows),
        key=lambda item: _coerce_int(item.get(key_field)) or 0,
    )


def _param_groups(params: Sequence[dict[str, object]]) -> _ParamGroups:
    positional: list[dict[str, object]] = []
    keyword: dict[str, dict[str, object]] = {}
    var_positional: dict[str, object] | None = None
    var_keyword: dict[str, object] | None = None
    for param in params:
        kind = _coerce_str(param.get("kind"))
        if kind in {"POSITIONAL_ONLY", "POSITIONAL_OR_KEYWORD"}:
            positional.append(param)
        if kind in {"POSITIONAL_OR_KEYWORD", "KEYWORD_ONLY"}:
            name = _coerce_str(param.get("name"))
            if name is not None:
                keyword[name] = param
        if kind == "VAR_POSITIONAL":
            var_positional = param
        elif kind == "VAR_KEYWORD":
            var_keyword = param
    return _ParamGroups(
        positional=positional,
        keyword=keyword,
        var_positional=var_positional,
        var_keyword=var_keyword,
    )


def _next_positional_param(
    params: Sequence[dict[str, object]],
    assigned: set[int],
    start_index: int,
) -> tuple[dict[str, object] | None, int]:
    index = start_index
    while index < len(params):
        param = params[index]
        param_index = _coerce_int(param.get("param_index"))
        if param_index is not None and param_index not in assigned:
            return param, index + 1
        index += 1
    return None, index


def _map_positional_arg(
    arg: dict[str, object],
    *,
    groups: _ParamGroups,
    assigned: set[int],
    pos_index: int,
) -> tuple[list[tuple[dict[str, object], dict[str, object], str]], int]:
    mappings: list[tuple[dict[str, object], dict[str, object], str]] = []
    param, next_index = _next_positional_param(groups.positional, assigned, pos_index)
    if param is not None:
        param_index = _coerce_int(param.get("param_index"))
        if param_index is not None:
            assigned.add(param_index)
            mappings.append((arg, param, "positional"))
        return mappings, next_index
    if groups.var_positional is not None:
        mappings.append((arg, groups.var_positional, "varargs"))
    return mappings, next_index


def _map_keyword_arg(
    arg: dict[str, object],
    *,
    groups: _ParamGroups,
    assigned: set[int],
) -> list[tuple[dict[str, object], dict[str, object], str]]:
    arg_name = _coerce_str(arg.get("arg_name"))
    if arg_name is None:
        return []
    param = groups.keyword.get(arg_name)
    if param is None:
        if groups.var_keyword is None:
            return []
        return [(arg, groups.var_keyword, "varkw")]
    param_index = _coerce_int(param.get("param_index"))
    if param_index is None or param_index in assigned:
        if groups.var_keyword is None:
            return []
        return [(arg, groups.var_keyword, "varkw")]
    assigned.add(param_index)
    return [(arg, param, "keyword")]


def _map_starargs_arg(
    arg: dict[str, object],
    *,
    groups: _ParamGroups,
) -> list[tuple[dict[str, object], dict[str, object], str]]:
    if groups.var_positional is None:
        return []
    return [(arg, groups.var_positional, "varargs")]


def _map_kwargs_arg(
    arg: dict[str, object],
    *,
    groups: _ParamGroups,
) -> list[tuple[dict[str, object], dict[str, object], str]]:
    if groups.var_keyword is None:
        return []
    return [(arg, groups.var_keyword, "varkw")]


def _arg_mappings_for_arg(
    arg: dict[str, object],
    *,
    groups: _ParamGroups,
    assigned: set[int],
    pos_index: int,
) -> tuple[list[tuple[dict[str, object], dict[str, object], str]], int]:
    arg_kind = _coerce_str(arg.get("arg_kind"))
    if arg_kind == "positional":
        return _map_positional_arg(arg, groups=groups, assigned=assigned, pos_index=pos_index)
    if arg_kind == "keyword":
        return _map_keyword_arg(arg, groups=groups, assigned=assigned), pos_index
    if arg_kind == "starargs":
        return _map_starargs_arg(arg, groups=groups), pos_index
    if arg_kind == "kwargs":
        return _map_kwargs_arg(arg, groups=groups), pos_index
    return [], pos_index


@dataclass(frozen=True)
class _InspectArgToParamContext:
    repo: str
    commit: str
    rel_path: str
    producer: str
    call_id: str
    signature_id: str


def _assign_args_to_params(
    args: Sequence[Mapping[str, object]],
    params: Sequence[Mapping[str, object]],
) -> list[tuple[dict[str, object], dict[str, object], str]]:
    ordered_args = _sorted_rows(args, key_field="arg_ordinal")
    ordered_params = _sorted_rows(params, key_field="param_index")
    groups = _param_groups(ordered_params)
    assigned: set[int] = set()
    pos_index = 0
    mappings: list[tuple[dict[str, object], dict[str, object], str]] = []
    for arg in ordered_args:
        new_mappings, pos_index = _arg_mappings_for_arg(
            arg,
            groups=groups,
            assigned=assigned,
            pos_index=pos_index,
        )
        mappings.extend(new_mappings)
    return mappings


def _signature_from_params(
    params: Sequence[Mapping[str, object]],
) -> inspect.Signature | None:
    kind_map = {
        "POSITIONAL_ONLY": inspect.Parameter.POSITIONAL_ONLY,
        "POSITIONAL_OR_KEYWORD": inspect.Parameter.POSITIONAL_OR_KEYWORD,
        "VAR_POSITIONAL": inspect.Parameter.VAR_POSITIONAL,
        "KEYWORD_ONLY": inspect.Parameter.KEYWORD_ONLY,
        "VAR_KEYWORD": inspect.Parameter.VAR_KEYWORD,
    }
    parameters: list[inspect.Parameter] = []
    for row in _sorted_rows(params, key_field="param_index"):
        name = _coerce_str(row.get("name"))
        kind_name = _coerce_str(row.get("kind"))
        if name is None or kind_name is None:
            return None
        kind = kind_map.get(kind_name)
        if kind is None:
            return None
        parameters.append(inspect.Parameter(name, kind))
    if not parameters:
        return None
    try:
        return inspect.Signature(parameters)
    except ValueError:
        return None


def _arg_identity(arg: Mapping[str, object]) -> tuple[object, object, object, object]:
    return (
        arg.get("arg_expr_node_id"),
        arg.get("arg_ordinal"),
        arg.get("arg_kind"),
        arg.get("arg_name"),
    )


def _bound_arg_mappings(
    args: Sequence[Mapping[str, object]],
    params: Sequence[Mapping[str, object]],
) -> list[tuple[dict[str, object], dict[str, object], str]]:
    signature = _signature_from_params(params)
    if signature is None:
        return []
    ordered_args = _sorted_rows(args, key_field="arg_ordinal")
    if any(_coerce_str(arg.get("arg_kind")) in {"starargs", "kwargs"} for arg in ordered_args):
        return []
    param_by_name = {
        cast("str", _coerce_str(row.get("name"))): dict(row)
        for row in params
        if _coerce_str(row.get("name")) is not None
    }
    token_result = _arg_tokens(ordered_args)
    if token_result is None:
        return []
    tokens_by_arg, positional_tokens, keyword_tokens = token_result
    try:
        bound = signature.bind_partial(*positional_tokens, **keyword_tokens)
    except TypeError:
        return []
    mappings: list[tuple[dict[str, object], dict[str, object], str]] = []
    for param_name, value in bound.arguments.items():
        param = param_by_name.get(param_name)
        if param is None:
            continue
        _append_bound_mappings(
            value=value,
            tokens_by_arg=tokens_by_arg,
            param=param,
            mappings=mappings,
        )
    return mappings


def _arg_tokens(
    ordered_args: Sequence[Mapping[str, object]],
) -> tuple[dict[object, dict[str, object]], list[object], dict[str, object]] | None:
    tokens_by_arg: dict[object, dict[str, object]] = {}
    positional_tokens: list[object] = []
    keyword_tokens: dict[str, object] = {}
    for arg in ordered_args:
        arg_kind = _coerce_str(arg.get("arg_kind"))
        token = object()
        tokens_by_arg[token] = dict(arg)
        if arg_kind == "positional":
            positional_tokens.append(token)
            continue
        if arg_kind == "keyword":
            arg_name = _coerce_str(arg.get("arg_name"))
            if arg_name is None:
                return None
            keyword_tokens[arg_name] = token
            continue
        return None
    return tokens_by_arg, positional_tokens, keyword_tokens


def _append_bound_mappings(
    *,
    value: object,
    tokens_by_arg: Mapping[object, dict[str, object]],
    param: dict[str, object],
    mappings: list[tuple[dict[str, object], dict[str, object], str]],
) -> None:
    if isinstance(value, tuple):
        _append_token_mappings(
            tokens=value,
            tokens_by_arg=tokens_by_arg,
            param=param,
            mapping_kind="bound_varargs",
            mappings=mappings,
        )
        return
    if isinstance(value, dict):
        _append_token_mappings(
            tokens=value.values(),
            tokens_by_arg=tokens_by_arg,
            param=param,
            mapping_kind="bound_varkw",
            mappings=mappings,
        )
        return
    arg = tokens_by_arg.get(value)
    if arg is None:
        return
    arg_kind = _coerce_str(arg.get("arg_kind"))
    mapping_kind = "bound_keyword" if arg_kind == "keyword" else "bound_positional"
    mappings.append((arg, param, mapping_kind))


def _append_token_mappings(
    *,
    tokens: Iterable[object],
    tokens_by_arg: Mapping[object, dict[str, object]],
    param: dict[str, object],
    mapping_kind: str,
    mappings: list[tuple[dict[str, object], dict[str, object], str]],
) -> None:
    for item in tokens:
        arg = tokens_by_arg.get(item)
        if arg is not None:
            mappings.append((arg, param, mapping_kind))


def _inspect_arg_to_param_edge_row(
    *,
    arg: Mapping[str, object],
    param: Mapping[str, object],
    mapping_kind: str,
    context: _InspectArgToParamContext,
) -> dict[str, object]:
    arg_node_id = _coerce_str(arg.get("arg_expr_node_id"))
    if arg_node_id is None:
        return {}
    param_index = _coerce_int(param.get("param_index"))
    if param_index is None:
        return {}
    src_cpg_node_id = _syntax_node_cpg_id(
        repo=context.repo,
        commit=context.commit,
        rel_path=context.rel_path,
        producer=context.producer,
        node_id=arg_node_id,
    )
    dst_cpg_node_id = _inspect_signature_param_cpg_id(
        repo=context.repo,
        commit=context.commit,
        signature_id=context.signature_id,
        param_index=param_index,
    )
    extras = {
        "call_id": context.call_id,
        "signature_id": context.signature_id,
        "arg_kind": arg.get("arg_kind"),
        "arg_name": arg.get("arg_name"),
        "param_name": param.get("name"),
        "param_kind": param.get("kind"),
        "mapping_kind": mapping_kind,
    }
    ordinal = _stable_ordinal(
        "graph.cpg_edges_arg_to_param_inspect",
        {
            "call_id": context.call_id,
            "signature_id": context.signature_id,
            "arg_ordinal": arg.get("arg_ordinal"),
            "param_index": param_index,
            "mapping_kind": mapping_kind,
        },
    )
    return {
        "repo": context.repo,
        "commit": context.commit,
        "src_cpg_node_id": src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": "ARG_TO_PARAM_INSPECT",
        "edge_layer": "CALL",
        "rel_path": context.rel_path,
        "ordinal": ordinal,
        "extras_json": _row_to_payload(extras),
    }


def _inspect_object_by_name(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], str]:
    entries: list[tuple[tuple[str, str, str], str]] = []
    for row in rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        object_id = _coerce_str(row.get("object_id"))
        full_name = _inspect_full_qualname(
            _coerce_str(row.get("module_name")),
            _coerce_str(row.get("qualname")),
        )
        if None in {repo, commit, object_id, full_name}:
            continue
        repo_value = cast("str", repo)
        commit_value = cast("str", commit)
        object_id_value = cast("str", object_id)
        full_name_value = cast("str", full_name)
        entries.append(((repo_value, commit_value, full_name_value), object_id_value))
    entries.sort(key=lambda item: item[0])
    return dict(entries)


def _signature_by_object(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], str]:
    signature_by_object: dict[tuple[str, str, str], str] = {}
    for row in rows:
        if not _inspect_status_ok(row.get("status")):
            continue
        variant = _coerce_str(row.get("variant"))
        if variant is not None and variant != "primary":
            continue
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        object_id = _coerce_str(row.get("object_id"))
        signature_id = _coerce_str(row.get("signature_id"))
        if None in {repo, commit, object_id, signature_id}:
            continue
        signature_by_object[
            cast("str", repo),
            cast("str", commit),
            cast("str", object_id),
        ] = cast("str", signature_id)
    return signature_by_object


def _params_by_signature(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, list[dict[str, object]]]:
    params_by_signature: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        if not _inspect_status_ok(row.get("status")):
            continue
        signature_id = _coerce_str(row.get("signature_id"))
        if signature_id is None:
            continue
        params_by_signature[signature_id].append(dict(row))
    for params in params_by_signature.values():
        params.sort(key=lambda item: _coerce_int(item.get("param_index")) or 0)
    return params_by_signature


def _signature_by_call(
    call_rows: Sequence[Mapping[str, object]],
    *,
    object_by_name: Mapping[tuple[str, str, str], str],
    signature_by_object: Mapping[tuple[str, str, str], str],
) -> dict[tuple[str, str, str], str]:
    signature_by_call: dict[tuple[str, str, str], str] = {}
    for row in call_rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        call_id = _coerce_str(row.get("call_id"))
        if None in {repo, commit, call_id}:
            continue
        repo_value = cast("str", repo)
        commit_value = cast("str", commit)
        call_id_value = cast("str", call_id)
        candidates = _call_callee_candidates(
            row.get("extras_json"),
            _coerce_str(row.get("callee_text")),
        )
        for candidate in candidates:
            object_id = object_by_name.get((repo_value, commit_value, candidate))
            if object_id is None:
                continue
            signature_id = signature_by_object.get((repo_value, commit_value, object_id))
            if signature_id is None:
                continue
            signature_by_call[repo_value, commit_value, call_id_value] = signature_id
            break
    return signature_by_call


def _args_by_call(
    rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], list[dict[str, object]]]:
    args_by_call: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        call_id = _coerce_str(row.get("call_id"))
        if None in {repo, commit, call_id}:
            continue
        args_by_call[cast("str", repo), cast("str", commit), cast("str", call_id)].append(dict(row))
    return args_by_call


def _inspect_arg_context_for_row(
    row: Mapping[str, object],
    *,
    signature_id: str,
) -> _InspectArgToParamContext | None:
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    rel_path = _coerce_str(row.get("rel_path"))
    producer = _coerce_str(row.get("producer"))
    call_id = _coerce_str(row.get("call_id"))
    if None in {repo, commit, rel_path, producer, call_id}:
        return None
    return _InspectArgToParamContext(
        repo=cast("str", repo),
        commit=cast("str", commit),
        rel_path=cast("str", rel_path),
        producer=cast("str", producer),
        call_id=cast("str", call_id),
        signature_id=signature_id,
    )


def _inspect_arg_to_param_edges_for_call(
    args: Sequence[dict[str, object]],
    *,
    signature_id: str,
    params: Sequence[dict[str, object]],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    bound_mappings = _bound_arg_mappings(args, params)
    assigned = {_arg_identity(arg) for arg, _, _ in bound_mappings}
    remaining_args = [arg for arg in args if _arg_identity(arg) not in assigned]
    fallback_mappings = _assign_args_to_params(remaining_args, params) if remaining_args else []
    mappings = [*bound_mappings, *fallback_mappings]
    for arg, param, mapping_kind in mappings:
        context = _inspect_arg_context_for_row(arg, signature_id=signature_id)
        if context is None:
            continue
        edge = _inspect_arg_to_param_edge_row(
            arg=arg,
            param=param,
            mapping_kind=mapping_kind,
            context=context,
        )
        if edge:
            edges.append(edge)
    return edges


def _inspect_arg_to_param_edges_to_cpg(
    syntax_calls: pa.Table,
    syntax_call_args: pa.Table,
    inspect_objects: pa.Table,
    inspect_signatures: pa.Table,
    inspect_signature_params: pa.Table,
) -> pa.Table:
    call_rows = _collect_rows(
        syntax_calls,
        columns=("repo", "commit", "rel_path", "producer", "call_id", "callee_text", "extras_json"),
    )
    arg_rows = _collect_rows(
        syntax_call_args,
        columns=(
            "repo",
            "commit",
            "rel_path",
            "producer",
            "call_id",
            "arg_ordinal",
            "arg_kind",
            "arg_name",
            "arg_expr_node_id",
        ),
    )
    inspect_rows = _collect_rows(
        inspect_objects,
        columns=("repo", "commit", "object_id", "module_name", "qualname"),
    )
    signature_rows = _collect_rows(
        inspect_signatures,
        columns=("repo", "commit", "signature_id", "object_id", "variant", "status"),
    )
    param_rows = _collect_rows(
        inspect_signature_params,
        columns=("repo", "commit", "signature_id", "param_index", "name", "kind", "status"),
    )
    if not call_rows or not arg_rows or not inspect_rows or not signature_rows or not param_rows:
        return _empty_table(_CPG_EDGE_COLUMNS)
    object_by_name = _inspect_object_by_name(inspect_rows)
    signature_by_object = _signature_by_object(signature_rows)
    params_by_signature = _params_by_signature(param_rows)
    signature_by_call = _signature_by_call(
        call_rows,
        object_by_name=object_by_name,
        signature_by_object=signature_by_object,
    )
    args_by_call = _args_by_call(arg_rows)
    edges: list[dict[str, object]] = []
    for call_key, args in args_by_call.items():
        signature_id = signature_by_call.get(call_key)
        if signature_id is None:
            continue
        params = params_by_signature.get(signature_id)
        if not params:
            continue
        edges.extend(
            _inspect_arg_to_param_edges_for_call(
                args,
                signature_id=signature_id,
                params=params,
            )
        )
    return _edge_rows_to_lazyframe(edges)


def _py_inspect_signature_edges_to_cpg(
    signatures: pa.Table,
    params: pa.Table,
) -> pa.Table:
    required_signatures = {"repo", "commit", "signature_id", "object_id"}
    required_params = {"repo", "commit", "signature_id", "param_index"}
    if not required_signatures.issubset(signatures.column_names) or not required_params.issubset(
        params.column_names
    ):
        return _empty_table(_CPG_EDGE_COLUMNS)
    edges: list[dict[str, object]] = []
    for row in _table_rows(signatures):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        signature_id = _coerce_str(row.get("signature_id"))
        object_id = _coerce_str(row.get("object_id"))
        if _has_missing(repo, commit, signature_id, object_id):
            continue
        src_cpg_node_id = _stable_cpg_id(
            PY_INSPECT_OBJECTS_TABLE_KEY,
            {"repo": repo, "commit": commit, "object_id": object_id},
        )
        dst_cpg_node_id = _stable_cpg_id(
            PY_INSPECT_SIGNATURES_TABLE_KEY,
            {"repo": repo, "commit": commit, "signature_id": signature_id},
        )
        extras = {
            "variant": row.get("variant"),
            "follow_wrapped": row.get("follow_wrapped"),
            "eval_str": row.get("eval_str"),
            "status": row.get("status"),
        }
        ordinal = _stable_ordinal(
            "graph.cpg_edges_inspect_signature",
            {"signature_id": signature_id},
        )
        edges.append(
            {
                "repo": repo,
                "commit": commit,
                "src_cpg_node_id": src_cpg_node_id,
                "dst_cpg_node_id": dst_cpg_node_id,
                "edge_kind": "HAS_SIGNATURE",
                "edge_layer": "SYMBOL",
                "rel_path": None,
                "ordinal": ordinal,
                "extras_json": _row_to_payload(extras),
            }
        )
    for row in _table_rows(params):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        signature_id = _coerce_str(row.get("signature_id"))
        param_index = _coerce_int(row.get("param_index"))
        if _has_missing(repo, commit, signature_id, param_index):
            continue
        src_cpg_node_id = _stable_cpg_id(
            PY_INSPECT_SIGNATURES_TABLE_KEY,
            {"repo": repo, "commit": commit, "signature_id": signature_id},
        )
        dst_cpg_node_id = _inspect_signature_param_cpg_id(
            repo=cast("str", repo),
            commit=cast("str", commit),
            signature_id=cast("str", signature_id),
            param_index=cast("int", param_index),
        )
        extras = {
            "param_index": param_index,
            "name": row.get("name"),
            "kind": row.get("kind"),
            "status": row.get("status"),
        }
        ordinal = _stable_ordinal(
            "graph.cpg_edges_inspect_signature_param",
            {"signature_id": signature_id, "param_index": param_index},
        )
        edges.append(
            {
                "repo": repo,
                "commit": commit,
                "src_cpg_node_id": src_cpg_node_id,
                "dst_cpg_node_id": dst_cpg_node_id,
                "edge_kind": "HAS_PARAM",
                "edge_layer": "SYMBOL",
                "rel_path": None,
                "ordinal": ordinal,
                "extras_json": _row_to_payload(extras),
            }
        )
    return _edge_rows_to_lazyframe(edges)


@dataclass(frozen=True, slots=True)
class _InspectAstIndex:
    by_qualname: dict[str, list[dict[str, object]]]
    by_norm_path: dict[str, list[dict[str, object]]]
    paths: list[str]


@dataclass(frozen=True, slots=True)
class _InspectAstContext:
    repo: str
    commit: str
    object_id: str


def _inspect_ast_indices(ast_rows: Sequence[dict[str, object]]) -> _InspectAstIndex:
    ast_by_qualname: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in ast_rows:
        qualname = _coerce_str(row.get("qualname"))
        if qualname is None:
            continue
        ast_by_qualname[qualname].append(row)
    ast_by_path = _ast_nodes_by_path(ast_rows)
    ast_by_norm_path = {_normalize_path(path): rows for path, rows in ast_by_path.items()}
    ast_paths = sorted(ast_by_norm_path.keys(), key=len, reverse=True)
    return _InspectAstIndex(
        by_qualname=ast_by_qualname,
        by_norm_path=ast_by_norm_path,
        paths=ast_paths,
    )


def _inspect_sources_by_object(
    source_rows: Sequence[Mapping[str, object]],
) -> dict[str, dict[str, object]]:
    sources_by_object: dict[str, dict[str, object]] = {}
    for row in source_rows:
        object_id = _coerce_str(row.get("object_id"))
        file_name = _coerce_str(row.get("file_name"))
        start_line = _coerce_int(row.get("start_line"))
        line_count = _coerce_int(row.get("line_count"))
        if _has_missing(object_id, file_name, start_line):
            continue
        start_line_value = cast("int", start_line)
        end_line = (
            start_line_value
            if line_count is None or line_count <= 0
            else start_line_value + line_count - 1
        )
        sources_by_object[cast("str", object_id)] = {
            "file_name": cast("str", file_name),
            "start_line": start_line_value,
            "end_line": end_line,
        }
    return sources_by_object


def _inspect_ast_edge_row(
    context: _InspectAstContext,
    *,
    node_hash: str,
    rel_path: object,
    extras: Mapping[str, object],
) -> dict[str, object]:
    ordinal = _stable_ordinal(
        "graph.cpg_edges_inspect_ast",
        {"object_id": context.object_id, "ast_hash": node_hash},
    )
    return {
        "repo": context.repo,
        "commit": context.commit,
        "src_cpg_node_id": _stable_cpg_id(
            PY_INSPECT_OBJECTS_TABLE_KEY,
            {
                "repo": context.repo,
                "commit": context.commit,
                "object_id": context.object_id,
            },
        ),
        "dst_cpg_node_id": _ast_cpg_id(node_hash),
        "edge_kind": "INSPECT_ANCHORS_AST",
        "edge_layer": "SYMBOL",
        "rel_path": rel_path,
        "ordinal": ordinal,
        "extras_json": _row_to_payload(extras),
    }


def _inspect_ast_edges_for_source(
    context: _InspectAstContext,
    *,
    source: Mapping[str, object],
    ast_index: _InspectAstIndex,
    seen: set[tuple[str, str]],
) -> list[dict[str, object]]:
    file_name = cast("str", source["file_name"])
    path = _best_source_path(file_name, ast_index.paths)
    if path is None:
        return []
    nodes = ast_index.by_norm_path.get(path, [])
    match = _select_ast_anchor_for_source(
        nodes,
        source_start=cast("int", source["start_line"]),
        source_end=cast("int", source["end_line"]),
    )
    if match is None:
        return []
    node, confidence, match_kind = match
    node_hash = _coerce_str(node.get("hash"))
    if node_hash is None:
        return []
    key = (context.object_id, node_hash)
    if key in seen:
        return []
    extras = {
        "match_kind": match_kind,
        "ast_kind": node.get("node_type"),
        "match_confidence": confidence,
        "source_start_line": source["start_line"],
        "source_end_line": source["end_line"],
    }
    seen.add(key)
    return [
        _inspect_ast_edge_row(
            context,
            node_hash=node_hash,
            rel_path=node.get("path"),
            extras=extras,
        )
    ]


def _inspect_ast_edges_for_qualname(
    context: _InspectAstContext,
    *,
    ast_rows: Sequence[Mapping[str, object]],
    seen: set[tuple[str, str]],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for ast_row in ast_rows:
        node_hash = _coerce_str(ast_row.get("hash"))
        if node_hash is None:
            continue
        key = (context.object_id, node_hash)
        if key in seen:
            continue
        extras = {
            "match_kind": "QUALNAME",
            "ast_kind": ast_row.get("node_type"),
            "match_confidence": 0.6,
        }
        edges.append(
            _inspect_ast_edge_row(
                context,
                node_hash=node_hash,
                rel_path=ast_row.get("path"),
                extras=extras,
            )
        )
        seen.add(key)
    return edges


_INSPECT_REQUIRED_COLUMNS = frozenset({"repo", "commit", "object_id", "module_name", "qualname"})
_INSPECT_SOURCE_REQUIRED_COLUMNS = frozenset({"object_id", "file_name", "start_line", "line_count"})
_INSPECT_AST_REQUIRED_COLUMNS = frozenset(
    {
        "hash",
        "qualname",
        "node_type",
        "path",
        "lineno",
        "end_lineno",
        "decorator_start_line",
        "decorator_end_line",
    }
)


def _inspect_to_ast_edges_to_cpg(
    inspect_objects: pa.Table,
    inspect_source: pa.Table,
    ast_nodes: pa.Table,
) -> pa.Table:
    if (
        not _INSPECT_REQUIRED_COLUMNS.issubset(inspect_objects.column_names)
        or not _INSPECT_SOURCE_REQUIRED_COLUMNS.issubset(inspect_source.column_names)
        or not _INSPECT_AST_REQUIRED_COLUMNS.issubset(ast_nodes.column_names)
    ):
        return _empty_table(_CPG_EDGE_COLUMNS)
    ast_rows = _collect_rows(
        ast_nodes,
        columns=(
            "path",
            "hash",
            "node_type",
            "qualname",
            "lineno",
            "end_lineno",
            "decorator_start_line",
            "decorator_end_line",
        ),
    )
    if not ast_rows:
        return _empty_table(_CPG_EDGE_COLUMNS)
    ast_index = _inspect_ast_indices(ast_rows)
    source_rows = _collect_rows(
        inspect_source,
        columns=("object_id", "file_name", "start_line", "line_count"),
    )
    sources_by_object = _inspect_sources_by_object(source_rows)
    edges: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for row in _table_rows(inspect_objects):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        object_id = _coerce_str(row.get("object_id"))
        if _has_missing(repo, commit, object_id):
            continue
        context = _InspectAstContext(
            repo=cast("str", repo),
            commit=cast("str", commit),
            object_id=cast("str", object_id),
        )
        source = sources_by_object.get(context.object_id)
        if source is not None:
            edges.extend(
                _inspect_ast_edges_for_source(
                    context,
                    source=source,
                    ast_index=ast_index,
                    seen=seen,
                )
            )
        full_qualname = _inspect_full_qualname(
            _coerce_str(row.get("module_name")),
            _coerce_str(row.get("qualname")),
        )
        if full_qualname is None:
            continue
        matches = ast_index.by_qualname.get(full_qualname)
        if not matches:
            continue
        edges.extend(
            _inspect_ast_edges_for_qualname(
                context,
                ast_rows=matches,
                seen=seen,
            )
        )
    return _edge_rows_to_lazyframe(edges)


def _inspect_to_scip_edges_to_cpg(
    inspect_objects: pa.Table,
    scip_symbols: pa.Table,
) -> pa.Table:
    required_inspect = {"repo", "commit", "object_id", "module_name", "qualname"}
    required_symbols = {"repo", "commit", "symbol", "display_name"}
    if not required_inspect.issubset(inspect_objects.column_names) or not required_symbols.issubset(
        scip_symbols.column_names
    ):
        return _empty_table(_CPG_EDGE_COLUMNS)
    symbols_by_key: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in _table_rows(scip_symbols):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        display_name = _coerce_str(row.get("display_name"))
        if _has_missing(repo, commit, display_name):
            continue
        symbols_by_key[cast("str", repo), cast("str", commit), cast("str", display_name)].append(
            row
        )
    edges: list[dict[str, object]] = []
    for row in _table_rows(inspect_objects):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        object_id = _coerce_str(row.get("object_id"))
        if _has_missing(repo, commit, object_id):
            continue
        full_qualname = _inspect_full_qualname(
            _coerce_str(row.get("module_name")),
            _coerce_str(row.get("qualname")),
        )
        if full_qualname is None:
            continue
        matches = symbols_by_key.get((cast("str", repo), cast("str", commit), full_qualname))
        if not matches:
            continue
        for symbol_row in matches:
            symbol = _coerce_str(symbol_row.get("symbol"))
            if symbol is None:
                continue
            extras = {"match_kind": "display_name"}
            ordinal = _stable_ordinal(
                "graph.cpg_edges_inspect_symbol",
                {"object_id": object_id, "symbol": symbol},
            )
            edges.append(
                {
                    "repo": repo,
                    "commit": commit,
                    "src_cpg_node_id": _stable_cpg_id(
                        PY_INSPECT_OBJECTS_TABLE_KEY,
                        {"repo": repo, "commit": commit, "object_id": object_id},
                    ),
                    "dst_cpg_node_id": _stable_cpg_id(
                        SCIP_SYMBOLS_TABLE_KEY,
                        {"repo": repo, "commit": commit, "symbol": symbol},
                    ),
                    "edge_kind": "INSPECT_SYMBOL",
                    "edge_layer": "SYMBOL",
                    "rel_path": None,
                    "ordinal": ordinal,
                    "extras_json": _row_to_payload(extras),
                }
            )
    return _edge_rows_to_lazyframe(edges)


def _py_inspect_class_mro_edges_to_cpg(class_mro: pa.Table) -> pa.Table:
    rows = _collect_rows(
        class_mro,
        columns=(
            "repo",
            "commit",
            "class_object_id",
            "mro_index",
            "base_object_id",
            "base_kind",
            "status",
        ),
    )
    if not rows:
        return _empty_table(_CPG_EDGE_COLUMNS)
    edges: list[dict[str, object]] = []
    for row in rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        class_object_id = _coerce_str(row.get("class_object_id"))
        base_object_id = _coerce_str(row.get("base_object_id"))
        mro_index = _coerce_int(row.get("mro_index"))
        if _has_missing(repo, commit, class_object_id, base_object_id, mro_index):
            continue
        src_cpg_node_id = _stable_cpg_id(
            PY_INSPECT_OBJECTS_TABLE_KEY,
            {"repo": repo, "commit": commit, "object_id": class_object_id},
        )
        dst_cpg_node_id = _stable_cpg_id(
            PY_INSPECT_OBJECTS_TABLE_KEY,
            {"repo": repo, "commit": commit, "object_id": base_object_id},
        )
        extras = {
            "mro_index": mro_index,
            "base_kind": row.get("base_kind"),
            "status": row.get("status"),
        }
        ordinal = _stable_ordinal(
            "graph.cpg_edges_inspect_class_mro",
            {
                "class_object_id": class_object_id,
                "base_object_id": base_object_id,
                "mro_index": mro_index,
            },
        )
        edges.append(
            {
                "repo": repo,
                "commit": commit,
                "src_cpg_node_id": src_cpg_node_id,
                "dst_cpg_node_id": dst_cpg_node_id,
                "edge_kind": "INHERITS",
                "edge_layer": "SYMBOL",
                "rel_path": None,
                "ordinal": ordinal,
                "extras_json": _row_to_payload(extras),
            }
        )
    return _edge_rows_to_lazyframe(edges)


def _py_inspect_class_attr_edges_to_cpg(class_attrs: pa.Table) -> pa.Table:
    rows = _collect_rows(
        class_attrs,
        columns=(
            "repo",
            "commit",
            "class_object_id",
            "attr_name",
            "attr_kind",
            "defining_object_id",
            "value_kind",
            "value_object_id",
            "desc_is_data",
            "desc_is_methoddesc",
            "desc_is_getset",
            "desc_is_member",
            "status",
        ),
    )
    if not rows:
        return _empty_table(_CPG_EDGE_COLUMNS)
    edges: list[dict[str, object]] = []
    for row in rows:
        context = _inspect_class_attr_context(row)
        if context is None:
            continue
        edges.extend(_inspect_class_attr_edges(context))
    return _edge_rows_to_lazyframe(edges)


@dataclass(frozen=True)
class _InspectClassAttrContext:
    repo: str
    commit: str
    class_object_id: str
    attr_name: str
    defining_object_id: str | None
    value_object_id: str | None
    extras: dict[str, object]
    src_cpg_node_id: int
    is_descriptor: bool


def _inspect_class_attr_context(row: Mapping[str, object]) -> _InspectClassAttrContext | None:
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    class_object_id = _coerce_str(row.get("class_object_id"))
    attr_name = _coerce_str(row.get("attr_name"))
    defining_object_id = _coerce_str(row.get("defining_object_id"))
    value_object_id = _coerce_str(row.get("value_object_id"))
    if _has_missing(repo, commit, class_object_id, attr_name):
        return None
    extras = {
        "attr_name": attr_name,
        "attr_kind": row.get("attr_kind"),
        "defining_object_id": defining_object_id,
        "value_kind": row.get("value_kind"),
        "desc_is_data": row.get("desc_is_data"),
        "desc_is_methoddesc": row.get("desc_is_methoddesc"),
        "desc_is_getset": row.get("desc_is_getset"),
        "desc_is_member": row.get("desc_is_member"),
        "status": row.get("status"),
    }
    src_cpg_node_id = _stable_cpg_id(
        PY_INSPECT_OBJECTS_TABLE_KEY,
        {"repo": repo, "commit": commit, "object_id": class_object_id},
    )
    is_descriptor = any(
        _coerce_bool(row.get(flag))
        for flag in ("desc_is_data", "desc_is_methoddesc", "desc_is_getset", "desc_is_member")
    )
    return _InspectClassAttrContext(
        repo=cast("str", repo),
        commit=cast("str", commit),
        class_object_id=cast("str", class_object_id),
        attr_name=cast("str", attr_name),
        defining_object_id=defining_object_id,
        value_object_id=value_object_id,
        extras=extras,
        src_cpg_node_id=src_cpg_node_id,
        is_descriptor=is_descriptor,
    )


def _inspect_class_attr_edge(
    context: _InspectClassAttrContext,
    *,
    target_object_id: str,
    edge_kind: str,
    ordinal_values: Mapping[str, object],
) -> dict[str, object]:
    dst_cpg_node_id = _stable_cpg_id(
        PY_INSPECT_OBJECTS_TABLE_KEY,
        {"repo": context.repo, "commit": context.commit, "object_id": target_object_id},
    )
    ordinal = _stable_ordinal("graph.cpg_edges_inspect_class_attr", ordinal_values)
    return {
        "repo": context.repo,
        "commit": context.commit,
        "src_cpg_node_id": context.src_cpg_node_id,
        "dst_cpg_node_id": dst_cpg_node_id,
        "edge_kind": edge_kind,
        "edge_layer": "SYMBOL",
        "rel_path": None,
        "ordinal": ordinal,
        "extras_json": _row_to_payload(context.extras),
    }


def _inspect_class_attr_edges(context: _InspectClassAttrContext) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    if context.value_object_id is not None:
        edges.append(
            _inspect_class_attr_edge(
                context,
                target_object_id=context.value_object_id,
                edge_kind="DECLARES_ATTR",
                ordinal_values={
                    "class_object_id": context.class_object_id,
                    "attr_name": context.attr_name,
                    "value_object_id": context.value_object_id,
                    "edge_kind": "DECLARES_ATTR",
                },
            )
        )
    if (
        context.defining_object_id is not None
        and context.defining_object_id != context.class_object_id
    ):
        edges.append(
            _inspect_class_attr_edge(
                context,
                target_object_id=context.defining_object_id,
                edge_kind="OVERRIDES",
                ordinal_values={
                    "class_object_id": context.class_object_id,
                    "attr_name": context.attr_name,
                    "defining_object_id": context.defining_object_id,
                    "edge_kind": "OVERRIDES",
                },
            )
        )
    if context.value_object_id is not None and context.is_descriptor:
        edges.append(
            _inspect_class_attr_edge(
                context,
                target_object_id=context.value_object_id,
                edge_kind="DESCRIPTOR",
                ordinal_values={
                    "class_object_id": context.class_object_id,
                    "attr_name": context.attr_name,
                    "value_object_id": context.value_object_id,
                    "edge_kind": "DESCRIPTOR",
                },
            )
        )
    return edges


def _runtime_state_extras(row: Mapping[str, object]) -> dict[str, object]:
    return {
        "state_kind": row.get("state_kind"),
        "state": row.get("state"),
        "object_kind": row.get("object_kind"),
        "frame_file": row.get("frame_file"),
        "frame_module": row.get("frame_module"),
        "frame_line": row.get("frame_line"),
        "frame_offset": row.get("frame_offset"),
        "status": row.get("status"),
    }


def _runtime_state_has_state_edges(runtime_state: pa.Table) -> pa.Table:
    edges: list[dict[str, object]] = []
    for row in _table_rows(runtime_state):
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        object_id = _coerce_str(row.get("object_id"))
        frame_object_id = _coerce_str(row.get("frame_object_id"))
        if _has_missing(repo, commit, object_id, frame_object_id):
            continue
        extras = _runtime_state_extras(row)
        ordinal = _stable_ordinal(
            "graph.cpg_edges_inspect_runtime_state",
            {
                "object_id": object_id,
                "state_kind": row.get("state_kind"),
                "frame_object_id": frame_object_id,
            },
        )
        edges.append(
            {
                "repo": repo,
                "commit": commit,
                "src_cpg_node_id": _stable_cpg_id(
                    PY_INSPECT_OBJECTS_TABLE_KEY,
                    {"repo": repo, "commit": commit, "object_id": object_id},
                ),
                "dst_cpg_node_id": _stable_cpg_id(
                    PY_INSPECT_OBJECTS_TABLE_KEY,
                    {"repo": repo, "commit": commit, "object_id": frame_object_id},
                ),
                "edge_kind": "HAS_STATE",
                "edge_layer": "FLOW",
                "rel_path": None,
                "ordinal": ordinal,
                "extras_json": _row_to_payload(extras),
            }
        )
    return _edge_rows_to_lazyframe(edges)


def _runtime_state_frame_name(row: Mapping[str, object]) -> str | None:
    return _coerce_str(row.get("frame_code_qualname")) or _coerce_str(row.get("frame_code_name"))


@dataclass(frozen=True, slots=True)
class _RuntimeStateInstrContext:
    repo: str
    commit: str
    object_id: str
    state_kind: str | None
    frame_qualpath: str
    frame_offset: int
    extras: dict[str, object]


def _runtime_state_context(row: Mapping[str, object]) -> _RuntimeStateInstrContext | None:
    repo = _coerce_str(row.get("repo"))
    commit = _coerce_str(row.get("commit"))
    object_id = _coerce_str(row.get("object_id"))
    if _has_missing(repo, commit, object_id):
        return None
    frame_module = _coerce_str(row.get("frame_module"))
    frame_name = _runtime_state_frame_name(row)
    frame_offset = _coerce_int(row.get("frame_offset"))
    if _has_missing(frame_module, frame_name, frame_offset):
        return None
    frame_offset_value = cast("int", frame_offset)
    if frame_offset_value < 0:
        return None
    frame_qualpath = f"{frame_module}::{frame_name}"
    return _RuntimeStateInstrContext(
        repo=cast("str", repo),
        commit=cast("str", commit),
        object_id=cast("str", object_id),
        state_kind=_coerce_str(row.get("state_kind")),
        frame_qualpath=frame_qualpath,
        frame_offset=frame_offset_value,
        extras=_runtime_state_extras(row),
    )


def _runtime_state_instr_edges(
    runtime_state: pa.Table,
    code_units: pa.Table,
    instructions: pa.Table,
) -> pa.Table:
    code_unit_rows = _collect_rows(
        code_units,
        columns=("repo", "commit", "rel_path", "code_unit_id", "qualpath"),
    )
    instr_rows = _collect_rows(
        instructions,
        columns=("repo", "commit", "rel_path", "code_unit_id", "instr_id", "offset"),
    )
    if not code_unit_rows or not instr_rows:
        return _empty_table(_CPG_EDGE_COLUMNS)
    units_by_key = _runtime_units_index(code_unit_rows)
    instr_by_key = _runtime_instr_index(instr_rows)
    edges: list[dict[str, object]] = []
    for row in _table_rows(runtime_state):
        edges.extend(_runtime_state_instr_edge_rows(row, units_by_key, instr_by_key))
    return _edge_rows_to_lazyframe(edges)


def _runtime_units_index(
    code_unit_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str], list[tuple[str, str]]]:
    units_by_key: dict[tuple[str, str, str], list[tuple[str, str]]] = defaultdict(list)
    for row in code_unit_rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        qualpath = _coerce_str(row.get("qualpath"))
        rel_path = _coerce_str(row.get("rel_path"))
        code_unit_id = _coerce_str(row.get("code_unit_id"))
        if _has_missing(repo, commit, qualpath, rel_path, code_unit_id):
            continue
        units_by_key[cast("str", repo), cast("str", commit), cast("str", qualpath)].append(
            (cast("str", rel_path), cast("str", code_unit_id))
        )
    return units_by_key


def _runtime_instr_index(
    instr_rows: Sequence[Mapping[str, object]],
) -> dict[tuple[str, str, str, str, int], list[str]]:
    instr_by_key: dict[tuple[str, str, str, str, int], list[str]] = defaultdict(list)
    for row in instr_rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        rel_path = _coerce_str(row.get("rel_path"))
        code_unit_id = _coerce_str(row.get("code_unit_id"))
        instr_id = _coerce_str(row.get("instr_id"))
        offset = _coerce_int(row.get("offset"))
        if _has_missing(repo, commit, rel_path, code_unit_id, instr_id, offset):
            continue
        key = (
            cast("str", repo),
            cast("str", commit),
            cast("str", rel_path),
            cast("str", code_unit_id),
            cast("int", offset),
        )
        instr_by_key[key].append(cast("str", instr_id))
    return instr_by_key


def _runtime_state_instr_edge_rows(
    row: Mapping[str, object],
    units_by_key: Mapping[tuple[str, str, str], Sequence[tuple[str, str]]],
    instr_by_key: Mapping[tuple[str, str, str, str, int], Sequence[str]],
) -> list[dict[str, object]]:
    context = _runtime_state_context(row)
    if context is None:
        return []
    unit_matches = units_by_key.get((context.repo, context.commit, context.frame_qualpath))
    if not unit_matches:
        return []
    return _runtime_state_edges_for_units(context, unit_matches, instr_by_key)


def _runtime_state_edges_for_units(
    context: _RuntimeStateInstrContext,
    unit_matches: Sequence[tuple[str, str]],
    instr_by_key: Mapping[tuple[str, str, str, str, int], Sequence[str]],
) -> list[dict[str, object]]:
    edge_kind = "TRACEBACK_AT_INSTR" if context.state_kind == "traceback" else "FRAME_AT_INSTR"
    edges: list[dict[str, object]] = []
    for rel_path, code_unit_id in unit_matches:
        instr_matches = instr_by_key.get(
            (context.repo, context.commit, rel_path, code_unit_id, context.frame_offset)
        )
        if not instr_matches:
            continue
        edges.extend(
            _runtime_state_edges_for_instr(
                context,
                edge_kind=edge_kind,
                rel_path=rel_path,
                code_unit_id=code_unit_id,
                instr_matches=instr_matches,
            )
        )
    return edges


def _runtime_state_edges_for_instr(
    context: _RuntimeStateInstrContext,
    *,
    edge_kind: str,
    rel_path: str,
    code_unit_id: str,
    instr_matches: Sequence[str],
) -> list[dict[str, object]]:
    edges: list[dict[str, object]] = []
    for instr_id in instr_matches:
        ordinal = _stable_ordinal(
            "graph.cpg_edges_inspect_runtime_state",
            {
                "object_id": context.object_id,
                "state_kind": context.state_kind,
                "instr_id": instr_id,
                "frame_offset": context.frame_offset,
            },
        )
        edges.append(
            {
                "repo": context.repo,
                "commit": context.commit,
                "src_cpg_node_id": _stable_cpg_id(
                    PY_INSPECT_OBJECTS_TABLE_KEY,
                    {
                        "repo": context.repo,
                        "commit": context.commit,
                        "object_id": context.object_id,
                    },
                ),
                "dst_cpg_node_id": _instruction_cpg_id(
                    repo=context.repo,
                    commit=context.commit,
                    rel_path=rel_path,
                    code_unit_id=code_unit_id,
                    instr_id=instr_id,
                ),
                "edge_kind": edge_kind,
                "edge_layer": "FLOW",
                "rel_path": rel_path,
                "ordinal": ordinal,
                "extras_json": _row_to_payload(context.extras),
            }
        )
    return edges


def _py_inspect_runtime_state_edges_to_cpg(
    runtime_state: pa.Table,
    code_units: pa.Table,
    instructions: pa.Table,
) -> pa.Table:
    required_state = {
        "repo",
        "commit",
        "object_id",
        "object_kind",
        "state_kind",
        "state",
        "frame_object_id",
        "frame_file",
        "frame_module",
        "frame_code_qualname",
        "frame_code_name",
        "frame_line",
        "frame_offset",
        "status",
    }
    required_units = {"repo", "commit", "rel_path", "code_unit_id", "qualpath"}
    required_instr = {"repo", "commit", "rel_path", "code_unit_id", "instr_id", "offset"}
    if (
        not required_state.issubset(runtime_state.column_names)
        or not required_units.issubset(code_units.column_names)
        or not required_instr.issubset(instructions.column_names)
    ):
        return _empty_table(_CPG_EDGE_COLUMNS)
    has_state_edges = _runtime_state_has_state_edges(runtime_state)
    instr_edges = _runtime_state_instr_edges(runtime_state, code_units, instructions)
    tables = [table for table in (has_state_edges, instr_edges) if table.num_rows > 0]
    if not tables:
        return _empty_table(_CPG_EDGE_COLUMNS)
    combined = concat_tables_unified(tables)
    return _select_edge_columns(combined)


def _py_inspect_unwrap_edges_to_cpg(unwrap_hops: pa.Table) -> pa.Table:
    rows = _collect_rows(
        unwrap_hops,
        columns=(
            "repo",
            "commit",
            "root_object_id",
            "hop",
            "object_id",
            "has_wrapped",
            "has_signature_override",
            "stop_reason",
        ),
    )
    if not rows:
        return _empty_table(_CPG_EDGE_COLUMNS)
    grouped: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        repo = _coerce_str(row.get("repo"))
        commit = _coerce_str(row.get("commit"))
        root_object_id = _coerce_str(row.get("root_object_id"))
        hop = _coerce_int(row.get("hop"))
        obj_id = _coerce_str(row.get("object_id"))
        if _has_missing(repo, commit, root_object_id, hop, obj_id):
            continue
        grouped[cast("str", repo), cast("str", commit), cast("str", root_object_id)].append(row)
    edges: list[dict[str, object]] = []
    for (repo, commit, root_object_id), items in grouped.items():
        items.sort(key=lambda item: _coerce_int(item.get("hop")) or 0)
        for idx in range(len(items) - 1):
            src_id = _coerce_str(items[idx].get("object_id"))
            dst_id = _coerce_str(items[idx + 1].get("object_id"))
            if _has_missing(src_id, dst_id):
                continue
            edge_kind = "DECORATES" if idx == 0 else "WRAPS"
            src_pk = {"repo": repo, "commit": commit, "object_id": src_id}
            dst_pk = {"repo": repo, "commit": commit, "object_id": dst_id}
            extras = {
                "root_object_id": root_object_id,
                "hop": items[idx].get("hop"),
                "has_wrapped": items[idx].get("has_wrapped"),
                "has_signature_override": items[idx].get("has_signature_override"),
                "stop_reason": items[idx].get("stop_reason"),
                "edge_kind": edge_kind,
            }
            hop_value = _coerce_int(items[idx].get("hop"))
            ordinal = _stable_ordinal(
                "graph.cpg_edges_inspect_wraps",
                {
                    "root_object_id": root_object_id,
                    "hop": hop_value,
                    "edge_kind": edge_kind,
                },
            )
            edges.append(
                {
                    "repo": repo,
                    "commit": commit,
                    "src_cpg_node_id": _stable_cpg_id(PY_INSPECT_OBJECTS_TABLE_KEY, src_pk),
                    "dst_cpg_node_id": _stable_cpg_id(PY_INSPECT_OBJECTS_TABLE_KEY, dst_pk),
                    "edge_kind": edge_kind,
                    "edge_layer": "SYMBOL",
                    "rel_path": None,
                    "ordinal": ordinal,
                    "extras_json": _row_to_payload(extras),
                }
            )
    return _edge_rows_to_lazyframe(edges)


def cpg_edge_symbol_inputs(
    q__core__syntax_edges: InferableTabularInput,
    q__core__scip_occurrence_syntax_xref: InferableTabularInput,
    q__core__scip_occurrence_span_xref: InferableTabularInput,
    q__core__scip_symbol_relationships: InferableTabularInput,
    q__core__scip_symbol_goid_xref: InferableTabularInput,
) -> _CpgSymbolInputs:
    """Collect symbol-layer inputs for CPG edge assembly.

    Returns
    -------
    _CpgSymbolInputs
        Symbol inputs for CPG edge assembly.
    """
    return _CpgSymbolInputs(
        syntax_edges=tabular_to_table(q__core__syntax_edges),
        occ_syntax=tabular_to_table(q__core__scip_occurrence_syntax_xref),
        occ_span=tabular_to_table(q__core__scip_occurrence_span_xref),
        symbol_rels=tabular_to_table(q__core__scip_symbol_relationships),
        symbol_goid=tabular_to_table(q__core__scip_symbol_goid_xref),
    )


def cpg_edge_flow_inputs(
    q__core__goids: InferableTabularInput,
    q__graph__cfg_edges: InferableTabularInput,
    q__graph__dfg_edges: InferableTabularInput,
    q__graph__cfg_blocks: InferableTabularInput,
    q__graph__cdg_edges: InferableTabularInput,
) -> _CpgFlowInputs:
    """Collect flow-layer inputs for CPG edge assembly.

    Returns
    -------
    _CpgFlowInputs
        Flow inputs for CPG edge assembly.
    """
    return _CpgFlowInputs(
        goids=tabular_to_table(q__core__goids),
        cfg_edges=tabular_to_table(q__graph__cfg_edges),
        dfg_edges=tabular_to_table(q__graph__dfg_edges),
        cfg_blocks=tabular_to_table(q__graph__cfg_blocks),
        cdg_edges=tabular_to_table(q__graph__cdg_edges),
    )


def cpg_edge_link_inputs(
    q__graph__call_graph_edges: InferableTabularInput,
    q__graph__import_graph_edges: InferableTabularInput,
) -> _CpgLinkInputs:
    """Collect graph-link inputs for CPG edge assembly.

    Returns
    -------
    _CpgLinkInputs
        Graph-link inputs for CPG edge assembly.
    """
    return _CpgLinkInputs(
        call_edges=tabular_to_table(q__graph__call_graph_edges),
        import_edges=tabular_to_table(q__graph__import_graph_edges),
    )


def cpg_edge_call_wiring_inputs(
    q__graph__cpg_edges_calls: InferableTabularInput,
    q__graph__cpg_edges_arg_to_param: InferableTabularInput,
    q__graph__cpg_edges_ret_to_call: InferableTabularInput,
) -> _CpgCallWiringInputs:
    """Collect call wiring inputs for CPG edge assembly.

    Returns
    -------
    _CpgCallWiringInputs
        Call wiring inputs for CPG edge assembly.
    """
    return _CpgCallWiringInputs(
        call_edges=tabular_to_table(q__graph__cpg_edges_calls),
        arg_to_param_edges=tabular_to_table(q__graph__cpg_edges_arg_to_param),
        ret_to_call_edges=tabular_to_table(q__graph__cpg_edges_ret_to_call),
    )


def cpg_edge_syntax_node_inputs(
    q__core__syntax_nodes: InferableTabularInput,
) -> _CpgSyntaxNodeInputs:
    """Collect syntax node inputs for CPG edge assembly.

    Returns
    -------
    _CpgSyntaxNodeInputs
        Syntax node inputs for CPG edge assembly.
    """
    return _CpgSyntaxNodeInputs(
        syntax_nodes=tabular_to_table(q__core__syntax_nodes),
    )


def cpg_edge_overlay_scope_inputs(
    q__core__py_sym_scopes: InferableTabularInput,
    q__core__py_sym_bindings: InferableTabularInput,
    q__core__py_sym_scope_edges: InferableTabularInput,
    q__core__py_sym_namespace_edges: InferableTabularInput,
    q__core__py_sym_resolution_edges: InferableTabularInput,
) -> _CpgOverlayScopeInputs:
    """Collect scope overlay inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayScopeInputs
        Scope overlay inputs for CPG edge assembly.
    """
    return _CpgOverlayScopeInputs(
        py_sym_scopes=tabular_to_table(q__core__py_sym_scopes),
        py_sym_bindings=tabular_to_table(q__core__py_sym_bindings),
        py_sym_scope_edges=tabular_to_table(q__core__py_sym_scope_edges),
        py_sym_namespace_edges=tabular_to_table(q__core__py_sym_namespace_edges),
        py_sym_resolution_edges=tabular_to_table(q__core__py_sym_resolution_edges),
    )


def cpg_edge_overlay_symbol_inputs(
    q__core__ast_nodes: InferableTabularInput,
    q__core__scip_symbol_information: InferableTabularInput,
    cpg_edge_overlay_scope_inputs: _CpgOverlayScopeInputs,
) -> _CpgOverlaySymbolInputs:
    """Collect symbol overlay inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlaySymbolInputs
        Symbol overlay inputs for CPG edge assembly.
    """
    return _CpgOverlaySymbolInputs(
        ast_nodes=tabular_to_table(q__core__ast_nodes),
        scip_symbols=tabular_to_table(q__core__scip_symbol_information),
        scope_inputs=cpg_edge_overlay_scope_inputs,
    )


def cpg_edge_overlay_bytecode_inputs(
    q__core__py_bc_code_units: InferableTabularInput,
    q__core__py_bc_instructions: InferableTabularInput,
    q__core__py_bc_blocks: InferableTabularInput,
    q__core__py_bc_cfg_edges: InferableTabularInput,
    q__core__py_bc_defuse_events: InferableTabularInput,
) -> _CpgOverlayBytecodeInputs:
    """Collect bytecode overlay inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayBytecodeInputs
        Bytecode overlay inputs for CPG edge assembly.
    """
    return _CpgOverlayBytecodeInputs(
        py_bc_code_units=tabular_to_table(q__core__py_bc_code_units),
        py_bc_instructions=tabular_to_table(q__core__py_bc_instructions),
        py_bc_blocks=tabular_to_table(q__core__py_bc_blocks),
        py_bc_cfg_edges=tabular_to_table(q__core__py_bc_cfg_edges),
        py_bc_defuse_events=tabular_to_table(q__core__py_bc_defuse_events),
    )


def cpg_edge_overlay_syntax_call_inputs(
    q__core__syntax_calls: InferableTabularInput,
    q__core__syntax_call_args: InferableTabularInput,
) -> _CpgOverlaySyntaxCallInputs:
    """Collect syntax call inputs for CPG inspect overlays.

    Returns
    -------
    _CpgOverlaySyntaxCallInputs
        Syntax call inputs for inspect overlays.
    """
    return _CpgOverlaySyntaxCallInputs(
        syntax_calls=tabular_to_table(q__core__syntax_calls),
        syntax_call_args=tabular_to_table(q__core__syntax_call_args),
    )


def cpg_edge_overlay_inspect_core_inputs(
    q__core__py_inspect_objects: InferableTabularInput,
    q__core__py_inspect_class_mro: InferableTabularInput,
    q__core__py_inspect_class_attrs: InferableTabularInput,
    q__core__py_inspect_unwrap_hops: InferableTabularInput,
    q__core__py_inspect_source: InferableTabularInput,
) -> _CpgOverlayInspectCoreInputs:
    """Collect core inspect inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayInspectCoreInputs
        Core inspect inputs for CPG edge assembly.
    """
    return _CpgOverlayInspectCoreInputs(
        py_inspect_objects=tabular_to_table(q__core__py_inspect_objects),
        py_inspect_class_mro=tabular_to_table(q__core__py_inspect_class_mro),
        py_inspect_class_attrs=tabular_to_table(q__core__py_inspect_class_attrs),
        py_inspect_unwrap_hops=tabular_to_table(q__core__py_inspect_unwrap_hops),
        py_inspect_source=tabular_to_table(q__core__py_inspect_source),
    )


def cpg_edge_overlay_inspect_runtime_inputs(
    q__core__py_inspect_signatures: InferableTabularInput,
    q__core__py_inspect_signature_params: InferableTabularInput,
    q__core__py_inspect_runtime_state: InferableTabularInput,
) -> _CpgOverlayInspectRuntimeInputs:
    """Collect runtime inspect inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayInspectRuntimeInputs
        Runtime inspect inputs for CPG edge assembly.
    """
    return _CpgOverlayInspectRuntimeInputs(
        py_inspect_signatures=tabular_to_table(q__core__py_inspect_signatures),
        py_inspect_signature_params=tabular_to_table(q__core__py_inspect_signature_params),
        py_inspect_runtime_state=tabular_to_table(q__core__py_inspect_runtime_state),
    )


def cpg_edge_overlay_inspect_inputs(
    cpg_edge_overlay_inspect_core_inputs: _CpgOverlayInspectCoreInputs,
    cpg_edge_overlay_inspect_runtime_inputs: _CpgOverlayInspectRuntimeInputs,
    cpg_edge_overlay_syntax_call_inputs: _CpgOverlaySyntaxCallInputs,
) -> _CpgOverlayInspectInputs:
    """Collect inspect overlay inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayInspectInputs
        Inspect overlay inputs for CPG edge assembly.
    """
    return _CpgOverlayInspectInputs(
        py_inspect_objects=cpg_edge_overlay_inspect_core_inputs.py_inspect_objects,
        py_inspect_class_mro=cpg_edge_overlay_inspect_core_inputs.py_inspect_class_mro,
        py_inspect_class_attrs=cpg_edge_overlay_inspect_core_inputs.py_inspect_class_attrs,
        py_inspect_unwrap_hops=cpg_edge_overlay_inspect_core_inputs.py_inspect_unwrap_hops,
        py_inspect_signatures=cpg_edge_overlay_inspect_runtime_inputs.py_inspect_signatures,
        py_inspect_signature_params=(
            cpg_edge_overlay_inspect_runtime_inputs.py_inspect_signature_params
        ),
        py_inspect_source=cpg_edge_overlay_inspect_core_inputs.py_inspect_source,
        py_inspect_runtime_state=cpg_edge_overlay_inspect_runtime_inputs.py_inspect_runtime_state,
        syntax_calls=cpg_edge_overlay_syntax_call_inputs.syntax_calls,
        syntax_call_args=cpg_edge_overlay_syntax_call_inputs.syntax_call_args,
    )


def cpg_edge_overlay_inputs(
    cpg_edge_overlay_symbol_inputs: _CpgOverlaySymbolInputs,
    cpg_edge_overlay_bytecode_inputs: _CpgOverlayBytecodeInputs,
    cpg_edge_overlay_inspect_inputs: _CpgOverlayInspectInputs,
) -> _CpgOverlayEdgeInputs:
    """Collect overlay inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayEdgeInputs
        Overlay inputs for CPG edge assembly.
    """
    return _CpgOverlayEdgeInputs(
        ast_nodes=cpg_edge_overlay_symbol_inputs.ast_nodes,
        syntax_calls=cpg_edge_overlay_inspect_inputs.syntax_calls,
        syntax_call_args=cpg_edge_overlay_inspect_inputs.syntax_call_args,
        scip_symbols=cpg_edge_overlay_symbol_inputs.scip_symbols,
        py_sym_scopes=cpg_edge_overlay_symbol_inputs.scope_inputs.py_sym_scopes,
        py_sym_bindings=cpg_edge_overlay_symbol_inputs.scope_inputs.py_sym_bindings,
        py_sym_scope_edges=cpg_edge_overlay_symbol_inputs.scope_inputs.py_sym_scope_edges,
        py_sym_namespace_edges=cpg_edge_overlay_symbol_inputs.scope_inputs.py_sym_namespace_edges,
        py_sym_resolution_edges=cpg_edge_overlay_symbol_inputs.scope_inputs.py_sym_resolution_edges,
        py_bc_code_units=cpg_edge_overlay_bytecode_inputs.py_bc_code_units,
        py_bc_instructions=cpg_edge_overlay_bytecode_inputs.py_bc_instructions,
        py_bc_blocks=cpg_edge_overlay_bytecode_inputs.py_bc_blocks,
        py_bc_cfg_edges=cpg_edge_overlay_bytecode_inputs.py_bc_cfg_edges,
        py_bc_defuse_events=cpg_edge_overlay_bytecode_inputs.py_bc_defuse_events,
        py_inspect_objects=cpg_edge_overlay_inspect_inputs.py_inspect_objects,
        py_inspect_class_mro=cpg_edge_overlay_inspect_inputs.py_inspect_class_mro,
        py_inspect_class_attrs=cpg_edge_overlay_inspect_inputs.py_inspect_class_attrs,
        py_inspect_unwrap_hops=cpg_edge_overlay_inspect_inputs.py_inspect_unwrap_hops,
        py_inspect_signatures=cpg_edge_overlay_inspect_inputs.py_inspect_signatures,
        py_inspect_signature_params=cpg_edge_overlay_inspect_inputs.py_inspect_signature_params,
        py_inspect_source=cpg_edge_overlay_inspect_inputs.py_inspect_source,
        py_inspect_runtime_state=cpg_edge_overlay_inspect_inputs.py_inspect_runtime_state,
    )


def cpg_edge_core_inputs(
    cpg_edge_symbol_inputs: _CpgSymbolInputs,
    cpg_edge_flow_inputs: _CpgFlowInputs,
    cpg_edge_link_inputs: _CpgLinkInputs,
    cpg_edge_call_wiring_inputs: _CpgCallWiringInputs,
    cpg_edge_syntax_node_inputs: _CpgSyntaxNodeInputs,
) -> _CpgEdgeCoreInputs:
    """Collect core edge inputs for CPG edge assembly.

    Returns
    -------
    _CpgEdgeCoreInputs
        Core inputs for CPG edge assembly.
    """
    return _CpgEdgeCoreInputs(
        symbol=cpg_edge_symbol_inputs,
        flow=cpg_edge_flow_inputs,
        link=cpg_edge_link_inputs,
        call_wiring=cpg_edge_call_wiring_inputs,
        syntax_nodes=cpg_edge_syntax_node_inputs,
    )


def _overlay_frames(
    *,
    overlay_inputs: _CpgOverlayEdgeInputs,
    overlay_options: CpgOverlayOptions,
    cpg_options: CpgOptions,
) -> list[pa.Table]:
    def _symtable_frames() -> list[pa.Table]:
        return [
            _py_sym_scope_edges_to_cpg(overlay_inputs.py_sym_scope_edges),
            _py_sym_namespace_edges_to_cpg(
                overlay_inputs.py_sym_namespace_edges,
                overlay_inputs.py_sym_bindings,
            ),
            _py_sym_binding_edges_to_cpg(overlay_inputs.py_sym_bindings),
            _py_sym_resolution_edges_to_cpg(overlay_inputs.py_sym_resolution_edges),
            _py_sym_binding_symbol_edges_to_cpg(
                overlay_inputs.py_sym_bindings,
                overlay_inputs.py_sym_scopes,
                overlay_inputs.scip_symbols,
            ),
            _ast_binding_edges_to_cpg(
                overlay_inputs.ast_nodes,
                overlay_inputs.py_sym_scopes,
                overlay_inputs.py_sym_bindings,
                overlay_inputs.py_sym_resolution_edges,
            ),
        ]

    def _bytecode_frames() -> list[pa.Table]:
        frames = [
            _py_bc_instruction_ast_edges_to_cpg(
                overlay_inputs.py_bc_instructions,
                overlay_inputs.ast_nodes,
            ),
            _py_bc_callsite_edges_to_cpg(
                overlay_inputs.py_bc_instructions,
                overlay_inputs.syntax_calls,
            ),
            _py_bc_callsite_symbol_edges_to_cpg(
                overlay_inputs.py_bc_instructions,
                overlay_inputs.syntax_calls,
                overlay_inputs.scip_symbols,
            ),
            _py_bc_cfg_edges_to_cpg(overlay_inputs.py_bc_cfg_edges),
            _py_bc_defuse_binding_edges_to_cpg(
                overlay_inputs.py_bc_defuse_events,
                overlay_inputs.py_bc_code_units,
                overlay_inputs.py_sym_scopes,
                overlay_inputs.py_sym_bindings,
                overlay_inputs.py_sym_resolution_edges,
            ),
            _py_bc_memory_edges_to_cpg(
                overlay_inputs.py_bc_defuse_events,
                overlay_inputs.py_bc_instructions,
                overlay_inputs.ast_nodes,
            ),
            _py_bc_stack_edges_to_cpg(
                overlay_inputs.py_bc_instructions,
                overlay_inputs.py_bc_blocks,
            ),
        ]
        if cpg_options.enable_reaches:
            frames.append(
                _py_bc_reaches_edges_to_cpg(
                    _PyBcReachesInputs(
                        defuse_events=overlay_inputs.py_bc_defuse_events,
                        code_units=overlay_inputs.py_bc_code_units,
                        scopes=overlay_inputs.py_sym_scopes,
                        bindings=overlay_inputs.py_sym_bindings,
                        resolution_edges=overlay_inputs.py_sym_resolution_edges,
                        blocks=overlay_inputs.py_bc_blocks,
                        cfg_edges=overlay_inputs.py_bc_cfg_edges,
                    )
                )
            )
        return frames

    def _inspect_frames() -> list[pa.Table]:
        return [
            _py_inspect_signature_edges_to_cpg(
                overlay_inputs.py_inspect_signatures,
                overlay_inputs.py_inspect_signature_params,
            ),
            _inspect_arg_to_param_edges_to_cpg(
                overlay_inputs.syntax_calls,
                overlay_inputs.syntax_call_args,
                overlay_inputs.py_inspect_objects,
                overlay_inputs.py_inspect_signatures,
                overlay_inputs.py_inspect_signature_params,
            ),
            _py_inspect_unwrap_edges_to_cpg(overlay_inputs.py_inspect_unwrap_hops),
            _py_inspect_class_mro_edges_to_cpg(overlay_inputs.py_inspect_class_mro),
            _py_inspect_class_attr_edges_to_cpg(overlay_inputs.py_inspect_class_attrs),
            _py_inspect_runtime_state_edges_to_cpg(
                overlay_inputs.py_inspect_runtime_state,
                overlay_inputs.py_bc_code_units,
                overlay_inputs.py_bc_instructions,
            ),
            _inspect_to_ast_edges_to_cpg(
                overlay_inputs.py_inspect_objects,
                overlay_inputs.py_inspect_source,
                overlay_inputs.ast_nodes,
            ),
            _inspect_to_scip_edges_to_cpg(
                overlay_inputs.py_inspect_objects,
                overlay_inputs.scip_symbols,
            ),
        ]

    registry = [
        _CpgOverlayRegistryEntry(
            name="symtable",
            enabled=overlay_options.enable_symtable,
            builder=_symtable_frames,
        ),
        _CpgOverlayRegistryEntry(
            name="bytecode",
            enabled=overlay_options.enable_bytecode,
            builder=_bytecode_frames,
        ),
        _CpgOverlayRegistryEntry(
            name="inspect",
            enabled=overlay_options.enable_inspect,
            builder=_inspect_frames,
        ),
    ]
    frames: list[pa.Table] = []
    for entry in registry:
        if not entry.enabled:
            continue
        frames.extend(entry.builder())
    return frames


def cpg_edges(
    cpg_edge_core_inputs: _CpgEdgeCoreInputs,
    cpg_edge_overlay_inputs: _CpgOverlayEdgeInputs,
    cpg__overlay_options: CpgOverlayOptions,
    cpg__options: CpgOptions,
) -> InferableTabularInput:
    """Build CPG edges from syntax, symbol, and flow sources.

    Returns
    -------
    InferableTabularInput
        Arrow reader for graph.cpg_edges.
    """
    cfg_blocks = cpg_edge_core_inputs.flow.cfg_blocks
    syntax_nodes = cpg_edge_core_inputs.syntax_nodes.syntax_nodes
    overlay_inputs = cpg_edge_overlay_inputs

    frames = [
        _syntax_edges_to_cpg(cpg_edge_core_inputs.symbol.syntax_edges),
        _scip_occurrence_edges_to_cpg(
            cpg_edge_core_inputs.symbol.occ_syntax,
            cpg_edge_core_inputs.symbol.occ_span,
        ),
        _scip_symbol_relationships_to_cpg(cpg_edge_core_inputs.symbol.symbol_rels),
        _scip_symbol_goid_edges_to_cpg(cpg_edge_core_inputs.symbol.symbol_goid),
        _call_graph_edges_to_cpg(cpg_edge_core_inputs.link.call_edges),
        _import_graph_edges_to_cpg(cpg_edge_core_inputs.link.import_edges),
        _cfg_edges_to_cpg(
            cpg_edge_core_inputs.flow.cfg_edges,
            cfg_blocks,
            cpg_edge_core_inputs.flow.goids,
        ),
        _dfg_edges_to_cpg(
            cpg_edge_core_inputs.flow.dfg_edges,
            cfg_blocks,
            cpg_edge_core_inputs.flow.goids,
        ),
        _cdg_edges_to_cpg(
            cpg_edge_core_inputs.flow.cdg_edges,
            cfg_blocks,
            cpg_edge_core_inputs.flow.goids,
        ),
        _call_wiring_calls_to_cpg(
            cpg_edge_core_inputs.call_wiring.call_edges,
            cfg_blocks,
            syntax_nodes,
        ),
        _call_wiring_arg_to_param_to_cpg(
            cpg_edge_core_inputs.call_wiring.arg_to_param_edges,
            syntax_nodes,
        ),
        _call_wiring_ret_to_call_to_cpg(
            cpg_edge_core_inputs.call_wiring.ret_to_call_edges,
            cfg_blocks,
            syntax_nodes,
        ),
    ]
    frames.extend(
        _overlay_frames(
            overlay_inputs=overlay_inputs,
            overlay_options=cpg__overlay_options,
            cpg_options=cpg__options,
        )
    )
    tables = [frame for frame in frames if frame.num_rows > 0]
    if tables:
        combined = concat_tables_unified(tables)
        combined = _select_edge_columns(combined)
        combined = dedupe_table_for_table(CPG_EDGES_TABLE_KEY, combined)
        return _frame_to_reader(CPG_EDGES_TABLE_KEY, combined)
    return empty_table_for_table(CPG_EDGES_TABLE_KEY)


def instruction_cpg_id(
    *,
    repo: str,
    commit: str,
    rel_path: str,
    code_unit_id: str,
    instr_id: str,
) -> int:
    """Public wrapper for instruction CPG node IDs.

    Returns
    -------
    int
        Stable CPG node identifier.
    """
    return _instruction_cpg_id(
        repo=repo,
        commit=commit,
        rel_path=rel_path,
        code_unit_id=code_unit_id,
        instr_id=instr_id,
    )


def stable_cpg_id(table_key: str, pk: Mapping[str, object]) -> int:
    """Public wrapper for stable CPG node IDs.

    Returns
    -------
    int
        Stable CPG node identifier.
    """
    return _stable_cpg_id(table_key, pk)


def py_bc_callsite_symbol_edges_to_cpg(
    instructions: pa.Table,
    syntax_calls: pa.Table,
    scip_symbols: pa.Table,
) -> pa.Table:
    """Public wrapper for bytecode callsite symbol edges.

    Returns
    -------
    pyarrow.Table
        Arrow table of callsite symbol edges.
    """
    return _py_bc_callsite_symbol_edges_to_cpg(instructions, syntax_calls, scip_symbols)


def py_bc_callsite_edges_to_cpg(
    instructions: pa.Table,
    syntax_calls: pa.Table,
) -> pa.Table:
    """Public wrapper for bytecode callsite edges.

    Returns
    -------
    pyarrow.Table
        Arrow table of callsite edges.
    """
    return _py_bc_callsite_edges_to_cpg(instructions, syntax_calls)


def py_bc_stack_edges_to_cpg(
    instructions: pa.Table,
    blocks: pa.Table,
) -> pa.Table:
    """Public wrapper for bytecode stack edges.

    Returns
    -------
    pyarrow.Table
        Arrow table of stack edges.
    """
    return _py_bc_stack_edges_to_cpg(instructions, blocks)


def py_inspect_unwrap_edges_to_cpg(
    unwrap_hops: pa.Table,
) -> pa.Table:
    """Public wrapper for inspect unwrap edges.

    Returns
    -------
    pyarrow.Table
        Arrow table of unwrap edges.
    """
    return _py_inspect_unwrap_edges_to_cpg(unwrap_hops)


def inspect_to_ast_edges_to_cpg(
    inspect_objects: pa.Table,
    inspect_source: pa.Table,
    ast_nodes: pa.Table,
) -> pa.Table:
    """Public wrapper for inspect-to-AST anchor edges.

    Returns
    -------
    pyarrow.Table
        Arrow table of inspect anchor edges.
    """
    return _inspect_to_ast_edges_to_cpg(inspect_objects, inspect_source, ast_nodes)


__all__ = [
    "CPG_EDGES_TABLE_KEY",
    "CPG_NODES_TABLE_KEY",
    "CPG_TARGET_NAME",
    "cpg_edges",
    "cpg_nodes",
    "inspect_to_ast_edges_to_cpg",
    "instruction_cpg_id",
    "py_bc_callsite_edges_to_cpg",
    "py_bc_callsite_symbol_edges_to_cpg",
    "py_bc_stack_edges_to_cpg",
    "py_inspect_unwrap_edges_to_cpg",
    "stable_cpg_id",
]
