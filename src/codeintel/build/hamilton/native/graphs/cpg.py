"""Unified CPG node and edge assembly for property graph exports."""

from __future__ import annotations

import hashlib
import json
import opcode
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial
from typing import TypedDict, cast

import polars as pl

from codeintel.build.graphs.compute.goid import DECIMAL_38_MAX
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.arrow_ops import (
    ArrowJoinSpec,
    arrow_join_frames,
    arrow_join_lazyframes,
    arrow_join_tables,
    arrow_table_from_lazyframe,
)
from codeintel.build.tabular.conversion import table_to_frame, tabular_to_lazyframe
from codeintel.build.tabular.frames import JoinSpec, dedupe_frame_for_table, empty_frame_for_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.intervals.span_resolver import SpanResolver
from codeintel.core.schemas.generated_rows import columns_for_table_key
from codeintel.core.serialization.payload import decode_payload, encode_payload

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

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


@dataclass(frozen=True)
class _CpgSymbolInputs:
    syntax_edges: pl.LazyFrame
    occ_syntax: pl.LazyFrame
    occ_span: pl.LazyFrame
    symbol_rels: pl.LazyFrame
    symbol_goid: pl.LazyFrame


@dataclass(frozen=True)
class _CpgFlowInputs:
    goids: pl.LazyFrame
    cfg_edges: pl.LazyFrame
    dfg_edges: pl.LazyFrame
    cfg_blocks: pl.LazyFrame
    cdg_edges: pl.LazyFrame


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
    syntax_nodes: pl.LazyFrame
    ast_nodes: pl.LazyFrame
    scip_symbol_information: pl.LazyFrame
    goids: pl.LazyFrame
    py_sym_scopes: pl.LazyFrame
    py_sym_bindings: pl.LazyFrame
    py_bc_code_units: pl.LazyFrame
    py_bc_instructions: pl.LazyFrame
    py_bc_blocks: pl.LazyFrame
    py_inspect_objects: pl.LazyFrame
    py_inspect_signatures: pl.LazyFrame
    py_inspect_signature_params: pl.LazyFrame
    ts_tokens: pl.LazyFrame
    ts_trivia: pl.LazyFrame


@dataclass(frozen=True)
class _CpgNodeGraphLazyFrames:
    cfg_blocks: pl.LazyFrame
    import_modules: pl.LazyFrame


@dataclass(frozen=True)
class _CpgLinkInputs:
    call_edges: pl.LazyFrame
    import_edges: pl.LazyFrame


@dataclass(frozen=True)
class _CpgCallWiringInputs:
    call_edges: pl.LazyFrame
    arg_to_param_edges: pl.LazyFrame
    ret_to_call_edges: pl.LazyFrame


@dataclass(frozen=True)
class _CpgSyntaxNodeInputs:
    syntax_nodes: pl.LazyFrame


@dataclass(frozen=True)
class _CpgOverlayEdgeInputs:
    ast_nodes: pl.LazyFrame
    syntax_calls: pl.LazyFrame
    syntax_call_args: pl.LazyFrame
    scip_symbols: pl.LazyFrame
    py_sym_scopes: pl.LazyFrame
    py_sym_bindings: pl.LazyFrame
    py_sym_scope_edges: pl.LazyFrame
    py_sym_namespace_edges: pl.LazyFrame
    py_sym_resolution_edges: pl.LazyFrame
    py_bc_code_units: pl.LazyFrame
    py_bc_instructions: pl.LazyFrame
    py_bc_blocks: pl.LazyFrame
    py_bc_cfg_edges: pl.LazyFrame
    py_bc_defuse_events: pl.LazyFrame
    py_inspect_objects: pl.LazyFrame
    py_inspect_class_mro: pl.LazyFrame
    py_inspect_class_attrs: pl.LazyFrame
    py_inspect_unwrap_hops: pl.LazyFrame
    py_inspect_signatures: pl.LazyFrame
    py_inspect_signature_params: pl.LazyFrame
    py_inspect_runtime_state: pl.LazyFrame


@dataclass(frozen=True)
class _CpgOverlayScopeInputs:
    py_sym_scopes: pl.LazyFrame
    py_sym_bindings: pl.LazyFrame
    py_sym_scope_edges: pl.LazyFrame
    py_sym_namespace_edges: pl.LazyFrame
    py_sym_resolution_edges: pl.LazyFrame


@dataclass(frozen=True)
class _CpgOverlaySymbolInputs:
    ast_nodes: pl.LazyFrame
    scip_symbols: pl.LazyFrame
    scope_inputs: _CpgOverlayScopeInputs


@dataclass(frozen=True)
class _CpgOverlayBytecodeInputs:
    py_bc_code_units: pl.LazyFrame
    py_bc_instructions: pl.LazyFrame
    py_bc_blocks: pl.LazyFrame
    py_bc_cfg_edges: pl.LazyFrame
    py_bc_defuse_events: pl.LazyFrame


@dataclass(frozen=True)
class _CpgOverlaySyntaxCallInputs:
    syntax_calls: pl.LazyFrame
    syntax_call_args: pl.LazyFrame


@dataclass(frozen=True)
class _CpgOverlayInspectCoreInputs:
    py_inspect_objects: pl.LazyFrame
    py_inspect_class_mro: pl.LazyFrame
    py_inspect_class_attrs: pl.LazyFrame
    py_inspect_unwrap_hops: pl.LazyFrame


@dataclass(frozen=True)
class _CpgOverlayInspectRuntimeInputs:
    py_inspect_signatures: pl.LazyFrame
    py_inspect_signature_params: pl.LazyFrame
    py_inspect_runtime_state: pl.LazyFrame


@dataclass(frozen=True)
class _CpgOverlayInspectInputs:
    py_inspect_objects: pl.LazyFrame
    py_inspect_class_mro: pl.LazyFrame
    py_inspect_class_attrs: pl.LazyFrame
    py_inspect_unwrap_hops: pl.LazyFrame
    py_inspect_signatures: pl.LazyFrame
    py_inspect_signature_params: pl.LazyFrame
    py_inspect_runtime_state: pl.LazyFrame
    syntax_calls: pl.LazyFrame
    syntax_call_args: pl.LazyFrame


@dataclass(frozen=True)
class _CpgEdgeCoreInputs:
    symbol: _CpgSymbolInputs
    flow: _CpgFlowInputs
    link: _CpgLinkInputs
    call_wiring: _CpgCallWiringInputs
    syntax_nodes: _CpgSyntaxNodeInputs


def _stable_int_hash(
    payload: object,
    *,
    digest_size: int,
    modulus: int,
) -> int:
    serialized = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    digest = hashlib.blake2b(serialized.encode("utf-8"), digest_size=digest_size).digest()
    return int.from_bytes(digest, "big") % modulus


def _stable_cpg_id(table_key: str, pk: Mapping[str, object]) -> int:
    payload = {"table_key": table_key, "pk": dict(pk)}
    return _stable_int_hash(payload, digest_size=16, modulus=DECIMAL_38_MAX)


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


def _struct_expr(values: Mapping[str, pl.Expr]) -> pl.Expr:
    fields = [expr.alias(name) for name, expr in values.items()]
    return pl.struct(fields)


def _pk_expr(table_key: str, values: Mapping[str, pl.Expr]) -> pl.Expr:
    return _struct_expr(values).map_elements(
        partial(_stable_cpg_id_from_row, table_key),
        return_dtype=pl.Object,
    )


def _ordinal_expr(table_key: str, values: Mapping[str, pl.Expr]) -> pl.Expr:
    return _struct_expr(values).map_elements(
        partial(_stable_ordinal_from_row, table_key),
        return_dtype=pl.Int64,
    )


def _pk_json_expr(values: Mapping[str, pl.Expr]) -> pl.Expr:
    return _struct_expr(values).map_elements(_row_to_payload, return_dtype=pl.Binary)


def _payload_json_expr(values: Mapping[str, pl.Expr]) -> pl.Expr:
    return _struct_expr(values).map_elements(_row_to_payload, return_dtype=pl.Binary)


def _select_node_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    missing = [name for name in _CPG_NODE_COLUMNS if name not in frame.columns]
    if missing:
        frame = frame.with_columns([pl.lit(None).alias(name) for name in missing])
    return frame.select(_CPG_NODE_COLUMNS)


def _select_edge_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    missing = [name for name in _CPG_EDGE_COLUMNS if name not in frame.columns]
    if missing:
        frame = frame.with_columns([pl.lit(None).alias(name) for name in missing])
    return frame.select(_CPG_EDGE_COLUMNS)


def _syntax_node_keys(syntax_nodes: pl.LazyFrame) -> pl.LazyFrame:
    return syntax_nodes.select("repo", "commit", "rel_path", "producer", "node_id")


def _syntax_nodes_to_cpg(syntax_nodes: pl.LazyFrame) -> pl.LazyFrame:
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "producer": pl.col("producer"),
        "node_id": pl.col("node_id"),
    }
    return syntax_nodes.with_columns(
        _pk_expr(SYNTAX_NODES_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("SYNTAX_NODE").alias("node_kind"),
        pl.lit(SYNTAX_NODES_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.col("rel_path"),
        pl.col("start_byte"),
        pl.col("end_byte"),
        pl.col("extras_json")
        .map_elements(_encode_optional_payload, return_dtype=pl.Binary)
        .alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _scip_symbols_to_cpg(symbols: pl.LazyFrame) -> pl.LazyFrame:
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "symbol": pl.col("symbol"),
    }
    return symbols.with_columns(
        _pk_expr(SCIP_SYMBOLS_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("SCIP_SYMBOL").alias("node_kind"),
        pl.lit(SCIP_SYMBOLS_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.lit(None).alias("rel_path"),
        pl.lit(None).alias("start_byte"),
        pl.lit(None).alias("end_byte"),
        pl.lit(None).cast(pl.Binary).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _goids_to_cpg(goids: pl.LazyFrame) -> pl.LazyFrame:
    pk_values = {"goid_h128": pl.col("goid_h128")}
    return goids.with_columns(
        _pk_expr(GOIDS_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("GOID").alias("node_kind"),
        pl.lit(GOIDS_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.col("rel_path"),
        pl.lit(None).alias("start_byte"),
        pl.lit(None).alias("end_byte"),
        pl.lit(None).cast(pl.Binary).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _cfg_blocks_to_cpg(cfg_blocks: pl.LazyFrame, goids: pl.LazyFrame) -> pl.LazyFrame:
    goid_ctx = goids.select(
        pl.col("goid_h128").alias("function_goid_h128"),
        "repo",
        "commit",
    )
    blocks = arrow_join_lazyframes(
        cfg_blocks,
        goid_ctx,
        spec=JoinSpec(on=["function_goid_h128"], how="left"),
    )
    pk_values = {
        "function_goid_h128": pl.col("function_goid_h128"),
        "block_idx": pl.col("block_idx"),
    }
    return blocks.with_columns(
        _pk_expr(CFG_BLOCKS_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("CFG_BLOCK").alias("node_kind"),
        pl.lit(CFG_BLOCKS_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.col("file_path").alias("rel_path"),
        pl.lit(None).alias("start_byte"),
        pl.lit(None).alias("end_byte"),
        pl.lit(None).cast(pl.Binary).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _import_modules_to_cpg(import_modules: pl.LazyFrame) -> pl.LazyFrame:
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "module": pl.col("module"),
    }
    return import_modules.with_columns(
        _pk_expr(IMPORT_MODULES_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("MODULE").alias("node_kind"),
        pl.lit(IMPORT_MODULES_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.lit(None).alias("rel_path"),
        pl.lit(None).alias("start_byte"),
        pl.lit(None).alias("end_byte"),
        pl.lit(None).cast(pl.Binary).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _ts_tokens_to_cpg(tokens: pl.LazyFrame) -> pl.LazyFrame:
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
    if not required.issubset(tokens.columns):
        return empty_frame_for_table(CPG_NODES_TABLE_KEY)
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "language": pl.col("language"),
        "token_id": pl.col("token_id"),
    }
    extras_values = {
        "token_kind": pl.col("token_kind"),
        "node_type": pl.col("node_type"),
        "text_preview": pl.col("text_preview"),
        "token_extras": pl.col("extras_json"),
    }
    return tokens.with_columns(
        _pk_expr(TS_TOKENS_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("TS_TOKEN").alias("node_kind"),
        pl.lit(TS_TOKENS_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.col("rel_path"),
        pl.col("start_byte"),
        pl.col("end_byte"),
        _payload_json_expr(extras_values).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _ts_trivia_to_cpg(trivia: pl.LazyFrame) -> pl.LazyFrame:
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
    if not required.issubset(trivia.columns):
        return empty_frame_for_table(CPG_NODES_TABLE_KEY)
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "language": pl.col("language"),
        "trivia_id": pl.col("trivia_id"),
    }
    extras_values = {
        "trivia_kind": pl.col("trivia_kind"),
        "node_type": pl.col("node_type"),
        "text_preview": pl.col("text_preview"),
        "trivia_extras": pl.col("extras_json"),
    }
    return trivia.with_columns(
        _pk_expr(TS_TRIVIA_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("TS_TRIVIA").alias("node_kind"),
        pl.lit(TS_TRIVIA_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.col("rel_path"),
        pl.col("start_byte"),
        pl.col("end_byte"),
        _payload_json_expr(extras_values).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _ast_nodes_to_cpg(ast_nodes: pl.LazyFrame, env: BuildEnv) -> pl.LazyFrame:
    required = {
        "path",
        "node_type",
        "hash",
        "start_byte",
        "end_byte",
    }
    if not required.issubset(ast_nodes.columns):
        return empty_frame_for_table(CPG_NODES_TABLE_KEY)
    pk_values = {"hash": pl.col("hash")}
    extras_values = {
        "node_type": pl.col("node_type"),
        "name": pl.col("name"),
        "qualname": pl.col("qualname"),
        "parent_qualname": pl.col("parent_qualname"),
        "lineno": pl.col("lineno"),
        "end_lineno": pl.col("end_lineno"),
        "col_offset": pl.col("col_offset"),
        "end_col_offset": pl.col("end_col_offset"),
        "decorator_start_line": pl.col("decorator_start_line"),
        "decorator_end_line": pl.col("decorator_end_line"),
        "decorators": pl.col("decorators"),
        "docstring": pl.col("docstring"),
        "ctx": pl.col("ctx"),
        "type_comment": pl.col("type_comment"),
        "type_ignores": pl.col("type_ignores"),
        "identifier": pl.col("identifier"),
        "attribute": pl.col("attribute"),
        "imported": pl.col("imported"),
        "asname": pl.col("asname"),
        "module": pl.col("module"),
        "level": pl.col("level"),
        "constant_kind": pl.col("constant_kind"),
    }
    return ast_nodes.with_columns(
        pl.lit(env.repo).alias("repo"),
        pl.lit(env.commit).alias("commit"),
        _pk_expr(AST_NODES_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("AST_NODE").alias("node_kind"),
        pl.lit(AST_NODES_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.col("path").alias("rel_path"),
        pl.col("start_byte"),
        pl.col("end_byte"),
        _payload_json_expr(extras_values).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _py_sym_scopes_to_cpg(scopes: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "rel_path",
        "scope_id",
        "scope_type",
    }
    if not required.issubset(scopes.columns):
        return empty_frame_for_table(CPG_NODES_TABLE_KEY)
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "scope_id": pl.col("scope_id"),
    }
    extras_values = {
        "scope_type": pl.col("scope_type"),
        "scope_name": pl.col("scope_name"),
        "qualpath": pl.col("qualpath"),
        "lineno": pl.col("lineno"),
        "is_nested": pl.col("is_nested"),
        "is_optimized": pl.col("is_optimized"),
        "has_children": pl.col("has_children"),
        "parent_scope_id": pl.col("parent_scope_id"),
        "anchor_ast_node_id": pl.col("anchor_ast_node_id"),
        "anchor_confidence": pl.col("anchor_confidence"),
        "anchor_reason": pl.col("anchor_reason"),
        "scope_local_id": pl.col("scope_local_id"),
    }
    return scopes.with_columns(
        _pk_expr(PY_SYM_SCOPES_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("SCOPE").alias("node_kind"),
        pl.lit(PY_SYM_SCOPES_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.col("rel_path"),
        pl.col("span_start_byte").alias("start_byte"),
        pl.col("span_end_byte").alias("end_byte"),
        _payload_json_expr(extras_values).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _py_sym_bindings_to_cpg(bindings: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "rel_path",
        "binding_id",
        "scope_id",
        "name",
        "binding_kind",
    }
    if not required.issubset(bindings.columns):
        return empty_frame_for_table(CPG_NODES_TABLE_KEY)
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "binding_id": pl.col("binding_id"),
    }
    extras_values = {
        "scope_id": pl.col("scope_id"),
        "name": pl.col("name"),
        "binding_kind": pl.col("binding_kind"),
        "declared_here": pl.col("declared_here"),
        "referenced_here": pl.col("referenced_here"),
        "assigned_here": pl.col("assigned_here"),
        "annotated_here": pl.col("annotated_here"),
        "scoping_class": pl.col("scoping_class"),
    }
    return bindings.with_columns(
        _pk_expr(PY_SYM_BINDINGS_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("BINDING").alias("node_kind"),
        pl.lit(PY_SYM_BINDINGS_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.col("rel_path"),
        pl.lit(None).alias("start_byte"),
        pl.lit(None).alias("end_byte"),
        _payload_json_expr(extras_values).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _py_bc_code_units_to_cpg(code_units: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "rel_path",
        "code_unit_id",
    }
    if not required.issubset(code_units.columns):
        return empty_frame_for_table(CPG_NODES_TABLE_KEY)
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "code_unit_id": pl.col("code_unit_id"),
    }
    extras_values = {
        "qualpath": pl.col("qualpath"),
        "co_name": pl.col("co_name"),
        "co_qualname": pl.col("co_qualname"),
        "kind": pl.col("kind"),
        "co_firstlineno": pl.col("co_firstlineno"),
        "flags": pl.col("flags"),
        "argcount": pl.col("argcount"),
        "posonlyargcount": pl.col("posonlyargcount"),
        "kwonlyargcount": pl.col("kwonlyargcount"),
        "nlocals": pl.col("nlocals"),
        "stacksize": pl.col("stacksize"),
        "varnames": pl.col("varnames"),
        "names": pl.col("names"),
        "freevars": pl.col("freevars"),
        "cellvars": pl.col("cellvars"),
        "bytecode_len": pl.col("bytecode_len"),
        "exceptiontable_len": pl.col("exceptiontable_len"),
        "python_version": pl.col("python_version"),
    }
    return code_units.with_columns(
        _pk_expr(PY_BC_CODE_UNITS_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("BC_CODE_UNIT").alias("node_kind"),
        pl.lit(PY_BC_CODE_UNITS_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.col("rel_path"),
        pl.col("span_start_byte").alias("start_byte"),
        pl.col("span_end_byte").alias("end_byte"),
        _payload_json_expr(extras_values).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _py_bc_instructions_to_cpg(instructions: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "rel_path",
        "code_unit_id",
        "instr_id",
    }
    if not required.issubset(instructions.columns):
        return empty_frame_for_table(CPG_NODES_TABLE_KEY)
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "code_unit_id": pl.col("code_unit_id"),
        "instr_id": pl.col("instr_id"),
    }
    extras_values = {
        "instr_index": pl.col("instr_index"),
        "start_offset": pl.col("start_offset"),
        "offset": pl.col("offset"),
        "end_offset": pl.col("end_offset"),
        "opcode": pl.col("opcode"),
        "opname": pl.col("opname"),
        "baseopname": pl.col("baseopname"),
        "arg": pl.col("arg"),
        "argrepr": pl.col("argrepr"),
        "argval_kind": pl.col("argval_kind"),
        "argval_str": pl.col("argval_str"),
        "argval_int": pl.col("argval_int"),
        "argval_repr": pl.col("argval_repr"),
        "is_jump_target": pl.col("is_jump_target"),
        "jump_target_offset": pl.col("jump_target_offset"),
        "jump_target_label": pl.col("jump_target_label"),
        "label": pl.col("label"),
        "starts_line": pl.col("starts_line"),
        "line_number": pl.col("line_number"),
        "pos": pl.col("pos"),
    }
    return instructions.with_columns(
        _pk_expr(PY_BC_INSTRUCTIONS_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("BC_INSTR").alias("node_kind"),
        pl.lit(PY_BC_INSTRUCTIONS_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.col("rel_path"),
        pl.col("span_start_byte").alias("start_byte"),
        pl.col("span_end_byte").alias("end_byte"),
        _payload_json_expr(extras_values).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _py_bc_blocks_to_cpg(blocks: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "rel_path",
        "block_id",
        "code_unit_id",
    }
    if not required.issubset(blocks.columns):
        return empty_frame_for_table(CPG_NODES_TABLE_KEY)
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "block_id": pl.col("block_id"),
    }
    extras_values = {
        "code_unit_id": pl.col("code_unit_id"),
        "start_offset": pl.col("start_offset"),
        "end_offset": pl.col("end_offset"),
        "start_label": pl.col("start_label"),
        "kind": pl.col("kind"),
        "first_instr_index": pl.col("first_instr_index"),
        "last_instr_index": pl.col("last_instr_index"),
    }
    return blocks.with_columns(
        _pk_expr(PY_BC_BLOCKS_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("BC_BLOCK").alias("node_kind"),
        pl.lit(PY_BC_BLOCKS_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.col("rel_path"),
        pl.col("anchor_span_start_byte").alias("start_byte"),
        pl.col("anchor_span_end_byte").alias("end_byte"),
        _payload_json_expr(extras_values).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _py_inspect_objects_to_cpg(objects: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "object_id",
        "kind",
    }
    if not required.issubset(objects.columns):
        return empty_frame_for_table(CPG_NODES_TABLE_KEY)
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "object_id": pl.col("object_id"),
    }
    extras_values = {
        "kind": pl.col("kind"),
        "module_name": pl.col("module_name"),
        "qualname": pl.col("qualname"),
        "name": pl.col("name"),
        "type_qualname": pl.col("type_qualname"),
        "object_addr": pl.col("object_addr"),
        "is_builtin": pl.col("is_builtin"),
        "is_callable": pl.col("is_callable"),
        "is_descriptor": pl.col("is_descriptor"),
        "has_wrapped": pl.col("has_wrapped"),
        "has_signature_override": pl.col("has_signature_override"),
        "has_annotations": pl.col("has_annotations"),
        "status": pl.col("status"),
    }
    return objects.with_columns(
        _pk_expr(PY_INSPECT_OBJECTS_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("INSPECT_OBJECT").alias("node_kind"),
        pl.lit(PY_INSPECT_OBJECTS_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.lit(None).alias("rel_path"),
        pl.lit(None).alias("start_byte"),
        pl.lit(None).alias("end_byte"),
        _payload_json_expr(extras_values).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _py_inspect_signatures_to_cpg(signatures: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "signature_id",
        "object_id",
    }
    if not required.issubset(signatures.columns):
        return empty_frame_for_table(CPG_NODES_TABLE_KEY)
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "signature_id": pl.col("signature_id"),
    }
    extras_values = {
        "object_id": pl.col("object_id"),
        "mode": pl.col("mode"),
        "variant": pl.col("variant"),
        "follow_wrapped": pl.col("follow_wrapped"),
        "eval_str": pl.col("eval_str"),
        "effective_object_id": pl.col("effective_object_id"),
        "sig_text": pl.col("sig_text"),
        "sig_format": pl.col("sig_format"),
        "has_varargs": pl.col("has_varargs"),
        "has_varkw": pl.col("has_varkw"),
        "status": pl.col("status"),
    }
    return signatures.with_columns(
        _pk_expr(PY_INSPECT_SIGNATURES_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("INSPECT_SIGNATURE").alias("node_kind"),
        pl.lit(PY_INSPECT_SIGNATURES_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.lit(None).alias("rel_path"),
        pl.lit(None).alias("start_byte"),
        pl.lit(None).alias("end_byte"),
        _payload_json_expr(extras_values).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


def _py_inspect_signature_params_to_cpg(params: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "signature_id",
        "param_index",
    }
    if not required.issubset(params.columns):
        return empty_frame_for_table(CPG_NODES_TABLE_KEY)
    pk_values = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "signature_id": pl.col("signature_id"),
        "param_index": pl.col("param_index"),
    }
    extras_values = {
        "mode": pl.col("mode"),
        "name": pl.col("name"),
        "kind": pl.col("kind"),
        "default_present": pl.col("default_present"),
        "default_value": pl.col("default_value"),
        "annotation_present": pl.col("annotation_present"),
        "annotation_value": pl.col("annotation_value"),
        "status": pl.col("status"),
    }
    return params.with_columns(
        _pk_expr(PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY, pk_values).alias("cpg_node_id"),
        pl.lit("INSPECT_SIGNATURE_PARAM").alias("node_kind"),
        pl.lit(PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY).alias("source_table_key"),
        _pk_json_expr(pk_values).alias("source_pk_json"),
        pl.lit(None).alias("rel_path"),
        pl.lit(None).alias("start_byte"),
        pl.lit(None).alias("end_byte"),
        _payload_json_expr(extras_values).alias("extras_json"),
    ).select(_CPG_NODE_COLUMNS)


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
        syntax_nodes=tabular_to_lazyframe(core_inputs.syntax_nodes),
        ast_nodes=tabular_to_lazyframe(core_inputs.ast_nodes),
        scip_symbol_information=tabular_to_lazyframe(core_inputs.scip_symbol_information),
        goids=tabular_to_lazyframe(core_inputs.goids),
        py_sym_scopes=tabular_to_lazyframe(core_inputs.py_sym_scopes),
        py_sym_bindings=tabular_to_lazyframe(core_inputs.py_sym_bindings),
        py_bc_code_units=tabular_to_lazyframe(core_inputs.py_bc_code_units),
        py_bc_instructions=tabular_to_lazyframe(core_inputs.py_bc_instructions),
        py_bc_blocks=tabular_to_lazyframe(core_inputs.py_bc_blocks),
        py_inspect_objects=tabular_to_lazyframe(core_inputs.py_inspect_objects),
        py_inspect_signatures=tabular_to_lazyframe(core_inputs.py_inspect_signatures),
        py_inspect_signature_params=tabular_to_lazyframe(core_inputs.py_inspect_signature_params),
        ts_tokens=tabular_to_lazyframe(core_inputs.ts_tokens),
        ts_trivia=tabular_to_lazyframe(core_inputs.ts_trivia),
    )


def _graph_lazyframes(graph_inputs: _CpgNodeGraphInputs) -> _CpgNodeGraphLazyFrames:
    return _CpgNodeGraphLazyFrames(
        cfg_blocks=tabular_to_lazyframe(graph_inputs.cfg_blocks),
        import_modules=tabular_to_lazyframe(graph_inputs.import_modules),
    )


def cpg_nodes(
    env: BuildEnv,
    cpg_nodes__inputs: _CpgNodeInputs,
) -> pl.LazyFrame:
    """Build CPG nodes from syntax, symbol, and flow inventories.

    Returns
    -------
    pl.LazyFrame
        LazyFrame for graph.cpg_nodes.
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
    combined = pl.concat(frames, how="vertical_relaxed")
    if combined.columns:
        combined = dedupe_frame_for_table(combined, table_key=CPG_NODES_TABLE_KEY)
        return _select_node_columns(combined)
    return empty_frame_for_table(CPG_NODES_TABLE_KEY)


def _syntax_edges_to_cpg(syntax_edges: pl.LazyFrame) -> pl.LazyFrame:
    parent_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "producer": pl.col("producer"),
        "node_id": pl.col("parent_node_id"),
    }
    child_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "producer": pl.col("producer"),
        "node_id": pl.col("child_node_id"),
    }
    return syntax_edges.with_columns(
        _pk_expr(SYNTAX_NODES_TABLE_KEY, parent_pk).alias("src_cpg_node_id"),
        _pk_expr(SYNTAX_NODES_TABLE_KEY, child_pk).alias("dst_cpg_node_id"),
        pl.lit("AST").alias("edge_kind"),
        pl.lit("SYNTAX").alias("edge_layer"),
        pl.col("rel_path"),
        pl.col("child_ordinal").alias("ordinal"),
        pl.lit(None).cast(pl.Binary).alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)


def _occurrence_role_resolvers(
    span_frame: pl.DataFrame,
) -> dict[tuple[str, str], SpanResolver[_OccurrenceRolePayload]]:
    resolvers: dict[tuple[str, str], SpanResolver[_OccurrenceRolePayload]] = {}
    for row in span_frame.iter_rows(named=True):
        rel_path = row.get("rel_path")
        scip_symbol = row.get("scip_symbol")
        if not isinstance(rel_path, str) or not isinstance(scip_symbol, str):
            continue
        start_line = row.get("occ_start_line")
        end_line = row.get("occ_end_line")
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
                scip_roles=_coerce_int(row.get("scip_roles")),
                is_definition=_coerce_bool(row.get("is_definition")),
                is_reference=_coerce_bool(row.get("is_reference")),
                is_import=_coerce_bool(row.get("is_import")),
                is_write=_coerce_bool(row.get("is_write")),
                is_read=_coerce_bool(row.get("is_read")),
            ),
        )
    return resolvers


def _occurrence_fallback_rows(
    joined_frame: pl.DataFrame,
    span_frame: pl.DataFrame,
) -> list[dict[str, object]]:
    if "scip_roles" not in joined_frame.columns:
        return []
    missing = joined_frame.filter(pl.col("scip_roles").is_null())
    if missing.is_empty():
        return []
    resolvers = _occurrence_role_resolvers(span_frame)
    rows: list[dict[str, object]] = []
    for row in missing.iter_rows(named=True):
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
    occ_syntax: pl.LazyFrame,
    occ_span: pl.LazyFrame,
) -> pl.LazyFrame:
    span_lf = occ_span.select(
        "repo",
        "commit",
        "rel_path",
        "scip_symbol",
        pl.col("roles").alias("scip_roles"),
        "is_definition",
        "is_reference",
        "is_import",
        "is_write",
        "is_read",
        pl.col("start_line").alias("occ_start_line"),
        pl.col("start_col").alias("occ_start_col"),
        pl.col("end_line").alias("occ_end_line"),
        pl.col("end_col").alias("occ_end_col"),
    ).with_columns(
        pl.col("occ_start_line").cast(pl.Int64),
        pl.col("occ_start_col").cast(pl.Int64),
        pl.col("occ_end_line").cast(pl.Int64),
        pl.col("occ_end_col").cast(pl.Int64),
    )
    syntax_lf = occ_syntax.select(
        "repo",
        "commit",
        "rel_path",
        "producer",
        "scip_symbol",
        "scip_occurrence_id",
        "occ_start_line",
        "occ_start_col",
        "occ_end_line",
        "occ_end_col",
        "syntax_node_id",
        "match_kind",
        "candidate_count",
    ).with_columns(
        pl.col("occ_start_line").cast(pl.Int64),
        pl.col("occ_start_col").cast(pl.Int64),
        pl.col("occ_end_line").cast(pl.Int64),
        pl.col("occ_end_col").cast(pl.Int64),
    )
    join_keys = [
        "repo",
        "commit",
        "rel_path",
        "scip_symbol",
        "occ_start_line",
        "occ_start_col",
        "occ_end_line",
        "occ_end_col",
    ]
    # Contract: span rows are unique per occurrence join key.
    span_table = arrow_table_from_lazyframe(span_lf)
    syntax_table = arrow_table_from_lazyframe(syntax_lf)
    joined_table = arrow_join_tables(
        syntax_table,
        span_table,
        spec=ArrowJoinSpec(on=join_keys, how="left", validate="m:1"),
    )
    if joined_table.num_rows == 0:
        return table_to_frame(joined_table).lazy()
    joined = table_to_frame(joined_table)
    span_frame = table_to_frame(span_table)
    joined = joined.with_row_index(name="__row_id").with_columns(
        pl.lit(None).cast(pl.Utf8).alias("span_match_kind"),
        pl.lit(None).cast(pl.Int64).alias("span_candidate_count"),
    )
    fallback_rows = _occurrence_fallback_rows(joined, span_frame)
    if fallback_rows:
        fallback = pl.DataFrame(fallback_rows)
        joined = arrow_join_frames(
            joined,
            fallback,
            spec=JoinSpec(on=["__row_id"], how="left"),
        )
        joined = joined.with_columns(
            pl.coalesce([pl.col("scip_roles"), pl.col("scip_roles_fallback")]).alias("scip_roles"),
            pl.coalesce([pl.col("is_definition"), pl.col("is_definition_fallback")]).alias(
                "is_definition"
            ),
            pl.coalesce([pl.col("is_reference"), pl.col("is_reference_fallback")]).alias(
                "is_reference"
            ),
            pl.coalesce([pl.col("is_import"), pl.col("is_import_fallback")]).alias("is_import"),
            pl.coalesce([pl.col("is_write"), pl.col("is_write_fallback")]).alias("is_write"),
            pl.coalesce([pl.col("is_read"), pl.col("is_read_fallback")]).alias("is_read"),
            pl.coalesce([pl.col("span_match_kind"), pl.col("span_match_kind_fallback")]).alias(
                "span_match_kind"
            ),
            pl.coalesce([pl.col("span_candidate_count"), pl.col("span_candidate_count_fallback")])
            .alias("span_candidate_count"),
        ).drop(
            [
                "scip_roles_fallback",
                "is_definition_fallback",
                "is_reference_fallback",
                "is_import_fallback",
                "is_write_fallback",
                "is_read_fallback",
                "span_match_kind_fallback",
                "span_candidate_count_fallback",
            ]
        )
    return joined.drop("__row_id").lazy()


def _scip_occurrence_edges_to_cpg(
    occ_syntax: pl.LazyFrame,
    occ_span: pl.LazyFrame,
) -> pl.LazyFrame:
    joined = _occurrence_roles(occ_syntax, occ_span)
    syntax_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "producer": pl.col("producer"),
        "node_id": pl.col("syntax_node_id"),
    }
    symbol_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "symbol": pl.col("scip_symbol"),
    }
    is_def = pl.col("is_definition").fill_null(value=False)
    is_import = pl.col("is_import").fill_null(value=False)
    is_write = pl.col("is_write").fill_null(value=False)
    is_read = pl.col("is_read").fill_null(value=False)
    edge_kind = (
        pl.when(is_def)
        .then(pl.lit("DEFINES"))
        .when(is_import)
        .then(pl.lit("IMPORTS"))
        .when(is_write)
        .then(pl.lit("WRITES"))
        .when(is_read)
        .then(pl.lit("REFERS_TO"))
        .otherwise(pl.lit("REFERS_TO"))
    )
    extras = _pk_json_expr(
        {
            "scip_occurrence_id": pl.col("scip_occurrence_id"),
            "match_kind": pl.col("match_kind"),
            "candidate_count": pl.col("candidate_count"),
            "scip_roles": pl.col("scip_roles"),
            "span_match_kind": pl.col("span_match_kind"),
            "span_candidate_count": pl.col("span_candidate_count"),
        }
    )
    ordinal = _ordinal_expr(
        "core.scip_occurrence_syntax_xref",
        {"scip_occurrence_id": pl.col("scip_occurrence_id")},
    )
    return (
        joined.filter(pl.col("syntax_node_id").is_not_null())
        .with_columns(
            _pk_expr(SYNTAX_NODES_TABLE_KEY, syntax_pk).alias("src_cpg_node_id"),
            _pk_expr(SCIP_SYMBOLS_TABLE_KEY, symbol_pk).alias("dst_cpg_node_id"),
            edge_kind.alias("edge_kind"),
            pl.lit("SYMBOL").alias("edge_layer"),
            pl.col("rel_path"),
            ordinal.alias("ordinal"),
            extras.alias("extras_json"),
        )
        .select(_CPG_EDGE_COLUMNS)
    )


def _scip_symbol_relationships_to_cpg(symbol_rels: pl.LazyFrame) -> pl.LazyFrame:
    src_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "symbol": pl.col("symbol"),
    }
    dst_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "symbol": pl.col("related_symbol"),
    }
    ordinal = _ordinal_expr(
        "core.scip_symbol_relationships",
        {
            "symbol": pl.col("symbol"),
            "related_symbol": pl.col("related_symbol"),
            "relationship_kind": pl.col("relationship_kind"),
        },
    )
    return symbol_rels.with_columns(
        _pk_expr(SCIP_SYMBOLS_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
        _pk_expr(SCIP_SYMBOLS_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
        pl.col("relationship_kind").alias("edge_kind"),
        pl.lit("SYMBOL").alias("edge_layer"),
        pl.lit(None).alias("rel_path"),
        ordinal.alias("ordinal"),
        pl.lit(None).cast(pl.Binary).alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)


def _scip_symbol_goid_edges_to_cpg(symbol_goid: pl.LazyFrame) -> pl.LazyFrame:
    src_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "symbol": pl.col("scip_symbol"),
    }
    dst_pk = {"goid_h128": pl.col("goid_h128")}
    extras = _pk_json_expr(
        {
            "def_rel_path": pl.col("def_rel_path"),
            "def_start_line": pl.col("def_start_line"),
            "def_start_col": pl.col("def_start_col"),
            "def_end_line": pl.col("def_end_line"),
            "def_end_col": pl.col("def_end_col"),
        }
    )
    ordinal = _ordinal_expr(
        "core.scip_symbol_goid_xref",
        {"scip_symbol": pl.col("scip_symbol"), "goid_h128": pl.col("goid_h128")},
    )
    return (
        symbol_goid.filter(pl.col("goid_h128").is_not_null())
        .with_columns(
            _pk_expr(SCIP_SYMBOLS_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
            _pk_expr(GOIDS_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
            pl.lit("RESOLVES_TO").alias("edge_kind"),
            pl.lit("SYMBOL").alias("edge_layer"),
            pl.col("def_rel_path").alias("rel_path"),
            ordinal.alias("ordinal"),
            extras.alias("extras_json"),
        )
        .select(_CPG_EDGE_COLUMNS)
    )


def _call_graph_edges_to_cpg(call_edges: pl.LazyFrame) -> pl.LazyFrame:
    src_pk = {"goid_h128": pl.col("caller_goid_h128")}
    dst_pk = {"goid_h128": pl.col("callee_goid_h128")}
    extras = _pk_json_expr(
        {
            "resolved_via": pl.col("resolved_via"),
            "confidence": pl.col("confidence"),
            "kind": pl.col("kind"),
        }
    )
    ordinal = _ordinal_expr(
        "graph.call_graph_edges",
        {
            "caller_goid_h128": pl.col("caller_goid_h128"),
            "callee_goid_h128": pl.col("callee_goid_h128"),
            "callsite_path": pl.col("callsite_path"),
            "callsite_line": pl.col("callsite_line"),
            "callsite_col": pl.col("callsite_col"),
        },
    )
    return (
        call_edges.filter(pl.col("caller_goid_h128").is_not_null())
        .filter(pl.col("callee_goid_h128").is_not_null())
        .with_columns(
            _pk_expr(GOIDS_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
            _pk_expr(GOIDS_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
            pl.lit("CALLS").alias("edge_kind"),
            pl.lit("FLOW").alias("edge_layer"),
            pl.col("callsite_path").alias("rel_path"),
            ordinal.alias("ordinal"),
            extras.alias("extras_json"),
        )
        .select(_CPG_EDGE_COLUMNS)
    )


def _import_graph_edges_to_cpg(import_edges: pl.LazyFrame) -> pl.LazyFrame:
    src_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "module": pl.col("src_module"),
    }
    dst_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "module": pl.col("dst_module"),
    }
    extras = _pk_json_expr(
        {
            "src_fan_out": pl.col("src_fan_out"),
            "dst_fan_in": pl.col("dst_fan_in"),
            "cycle_group": pl.col("cycle_group"),
            "module_layer": pl.col("module_layer"),
        }
    )
    ordinal = _ordinal_expr(
        "graph.import_graph_edges",
        {
            "src_module": pl.col("src_module"),
            "dst_module": pl.col("dst_module"),
            "cycle_group": pl.col("cycle_group"),
        },
    )
    return import_edges.with_columns(
        _pk_expr(IMPORT_MODULES_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
        _pk_expr(IMPORT_MODULES_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
        pl.lit("IMPORTS").alias("edge_kind"),
        pl.lit("SYMBOL").alias("edge_layer"),
        pl.lit(None).alias("rel_path"),
        ordinal.alias("ordinal"),
        extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)


def _cfg_edges_to_cpg(
    cfg_edges: pl.LazyFrame,
    cfg_blocks: pl.LazyFrame,
    goids: pl.LazyFrame,
) -> pl.LazyFrame:
    goid_ctx = goids.select(
        pl.col("goid_h128").alias("function_goid_h128"),
        "repo",
        "commit",
    )
    blocks = arrow_join_lazyframes(
        cfg_blocks,
        goid_ctx,
        spec=JoinSpec(on=["function_goid_h128"], how="left"),
    ).select(
        "function_goid_h128",
        "block_id",
        "block_idx",
        "repo",
        "commit",
        pl.col("file_path").alias("rel_path"),
    )
    src_blocks = blocks.rename(
        {"block_id": "src_block_id", "block_idx": "src_block_idx", "rel_path": "src_path"}
    )
    dst_blocks = blocks.rename(
        {"block_id": "dst_block_id", "block_idx": "dst_block_idx", "rel_path": "dst_path"}
    )
    joined = arrow_join_lazyframes(
        cfg_edges,
        src_blocks,
        spec=JoinSpec(on=["function_goid_h128", "src_block_id"], how="left"),
    ).pipe(
        arrow_join_lazyframes,
        dst_blocks,
        spec=JoinSpec(on=["function_goid_h128", "dst_block_id"], how="left"),
    )
    src_pk = {
        "function_goid_h128": pl.col("function_goid_h128"),
        "block_idx": pl.col("src_block_idx"),
    }
    dst_pk = {
        "function_goid_h128": pl.col("function_goid_h128"),
        "block_idx": pl.col("dst_block_idx"),
    }
    extras = _pk_json_expr({"cfg_edge_kind": pl.col("edge_kind")})
    ordinal = _ordinal_expr(
        "graph.cfg_edges",
        {
            "function_goid_h128": pl.col("function_goid_h128"),
            "src_block_id": pl.col("src_block_id"),
            "dst_block_id": pl.col("dst_block_id"),
            "edge_kind": pl.col("edge_kind"),
        },
    )
    rel_path = pl.coalesce([pl.col("src_path"), pl.col("dst_path")])
    return (
        joined.filter(pl.col("src_block_idx").is_not_null())
        .filter(pl.col("dst_block_idx").is_not_null())
        .with_columns(
            _pk_expr(CFG_BLOCKS_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
            _pk_expr(CFG_BLOCKS_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
            pl.lit("CFG").alias("edge_kind"),
            pl.lit("FLOW").alias("edge_layer"),
            rel_path.alias("rel_path"),
            ordinal.alias("ordinal"),
            extras.alias("extras_json"),
        )
        .select(_CPG_EDGE_COLUMNS)
    )


def _dfg_edges_to_cpg(
    dfg_edges: pl.LazyFrame,
    cfg_blocks: pl.LazyFrame,
    goids: pl.LazyFrame,
) -> pl.LazyFrame:
    goid_ctx = goids.select(
        pl.col("goid_h128").alias("function_goid_h128"),
        "repo",
        "commit",
    )
    blocks = arrow_join_lazyframes(
        cfg_blocks,
        goid_ctx,
        spec=JoinSpec(on=["function_goid_h128"], how="left"),
    ).select(
        "function_goid_h128",
        "block_id",
        "block_idx",
        "repo",
        "commit",
        pl.col("file_path").alias("rel_path"),
    )
    src_blocks = blocks.rename(
        {"block_id": "src_block_id", "block_idx": "src_block_idx", "rel_path": "src_path"}
    )
    dst_blocks = blocks.rename(
        {"block_id": "dst_block_id", "block_idx": "dst_block_idx", "rel_path": "dst_path"}
    )
    joined = arrow_join_lazyframes(
        dfg_edges,
        src_blocks,
        spec=JoinSpec(on=["function_goid_h128", "src_block_id"], how="left"),
    ).pipe(
        arrow_join_lazyframes,
        dst_blocks,
        spec=JoinSpec(on=["function_goid_h128", "dst_block_id"], how="left"),
    )
    src_pk = {
        "function_goid_h128": pl.col("function_goid_h128"),
        "block_idx": pl.col("src_block_idx"),
    }
    dst_pk = {
        "function_goid_h128": pl.col("function_goid_h128"),
        "block_idx": pl.col("dst_block_idx"),
    }
    extras = _pk_json_expr(
        {
            "src_var": pl.col("src_var"),
            "dst_var": pl.col("dst_var"),
            "edge_kind": pl.col("edge_kind"),
            "via_phi": pl.col("via_phi"),
            "use_kind": pl.col("use_kind"),
        }
    )
    ordinal = _ordinal_expr(
        "graph.dfg_edges",
        {
            "function_goid_h128": pl.col("function_goid_h128"),
            "src_block_id": pl.col("src_block_id"),
            "dst_block_id": pl.col("dst_block_id"),
            "src_var": pl.col("src_var"),
            "dst_var": pl.col("dst_var"),
        },
    )
    rel_path = pl.coalesce([pl.col("src_path"), pl.col("dst_path")])
    return (
        joined.filter(pl.col("src_block_idx").is_not_null())
        .filter(pl.col("dst_block_idx").is_not_null())
        .with_columns(
            _pk_expr(CFG_BLOCKS_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
            _pk_expr(CFG_BLOCKS_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
            pl.lit("DFG").alias("edge_kind"),
            pl.lit("FLOW").alias("edge_layer"),
            rel_path.alias("rel_path"),
            ordinal.alias("ordinal"),
            extras.alias("extras_json"),
        )
        .select(_CPG_EDGE_COLUMNS)
    )


def _cdg_edges_to_cpg(
    cdg_edges: pl.LazyFrame,
    cfg_blocks: pl.LazyFrame,
    goids: pl.LazyFrame,
) -> pl.LazyFrame:
    goid_ctx = goids.select(
        pl.col("goid_h128").alias("function_goid_h128"),
        "repo",
        "commit",
    )
    blocks = arrow_join_lazyframes(
        cfg_blocks,
        goid_ctx,
        spec=JoinSpec(on=["function_goid_h128"], how="left"),
    ).select(
        "function_goid_h128",
        "block_id",
        "block_idx",
        "repo",
        "commit",
        pl.col("file_path").alias("rel_path"),
    )
    src_blocks = blocks.rename(
        {"block_id": "src_block_id", "block_idx": "src_block_idx", "rel_path": "src_path"}
    )
    dst_blocks = blocks.rename(
        {"block_id": "dst_block_id", "block_idx": "dst_block_idx", "rel_path": "dst_path"}
    )
    joined = arrow_join_lazyframes(
        cdg_edges,
        src_blocks,
        spec=JoinSpec(on=["function_goid_h128", "src_block_id"], how="left"),
    ).pipe(
        arrow_join_lazyframes,
        dst_blocks,
        spec=JoinSpec(on=["function_goid_h128", "dst_block_id"], how="left"),
    )
    src_pk = {
        "function_goid_h128": pl.col("function_goid_h128"),
        "block_idx": pl.col("src_block_idx"),
    }
    dst_pk = {
        "function_goid_h128": pl.col("function_goid_h128"),
        "block_idx": pl.col("dst_block_idx"),
    }
    extras = _pk_json_expr(
        {
            "via_succ_block_id": pl.col("via_succ_block_id"),
            "via_edge_kind": pl.col("via_edge_kind"),
        }
    )
    ordinal = _ordinal_expr(
        "graph.cdg_edges",
        {
            "function_goid_h128": pl.col("function_goid_h128"),
            "src_block_id": pl.col("src_block_id"),
            "dst_block_id": pl.col("dst_block_id"),
            "via_succ_block_id": pl.col("via_succ_block_id"),
        },
    )
    rel_path = pl.coalesce([pl.col("src_path"), pl.col("dst_path")])
    return (
        joined.filter(pl.col("src_block_idx").is_not_null())
        .filter(pl.col("dst_block_idx").is_not_null())
        .with_columns(
            _pk_expr(CFG_BLOCKS_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
            _pk_expr(CFG_BLOCKS_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
            pl.col("edge_kind").fill_null("CDG").alias("edge_kind"),
            pl.lit("FLOW").alias("edge_layer"),
            rel_path.alias("rel_path"),
            ordinal.alias("ordinal"),
            extras.alias("extras_json"),
        )
        .select(_CPG_EDGE_COLUMNS)
    )


def _call_wiring_calls_to_cpg(
    call_edges: pl.LazyFrame,
    cfg_blocks: pl.LazyFrame,
    syntax_nodes: pl.LazyFrame,
) -> pl.LazyFrame:
    syntax_keys = _syntax_node_keys(syntax_nodes).rename(
        {
            "rel_path": "call_rel_path",
            "producer": "call_producer",
            "node_id": "call_node_id",
        }
    )
    blocks = cfg_blocks.select("function_goid_h128", "block_id", "block_idx")
    joined = (
        arrow_join_lazyframes(
            call_edges,
            syntax_keys,
            spec=JoinSpec(on=["repo", "commit", "call_node_id"], how="left"),
        )
        .pipe(
            arrow_join_lazyframes,
            blocks,
            spec=JoinSpec(
                left_on=["callee_entry_block_id"],
                right_on=["block_id"],
                how="left",
            ),
        )
        .drop(["block_id"])
    )
    src_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("call_rel_path"),
        "producer": pl.col("call_producer"),
        "node_id": pl.col("call_node_id"),
    }
    dst_pk = {
        "function_goid_h128": pl.col("function_goid_h128"),
        "block_idx": pl.col("block_idx"),
    }
    call_extras = pl.col("extras_json").map_elements(decode_payload, return_dtype=pl.Object)
    extras = _pk_json_expr(
        {
            "call_id": pl.col("call_id"),
            "confidence": pl.col("confidence"),
            "call_extras": call_extras,
        }
    )
    ordinal = _ordinal_expr(
        "graph.cpg_edges_calls",
        {"call_id": pl.col("call_id"), "callee_entry_block_id": pl.col("callee_entry_block_id")},
    )
    return (
        joined.filter(pl.col("call_node_id").is_not_null())
        .filter(pl.col("block_idx").is_not_null())
        .with_columns(
            _pk_expr(SYNTAX_NODES_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
            _pk_expr(CFG_BLOCKS_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
            pl.col("edge_kind").fill_null("CALLS").alias("edge_kind"),
            pl.lit("FLOW").alias("edge_layer"),
            pl.col("call_rel_path").alias("rel_path"),
            ordinal.alias("ordinal"),
            extras.alias("extras_json"),
        )
        .select(_CPG_EDGE_COLUMNS)
    )


def _call_wiring_arg_to_param_to_cpg(
    arg_edges: pl.LazyFrame,
    syntax_nodes: pl.LazyFrame,
) -> pl.LazyFrame:
    syntax_keys = _syntax_node_keys(syntax_nodes)
    src_keys = syntax_keys.rename(
        {
            "rel_path": "src_rel_path",
            "producer": "src_producer",
            "node_id": "src_arg_node_id",
        }
    )
    dst_keys = syntax_keys.rename(
        {
            "rel_path": "dst_rel_path",
            "producer": "dst_producer",
            "node_id": "dst_param_node_id",
        }
    )
    joined = arrow_join_lazyframes(
        arg_edges,
        src_keys,
        spec=JoinSpec(on=["repo", "commit", "src_arg_node_id"], how="left"),
    ).pipe(
        arrow_join_lazyframes,
        dst_keys,
        spec=JoinSpec(on=["repo", "commit", "dst_param_node_id"], how="left"),
    )
    src_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("src_rel_path"),
        "producer": pl.col("src_producer"),
        "node_id": pl.col("src_arg_node_id"),
    }
    dst_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("dst_rel_path"),
        "producer": pl.col("dst_producer"),
        "node_id": pl.col("dst_param_node_id"),
    }
    extras = _pk_json_expr(
        {
            "call_id": pl.col("call_id"),
            "arg_ordinal": pl.col("arg_ordinal"),
            "param_ordinal": pl.col("param_ordinal"),
            "arg_name": pl.col("arg_name"),
            "param_name": pl.col("param_name"),
            "arg_slot": pl.col("arg_slot"),
            "arg_role": pl.col("arg_role"),
            "arg_is_implicit": pl.col("arg_is_implicit"),
            "call_kind": pl.col("call_kind"),
            "augop": pl.col("augop"),
            "confidence": pl.col("confidence"),
        }
    )
    ordinal = _ordinal_expr(
        "graph.cpg_edges_arg_to_param",
        {
            "call_id": pl.col("call_id"),
            "arg_ordinal": pl.col("arg_ordinal"),
            "param_ordinal": pl.col("param_ordinal"),
            "src_arg_node_id": pl.col("src_arg_node_id"),
            "dst_param_node_id": pl.col("dst_param_node_id"),
        },
    )
    return (
        joined.filter(pl.col("src_arg_node_id").is_not_null())
        .filter(pl.col("dst_param_node_id").is_not_null())
        .with_columns(
            _pk_expr(SYNTAX_NODES_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
            _pk_expr(SYNTAX_NODES_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
            pl.col("edge_kind").fill_null("ARG_TO_PARAM").alias("edge_kind"),
            pl.lit("FLOW").alias("edge_layer"),
            pl.col("src_rel_path").alias("rel_path"),
            ordinal.alias("ordinal"),
            extras.alias("extras_json"),
        )
        .select(_CPG_EDGE_COLUMNS)
    )


def _call_wiring_ret_to_call_to_cpg(
    ret_edges: pl.LazyFrame,
    cfg_blocks: pl.LazyFrame,
    syntax_nodes: pl.LazyFrame,
) -> pl.LazyFrame:
    syntax_keys = _syntax_node_keys(syntax_nodes).rename(
        {
            "rel_path": "call_rel_path",
            "producer": "call_producer",
            "node_id": "call_node_id",
        }
    )
    blocks = cfg_blocks.select("function_goid_h128", "block_id", "block_idx")
    joined = (
        arrow_join_lazyframes(
            ret_edges,
            syntax_keys,
            spec=JoinSpec(on=["repo", "commit", "call_node_id"], how="left"),
        )
        .pipe(
            arrow_join_lazyframes,
            blocks,
            spec=JoinSpec(
                left_on=["exit_block_id"],
                right_on=["block_id"],
                how="left",
            ),
        )
        .drop(["block_id"])
    )
    src_pk = {
        "function_goid_h128": pl.col("function_goid_h128"),
        "block_idx": pl.col("block_idx"),
    }
    dst_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("call_rel_path"),
        "producer": pl.col("call_producer"),
        "node_id": pl.col("call_node_id"),
    }
    extras = _pk_json_expr(
        {
            "call_id": pl.col("call_id"),
            "confidence": pl.col("confidence"),
            "target_role": pl.col("target_role"),
            "call_kind": pl.col("call_kind"),
            "origin": pl.col("origin"),
            "summary": pl.col("extras_json"),
        }
    )
    ordinal = _ordinal_expr(
        "graph.cpg_edges_ret_to_call",
        {"call_id": pl.col("call_id"), "exit_block_id": pl.col("exit_block_id")},
    )
    return (
        joined.filter(pl.col("call_node_id").is_not_null())
        .filter(pl.col("block_idx").is_not_null())
        .with_columns(
            _pk_expr(CFG_BLOCKS_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
            _pk_expr(SYNTAX_NODES_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
            pl.col("edge_kind").fill_null("RET_TO_CALL").alias("edge_kind"),
            pl.lit("FLOW").alias("edge_layer"),
            pl.col("call_rel_path").alias("rel_path"),
            ordinal.alias("ordinal"),
            extras.alias("extras_json"),
        )
        .select(_CPG_EDGE_COLUMNS)
    )


def _collect_rows(
    frame: pl.LazyFrame,
    *,
    columns: Sequence[str],
) -> list[dict[str, object]]:
    if not set(columns).issubset(frame.columns):
        return []
    return frame.select(list(columns)).collect().to_dicts()


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


def _coerce_var_key(value: object) -> tuple[str, str, str] | None:
    if not isinstance(value, tuple) or len(value) != _VAR_KEY_LEN:
        return None
    first, second, third = value
    if isinstance(first, str) and isinstance(second, str) and isinstance(third, str):
        return first, second, third
    return None


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


def _edge_rows_to_lazyframe(rows: list[dict[str, object]]) -> pl.LazyFrame:
    if not rows:
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
    frame = pl.DataFrame(rows).lazy()
    return _select_edge_columns(frame)


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


def _ast_nodes_by_path(ast_rows: list[dict[str, object]]) -> dict[str, list[dict[str, object]]]:
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


def _ast_binding_edges_to_cpg(
    ast_nodes: pl.LazyFrame,
    scopes: pl.LazyFrame,
    bindings: pl.LazyFrame,
    resolution_edges: pl.LazyFrame,
) -> pl.LazyFrame:
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
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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
    instructions: pl.LazyFrame,
    ast_nodes: pl.LazyFrame,
) -> pl.LazyFrame:
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
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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
    instructions: pl.LazyFrame,
    syntax_calls: pl.LazyFrame,
) -> pl.LazyFrame:
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
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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
    instructions: pl.LazyFrame,
    syntax_calls: pl.LazyFrame,
    scip_symbols: pl.LazyFrame,
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
    instructions: pl.LazyFrame,
    syntax_calls: pl.LazyFrame,
    scip_symbols: pl.LazyFrame,
) -> pl.LazyFrame:
    inputs = _callsite_symbol_inputs(instructions, syntax_calls, scip_symbols)
    if inputs is None:
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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
    defuse_events: pl.LazyFrame,
    instructions: pl.LazyFrame,
    ast_nodes: pl.LazyFrame,
) -> pl.LazyFrame:
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
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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
    instructions: pl.LazyFrame,
    blocks: pl.LazyFrame,
) -> pl.LazyFrame:
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
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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


def _py_sym_scope_edges_to_cpg(scope_edges: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "rel_path",
        "parent_scope_id",
        "child_scope_id",
        "edge_kind",
    }
    if not required.issubset(scope_edges.columns):
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
    base = scope_edges.filter(
        pl.col("parent_scope_id").is_not_null() & pl.col("child_scope_id").is_not_null()
    )
    parent_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "scope_id": pl.col("parent_scope_id"),
    }
    child_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "scope_id": pl.col("child_scope_id"),
    }
    extras = _pk_json_expr({"edge_kind": pl.col("edge_kind")})
    owns_ordinal = _ordinal_expr(
        "graph.cpg_edges_scope",
        {
            "parent_scope_id": pl.col("parent_scope_id"),
            "child_scope_id": pl.col("child_scope_id"),
            "edge_kind": pl.lit("OWNS_SCOPE"),
        },
    )
    parent_ordinal = _ordinal_expr(
        "graph.cpg_edges_scope",
        {
            "parent_scope_id": pl.col("parent_scope_id"),
            "child_scope_id": pl.col("child_scope_id"),
            "edge_kind": pl.lit("PARENT_SCOPE"),
        },
    )
    owns = base.with_columns(
        _pk_expr(PY_SYM_SCOPES_TABLE_KEY, parent_pk).alias("src_cpg_node_id"),
        _pk_expr(PY_SYM_SCOPES_TABLE_KEY, child_pk).alias("dst_cpg_node_id"),
        pl.lit("OWNS_SCOPE").alias("edge_kind"),
        pl.lit("SYMBOL").alias("edge_layer"),
        pl.col("rel_path"),
        owns_ordinal.alias("ordinal"),
        extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)
    parent = base.with_columns(
        _pk_expr(PY_SYM_SCOPES_TABLE_KEY, child_pk).alias("src_cpg_node_id"),
        _pk_expr(PY_SYM_SCOPES_TABLE_KEY, parent_pk).alias("dst_cpg_node_id"),
        pl.lit("PARENT_SCOPE").alias("edge_kind"),
        pl.lit("SYMBOL").alias("edge_layer"),
        pl.col("rel_path"),
        parent_ordinal.alias("ordinal"),
        extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)
    return pl.concat([owns, parent], how="vertical_relaxed")


def _py_sym_namespace_edges_to_cpg(
    namespace_edges: pl.LazyFrame,
    bindings: pl.LazyFrame,
) -> pl.LazyFrame:
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
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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
    binding = bindings_by_scope.get(
        (rel_path, scope_id, name)
    )
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


def _py_sym_binding_edges_to_cpg(bindings: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "rel_path",
        "scope_id",
        "binding_id",
        "binding_kind",
        "name",
    }
    if not required.issubset(bindings.columns):
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
    scope_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "scope_id": pl.col("scope_id"),
    }
    binding_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "binding_id": pl.col("binding_id"),
    }
    extras = _pk_json_expr(
        {
            "binding_kind": pl.col("binding_kind"),
            "declared_here": pl.col("declared_here"),
            "referenced_here": pl.col("referenced_here"),
            "assigned_here": pl.col("assigned_here"),
            "annotated_here": pl.col("annotated_here"),
            "scoping_class": pl.col("scoping_class"),
        }
    )
    ordinal = _ordinal_expr(
        "graph.cpg_edges_binding",
        {
            "scope_id": pl.col("scope_id"),
            "binding_id": pl.col("binding_id"),
        },
    )
    return bindings.with_columns(
        _pk_expr(PY_SYM_SCOPES_TABLE_KEY, scope_pk).alias("src_cpg_node_id"),
        _pk_expr(PY_SYM_BINDINGS_TABLE_KEY, binding_pk).alias("dst_cpg_node_id"),
        pl.lit("DECLARES").alias("edge_kind"),
        pl.lit("SYMBOL").alias("edge_layer"),
        pl.col("rel_path"),
        ordinal.alias("ordinal"),
        extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)


def _py_sym_resolution_edges_to_cpg(resolution_edges: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "rel_path",
        "edge_id",
        "src_binding_id",
        "dst_binding_id",
        "kind",
    }
    if not required.issubset(resolution_edges.columns):
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
    src_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "binding_id": pl.col("src_binding_id"),
    }
    dst_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "binding_id": pl.col("dst_binding_id"),
    }
    extras = _pk_json_expr(
        {
            "kind": pl.col("kind"),
            "confidence": pl.col("confidence"),
            "reason": pl.col("reason"),
        }
    )
    ordinal = _ordinal_expr(
        PY_SYM_RESOLUTION_EDGES_TABLE_KEY,
        {"edge_id": pl.col("edge_id")},
    )
    return resolution_edges.with_columns(
        _pk_expr(PY_SYM_BINDINGS_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
        _pk_expr(PY_SYM_BINDINGS_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
        pl.lit("RESOLVES_TO").alias("edge_kind"),
        pl.lit("SYMBOL").alias("edge_layer"),
        pl.col("rel_path"),
        ordinal.alias("ordinal"),
        extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)


def _py_sym_binding_symbol_edges_to_cpg(
    bindings: pl.LazyFrame,
    scopes: pl.LazyFrame,
    scip_symbols: pl.LazyFrame,
) -> pl.LazyFrame:
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
        not required_bindings.issubset(bindings.columns)
        or not required_scopes.issubset(scopes.columns)
        or not required_symbols.issubset(scip_symbols.columns)
    ):
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
    scope_fields = scopes.select("repo", "commit", "rel_path", "scope_id", "qualpath")
    joined = arrow_join_lazyframes(
        bindings,
        scope_fields,
        spec=JoinSpec(on=["repo", "commit", "rel_path", "scope_id"], how="left"),
    )
    scope_qualname = (
        pl.col("qualpath")
        .str.replace_all("::", ".")
        .str.replace_all(r"#\d+", "")
        .alias("scope_qualname")
    )
    qualified = joined.with_columns(scope_qualname)
    binding_qualname = (
        pl.when(pl.col("scope_qualname").is_not_null())
        .then(pl.concat_str([pl.col("scope_qualname"), pl.lit("."), pl.col("name")]))
        .otherwise(None)
        .alias("binding_qualname")
    )
    bindings_named = qualified.with_columns(binding_qualname).filter(
        pl.col("binding_qualname").is_not_null()
    )
    symbols = scip_symbols.select("repo", "commit", "symbol", "display_name")
    matched = arrow_join_lazyframes(
        bindings_named,
        symbols,
        spec=JoinSpec(
            left_on=["repo", "commit", "binding_qualname"],
            right_on=["repo", "commit", "display_name"],
            how="inner",
        ),
    )
    binding_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "binding_id": pl.col("binding_id"),
    }
    symbol_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "symbol": pl.col("symbol"),
    }
    extras = _pk_json_expr(
        {
            "binding_kind": pl.col("binding_kind"),
            "match_kind": pl.lit("qualpath"),
        }
    )
    ordinal = _ordinal_expr(
        "graph.cpg_edges_binding_symbol",
        {"binding_id": pl.col("binding_id"), "symbol": pl.col("symbol")},
    )
    return matched.with_columns(
        _pk_expr(PY_SYM_BINDINGS_TABLE_KEY, binding_pk).alias("src_cpg_node_id"),
        _pk_expr(SCIP_SYMBOLS_TABLE_KEY, symbol_pk).alias("dst_cpg_node_id"),
        pl.lit("BINDS_SYMBOL").alias("edge_kind"),
        pl.lit("SYMBOL").alias("edge_layer"),
        pl.col("rel_path"),
        ordinal.alias("ordinal"),
        extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)


def _py_bc_cfg_edges_to_cpg(cfg_edges: pl.LazyFrame) -> pl.LazyFrame:
    required = {
        "repo",
        "commit",
        "rel_path",
        "edge_id",
        "src_block_id",
        "dst_block_id",
        "kind",
    }
    if not required.issubset(cfg_edges.columns):
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
    src_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "block_id": pl.col("src_block_id"),
    }
    dst_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "rel_path": pl.col("rel_path"),
        "block_id": pl.col("dst_block_id"),
    }
    extras = _pk_json_expr(
        {
            "kind": pl.col("kind"),
            "cond_instr_id": pl.col("cond_instr_id"),
            "exc_entry_index": pl.col("exc_entry_index"),
        }
    )
    ordinal = _ordinal_expr(PY_BC_CFG_EDGES_TABLE_KEY, {"edge_id": pl.col("edge_id")})
    return cfg_edges.with_columns(
        _pk_expr(PY_BC_BLOCKS_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
        _pk_expr(PY_BC_BLOCKS_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
        pl.lit("CFG").alias("edge_kind"),
        pl.lit("FLOW").alias("edge_layer"),
        pl.col("rel_path"),
        ordinal.alias("ordinal"),
        extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)


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
    defuse_events: pl.LazyFrame
    code_units: pl.LazyFrame
    scopes: pl.LazyFrame
    bindings: pl.LazyFrame
    resolution_edges: pl.LazyFrame
    blocks: pl.LazyFrame
    cfg_edges: pl.LazyFrame


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
    defuse_events: pl.LazyFrame,
    code_units: pl.LazyFrame,
    scopes: pl.LazyFrame,
    bindings: pl.LazyFrame,
    resolution_edges: pl.LazyFrame,
) -> pl.LazyFrame:
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
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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


def _py_bc_reaches_edges_to_cpg(inputs: _PyBcReachesInputs) -> pl.LazyFrame:
    rows = _collect_py_bc_reaches_rows(inputs)
    if rows is None:
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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


def _inspect_full_qualname_expr() -> pl.Expr:
    module_name = pl.col("module_name")
    qualname = pl.col("qualname")
    module_prefix = pl.concat_str([module_name, pl.lit(".")])
    return (
        pl.when(module_name.is_not_null() & qualname.is_not_null() & (qualname == module_name))
        .then(module_name)
        .when(
            module_name.is_not_null()
            & qualname.is_not_null()
            & qualname.str.starts_with(module_prefix)
        )
        .then(qualname)
        .when(module_name.is_not_null() & qualname.is_not_null())
        .then(pl.concat_str([module_name, pl.lit("."), qualname]))
        .when(module_name.is_not_null())
        .then(module_name)
        .otherwise(None)
    )


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
    mappings = _assign_args_to_params(args, params)
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
    syntax_calls: pl.LazyFrame,
    syntax_call_args: pl.LazyFrame,
    inspect_objects: pl.LazyFrame,
    inspect_signatures: pl.LazyFrame,
    inspect_signature_params: pl.LazyFrame,
) -> pl.LazyFrame:
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
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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
    signatures: pl.LazyFrame,
    params: pl.LazyFrame,
) -> pl.LazyFrame:
    required_signatures = {"repo", "commit", "signature_id", "object_id"}
    required_params = {"repo", "commit", "signature_id", "param_index"}
    if not required_signatures.issubset(signatures.columns) or not required_params.issubset(
        params.columns
    ):
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
    sig_src = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "object_id": pl.col("object_id"),
    }
    sig_dst = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "signature_id": pl.col("signature_id"),
    }
    sig_extras = _pk_json_expr(
        {
            "variant": pl.col("variant"),
            "follow_wrapped": pl.col("follow_wrapped"),
            "eval_str": pl.col("eval_str"),
            "status": pl.col("status"),
        }
    )
    sig_ordinal = _ordinal_expr(
        "graph.cpg_edges_inspect_signature",
        {"signature_id": pl.col("signature_id")},
    )
    sig_edges = signatures.with_columns(
        _pk_expr(PY_INSPECT_OBJECTS_TABLE_KEY, sig_src).alias("src_cpg_node_id"),
        _pk_expr(PY_INSPECT_SIGNATURES_TABLE_KEY, sig_dst).alias("dst_cpg_node_id"),
        pl.lit("HAS_SIGNATURE").alias("edge_kind"),
        pl.lit("SYMBOL").alias("edge_layer"),
        pl.lit(None).alias("rel_path"),
        sig_ordinal.alias("ordinal"),
        sig_extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)
    param_src = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "signature_id": pl.col("signature_id"),
    }
    param_dst = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "signature_id": pl.col("signature_id"),
        "param_index": pl.col("param_index"),
    }
    param_extras = _pk_json_expr(
        {
            "param_index": pl.col("param_index"),
            "name": pl.col("name"),
            "kind": pl.col("kind"),
            "status": pl.col("status"),
        }
    )
    param_ordinal = _ordinal_expr(
        "graph.cpg_edges_inspect_signature_param",
        {
            "signature_id": pl.col("signature_id"),
            "param_index": pl.col("param_index"),
        },
    )
    param_edges = params.with_columns(
        _pk_expr(PY_INSPECT_SIGNATURES_TABLE_KEY, param_src).alias("src_cpg_node_id"),
        _pk_expr(PY_INSPECT_SIGNATURE_PARAMS_TABLE_KEY, param_dst).alias("dst_cpg_node_id"),
        pl.lit("HAS_PARAM").alias("edge_kind"),
        pl.lit("SYMBOL").alias("edge_layer"),
        pl.lit(None).alias("rel_path"),
        param_ordinal.alias("ordinal"),
        param_extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)
    return pl.concat([sig_edges, param_edges], how="vertical_relaxed")


def _inspect_to_ast_edges_to_cpg(
    inspect_objects: pl.LazyFrame,
    ast_nodes: pl.LazyFrame,
) -> pl.LazyFrame:
    required_inspect = {"repo", "commit", "object_id", "module_name", "qualname"}
    required_ast = {"hash", "qualname", "node_type", "path"}
    if not required_inspect.issubset(inspect_objects.columns) or not required_ast.issubset(
        ast_nodes.columns
    ):
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
    inspect_full = inspect_objects.with_columns(
        _inspect_full_qualname_expr().alias("full_qualname")
    ).filter(pl.col("full_qualname").is_not_null())
    ast_defs = ast_nodes.filter(pl.col("qualname").is_not_null())
    joined = arrow_join_lazyframes(
        inspect_full,
        ast_defs,
        spec=JoinSpec(left_on=["full_qualname"], right_on=["qualname"], how="inner"),
    )
    src_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "object_id": pl.col("object_id"),
    }
    dst_pk = {"hash": pl.col("hash")}
    extras = _pk_json_expr(
        {
            "match_kind": pl.lit("qualname"),
            "ast_kind": pl.col("node_type"),
        }
    )
    ordinal = _ordinal_expr(
        "graph.cpg_edges_inspect_ast",
        {"object_id": pl.col("object_id"), "ast_hash": pl.col("hash")},
    )
    return joined.with_columns(
        _pk_expr(PY_INSPECT_OBJECTS_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
        _pk_expr(AST_NODES_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
        pl.lit("INSPECT_ANCHORS_AST").alias("edge_kind"),
        pl.lit("SYMBOL").alias("edge_layer"),
        pl.col("path").alias("rel_path"),
        ordinal.alias("ordinal"),
        extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)


def _inspect_to_scip_edges_to_cpg(
    inspect_objects: pl.LazyFrame,
    scip_symbols: pl.LazyFrame,
) -> pl.LazyFrame:
    required_inspect = {"repo", "commit", "object_id", "module_name", "qualname"}
    required_symbols = {"repo", "commit", "symbol", "display_name"}
    if not required_inspect.issubset(inspect_objects.columns) or not required_symbols.issubset(
        scip_symbols.columns
    ):
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
    inspect_full = inspect_objects.with_columns(
        _inspect_full_qualname_expr().alias("full_qualname")
    ).filter(pl.col("full_qualname").is_not_null())
    symbols = scip_symbols.select("repo", "commit", "symbol", "display_name")
    joined = arrow_join_lazyframes(
        inspect_full,
        symbols,
        spec=JoinSpec(
            left_on=["repo", "commit", "full_qualname"],
            right_on=["repo", "commit", "display_name"],
            how="inner",
        ),
    )
    src_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "object_id": pl.col("object_id"),
    }
    dst_pk = {
        "repo": pl.col("repo"),
        "commit": pl.col("commit"),
        "symbol": pl.col("symbol"),
    }
    extras = _pk_json_expr({"match_kind": pl.lit("display_name")})
    ordinal = _ordinal_expr(
        "graph.cpg_edges_inspect_symbol",
        {"object_id": pl.col("object_id"), "symbol": pl.col("symbol")},
    )
    return joined.with_columns(
        _pk_expr(PY_INSPECT_OBJECTS_TABLE_KEY, src_pk).alias("src_cpg_node_id"),
        _pk_expr(SCIP_SYMBOLS_TABLE_KEY, dst_pk).alias("dst_cpg_node_id"),
        pl.lit("INSPECT_SYMBOL").alias("edge_kind"),
        pl.lit("SYMBOL").alias("edge_layer"),
        pl.lit(None).alias("rel_path"),
        ordinal.alias("ordinal"),
        extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)


def _py_inspect_class_mro_edges_to_cpg(class_mro: pl.LazyFrame) -> pl.LazyFrame:
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
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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


def _py_inspect_class_attr_edges_to_cpg(class_attrs: pl.LazyFrame) -> pl.LazyFrame:
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
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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


def _runtime_state_extras_expr() -> pl.Expr:
    return _pk_json_expr(
        {
            "state_kind": pl.col("state_kind"),
            "state": pl.col("state"),
            "object_kind": pl.col("object_kind"),
            "frame_file": pl.col("frame_file"),
            "frame_module": pl.col("frame_module"),
            "frame_line": pl.col("frame_line"),
            "frame_offset": pl.col("frame_offset"),
            "status": pl.col("status"),
        }
    )


def _runtime_state_has_state_edges(runtime_state: pl.LazyFrame) -> pl.LazyFrame:
    has_state = runtime_state.filter(pl.col("frame_object_id").is_not_null())
    extras = _runtime_state_extras_expr()
    ordinal = _ordinal_expr(
        "graph.cpg_edges_inspect_runtime_state",
        {
            "object_id": pl.col("object_id"),
            "state_kind": pl.col("state_kind"),
            "frame_object_id": pl.col("frame_object_id"),
        },
    )
    return has_state.with_columns(
        _pk_expr(
            PY_INSPECT_OBJECTS_TABLE_KEY,
            {"repo": pl.col("repo"), "commit": pl.col("commit"), "object_id": pl.col("object_id")},
        ).alias("src_cpg_node_id"),
        _pk_expr(
            PY_INSPECT_OBJECTS_TABLE_KEY,
            {
                "repo": pl.col("repo"),
                "commit": pl.col("commit"),
                "object_id": pl.col("frame_object_id"),
            },
        ).alias("dst_cpg_node_id"),
        pl.lit("HAS_STATE").alias("edge_kind"),
        pl.lit("FLOW").alias("edge_layer"),
        pl.lit(None).alias("rel_path"),
        ordinal.alias("ordinal"),
        extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)


def _runtime_state_with_path(runtime_state: pl.LazyFrame) -> pl.LazyFrame:
    frame_name = pl.coalesce([pl.col("frame_code_qualname"), pl.col("frame_code_name")])
    return (
        runtime_state.filter(pl.col("frame_module").is_not_null())
        .with_columns(frame_name.alias("frame_name"))
        .filter(pl.col("frame_name").is_not_null())
        .with_columns(
            pl.concat_str([pl.col("frame_module"), pl.lit("::"), pl.col("frame_name")]).alias(
                "frame_qualpath"
            )
        )
        .filter(pl.col("frame_offset").is_not_null())
        .filter(pl.col("frame_offset") >= 0)
    )


def _runtime_state_instr_edges(
    runtime_state: pl.LazyFrame,
    code_units: pl.LazyFrame,
    instructions: pl.LazyFrame,
) -> pl.LazyFrame:
    runtime_with_path = _runtime_state_with_path(runtime_state)
    joined_units = arrow_join_lazyframes(
        runtime_with_path,
        code_units.select("repo", "commit", "rel_path", "code_unit_id", "qualpath"),
        spec=JoinSpec(
            left_on=["repo", "commit", "frame_qualpath"],
            right_on=["repo", "commit", "qualpath"],
            how="left",
        ),
    )
    joined_instr = arrow_join_lazyframes(
        joined_units,
        instructions.select("repo", "commit", "rel_path", "code_unit_id", "instr_id", "offset"),
        spec=JoinSpec(
            left_on=["repo", "commit", "rel_path", "code_unit_id", "frame_offset"],
            right_on=["repo", "commit", "rel_path", "code_unit_id", "offset"],
            how="left",
        ),
    )
    instr_filtered = joined_instr.filter(pl.col("instr_id").is_not_null())
    edge_kind = (
        pl.when(pl.col("state_kind") == "traceback")
        .then(pl.lit("TRACEBACK_AT_INSTR"))
        .otherwise(pl.lit("FRAME_AT_INSTR"))
    )
    extras = _runtime_state_extras_expr()
    ordinal = _ordinal_expr(
        "graph.cpg_edges_inspect_runtime_state",
        {
            "object_id": pl.col("object_id"),
            "state_kind": pl.col("state_kind"),
            "instr_id": pl.col("instr_id"),
            "frame_offset": pl.col("frame_offset"),
        },
    )
    return instr_filtered.with_columns(
        _pk_expr(
            PY_INSPECT_OBJECTS_TABLE_KEY,
            {"repo": pl.col("repo"), "commit": pl.col("commit"), "object_id": pl.col("object_id")},
        ).alias("src_cpg_node_id"),
        _pk_expr(
            PY_BC_INSTRUCTIONS_TABLE_KEY,
            {
                "repo": pl.col("repo"),
                "commit": pl.col("commit"),
                "rel_path": pl.col("rel_path"),
                "code_unit_id": pl.col("code_unit_id"),
                "instr_id": pl.col("instr_id"),
            },
        ).alias("dst_cpg_node_id"),
        edge_kind.alias("edge_kind"),
        pl.lit("FLOW").alias("edge_layer"),
        pl.col("rel_path"),
        ordinal.alias("ordinal"),
        extras.alias("extras_json"),
    ).select(_CPG_EDGE_COLUMNS)


def _py_inspect_runtime_state_edges_to_cpg(
    runtime_state: pl.LazyFrame,
    code_units: pl.LazyFrame,
    instructions: pl.LazyFrame,
) -> pl.LazyFrame:
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
        not required_state.issubset(runtime_state.columns)
        or not required_units.issubset(code_units.columns)
        or not required_instr.issubset(instructions.columns)
    ):
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
    has_state_edges = _runtime_state_has_state_edges(runtime_state)
    instr_edges = _runtime_state_instr_edges(runtime_state, code_units, instructions)
    combined = pl.concat([has_state_edges, instr_edges], how="vertical_relaxed")
    if combined.columns:
        return combined.select(_CPG_EDGE_COLUMNS)
    return empty_frame_for_table(CPG_EDGES_TABLE_KEY)


def _py_inspect_unwrap_edges_to_cpg(unwrap_hops: pl.LazyFrame) -> pl.LazyFrame:
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
        return empty_frame_for_table(CPG_EDGES_TABLE_KEY)
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
            src_pk = {"repo": repo, "commit": commit, "object_id": src_id}
            dst_pk = {"repo": repo, "commit": commit, "object_id": dst_id}
            extras = {
                "root_object_id": root_object_id,
                "hop": items[idx].get("hop"),
                "has_wrapped": items[idx].get("has_wrapped"),
                "has_signature_override": items[idx].get("has_signature_override"),
                "stop_reason": items[idx].get("stop_reason"),
            }
            hop_value = _coerce_int(items[idx].get("hop"))
            ordinal = _stable_ordinal(
                "graph.cpg_edges_inspect_wraps",
                {"root_object_id": root_object_id, "hop": hop_value},
            )
            edges.append(
                {
                    "repo": repo,
                    "commit": commit,
                    "src_cpg_node_id": _stable_cpg_id(PY_INSPECT_OBJECTS_TABLE_KEY, src_pk),
                    "dst_cpg_node_id": _stable_cpg_id(PY_INSPECT_OBJECTS_TABLE_KEY, dst_pk),
                    "edge_kind": "WRAPS",
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
        syntax_edges=tabular_to_lazyframe(q__core__syntax_edges),
        occ_syntax=tabular_to_lazyframe(q__core__scip_occurrence_syntax_xref),
        occ_span=tabular_to_lazyframe(q__core__scip_occurrence_span_xref),
        symbol_rels=tabular_to_lazyframe(q__core__scip_symbol_relationships),
        symbol_goid=tabular_to_lazyframe(q__core__scip_symbol_goid_xref),
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
        goids=tabular_to_lazyframe(q__core__goids),
        cfg_edges=tabular_to_lazyframe(q__graph__cfg_edges),
        dfg_edges=tabular_to_lazyframe(q__graph__dfg_edges),
        cfg_blocks=tabular_to_lazyframe(q__graph__cfg_blocks),
        cdg_edges=tabular_to_lazyframe(q__graph__cdg_edges),
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
        call_edges=tabular_to_lazyframe(q__graph__call_graph_edges),
        import_edges=tabular_to_lazyframe(q__graph__import_graph_edges),
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
        call_edges=tabular_to_lazyframe(q__graph__cpg_edges_calls),
        arg_to_param_edges=tabular_to_lazyframe(q__graph__cpg_edges_arg_to_param),
        ret_to_call_edges=tabular_to_lazyframe(q__graph__cpg_edges_ret_to_call),
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
        syntax_nodes=tabular_to_lazyframe(q__core__syntax_nodes),
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
        py_sym_scopes=tabular_to_lazyframe(q__core__py_sym_scopes),
        py_sym_bindings=tabular_to_lazyframe(q__core__py_sym_bindings),
        py_sym_scope_edges=tabular_to_lazyframe(q__core__py_sym_scope_edges),
        py_sym_namespace_edges=tabular_to_lazyframe(q__core__py_sym_namespace_edges),
        py_sym_resolution_edges=tabular_to_lazyframe(q__core__py_sym_resolution_edges),
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
        ast_nodes=tabular_to_lazyframe(q__core__ast_nodes),
        scip_symbols=tabular_to_lazyframe(q__core__scip_symbol_information),
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
        py_bc_code_units=tabular_to_lazyframe(q__core__py_bc_code_units),
        py_bc_instructions=tabular_to_lazyframe(q__core__py_bc_instructions),
        py_bc_blocks=tabular_to_lazyframe(q__core__py_bc_blocks),
        py_bc_cfg_edges=tabular_to_lazyframe(q__core__py_bc_cfg_edges),
        py_bc_defuse_events=tabular_to_lazyframe(q__core__py_bc_defuse_events),
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
        syntax_calls=tabular_to_lazyframe(q__core__syntax_calls),
        syntax_call_args=tabular_to_lazyframe(q__core__syntax_call_args),
    )


def cpg_edge_overlay_inspect_core_inputs(
    q__core__py_inspect_objects: InferableTabularInput,
    q__core__py_inspect_class_mro: InferableTabularInput,
    q__core__py_inspect_class_attrs: InferableTabularInput,
    q__core__py_inspect_unwrap_hops: InferableTabularInput,
) -> _CpgOverlayInspectCoreInputs:
    """Collect core inspect inputs for CPG edge assembly.

    Returns
    -------
    _CpgOverlayInspectCoreInputs
        Core inspect inputs for CPG edge assembly.
    """
    return _CpgOverlayInspectCoreInputs(
        py_inspect_objects=tabular_to_lazyframe(q__core__py_inspect_objects),
        py_inspect_class_mro=tabular_to_lazyframe(q__core__py_inspect_class_mro),
        py_inspect_class_attrs=tabular_to_lazyframe(q__core__py_inspect_class_attrs),
        py_inspect_unwrap_hops=tabular_to_lazyframe(q__core__py_inspect_unwrap_hops),
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
        py_inspect_signatures=tabular_to_lazyframe(q__core__py_inspect_signatures),
        py_inspect_signature_params=tabular_to_lazyframe(q__core__py_inspect_signature_params),
        py_inspect_runtime_state=tabular_to_lazyframe(q__core__py_inspect_runtime_state),
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
        py_inspect_signature_params=cpg_edge_overlay_inspect_runtime_inputs.py_inspect_signature_params,
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


def cpg_edges(
    cpg_edge_core_inputs: _CpgEdgeCoreInputs,
    cpg_edge_overlay_inputs: _CpgOverlayEdgeInputs,
) -> pl.LazyFrame:
    """Build CPG edges from syntax, symbol, and flow sources.

    Returns
    -------
    pl.LazyFrame
        LazyFrame for graph.cpg_edges.
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
        ),
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
            overlay_inputs.ast_nodes,
        ),
        _inspect_to_scip_edges_to_cpg(
            overlay_inputs.py_inspect_objects,
            overlay_inputs.scip_symbols,
        ),
    ]
    combined = pl.concat(frames, how="vertical_relaxed")
    if combined.columns:
        combined = dedupe_frame_for_table(combined, table_key=CPG_EDGES_TABLE_KEY)
        return _select_edge_columns(combined)
    return empty_frame_for_table(CPG_EDGES_TABLE_KEY)


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
    instructions: pl.LazyFrame,
    syntax_calls: pl.LazyFrame,
    scip_symbols: pl.LazyFrame,
) -> pl.LazyFrame:
    """Public wrapper for bytecode callsite symbol edges.

    Returns
    -------
    polars.LazyFrame
        LazyFrame of callsite symbol edges.
    """
    return _py_bc_callsite_symbol_edges_to_cpg(instructions, syntax_calls, scip_symbols)


def py_bc_stack_edges_to_cpg(
    instructions: pl.LazyFrame,
    blocks: pl.LazyFrame,
) -> pl.LazyFrame:
    """Public wrapper for bytecode stack edges.

    Returns
    -------
    polars.LazyFrame
        LazyFrame of stack edges.
    """
    return _py_bc_stack_edges_to_cpg(instructions, blocks)


__all__ = [
    "CPG_EDGES_TABLE_KEY",
    "CPG_NODES_TABLE_KEY",
    "CPG_TARGET_NAME",
    "cpg_edges",
    "cpg_nodes",
    "instruction_cpg_id",
    "py_bc_callsite_symbol_edges_to_cpg",
    "py_bc_stack_edges_to_cpg",
    "stable_cpg_id",
]
