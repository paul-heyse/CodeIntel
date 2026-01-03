"""Unified CPG node and edge assembly for property graph exports."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial

import polars as pl

from codeintel.build.graphs.compute.goid import DECIMAL_38_MAX
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.frames import dedupe_frame_for_table, empty_frame_for_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.helpers.payload import encode_payload
from codeintel.core.schemas.generated_rows import columns_for_table_key

_HAMILTON_TYPE_HINTS = (BuildEnv, DagCatalog, TargetRunRecord, InferableTabularInput)

CPG_TARGET_NAME = "cpg"
CPG_NODES_TABLE_KEY = "graph.cpg_nodes"
CPG_EDGES_TABLE_KEY = "graph.cpg_edges"

SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
SCIP_SYMBOLS_TABLE_KEY = "core.scip_symbol_information"
GOIDS_TABLE_KEY = "core.goids"
CFG_BLOCKS_TABLE_KEY = "graph.cfg_blocks"
IMPORT_MODULES_TABLE_KEY = "graph.import_modules"
TS_TOKENS_TABLE_KEY = "core.ts_tokens"
TS_TRIVIA_TABLE_KEY = "core.ts_trivia"

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
class _CpgNodeCoreInputs:
    syntax_nodes: InferableTabularInput
    scip_symbol_information: InferableTabularInput
    goids: InferableTabularInput
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
    blocks = cfg_blocks.join(goid_ctx, on="function_goid_h128", how="left")
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


def cpg_nodes__core_inputs(
    q__core__syntax_nodes: InferableTabularInput,
    q__core__scip_symbol_information: InferableTabularInput,
    q__core__goids: InferableTabularInput,
    q__core__ts_tokens: InferableTabularInput,
    q__core__ts_trivia: InferableTabularInput,
) -> _CpgNodeCoreInputs:
    """Bundle core tables for CPG node assembly.

    Returns
    -------
    _CpgNodeCoreInputs
        Core inputs for CPG node assembly.
    """
    return _CpgNodeCoreInputs(
        syntax_nodes=q__core__syntax_nodes,
        scip_symbol_information=q__core__scip_symbol_information,
        goids=q__core__goids,
        ts_tokens=q__core__ts_tokens,
        ts_trivia=q__core__ts_trivia,
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


def cpg_nodes(
    cpg_nodes__inputs: _CpgNodeInputs,
) -> pl.LazyFrame:
    """Build CPG nodes from syntax, symbol, and flow inventories.

    Returns
    -------
    pl.LazyFrame
        LazyFrame for graph.cpg_nodes.
    """
    core_inputs = cpg_nodes__inputs.core
    graph_inputs = cpg_nodes__inputs.graph
    syntax_nodes = tabular_to_lazyframe(core_inputs.syntax_nodes)
    scip_symbols = tabular_to_lazyframe(core_inputs.scip_symbol_information)
    goids = tabular_to_lazyframe(core_inputs.goids)
    ts_tokens = tabular_to_lazyframe(core_inputs.ts_tokens)
    ts_trivia = tabular_to_lazyframe(core_inputs.ts_trivia)
    cfg_blocks = tabular_to_lazyframe(graph_inputs.cfg_blocks)
    import_modules = tabular_to_lazyframe(graph_inputs.import_modules)

    frames = [
        _syntax_nodes_to_cpg(syntax_nodes),
        _scip_symbols_to_cpg(scip_symbols),
        _goids_to_cpg(goids),
        _ts_tokens_to_cpg(ts_tokens),
        _ts_trivia_to_cpg(ts_trivia),
        _cfg_blocks_to_cpg(cfg_blocks, goids),
        _import_modules_to_cpg(import_modules),
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


def _occurrence_roles(
    occ_syntax: pl.LazyFrame,
    occ_span: pl.LazyFrame,
) -> pl.LazyFrame:
    span = occ_span.select(
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
    )
    syntax = occ_syntax.select(
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
    return syntax.join(span, on=join_keys, how="left")


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
    blocks = cfg_blocks.join(goid_ctx, on="function_goid_h128", how="left").select(
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
    joined = cfg_edges.join(src_blocks, on=["function_goid_h128", "src_block_id"], how="left").join(
        dst_blocks, on=["function_goid_h128", "dst_block_id"], how="left"
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
    blocks = cfg_blocks.join(goid_ctx, on="function_goid_h128", how="left").select(
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
    joined = dfg_edges.join(src_blocks, on=["function_goid_h128", "src_block_id"], how="left").join(
        dst_blocks, on=["function_goid_h128", "dst_block_id"], how="left"
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
    blocks = cfg_blocks.join(goid_ctx, on="function_goid_h128", how="left").select(
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
    joined = cdg_edges.join(src_blocks, on=["function_goid_h128", "src_block_id"], how="left").join(
        dst_blocks, on=["function_goid_h128", "dst_block_id"], how="left"
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
        call_edges.join(
            syntax_keys,
            on=["repo", "commit", "call_node_id"],
            how="left",
        )
        .join(
            blocks,
            left_on="callee_entry_block_id",
            right_on="block_id",
            how="left",
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
    extras = _pk_json_expr(
        {
            "call_id": pl.col("call_id"),
            "confidence": pl.col("confidence"),
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
    joined = arg_edges.join(src_keys, on=["repo", "commit", "src_arg_node_id"], how="left").join(
        dst_keys, on=["repo", "commit", "dst_param_node_id"], how="left"
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
        ret_edges.join(
            syntax_keys,
            on=["repo", "commit", "call_node_id"],
            how="left",
        )
        .join(
            blocks,
            left_on="exit_block_id",
            right_on="block_id",
            how="left",
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


def cpg_edges(
    cpg_edge_symbol_inputs: _CpgSymbolInputs,
    cpg_edge_flow_inputs: _CpgFlowInputs,
    cpg_edge_link_inputs: _CpgLinkInputs,
    cpg_edge_call_wiring_inputs: _CpgCallWiringInputs,
    cpg_edge_syntax_node_inputs: _CpgSyntaxNodeInputs,
) -> pl.LazyFrame:
    """Build CPG edges from syntax, symbol, and flow sources.

    Returns
    -------
    pl.LazyFrame
        LazyFrame for graph.cpg_edges.
    """
    cfg_blocks = cpg_edge_flow_inputs.cfg_blocks
    syntax_nodes = cpg_edge_syntax_node_inputs.syntax_nodes

    frames = [
        _syntax_edges_to_cpg(cpg_edge_symbol_inputs.syntax_edges),
        _scip_occurrence_edges_to_cpg(
            cpg_edge_symbol_inputs.occ_syntax,
            cpg_edge_symbol_inputs.occ_span,
        ),
        _scip_symbol_relationships_to_cpg(cpg_edge_symbol_inputs.symbol_rels),
        _scip_symbol_goid_edges_to_cpg(cpg_edge_symbol_inputs.symbol_goid),
        _call_graph_edges_to_cpg(cpg_edge_link_inputs.call_edges),
        _import_graph_edges_to_cpg(cpg_edge_link_inputs.import_edges),
        _cfg_edges_to_cpg(cpg_edge_flow_inputs.cfg_edges, cfg_blocks, cpg_edge_flow_inputs.goids),
        _dfg_edges_to_cpg(cpg_edge_flow_inputs.dfg_edges, cfg_blocks, cpg_edge_flow_inputs.goids),
        _cdg_edges_to_cpg(cpg_edge_flow_inputs.cdg_edges, cfg_blocks, cpg_edge_flow_inputs.goids),
        _call_wiring_calls_to_cpg(
            cpg_edge_call_wiring_inputs.call_edges,
            cfg_blocks,
            syntax_nodes,
        ),
        _call_wiring_arg_to_param_to_cpg(
            cpg_edge_call_wiring_inputs.arg_to_param_edges,
            syntax_nodes,
        ),
        _call_wiring_ret_to_call_to_cpg(
            cpg_edge_call_wiring_inputs.ret_to_call_edges,
            cfg_blocks,
            syntax_nodes,
        ),
    ]
    combined = pl.concat(frames, how="vertical_relaxed")
    if combined.columns:
        combined = dedupe_frame_for_table(combined, table_key=CPG_EDGES_TABLE_KEY)
        return _select_edge_columns(combined)
    return empty_frame_for_table(CPG_EDGES_TABLE_KEY)


__all__ = [
    "CPG_EDGES_TABLE_KEY",
    "CPG_NODES_TABLE_KEY",
    "CPG_TARGET_NAME",
    "cpg_edges",
    "cpg_nodes",
]
