"""Call wiring relation sources for CPG interprocedural edges."""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl
from intervaltree import IntervalTree

from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.frames import dedupe_frame_for_table, empty_frame_for_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.helpers.payload import PayloadValue, encode_payload

CALL_WIRING_TARGET_NAME = "call_wiring"
CPG_CALL_TARGETS_TABLE_KEY = "graph.cpg_call_targets"
CPG_CALL_EDGES_TABLE_KEY = "graph.cpg_edges_calls"
CPG_ARG_TO_PARAM_EDGES_TABLE_KEY = "graph.cpg_edges_arg_to_param"
CPG_RET_TO_CALL_EDGES_TABLE_KEY = "graph.cpg_edges_ret_to_call"

_ROLE_DEFINITION = 0x1
_OVERLAP_CONFIDENCE_THRESHOLD = 3


@dataclass(frozen=True)
class _OccurrenceCandidate:
    start: int
    end: int
    symbol: str
    roles: int


def _score_occurrence(candidate: _OccurrenceCandidate, callee_end: int) -> tuple[int, int, int]:
    is_def = 1 if (candidate.roles & _ROLE_DEFINITION) else 0
    span_len = candidate.end - candidate.start
    dist = abs(callee_end - candidate.end)
    return is_def, span_len, dist


def _pick_best_symbol(
    candidates: list[_OccurrenceCandidate],
    callee_end: int,
) -> tuple[str | None, float, str, int, list[str]]:
    if not candidates:
        return None, 0.0, "scip_none", 0, []
    best = min(candidates, key=lambda item: _score_occurrence(item, callee_end))
    confidence = 1.0
    if best.roles & _ROLE_DEFINITION:
        confidence *= 0.4
    if len(candidates) > _OVERLAP_CONFIDENCE_THRESHOLD:
        confidence *= 0.7
    symbols = sorted({candidate.symbol for candidate in candidates})
    return best.symbol, confidence, "scip_overlap_best", len(candidates), symbols


def _payload_literal(value: PayloadValue | bytes | bytearray | memoryview | None) -> pl.Expr:
    return pl.lit(encode_payload(value)).cast(pl.Binary)


def _rel_path_key(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], str):
        return value[0]
    return None


def _build_occurrence_tree(occ_df: pl.DataFrame | None) -> IntervalTree:
    tree = IntervalTree()
    if occ_df is None or occ_df.is_empty():
        return tree
    for row in occ_df.iter_rows(named=True):
        symbol = row.get("scip_symbol")
        start = row.get("start_byte")
        end = row.get("end_byte")
        roles = row.get("roles")
        if not isinstance(symbol, str) or not isinstance(start, int) or not isinstance(end, int):
            continue
        tree.addi(start, end, _OccurrenceCandidate(start, end, symbol, int(roles or 0)))
    return tree


def _call_target_row(
    call_row: dict[str, object],
    *,
    rel_path: str,
    tree: IntervalTree,
) -> dict[str, object]:
    call_id = call_row.get("call_id")
    callee_start = call_row.get("callee_start_byte")
    callee_end = call_row.get("callee_end_byte")
    if (
        not isinstance(call_id, str)
        or not isinstance(callee_start, int)
        or not isinstance(callee_end, int)
    ):
        return {
            "repo": call_row.get("repo"),
            "commit": call_row.get("commit"),
            "rel_path": rel_path,
            "call_id": call_id,
            "call_node_id": call_row.get("call_node_id"),
            "callee_symbol": None,
            "resolution_kind": "scip_none",
            "confidence": 0.0,
            "candidate_count": 0,
            "extras_json": encode_payload(None),
        }
    candidates = [interval.data for interval in tree.overlap(callee_start, callee_end)]
    symbol, confidence, kind, candidate_count, candidate_symbols = _pick_best_symbol(
        candidates,
        callee_end,
    )
    extras_json = {"candidate_symbols": candidate_symbols} if candidate_symbols else None
    return {
        "repo": call_row.get("repo"),
        "commit": call_row.get("commit"),
        "rel_path": rel_path,
        "call_id": call_id,
        "call_node_id": call_row.get("call_node_id"),
        "callee_symbol": symbol,
        "resolution_kind": kind,
        "confidence": confidence,
        "candidate_count": candidate_count,
        "extras_json": encode_payload(extras_json),
    }


def _resolve_call_targets(
    calls: pl.DataFrame,
    occurrences: pl.DataFrame,
) -> pl.DataFrame:
    out_rows: list[dict[str, object]] = []
    calls_by_path = calls.partition_by("rel_path", as_dict=True)
    occs_by_path = occurrences.partition_by("rel_path", as_dict=True)

    for rel_path_key, calls_df in calls_by_path.items():
        rel_path = _rel_path_key(rel_path_key)
        if rel_path is None:
            continue
        tree = _build_occurrence_tree(occs_by_path.get(rel_path_key))
        out_rows.extend(
            [
                _call_target_row(call_row, rel_path=rel_path, tree=tree)
                for call_row in calls_df.iter_rows(named=True)
            ]
        )

    if not out_rows:
        return pl.DataFrame()
    return pl.DataFrame(out_rows)


def _call_targets_defs(
    defs_resolved: pl.LazyFrame,
) -> pl.LazyFrame:
    return (
        defs_resolved.filter(pl.col("scip_symbol").is_not_null())
        .filter(pl.col("def_kind").is_in(["function", "async_function"]))
        .select(
            "repo",
            "commit",
            "scip_symbol",
            "def_id",
            "syntax_node_id",
            "goid_h128",
        )
        .group_by(["repo", "commit", "scip_symbol"])
        .agg(
            [
                pl.first("def_id").alias("def_id"),
                pl.first("syntax_node_id").alias("syntax_node_id"),
                pl.first("goid_h128").alias("goid_h128"),
            ]
        )
    )


def _entry_blocks(cfg_blocks: pl.LazyFrame) -> pl.LazyFrame:
    return (
        cfg_blocks.filter(pl.col("kind") == "entry")
        .select("function_goid_h128", pl.col("block_id").alias("entry_block_id"))
        .unique(subset=["function_goid_h128"])
    )


def _exit_blocks(cfg_blocks: pl.LazyFrame) -> pl.LazyFrame:
    return (
        cfg_blocks.filter(pl.col("kind") == "exit")
        .select("function_goid_h128", pl.col("block_id").alias("exit_block_id"))
        .unique(subset=["function_goid_h128"])
    )


def cpg_call_targets(
    q__core__syntax_calls: InferableTabularInput,
    q__core__scip_occurrence_span_xref: InferableTabularInput,
    q__core__syntax_defs_resolved: InferableTabularInput,
    q__graph__cfg_blocks: InferableTabularInput,
) -> pl.LazyFrame:
    """Resolve call targets by welding callee spans to SCIP occurrences.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for graph.cpg_call_targets.
    """
    calls = tabular_to_lazyframe(q__core__syntax_calls).select(
        "repo",
        "commit",
        "rel_path",
        "call_id",
        "call_node_id",
        "callee_start_byte",
        "callee_end_byte",
    )
    occurrences = tabular_to_lazyframe(q__core__scip_occurrence_span_xref).select(
        "rel_path",
        "scip_symbol",
        "roles",
        "start_byte",
        "end_byte",
    )
    calls_df = calls.collect()
    if calls_df.is_empty():
        return empty_frame_for_table(CPG_CALL_TARGETS_TABLE_KEY)
    occ_df = occurrences.collect()
    resolved = _resolve_call_targets(calls_df, occ_df)
    if resolved.is_empty():
        return empty_frame_for_table(CPG_CALL_TARGETS_TABLE_KEY)

    targets = resolved.lazy()
    defs = _call_targets_defs(tabular_to_lazyframe(q__core__syntax_defs_resolved))
    blocks = tabular_to_lazyframe(q__graph__cfg_blocks)

    targets = targets.join(
        defs,
        left_on=["repo", "commit", "callee_symbol"],
        right_on=["repo", "commit", "scip_symbol"],
        how="left",
    ).drop(["scip_symbol"])
    targets = targets.with_columns(
        pl.col("def_id").alias("callee_def_id"),
        pl.col("syntax_node_id").alias("callee_def_node_id"),
        pl.col("goid_h128").alias("callee_goid_h128"),
    ).drop(["def_id", "syntax_node_id", "goid_h128"])
    targets = targets.join(
        _entry_blocks(blocks),
        left_on="callee_goid_h128",
        right_on="function_goid_h128",
        how="left",
    ).drop(["function_goid_h128"])
    targets = targets.join(
        _exit_blocks(blocks),
        left_on="callee_goid_h128",
        right_on="function_goid_h128",
        how="left",
    ).drop(["function_goid_h128"])
    targets = targets.rename(
        {
            "entry_block_id": "callee_entry_block_id",
            "exit_block_id": "callee_exit_block_id",
        }
    )

    targets = targets.select(
        [
            "repo",
            "commit",
            "rel_path",
            "call_id",
            "call_node_id",
            "callee_symbol",
            "callee_def_id",
            "callee_def_node_id",
            "callee_goid_h128",
            "callee_entry_block_id",
            "callee_exit_block_id",
            "resolution_kind",
            "confidence",
            "candidate_count",
            "extras_json",
        ]
    )
    return dedupe_frame_for_table(targets, table_key=CPG_CALL_TARGETS_TABLE_KEY)


def cpg_edges_calls(cpg_call_targets: pl.LazyFrame) -> pl.LazyFrame:
    """Build CALLS edges from call targets.

    Returns
    -------
    polars.LazyFrame
        Lazy frame for graph.cpg_edges_calls.
    """
    return (
        cpg_call_targets.filter(pl.col("callee_entry_block_id").is_not_null())
        .with_columns(
            pl.lit("CALLS").alias("edge_kind"),
        )
        .select(
            [
                "repo",
                "commit",
                "call_id",
                "call_node_id",
                "callee_entry_block_id",
                "edge_kind",
                "confidence",
                "extras_json",
            ]
        )
    )


def _arg_edges_positional(args: pl.LazyFrame, params: pl.LazyFrame) -> pl.LazyFrame:
    pos_args = args.filter(pl.col("arg_kind") == "positional")
    non_variadic = params.filter(~pl.col("param_kind").is_in(["varargs", "varkw"]))
    return (
        pos_args.join(
            non_variadic,
            left_on=["callee_def_id", "arg_ordinal"],
            right_on=["callee_def_id", "param_ordinal"],
            how="inner",
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
                "arg_ordinal",
                "param_ordinal",
                pl.col("arg_name").alias("arg_name"),
                pl.col("param_name").alias("param_name"),
                "confidence",
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )


def _arg_edges_keyword(args: pl.LazyFrame, params: pl.LazyFrame) -> pl.LazyFrame:
    kw_args = args.filter(pl.col("arg_kind") == "keyword")
    return (
        kw_args.join(
            params,
            left_on=["callee_def_id", "arg_name"],
            right_on=["callee_def_id", "param_name"],
            how="inner",
        )
        .with_columns(pl.lit("ARG_TO_PARAM").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
                "arg_ordinal",
                "param_ordinal",
                "arg_name",
                "param_name",
                "confidence",
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )


def _arg_edges_star(
    args: pl.LazyFrame,
    params: pl.LazyFrame,
    *,
    arg_kind: str,
    param_kind: str,
    confidence_scale: float,
) -> pl.LazyFrame:
    subset = args.filter(pl.col("arg_kind") == arg_kind)
    var_params = params.filter(pl.col("param_kind") == param_kind)
    return (
        subset.join(var_params, on="callee_def_id", how="inner")
        .with_columns(
            pl.lit("ARG_TO_PARAM").alias("edge_kind"),
            (pl.col("confidence") * pl.lit(confidence_scale)).alias("confidence"),
        )
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("arg_expr_node_id").alias("src_arg_node_id"),
                pl.col("param_node_id").alias("dst_param_node_id"),
                "edge_kind",
                "arg_ordinal",
                "param_ordinal",
                "arg_name",
                "param_name",
                "confidence",
                _payload_literal(None).alias("extras_json"),
            ]
        )
    )


def cpg_edges_arg_to_param(
    cpg_call_targets: pl.LazyFrame,
    q__core__syntax_call_args: InferableTabularInput,
    q__core__syntax_func_params: InferableTabularInput,
) -> pl.LazyFrame:
    """Build ARG_TO_PARAM edges from call arguments and function params.

    Returns
    -------
    pl.LazyFrame
        ARG_TO_PARAM edges for graph.cpg_edges_arg_to_param.
    """
    args = tabular_to_lazyframe(q__core__syntax_call_args).join(
        cpg_call_targets.select(["call_id", "callee_def_id", "confidence"]),
        on="call_id",
        how="left",
    )
    args = args.filter(pl.col("callee_def_id").is_not_null())
    args = args.with_columns(pl.col("confidence").fill_null(0.0))

    params = tabular_to_lazyframe(q__core__syntax_func_params).select(
        [
            pl.col("func_def_id").alias("callee_def_id"),
            "param_ordinal",
            "param_kind",
            "param_name",
            "param_node_id",
        ]
    )

    frames = [
        _arg_edges_positional(args, params),
        _arg_edges_keyword(args, params),
        _arg_edges_star(
            args,
            params,
            arg_kind="starargs",
            param_kind="varargs",
            confidence_scale=0.7,
        ),
        _arg_edges_star(args, params, arg_kind="kwargs", param_kind="varkw", confidence_scale=0.6),
    ]
    combined = pl.concat(frames, how="vertical_relaxed")
    if not combined.columns:
        return empty_frame_for_table(CPG_ARG_TO_PARAM_EDGES_TABLE_KEY)
    return dedupe_frame_for_table(combined, table_key=CPG_ARG_TO_PARAM_EDGES_TABLE_KEY)


def cpg_edges_ret_to_call(cpg_call_targets: pl.LazyFrame) -> pl.LazyFrame:
    """Build RET_TO_CALL edges using callee exit block summaries.

    Returns
    -------
    pl.LazyFrame
        RET_TO_CALL edges for graph.cpg_edges_ret_to_call.
    """
    edges = (
        cpg_call_targets.filter(pl.col("callee_exit_block_id").is_not_null())
        .with_columns(pl.lit("RET_TO_CALL").alias("edge_kind"))
        .select(
            [
                "repo",
                "commit",
                "call_id",
                pl.col("callee_exit_block_id").alias("exit_block_id"),
                "call_node_id",
                "edge_kind",
                (pl.col("confidence") * pl.lit(0.9)).alias("confidence"),
                _payload_literal({"summary_kind": "exit_block"}).alias("extras_json"),
            ]
        )
    )
    return dedupe_frame_for_table(edges, table_key=CPG_RET_TO_CALL_EDGES_TABLE_KEY)


__all__ = [
    "CALL_WIRING_TARGET_NAME",
    "CPG_ARG_TO_PARAM_EDGES_TABLE_KEY",
    "CPG_CALL_EDGES_TABLE_KEY",
    "CPG_CALL_TARGETS_TABLE_KEY",
    "CPG_RET_TO_CALL_EDGES_TABLE_KEY",
    "cpg_call_targets",
    "cpg_edges_arg_to_param",
    "cpg_edges_calls",
    "cpg_edges_ret_to_call",
]
