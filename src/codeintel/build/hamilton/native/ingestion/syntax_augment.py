"""Tree-sitter welds and LibCST fallback for canonical syntax nodes/edges."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING

import polars as pl
from intervaltree import IntervalTree

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.options.ingestion import SyntaxAugmentOptions
from codeintel.build.hamilton.native.patterns import (
    RelationTableSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.frames import (
    dedupe_frame_for_table,
    empty_lazyframe_for_table,
    rows_to_frame,
)
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.helpers.payload import decode_payload, encode_payload
from codeintel.core.spans import normalize_byte_span

if TYPE_CHECKING:
    from collections.abc import Iterable

_HAMILTON_TYPE_HINTS = (BuildEnv, InferableTabularInput)

SYNTAX_AUGMENT_TARGET_NAME = "syntax_augment"
SYNTAX_NODES_TABLE_KEY = "core.syntax_nodes"
SYNTAX_EDGES_TABLE_KEY = "core.syntax_edges"
PARSE_MANIFEST_TABLE_KEY = "core.parse_manifest"
TS_NODES_TABLE_KEY = "core.ts_nodes"
TS_EDGES_TABLE_KEY = "core.ts_edges"
TS_XREF_TABLE_KEY = "core.ts_syntax_node_xref"

SYNTAX_PRODUCER_LIBCST = "libcst"
TS_PRODUCER = "tree_sitter"
EDGE_KIND = "AST_CHILD"


@dataclass(slots=True)
class _SyntaxNodeCandidate:
    node_id: str
    start_byte: int
    end_byte: int
    order: int


@dataclass(slots=True)
class _SyntaxNodeIndex:
    tree: IntervalTree
    exact: dict[tuple[int, int], list[_SyntaxNodeCandidate]]


@dataclass(frozen=True, slots=True)
class SyntaxAugmentFrames:
    syntax_nodes: pl.LazyFrame
    syntax_edges: pl.LazyFrame
    ts_syntax_node_xref: pl.LazyFrame


@dataclass(frozen=True, slots=True)
class _SyntaxAugmentInputs:
    syntax_nodes: pl.DataFrame
    syntax_edges: pl.DataFrame
    ts_nodes: pl.DataFrame
    ts_edges: pl.DataFrame
    parse_manifest: pl.DataFrame


def syntax_augment__options(env: BuildEnv) -> SyntaxAugmentOptions:
    """Load syntax augmentation options from the build environment.

    Returns
    -------
    SyntaxAugmentOptions
        Parsed options for syntax augmentation.
    """
    return load_target_options(
        env,
        target_name=SYNTAX_AUGMENT_TARGET_NAME,
        options_type=SyntaxAugmentOptions,
    )


def syntax_augment__inputs(
    q__core__syntax_nodes: InferableTabularInput,
    q__core__syntax_edges: InferableTabularInput,
    q__core__ts_nodes: InferableTabularInput,
    q__core__ts_edges: InferableTabularInput,
    q__core__parse_manifest: InferableTabularInput,
) -> _SyntaxAugmentInputs:
    """Collect syntax augmentation input frames.

    Returns
    -------
    _SyntaxAugmentInputs
        Collected input frames for syntax augmentation.
    """
    return _SyntaxAugmentInputs(
        syntax_nodes=tabular_to_lazyframe(q__core__syntax_nodes).collect(),
        syntax_edges=tabular_to_lazyframe(q__core__syntax_edges).collect(),
        ts_nodes=tabular_to_lazyframe(q__core__ts_nodes).collect(),
        ts_edges=tabular_to_lazyframe(q__core__ts_edges).collect(),
        parse_manifest=tabular_to_lazyframe(q__core__parse_manifest).collect(),
    )


def _failure_paths(parse_manifest: pl.DataFrame) -> set[str]:
    if not parse_manifest.columns:
        return set()
    if "producer" not in parse_manifest.columns or "parse_ok" not in parse_manifest.columns:
        return set()
    failures = parse_manifest.filter(
        (pl.col("producer") == SYNTAX_PRODUCER_LIBCST)
        & (~pl.col("parse_ok").fill_null(value=False))
    )
    paths = failures.get_column("rel_path") if "rel_path" in failures.columns else []
    return {path for path in paths if isinstance(path, str)}


def _build_syntax_index(nodes_frame: pl.DataFrame) -> dict[str, _SyntaxNodeIndex]:
    indexes: dict[str, _SyntaxNodeIndex] = {}
    for order, row in enumerate(nodes_frame.iter_rows(named=True)):
        rel_path = row.get("rel_path")
        node_id = row.get("node_id")
        if not isinstance(rel_path, str) or not isinstance(node_id, str):
            continue
        start_byte = row.get("start_byte")
        end_byte = row.get("end_byte")
        if not isinstance(start_byte, Integral) or not isinstance(end_byte, Integral):
            continue
        span = normalize_byte_span(int(start_byte), int(end_byte))
        if span is None:
            continue
        start_byte, end_byte = span
        candidate = _SyntaxNodeCandidate(
            node_id=node_id,
            start_byte=start_byte,
            end_byte=end_byte,
            order=order,
        )
        index = indexes.get(rel_path)
        if index is None:
            index = _SyntaxNodeIndex(tree=IntervalTree(), exact={})
            indexes[rel_path] = index
        index.exact.setdefault((start_byte, end_byte), []).append(candidate)
        if start_byte < end_byte:
            index.tree.addi(start_byte, end_byte, candidate)
    return indexes


def _pick_candidate(candidates: Iterable[_SyntaxNodeCandidate]) -> _SyntaxNodeCandidate | None:
    items = list(candidates)
    if not items:
        return None
    return min(items, key=lambda item: (item.end_byte - item.start_byte, item.order))


def _interval_candidates(intervals: Iterable[object]) -> list[_SyntaxNodeCandidate]:
    candidates: list[_SyntaxNodeCandidate] = []
    for interval in intervals:
        data = getattr(interval, "data", None)
        if isinstance(data, _SyntaxNodeCandidate):
            candidates.append(data)
    return candidates


def _containing_intervals(tree: IntervalTree, start: int, end: int) -> list[object]:
    envelop = getattr(tree, "envelop", None)
    if callable(envelop):
        return list(envelop(start, end))
    return [
        interval
        for interval in tree.overlap(start, end)
        if interval.begin <= start and interval.end >= end
    ]


def _point_intervals(tree: IntervalTree, point: int) -> list[object]:
    at = getattr(tree, "at", None)
    if callable(at):
        return list(at(point))
    return list(tree.overlap(point, point + 1))


def _match_syntax_node(
    index: _SyntaxNodeIndex,
    start: int,
    end: int,
) -> tuple[str | None, str, int]:
    exact = index.exact.get((start, end))
    if exact:
        chosen = _pick_candidate(exact)
        return (chosen.node_id if chosen else None, "EXACT", len(exact))
    if start == end:
        point = _interval_candidates(_point_intervals(index.tree, start))
        chosen = _pick_candidate(point)
        if chosen is None:
            return None, "NONE", 0
        return chosen.node_id, "POINT", len(point)
    containing = _interval_candidates(_containing_intervals(index.tree, start, end))
    if containing:
        chosen = _pick_candidate(containing)
        return (chosen.node_id if chosen else None, "CONTAINS", len(containing))
    overlaps = _interval_candidates(index.tree.overlap(start, end))
    if overlaps:
        chosen = _pick_candidate(overlaps)
        return (chosen.node_id if chosen else None, "OVERLAP", len(overlaps))
    return None, "NONE", 0


def _producer_by_path(nodes_frame: pl.DataFrame) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not nodes_frame.columns:
        return mapping
    for row in nodes_frame.iter_rows(named=True):
        rel_path = row.get("rel_path")
        producer = row.get("producer")
        if isinstance(rel_path, str) and isinstance(producer, str):
            mapping.setdefault(rel_path, producer)
    return mapping


def _xref_rows(
    *,
    ts_nodes: pl.DataFrame,
    syntax_nodes: pl.DataFrame,
) -> list[dict[str, object]]:
    if ts_nodes.is_empty():
        return []
    index_by_path = _build_syntax_index(syntax_nodes)
    producer_by_path = _producer_by_path(syntax_nodes)
    rows: list[dict[str, object]] = []
    for row in ts_nodes.iter_rows(named=True):
        xref_row = _xref_row_for_ts_node(
            row,
            index_by_path=index_by_path,
            producer_by_path=producer_by_path,
        )
        if xref_row is not None:
            rows.append(xref_row)
    return rows


def _xref_row_for_ts_node(
    ts_row: dict[str, object],
    *,
    index_by_path: dict[str, _SyntaxNodeIndex],
    producer_by_path: dict[str, str],
) -> dict[str, object] | None:
    rel_path = ts_row.get("rel_path")
    ts_node_id = ts_row.get("node_id")
    language = ts_row.get("language")
    repo = ts_row.get("repo")
    commit = ts_row.get("commit")
    if not isinstance(rel_path, str) or not isinstance(ts_node_id, str):
        return None
    if not isinstance(language, str) or not isinstance(repo, str) or not isinstance(commit, str):
        return None
    producer = producer_by_path.get(rel_path, SYNTAX_PRODUCER_LIBCST)
    start_byte = ts_row.get("start_byte")
    end_byte = ts_row.get("end_byte")
    normalized = (
        normalize_byte_span(int(start_byte), int(end_byte))
        if isinstance(start_byte, Integral) and isinstance(end_byte, Integral)
        else None
    )
    if normalized is None:
        return _xref_row_payload(
            repo=repo,
            commit=commit,
            rel_path=rel_path,
            language=language,
            producer=producer,
            ts_node_id=ts_node_id,
            syntax_node_id=None,
            match_kind="NONE",
            candidate_count=0,
        )
    start_byte, end_byte = normalized
    index = index_by_path.get(rel_path)
    if index is None:
        return _xref_row_payload(
            repo=repo,
            commit=commit,
            rel_path=rel_path,
            language=language,
            producer=producer,
            ts_node_id=ts_node_id,
            syntax_node_id=None,
            match_kind="NONE",
            candidate_count=0,
        )
    syntax_node_id, match_kind, candidate_count = _match_syntax_node(
        index,
        start_byte,
        end_byte,
    )
    return _xref_row_payload(
        repo=repo,
        commit=commit,
        rel_path=rel_path,
        language=language,
        producer=producer,
        ts_node_id=ts_node_id,
        syntax_node_id=syntax_node_id,
        match_kind=match_kind,
        candidate_count=candidate_count,
    )


def _xref_row_payload(
    *,
    repo: str,
    commit: str,
    rel_path: str,
    language: str,
    producer: str,
    ts_node_id: str,
    syntax_node_id: str | None,
    match_kind: str,
    candidate_count: int,
) -> dict[str, object]:
    return {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "language": language,
        "producer": producer,
        "ts_node_id": ts_node_id,
        "syntax_node_id": syntax_node_id,
        "match_kind": match_kind,
        "candidate_count": candidate_count,
    }


def _ts_node_payload(ts_row: dict[str, object], match_kind: str) -> dict[str, object]:
    return {
        "ts_node_id": ts_row.get("node_id"),
        "ts_node_type": ts_row.get("node_type"),
        "start_byte": ts_row.get("start_byte"),
        "end_byte": ts_row.get("end_byte"),
        "start_row": ts_row.get("start_row"),
        "start_col": ts_row.get("start_col"),
        "end_row": ts_row.get("end_row"),
        "end_col": ts_row.get("end_col"),
        "grammar_id": ts_row.get("grammar_id"),
        "kind_id": ts_row.get("kind_id"),
        "parse_state": ts_row.get("parse_state"),
        "next_parse_state": ts_row.get("next_parse_state"),
        "is_named": ts_row.get("is_named"),
        "is_missing": ts_row.get("is_missing"),
        "is_error": ts_row.get("is_error"),
        "has_error": ts_row.get("has_error"),
        "match_kind": match_kind,
    }


def _merge_ts_extras(
    nodes_rows: list[dict[str, object]],
    ts_nodes: pl.DataFrame,
    xref_rows: list[dict[str, object]],
) -> None:
    ts_index = _ts_node_index(ts_nodes)
    payloads_by_node = _payloads_by_syntax_node(xref_rows, ts_index)
    _apply_ts_payloads(nodes_rows, payloads_by_node)


def _ts_node_index(ts_nodes: pl.DataFrame) -> dict[str, dict[str, object]]:
    ts_index: dict[str, dict[str, object]] = {}
    for row in ts_nodes.iter_rows(named=True):
        node_id = row.get("node_id")
        if isinstance(node_id, str):
            ts_index[node_id] = row
    return ts_index


def _payloads_by_syntax_node(
    xref_rows: list[dict[str, object]],
    ts_index: dict[str, dict[str, object]],
) -> dict[str, dict[str, dict[str, object]]]:
    payloads_by_node: dict[str, dict[str, dict[str, object]]] = {}
    for xref in xref_rows:
        syntax_node_id = xref.get("syntax_node_id")
        ts_node_id = xref.get("ts_node_id")
        match_kind = xref.get("match_kind")
        if not isinstance(syntax_node_id, str) or not isinstance(ts_node_id, str):
            continue
        if not isinstance(match_kind, str) or match_kind == "NONE":
            continue
        ts_row = ts_index.get(ts_node_id)
        if ts_row is None:
            continue
        payload = _ts_node_payload(ts_row, match_kind)
        payloads_by_node.setdefault(syntax_node_id, {})[ts_node_id] = payload
    return payloads_by_node


def _apply_ts_payloads(
    nodes_rows: list[dict[str, object]],
    payloads_by_node: dict[str, dict[str, dict[str, object]]],
) -> None:
    for row in nodes_rows:
        node_id = row.get("node_id")
        if not isinstance(node_id, str):
            continue
        payload_map = payloads_by_node.get(node_id)
        if not payload_map:
            continue
        extras = _merge_ts_node_payloads(row.get("extras_json"), payload_map)
        row["extras_json"] = encode_payload(extras)


def _merge_ts_node_payloads(
    extras_raw: object,
    payload_map: dict[str, dict[str, object]],
) -> dict[str, object]:
    existing = decode_payload(extras_raw)
    extras = dict(existing) if isinstance(existing, dict) else {}
    prior = extras.get("ts_nodes")
    merged: dict[str, dict[str, object]] = {}
    if isinstance(prior, list):
        for item in prior:
            if isinstance(item, dict):
                ts_node_id = item.get("ts_node_id")
                if isinstance(ts_node_id, str):
                    merged[ts_node_id] = item
    merged.update(payload_map)
    payloads = list(merged.values())
    payloads.sort(key=_ts_payload_sort_key)
    extras["ts_nodes"] = payloads
    return extras


def _ts_payload_sort_key(item: dict[str, object]) -> tuple[int, int]:
    start = item.get("start_byte")
    end = item.get("end_byte")
    start_value = int(start) if isinstance(start, Integral) else -1
    end_value = int(end) if isinstance(end, Integral) else -1
    return start_value, end_value


def _ts_nodes_to_syntax_nodes(ts_nodes: pl.DataFrame) -> pl.DataFrame:
    if ts_nodes.is_empty():
        return pl.DataFrame()
    return ts_nodes.select(
        pl.col("repo"),
        pl.col("commit"),
        pl.col("rel_path"),
        pl.lit(TS_PRODUCER).alias("producer"),
        pl.col("language"),
        pl.col("node_id"),
        pl.col("node_type").alias("node_kind"),
        pl.col("node_type").alias("raw_kind"),
        pl.col("start_row").alias("start_line"),
        pl.col("start_col"),
        pl.col("end_row").alias("end_line"),
        pl.col("end_col"),
        pl.col("start_byte"),
        pl.col("end_byte"),
        pl.col("text_preview"),
        pl.lit(None).alias("extras_json"),
    )


def _ts_edges_to_syntax_edges(ts_edges: pl.DataFrame) -> pl.DataFrame:
    if ts_edges.is_empty():
        return pl.DataFrame()
    return ts_edges.select(
        pl.col("repo"),
        pl.col("commit"),
        pl.col("rel_path"),
        pl.lit(TS_PRODUCER).alias("producer"),
        pl.col("parent_node_id"),
        pl.col("child_node_id"),
        pl.lit(EDGE_KIND).alias("edge_kind"),
        pl.col("field_name"),
        pl.col("child_ordinal"),
    )


def syntax_augment__frames(
    syntax_augment__inputs: _SyntaxAugmentInputs,
    syntax_augment__options: SyntaxAugmentOptions,
) -> SyntaxAugmentFrames:
    """Build canonical syntax nodes/edges and tree-sitter xref rows.

    Returns
    -------
    SyntaxAugmentFrames
        Canonical syntax nodes, edges, and optional tree-sitter xref rows.
    """
    syntax_nodes = syntax_augment__inputs.syntax_nodes
    syntax_edges = syntax_augment__inputs.syntax_edges
    ts_nodes = syntax_augment__inputs.ts_nodes
    ts_edges = syntax_augment__inputs.ts_edges
    parse_manifest = syntax_augment__inputs.parse_manifest

    if syntax_augment__options.fallback_on_libcst_failure:
        fallback_paths = _failure_paths(parse_manifest)
    else:
        fallback_paths = set()

    if fallback_paths:
        node_columns = set(syntax_nodes.columns)
        edge_columns = set(syntax_edges.columns)
        if {"producer", "rel_path"}.issubset(node_columns):
            libcst_mask = (pl.col("producer") == SYNTAX_PRODUCER_LIBCST) & (
                pl.col("rel_path").is_in(fallback_paths)
            )
            syntax_nodes = syntax_nodes.filter(~libcst_mask)
        if {"producer", "rel_path"}.issubset(edge_columns):
            libcst_mask = (pl.col("producer") == SYNTAX_PRODUCER_LIBCST) & (
                pl.col("rel_path").is_in(fallback_paths)
            )
            syntax_edges = syntax_edges.filter(~libcst_mask)
        ts_fallback_nodes = _ts_nodes_to_syntax_nodes(
            ts_nodes.filter(pl.col("rel_path").is_in(fallback_paths))
        )
        ts_fallback_edges = _ts_edges_to_syntax_edges(
            ts_edges.filter(pl.col("rel_path").is_in(fallback_paths))
        )
        if not ts_fallback_nodes.is_empty():
            syntax_nodes = pl.concat([syntax_nodes, ts_fallback_nodes], how="vertical_relaxed")
        if not ts_fallback_edges.is_empty():
            syntax_edges = pl.concat([syntax_edges, ts_fallback_edges], how="vertical_relaxed")

    xref_rows = _xref_rows(ts_nodes=ts_nodes, syntax_nodes=syntax_nodes)
    nodes_rows = syntax_nodes.to_dicts()
    _merge_ts_extras(nodes_rows, ts_nodes, xref_rows)

    syntax_nodes_frame = dedupe_frame_for_table(
        rows_to_frame(SYNTAX_NODES_TABLE_KEY, nodes_rows),
        table_key=SYNTAX_NODES_TABLE_KEY,
    )
    syntax_edges_frame = dedupe_frame_for_table(
        syntax_edges.lazy(),
        table_key=SYNTAX_EDGES_TABLE_KEY,
    )
    if syntax_augment__options.emit_ts_xref:
        xref_frame = dedupe_frame_for_table(
            rows_to_frame(TS_XREF_TABLE_KEY, xref_rows),
            table_key=TS_XREF_TABLE_KEY,
        )
    else:
        xref_frame = empty_lazyframe_for_table(TS_XREF_TABLE_KEY)

    return SyntaxAugmentFrames(
        syntax_nodes=syntax_nodes_frame,
        syntax_edges=syntax_edges_frame,
        ts_syntax_node_xref=xref_frame,
    )


def syntax_augment__syntax_nodes__base(
    syntax_augment__frames: SyntaxAugmentFrames,
) -> pl.LazyFrame:
    """Return canonical syntax nodes with tree-sitter augmentation.

    Returns
    -------
    pl.LazyFrame
        Canonical syntax node rows.
    """
    return syntax_augment__frames.syntax_nodes


def syntax_augment__syntax_edges__base(
    syntax_augment__frames: SyntaxAugmentFrames,
) -> pl.LazyFrame:
    """Return canonical syntax edges with tree-sitter fallback applied.

    Returns
    -------
    pl.LazyFrame
        Canonical syntax edge rows.
    """
    return syntax_augment__frames.syntax_edges


def syntax_augment__ts_syntax_node_xref__base(
    syntax_augment__frames: SyntaxAugmentFrames,
) -> pl.LazyFrame:
    """Return tree-sitter xref rows for canonical syntax nodes.

    Returns
    -------
    pl.LazyFrame
        Tree-sitter xref rows.
    """
    return syntax_augment__frames.ts_syntax_node_xref


_MODULE = sys.modules[__name__]
_SYNTAX_AUGMENT_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="ingestion",
    target_name=SYNTAX_AUGMENT_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=SYNTAX_NODES_TABLE_KEY,
            base_node="syntax_augment__syntax_nodes__base",
            save_spec=RelationTableSaveSpec(table_key=SYNTAX_NODES_TABLE_KEY),
            node_name="syntax_augment__syntax_nodes",
            input_type=pl.LazyFrame,
        ),
        TableTargetTableSpec(
            table_key=SYNTAX_EDGES_TABLE_KEY,
            base_node="syntax_augment__syntax_edges__base",
            save_spec=RelationTableSaveSpec(table_key=SYNTAX_EDGES_TABLE_KEY),
            node_name="syntax_augment__syntax_edges",
            input_type=pl.LazyFrame,
        ),
        TableTargetTableSpec(
            table_key=TS_XREF_TABLE_KEY,
            base_node="syntax_augment__ts_syntax_node_xref__base",
            save_spec=RelationTableSaveSpec(table_key=TS_XREF_TABLE_KEY),
            node_name="syntax_augment__ts_syntax_node_xref",
            input_type=pl.LazyFrame,
        ),
    ),
    table_materializations_node="syntax_augment__table_materializations",
    anchor_node_name="t__syntax_augment",
)
attach_table_target_template(_MODULE, spec=_SYNTAX_AUGMENT_TABLE_TARGET_SPEC)
syntax_augment__syntax_nodes = _MODULE.syntax_augment__syntax_nodes
syntax_augment__syntax_edges = _MODULE.syntax_augment__syntax_edges
syntax_augment__ts_syntax_node_xref = _MODULE.syntax_augment__ts_syntax_node_xref
syntax_augment__table_materializations = _MODULE.syntax_augment__table_materializations
t__syntax_augment = _MODULE.t__syntax_augment


__all__ = [
    "syntax_augment__syntax_edges",
    "syntax_augment__syntax_edges__base",
    "syntax_augment__syntax_nodes",
    "syntax_augment__syntax_nodes__base",
    "syntax_augment__table_materializations",
    "syntax_augment__ts_syntax_node_xref",
    "syntax_augment__ts_syntax_node_xref__base",
    "t__syntax_augment",
]
