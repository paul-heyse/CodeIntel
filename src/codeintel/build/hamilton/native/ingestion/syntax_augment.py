"""Tree-sitter welds and LibCST fallback for canonical syntax nodes/edges."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral

import polars as pl
import pyarrow as pa

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.options.ingestion import SyntaxAugmentOptions
from codeintel.build.hamilton.native.patterns import (
    RelationTableSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.tabular.arrow_ops import (
    align_table_to_contract,
    arrow_join_lazyframes,
    dedupe_table_for_table,
)
from codeintel.build.tabular.conversion import (
    reader_to_table,
    table_to_reader,
    tabular_to_arrow_table,
    tabular_to_frame,
)
from codeintel.build.tabular.frames import JoinSpec
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_reader_for_table, record_batch_reader_for_rows
from codeintel.core.intervals.span_resolver import SpanResolver
from codeintel.core.spans import normalize_byte_span

_HAMILTON_TYPE_HINTS = (BuildEnv, InferableTabularInput)

SYNTAX_AUGMENT_TARGET_NAME = "syntax_augment"
SYNTAX_NODES_AUGMENTED_TABLE_KEY = "core.syntax_nodes_augmented"
SYNTAX_EDGES_AUGMENTED_TABLE_KEY = "core.syntax_edges_augmented"
PARSE_MANIFEST_TABLE_KEY = "core.parse_manifest"
TS_NODES_TABLE_KEY = "core.ts_nodes"
TS_EDGES_TABLE_KEY = "core.ts_edges"
TS_XREF_TABLE_KEY = "core.ts_syntax_node_xref"
TS_WELD_COVERAGE_TABLE_KEY = "core.ts_weld_coverage"

SYNTAX_PRODUCER_LIBCST = "libcst"
TS_PRODUCER = "tree_sitter"
EDGE_KIND = "AST_CHILD"


@dataclass(slots=True)
class _SyntaxNodeIndex:
    resolver: SpanResolver[str]


@dataclass(frozen=True, slots=True)
class SyntaxAugmentFrames:
    syntax_nodes: pa.RecordBatchReader
    syntax_edges: pa.RecordBatchReader
    ts_syntax_node_xref: pa.RecordBatchReader
    ts_weld_coverage: pa.RecordBatchReader


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
        syntax_nodes=tabular_to_frame(q__core__syntax_nodes),
        syntax_edges=tabular_to_frame(q__core__syntax_edges),
        ts_nodes=tabular_to_frame(q__core__ts_nodes),
        ts_edges=tabular_to_frame(q__core__ts_edges),
        parse_manifest=tabular_to_frame(q__core__parse_manifest),
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
    for row in nodes_frame.iter_rows(named=True):
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
        index = indexes.get(rel_path)
        if index is None:
            index = _SyntaxNodeIndex(
                resolver=SpanResolver.for_bytes(path_normalizer=lambda value: value)
            )
            indexes[rel_path] = index
        index.resolver.add_span(rel_path, start_byte, end_byte, node_id)
    return indexes


def _match_syntax_node(
    index: _SyntaxNodeIndex,
    rel_path: str,
    start: int,
    end: int,
) -> tuple[str | None, str, int]:
    match = index.resolver.resolve(rel_path, start, end, allow_adjacent_point=True)
    return match.payload, match.match_kind, match.candidate_count


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
    context = _XrefRowContext(
        repo=repo,
        commit=commit,
        rel_path=rel_path,
        language=language,
        producer=producer,
        ts_node_id=ts_node_id,
    )
    start_byte = ts_row.get("start_byte")
    end_byte = ts_row.get("end_byte")
    normalized = (
        normalize_byte_span(int(start_byte), int(end_byte))
        if isinstance(start_byte, Integral) and isinstance(end_byte, Integral)
        else None
    )
    if normalized is None:
        return _xref_row_payload(
            context,
            syntax_node_id=None,
            match_kind="NONE",
            candidate_count=0,
        )
    start_byte, end_byte = normalized
    index = index_by_path.get(rel_path)
    if index is None:
        return _xref_row_payload(
            context,
            syntax_node_id=None,
            match_kind="NONE",
            candidate_count=0,
        )
    syntax_node_id, match_kind, candidate_count = _match_syntax_node(
        index,
        rel_path,
        start_byte,
        end_byte,
    )
    return _xref_row_payload(
        context,
        syntax_node_id=syntax_node_id,
        match_kind=match_kind,
        candidate_count=candidate_count,
    )


@dataclass(frozen=True, slots=True)
class _XrefRowContext:
    repo: str
    commit: str
    rel_path: str
    language: str
    producer: str
    ts_node_id: str


def _xref_row_payload(
    context: _XrefRowContext,
    *,
    syntax_node_id: str | None,
    match_kind: str,
    candidate_count: int,
) -> dict[str, object]:
    return {
        "repo": context.repo,
        "commit": context.commit,
        "rel_path": context.rel_path,
        "language": context.language,
        "producer": context.producer,
        "ts_node_id": context.ts_node_id,
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
        row["extras_json"] = extras


def _weld_coverage_frame(
    ts_nodes: pl.DataFrame,
    xref_rows: list[dict[str, object]],
) -> pl.DataFrame:
    if ts_nodes.is_empty():
        return pl.DataFrame()
    group_keys = ["repo", "commit", "rel_path", "language"]
    ts_counts = ts_nodes.lazy().group_by(group_keys).agg(pl.len().alias("ts_node_count"))
    mapped: pl.LazyFrame | None = None
    if xref_rows:
        xref_frame = pl.DataFrame(xref_rows)
        if set(group_keys).issubset(xref_frame.columns):
            mapped = (
                xref_frame.lazy()
                .filter(pl.col("syntax_node_id").is_not_null() & (pl.col("match_kind") != "NONE"))
                .group_by(group_keys)
                .agg(pl.len().alias("mapped_count"))
            )
    if mapped is None:
        coverage = ts_counts.with_columns(pl.lit(0).alias("mapped_count"))
    else:
        coverage = arrow_join_lazyframes(
            ts_counts,
            mapped,
            spec=JoinSpec(on=group_keys, how="left"),
        ).with_columns(pl.col("mapped_count").fill_null(0))
    coverage = coverage.with_columns(
        pl.col("mapped_count").cast(pl.Int64),
        pl.when(pl.col("ts_node_count") > 0)
        .then(pl.col("mapped_count") / pl.col("ts_node_count"))
        .otherwise(pl.lit(0.0))
        .alias("coverage_ratio"),
    )
    return coverage.collect()


def _merge_ts_node_payloads(
    extras_raw: object,
    payload_map: dict[str, dict[str, object]],
) -> dict[str, object]:
    extras = dict(extras_raw) if isinstance(extras_raw, Mapping) else {}
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


def _reader_from_rows(table_key: str, rows: list[dict[str, object]]) -> pa.RecordBatchReader:
    try:
        reader, _ = record_batch_reader_for_rows(table_key, rows)
        table = reader_to_table(reader)
    except (KeyError, RuntimeError):
        if not rows:
            return pa.RecordBatchReader.from_batches(pa.schema([]), [])
        table = pa.Table.from_pylist(rows)
    deduped = dedupe_table_for_table(table_key, table)
    return table_to_reader(deduped)


def _empty_reader(table_key: str) -> pa.RecordBatchReader:
    try:
        return empty_reader_for_table(table_key)
    except (KeyError, RuntimeError):
        return pa.RecordBatchReader.from_batches(pa.schema([]), [])


def _reader_from_frame(table_key: str, frame: pl.DataFrame) -> pa.RecordBatchReader:
    if frame.is_empty():
        return _empty_reader(table_key)
    table = tabular_to_arrow_table(frame)
    try:
        aligned = align_table_to_contract(table_key, table)
    except (KeyError, RuntimeError):
        aligned = table
    deduped = dedupe_table_for_table(table_key, aligned)
    return table_to_reader(deduped)


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


def _filter_libcst_rows(frame: pl.DataFrame, fallback_paths: set[str]) -> pl.DataFrame:
    if not {"producer", "rel_path"}.issubset(frame.columns):
        return frame
    libcst_mask = (pl.col("producer") == SYNTAX_PRODUCER_LIBCST) & (
        pl.col("rel_path").is_in(fallback_paths)
    )
    return frame.filter(~libcst_mask)


def _concat_if_non_empty(base: pl.DataFrame, extra: pl.DataFrame) -> pl.DataFrame:
    if extra.is_empty():
        return base
    return pl.concat([base, extra], how="vertical_relaxed")


def _apply_fallback_paths(
    syntax_nodes: pl.DataFrame,
    syntax_edges: pl.DataFrame,
    ts_nodes: pl.DataFrame,
    ts_edges: pl.DataFrame,
    fallback_paths: set[str],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    if not fallback_paths:
        return syntax_nodes, syntax_edges
    syntax_nodes = _filter_libcst_rows(syntax_nodes, fallback_paths)
    syntax_edges = _filter_libcst_rows(syntax_edges, fallback_paths)
    ts_fallback_nodes = _ts_nodes_to_syntax_nodes(
        ts_nodes.filter(pl.col("rel_path").is_in(fallback_paths))
    )
    ts_fallback_edges = _ts_edges_to_syntax_edges(
        ts_edges.filter(pl.col("rel_path").is_in(fallback_paths))
    )
    syntax_nodes = _concat_if_non_empty(syntax_nodes, ts_fallback_nodes)
    syntax_edges = _concat_if_non_empty(syntax_edges, ts_fallback_edges)
    return syntax_nodes, syntax_edges


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
    inputs = syntax_augment__inputs
    fallback_paths = (
        _failure_paths(inputs.parse_manifest)
        if syntax_augment__options.fallback_on_libcst_failure
        else set()
    )
    syntax_nodes, syntax_edges = _apply_fallback_paths(
        inputs.syntax_nodes,
        inputs.syntax_edges,
        inputs.ts_nodes,
        inputs.ts_edges,
        fallback_paths,
    )
    xref_rows = _xref_rows(ts_nodes=inputs.ts_nodes, syntax_nodes=syntax_nodes)
    nodes_rows = syntax_nodes.to_dicts()
    _merge_ts_extras(nodes_rows, inputs.ts_nodes, xref_rows)

    syntax_nodes_frame = _reader_from_rows(SYNTAX_NODES_AUGMENTED_TABLE_KEY, nodes_rows)
    if not syntax_edges.columns or syntax_edges.is_empty():
        syntax_edges_frame = _empty_reader(SYNTAX_EDGES_AUGMENTED_TABLE_KEY)
    else:
        syntax_edges_frame = _reader_from_frame(SYNTAX_EDGES_AUGMENTED_TABLE_KEY, syntax_edges)
    if syntax_augment__options.emit_ts_xref:
        xref_frame = _reader_from_rows(TS_XREF_TABLE_KEY, xref_rows)
    else:
        xref_frame = _empty_reader(TS_XREF_TABLE_KEY)

    coverage_rows = _weld_coverage_frame(inputs.ts_nodes, xref_rows)
    if coverage_rows.is_empty():
        coverage_frame = _empty_reader(TS_WELD_COVERAGE_TABLE_KEY)
    else:
        coverage_frame = _reader_from_frame(TS_WELD_COVERAGE_TABLE_KEY, coverage_rows)

    return SyntaxAugmentFrames(
        syntax_nodes=syntax_nodes_frame,
        syntax_edges=syntax_edges_frame,
        ts_syntax_node_xref=xref_frame,
        ts_weld_coverage=coverage_frame,
    )


def syntax_augment__syntax_nodes__base(
    syntax_augment__frames: SyntaxAugmentFrames,
) -> pa.RecordBatchReader:
    """Return canonical syntax nodes with tree-sitter augmentation.

    Returns
    -------
    pa.RecordBatchReader
        Canonical syntax node rows.
    """
    return syntax_augment__frames.syntax_nodes


def syntax_augment__syntax_edges__base(
    syntax_augment__frames: SyntaxAugmentFrames,
) -> pa.RecordBatchReader:
    """Return canonical syntax edges with tree-sitter fallback applied.

    Returns
    -------
    pa.RecordBatchReader
        Canonical syntax edge rows.
    """
    return syntax_augment__frames.syntax_edges


def syntax_augment__ts_syntax_node_xref__base(
    syntax_augment__frames: SyntaxAugmentFrames,
) -> pa.RecordBatchReader:
    """Return tree-sitter xref rows for canonical syntax nodes.

    Returns
    -------
    pa.RecordBatchReader
        Tree-sitter xref rows.
    """
    return syntax_augment__frames.ts_syntax_node_xref


def syntax_augment__ts_weld_coverage__base(
    syntax_augment__frames: SyntaxAugmentFrames,
) -> pa.RecordBatchReader:
    """Return per-file tree-sitter weld coverage rows.

    Returns
    -------
    pa.RecordBatchReader
        Weld coverage rows.
    """
    return syntax_augment__frames.ts_weld_coverage


_MODULE = sys.modules[__name__]
_SYNTAX_AUGMENT_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="ingestion",
    target_name=SYNTAX_AUGMENT_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=SYNTAX_NODES_AUGMENTED_TABLE_KEY,
            base_node="syntax_augment__syntax_nodes__base",
            save_spec=RelationTableSaveSpec(
                table_key=SYNTAX_NODES_AUGMENTED_TABLE_KEY,
                output_name=materialize_node(
                    f"{SYNTAX_NODES_AUGMENTED_TABLE_KEY}__{SYNTAX_AUGMENT_TARGET_NAME}"
                ),
            ),
            node_name="syntax_augment__syntax_nodes",
            input_type=pa.RecordBatchReader,
        ),
        TableTargetTableSpec(
            table_key=SYNTAX_EDGES_AUGMENTED_TABLE_KEY,
            base_node="syntax_augment__syntax_edges__base",
            save_spec=RelationTableSaveSpec(
                table_key=SYNTAX_EDGES_AUGMENTED_TABLE_KEY,
                output_name=materialize_node(
                    f"{SYNTAX_EDGES_AUGMENTED_TABLE_KEY}__{SYNTAX_AUGMENT_TARGET_NAME}"
                ),
            ),
            node_name="syntax_augment__syntax_edges",
            input_type=pa.RecordBatchReader,
        ),
        TableTargetTableSpec(
            table_key=TS_XREF_TABLE_KEY,
            base_node="syntax_augment__ts_syntax_node_xref__base",
            save_spec=RelationTableSaveSpec(table_key=TS_XREF_TABLE_KEY),
            node_name="syntax_augment__ts_syntax_node_xref",
            input_type=pa.RecordBatchReader,
        ),
        TableTargetTableSpec(
            table_key=TS_WELD_COVERAGE_TABLE_KEY,
            base_node="syntax_augment__ts_weld_coverage__base",
            save_spec=RelationTableSaveSpec(table_key=TS_WELD_COVERAGE_TABLE_KEY),
            node_name="syntax_augment__ts_weld_coverage",
            input_type=pa.RecordBatchReader,
        ),
    ),
    table_materializations_node="syntax_augment__table_materializations",
    anchor_node_name="t__syntax_augment",
)
attach_table_target_template(_MODULE, spec=_SYNTAX_AUGMENT_TABLE_TARGET_SPEC)
syntax_augment__syntax_nodes = _MODULE.syntax_augment__syntax_nodes
syntax_augment__syntax_edges = _MODULE.syntax_augment__syntax_edges
syntax_augment__ts_syntax_node_xref = _MODULE.syntax_augment__ts_syntax_node_xref
syntax_augment__ts_weld_coverage = _MODULE.syntax_augment__ts_weld_coverage
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
    "syntax_augment__ts_weld_coverage",
    "syntax_augment__ts_weld_coverage__base",
    "t__syntax_augment",
]
