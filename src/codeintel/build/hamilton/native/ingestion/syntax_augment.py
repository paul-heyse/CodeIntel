"""Tree-sitter welds and LibCST fallback for canonical syntax nodes/edges."""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from numbers import Integral

import pyarrow as pa
import pyarrow.compute as pc
from hamilton.function_modifiers import cache

from codeintel.build.graphs.assembly import select_table_columns
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.options.ingestion import SyntaxAugmentOptions
from codeintel.build.hamilton.native.patterns import (
    MultiTableTargetContext,
    RelationTableSaveSpec,
    TableTargetTableContext,
    attach_table_target_template,
    build_multi_table_target_spec_from_contexts,
)
from codeintel.build.hamilton.options_loading import load_target_options
from codeintel.build.scopes.snapshot import SnapshotScope
from codeintel.build.tabular.array_ops import ensure_array
from codeintel.build.tabular.arrow_ops import (
    ArrowJoinSpec,
    align_table_to_contract,
    arrow_join_tables,
    build_join_options,
    dedupe_table_for_table,
    emit_alignment_report,
    group_list_or_polars,
    iter_array_values,
    normalize_table_for_compute,
    normalize_table_for_join,
)
from codeintel.build.tabular.compute_columns import constant_array
from codeintel.build.tabular.compute_helpers import cast_array, safe_filter, take_array
from codeintel.build.tabular.compute_masks import (
    and_kleene,
    equal_mask,
    invert_mask,
    is_in_mask,
    is_null_mask,
    is_valid_mask,
)
from codeintel.build.tabular.conversion import tabular_to_scoped_table
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.columnar.rows import empty_table_for_table, table_for_rows
from codeintel.core.columnar.schema_ops import concat_tables_unified
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

_XrefRowValues = tuple[str, str, str, str, str, str, str | None, str, int]
_FuzzyRowScalars = tuple[
    pa.Scalar,
    pa.Scalar,
    pa.Scalar,
    pa.Scalar,
    pa.Scalar,
    pa.Scalar,
    pa.Scalar,
    pa.Scalar,
]


@dataclass(slots=True)
class _SyntaxNodeIndex:
    resolver: SpanResolver[str]


@dataclass(frozen=True, slots=True)
class _SyntaxIndexColumns:
    rel_path: int
    node_id: int
    start_byte: int
    end_byte: int


@dataclass(frozen=True, slots=True)
class SyntaxAugmentFrames:
    syntax_nodes: pa.Table
    syntax_edges: pa.Table
    ts_syntax_node_xref: pa.Table
    ts_weld_coverage: pa.Table


@dataclass(frozen=True, slots=True)
class _SyntaxAugmentInputs:
    syntax_nodes: pa.Table
    syntax_edges: pa.Table
    ts_nodes: pa.Table
    ts_edges: pa.Table
    parse_manifest: pa.Table


@cache(behavior="ignore")
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
        syntax_nodes=tabular_to_scoped_table(
            q__core__syntax_nodes,
            columns=None,
            scope=None,
            require_scope_columns=False,
        ),
        syntax_edges=tabular_to_scoped_table(
            q__core__syntax_edges,
            columns=None,
            scope=None,
            require_scope_columns=False,
        ),
        ts_nodes=tabular_to_scoped_table(
            q__core__ts_nodes,
            columns=None,
            scope=None,
            require_scope_columns=False,
        ),
        ts_edges=tabular_to_scoped_table(
            q__core__ts_edges,
            columns=None,
            scope=None,
            require_scope_columns=False,
        ),
        parse_manifest=tabular_to_scoped_table(
            q__core__parse_manifest,
            columns=None,
            scope=None,
            require_scope_columns=False,
        ),
    )


def _if_else(
    condition: pa.Array | pa.ChunkedArray,
    left: object,
    right: object,
) -> pa.Array | pa.ChunkedArray:
    return pc.call_function("if_else", [condition, left, right])


def _drop_nulls(values: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    return pc.call_function("drop_null", [values])


def _path_set(values: pa.Array | pa.ChunkedArray) -> set[str]:
    array = ensure_array(values)
    return {item for item in iter_array_values(array) if isinstance(item, str)}


def _struct_field(
    values: pa.Array | pa.ChunkedArray,
    name: str,
) -> pa.Array | pa.ChunkedArray:
    options = pc.StructFieldOptions(name)
    return pc.call_function("struct_field", [values], options=options)


def _divide(
    left: pa.Array | pa.ChunkedArray,
    right: pa.Array | pa.ChunkedArray,
) -> pa.Array | pa.ChunkedArray:
    return pc.call_function("divide", [left, right])


def _failure_paths(parse_manifest: pa.Table) -> pa.Array | pa.ChunkedArray:
    if not parse_manifest.column_names:
        return pa.array([], type=pa.string())
    if (
        "producer" not in parse_manifest.column_names
        or "parse_ok" not in parse_manifest.column_names
        or "rel_path" not in parse_manifest.column_names
    ):
        return pa.array([], type=pa.string())
    producer_mask = equal_mask(parse_manifest["producer"], pa.scalar(SYNTAX_PRODUCER_LIBCST))
    ok_mask = equal_mask(parse_manifest["parse_ok"], pa.scalar(value=True))
    failure_mask = and_kleene(producer_mask, invert_mask(ok_mask))
    filtered = safe_filter(parse_manifest, failure_mask)
    if filtered.num_rows == 0:
        return pa.array([], type=pa.string())
    rel_path = filtered["rel_path"]
    return ensure_array(_drop_nulls(rel_path))


def _normalize_span_bytes(
    start_byte: pa.Scalar,
    end_byte: pa.Scalar,
) -> tuple[int, int] | None:
    start_value = start_byte.as_py()
    end_value = end_byte.as_py()
    if not isinstance(start_value, Integral) or not isinstance(end_value, Integral):
        return None
    return normalize_byte_span(int(start_value), int(end_value))


def _syntax_index_columns(nodes_table: pa.Table) -> _SyntaxIndexColumns:
    return _SyntaxIndexColumns(
        rel_path=nodes_table.column_names.index("rel_path"),
        node_id=nodes_table.column_names.index("node_id"),
        start_byte=nodes_table.column_names.index("start_byte"),
        end_byte=nodes_table.column_names.index("end_byte"),
    )


def _index_syntax_batch(
    indexes: dict[str, _SyntaxNodeIndex],
    batch: pa.RecordBatch,
    columns: _SyntaxIndexColumns,
) -> None:
    rel_paths = batch.column(columns.rel_path)
    node_ids = batch.column(columns.node_id)
    starts = batch.column(columns.start_byte)
    ends = batch.column(columns.end_byte)
    for rel_path, node_id, start_byte, end_byte in zip(
        rel_paths, node_ids, starts, ends, strict=True
    ):
        rel_value = rel_path.as_py()
        node_value = node_id.as_py()
        if not isinstance(rel_value, str) or not isinstance(node_value, str):
            continue
        normalized = _normalize_span_bytes(start_byte, end_byte)
        if normalized is None:
            continue
        span_start, span_end = normalized
        index = indexes.get(rel_value)
        if index is None:
            index = _SyntaxNodeIndex(
                resolver=SpanResolver.for_bytes(path_normalizer=lambda value: value)
            )
            indexes[rel_value] = index
        index.resolver.add_span(rel_value, span_start, span_end, node_value)


def _build_syntax_index(nodes_table: pa.Table) -> dict[str, _SyntaxNodeIndex]:
    indexes: dict[str, _SyntaxNodeIndex] = {}
    if nodes_table.num_rows == 0:
        return indexes
    required = {"rel_path", "node_id", "start_byte", "end_byte"}
    if not required.issubset(set(nodes_table.column_names)):
        return indexes
    columns = _syntax_index_columns(nodes_table)
    for batch in nodes_table.to_batches():
        _index_syntax_batch(indexes, batch, columns)
    return indexes


def _producer_table(nodes_table: pa.Table) -> pa.Table:
    selected = select_table_columns(nodes_table, ["rel_path", "producer"])
    if selected.num_rows == 0:
        return pa.table({"rel_path": [], "producer": []})
    grouped = (
        normalize_table_for_compute(selected)
        .group_by(["rel_path"])
        .aggregate([("producer", "min")])
    )
    rename: dict[str, str] = {}
    for name in grouped.column_names:
        if name.startswith("producer"):
            rename[name] = "producer"
    return _rename_columns(grouped, rename)


def _xref_exact(ts_nodes: pa.Table, syntax_nodes: pa.Table) -> pa.Table:
    required_ts = {"repo", "commit", "rel_path", "language", "node_id", "start_byte", "end_byte"}
    required_syntax = {
        "repo",
        "commit",
        "rel_path",
        "node_id",
        "start_byte",
        "end_byte",
        "producer",
    }
    if not required_ts.issubset(set(ts_nodes.column_names)):
        return _empty_reader(TS_XREF_TABLE_KEY)
    if not required_syntax.issubset(set(syntax_nodes.column_names)):
        return _empty_reader(TS_XREF_TABLE_KEY)
    ts_selected = select_table_columns(
        ts_nodes,
        ["repo", "commit", "rel_path", "language", "node_id", "start_byte", "end_byte"],
    )
    ts_selected = _rename_columns(ts_selected, {"node_id": "ts_node_id"})
    ts_selected = normalize_table_for_join(ts_selected)
    syntax_selected = select_table_columns(
        syntax_nodes,
        ["repo", "commit", "rel_path", "node_id", "start_byte", "end_byte", "producer"],
    )
    syntax_selected = _rename_columns(syntax_selected, {"node_id": "syntax_node_id"})
    syntax_selected = normalize_table_for_join(syntax_selected)
    join_spec = ArrowJoinSpec(
        on=["repo", "commit", "rel_path", "start_byte", "end_byte"],
        how="left",
    )
    join_options = build_join_options(ts_selected, syntax_selected, normalize_inputs=False)
    joined = arrow_join_tables(
        ts_selected,
        syntax_selected,
        spec=join_spec,
        options=join_options,
    )
    syntax_null = is_null_mask(joined["syntax_node_id"])
    match_kind = _if_else(syntax_null, pa.scalar("NONE"), pa.scalar("EXACT"))
    candidate_count = _if_else(syntax_null, pa.scalar(0), pa.scalar(1))
    producer = joined["producer"]
    producer = _if_else(
        is_null_mask(producer),
        pa.scalar(SYNTAX_PRODUCER_LIBCST),
        producer,
    )
    table = pa.table(
        {
            "repo": joined["repo"],
            "commit": joined["commit"],
            "rel_path": joined["rel_path"],
            "language": joined["language"],
            "producer": producer,
            "ts_node_id": joined["ts_node_id"],
            "syntax_node_id": joined["syntax_node_id"],
            "match_kind": match_kind,
            "candidate_count": cast_array(candidate_count, pa.int64(), safe=True),
        }
    )
    return _reader_from_table(TS_XREF_TABLE_KEY, table)


def _unmatched_ts_nodes(ts_nodes: pa.Table, xref_exact: pa.Table) -> pa.Table:
    if ts_nodes.num_rows == 0:
        return pa.Table.from_pylist([])
    required = {"repo", "commit", "rel_path", "language", "node_id", "start_byte", "end_byte"}
    if not required.issubset(set(ts_nodes.column_names)):
        return pa.Table.from_pylist([])
    ts_selected = select_table_columns(
        ts_nodes,
        ["repo", "commit", "rel_path", "language", "node_id", "start_byte", "end_byte"],
    )
    ts_selected = _rename_columns(ts_selected, {"node_id": "ts_node_id"})
    ts_selected = normalize_table_for_join(ts_selected)
    if xref_exact.num_rows == 0 or "ts_node_id" not in xref_exact.column_names:
        return ts_selected
    xref_selected = select_table_columns(xref_exact, ["ts_node_id", "syntax_node_id"])
    xref_selected = normalize_table_for_join(xref_selected)
    join_spec = ArrowJoinSpec(on=["ts_node_id"], how="left")
    join_options = build_join_options(ts_selected, xref_selected, normalize_inputs=False)
    joined = arrow_join_tables(
        ts_selected,
        xref_selected,
        spec=join_spec,
        options=join_options,
    )
    mask = is_null_mask(joined["syntax_node_id"])
    return safe_filter(joined, mask)


def _xref_row_from_values(values: _XrefRowValues) -> dict[str, object]:
    (
        repo,
        commit,
        rel_path,
        language,
        producer,
        ts_node_id,
        syntax_node_id,
        match_kind,
        count,
    ) = values
    return {
        "repo": repo,
        "commit": commit,
        "rel_path": rel_path,
        "language": language,
        "producer": producer,
        "ts_node_id": ts_node_id,
        "syntax_node_id": syntax_node_id,
        "match_kind": match_kind,
        "candidate_count": count,
    }


def _batch_columns(batch: pa.RecordBatch, names: Sequence[str]) -> list[pa.Array]:
    schema = batch.schema
    arrays: list[pa.Array] = []
    for name in names:
        index = schema.get_field_index(name)
        if index < 0:
            arrays.append(pa.nulls(batch.num_rows))
        else:
            arrays.append(batch.column(index))
    return arrays


def _coerce_scalar_str(value: pa.Scalar) -> str | None:
    candidate = value.as_py()
    if isinstance(candidate, str):
        return candidate
    return None


def _parse_fuzzy_row(
    row: _FuzzyRowScalars,
) -> tuple[str, str, str, str, str, str, pa.Scalar, pa.Scalar] | None:
    (
        repo,
        commit,
        rel_path,
        language,
        ts_node_id,
        start_byte,
        end_byte,
        producer,
    ) = row
    repo_value = _coerce_scalar_str(repo)
    commit_value = _coerce_scalar_str(commit)
    rel_value = _coerce_scalar_str(rel_path)
    lang_value = _coerce_scalar_str(language)
    ts_value = _coerce_scalar_str(ts_node_id)
    if (
        repo_value is None
        or commit_value is None
        or rel_value is None
        or lang_value is None
        or ts_value is None
    ):
        return None
    producer_value = _coerce_scalar_str(producer)
    if producer_value is None:
        producer_value = SYNTAX_PRODUCER_LIBCST
    return (
        repo_value,
        commit_value,
        rel_value,
        lang_value,
        ts_value,
        producer_value,
        start_byte,
        end_byte,
    )


def _build_fuzzy_row(
    index_by_path: dict[str, _SyntaxNodeIndex],
    row: _FuzzyRowScalars,
) -> dict[str, object] | None:
    parsed = _parse_fuzzy_row(row)
    if parsed is None:
        return None
    (
        repo_value,
        commit_value,
        rel_value,
        lang_value,
        ts_value,
        producer_value,
        start_byte,
        end_byte,
    ) = parsed
    normalized = _normalize_span_bytes(start_byte, end_byte)
    if normalized is None:
        return _xref_row_from_values(
            (
                repo_value,
                commit_value,
                rel_value,
                lang_value,
                producer_value,
                ts_value,
                None,
                "NONE",
                0,
            )
        )
    index = index_by_path.get(rel_value)
    if index is None:
        return _xref_row_from_values(
            (
                repo_value,
                commit_value,
                rel_value,
                lang_value,
                producer_value,
                ts_value,
                None,
                "NONE",
                0,
            )
        )
    match = index.resolver.resolve(
        rel_value,
        normalized[0],
        normalized[1],
        allow_adjacent_point=True,
    )
    return _xref_row_from_values(
        (
            repo_value,
            commit_value,
            rel_value,
            lang_value,
            producer_value,
            ts_value,
            match.payload,
            match.match_kind,
            match.candidate_count,
        )
    )


def _xref_fuzzy(
    unmatched_ts_nodes: pa.Table,
    syntax_nodes: pa.Table,
    producer_table: pa.Table,
) -> pa.Table:
    if unmatched_ts_nodes.num_rows == 0:
        return _empty_reader(TS_XREF_TABLE_KEY)
    index_by_path = _build_syntax_index(syntax_nodes)
    unmatched_ts_nodes = normalize_table_for_join(unmatched_ts_nodes)
    producer_table = normalize_table_for_join(producer_table)
    join_spec = ArrowJoinSpec(on=["rel_path"], how="left")
    join_options = build_join_options(unmatched_ts_nodes, producer_table, normalize_inputs=False)
    joined = arrow_join_tables(
        unmatched_ts_nodes,
        producer_table,
        spec=join_spec,
        options=join_options,
    )
    rows: list[dict[str, object]] = []
    columns = (
        "repo",
        "commit",
        "rel_path",
        "language",
        "ts_node_id",
        "start_byte",
        "end_byte",
        "producer",
    )
    for batch in joined.to_batches():
        arrays = _batch_columns(batch, columns)
        for repo, commit, rel_path, language, ts_node_id, start_byte, end_byte, producer in zip(
            *arrays,
            strict=True,
        ):
            row = _build_fuzzy_row(
                index_by_path,
                (
                    repo,
                    commit,
                    rel_path,
                    language,
                    ts_node_id,
                    start_byte,
                    end_byte,
                    producer,
                ),
            )
            if row is not None:
                rows.append(row)
    return _reader_from_rows(TS_XREF_TABLE_KEY, rows)


def _xref_union(exact: pa.Table, fuzzy: pa.Table) -> pa.Table:
    if exact.num_rows == 0:
        return fuzzy
    if fuzzy.num_rows == 0:
        return exact
    combined = concat_tables_unified([exact, fuzzy])
    return dedupe_table_for_table(TS_XREF_TABLE_KEY, combined)


def _column_or_null(table: pa.Table, name: str) -> pa.Array | pa.ChunkedArray:
    if name in table.column_names:
        return ensure_array(table[name])
    return pa.nulls(table.num_rows)


def _group_payloads_by_syntax_node(
    syntax_node_ids: pa.Array | pa.ChunkedArray,
    payloads: pa.StructArray,
) -> pa.Table:
    ids = ensure_array(syntax_node_ids)
    if ids.num_rows == 0:
        empty_nodes = pa.array([], type=pa.string())
        empty_payloads = pa.array([], type=pa.list_(payloads.type))
        return pa.table({"syntax_node_id": empty_nodes, "ts_nodes": empty_payloads})
    if not pa.types.is_string(ids.type):
        index_by_id: dict[str, list[int]] = {}
        for idx, node_id in enumerate(iter_array_values(ids)):
            if not isinstance(node_id, str):
                continue
            index_by_id.setdefault(node_id, []).append(idx)
        if not index_by_id:
            empty_nodes = pa.array([], type=pa.string())
            empty_payloads = pa.array([], type=pa.list_(payloads.type))
            return pa.table({"syntax_node_id": empty_nodes, "ts_nodes": empty_payloads})
        keys: list[str] = []
        indices_flat: list[int] = []
        offsets: list[int] = [0]
        for node_id, indices in index_by_id.items():
            keys.append(node_id)
            indices_flat.extend(indices)
            offsets.append(len(indices_flat))
        offsets_array = pa.array(offsets, type=pa.int64())
        flat_indices = pa.array(indices_flat, type=pa.int64())
        flat_payloads = payloads.take(flat_indices)
        list_array = pa.ListArray.from_arrays(offsets_array, flat_payloads)
        return pa.table(
            {"syntax_node_id": pa.array(keys, type=pa.string()), "ts_nodes": list_array}
        )
    payload_table = pa.Table.from_arrays(
        [ids, payloads],
        names=["syntax_node_id", "ts_payload"],
    )
    payload_table = safe_filter(payload_table, is_valid_mask(payload_table["syntax_node_id"]))
    grouped = group_list_or_polars(
        payload_table,
        keys=["syntax_node_id"],
        value_col="ts_payload",
        maintain_order=True,
    )
    renamed = _rename_columns(grouped, {"ts_payload_list": "ts_nodes"})
    if "ts_nodes" not in renamed.column_names:
        return pa.table(
            {
                "syntax_node_id": pa.array([], type=pa.string()),
                "ts_nodes": pa.array([], type=pa.list_(payloads.type)),
            }
        )
    return renamed


def _index_lookup_indices(node_ids: pa.Array, payload_ids: pa.Array) -> pa.Array:
    id_to_index = {
        value: idx
        for idx, value in enumerate(iter_array_values(payload_ids))
        if isinstance(value, str)
    }
    indices = [
        id_to_index.get(value) if isinstance(value, str) else None
        for value in iter_array_values(node_ids)
    ]
    return pa.array(indices, type=pa.int32())


def _ast_nodes_from_extras(syntax_nodes: pa.Table) -> pa.Array:
    if "extras_json" not in syntax_nodes.column_names:
        return pa.nulls(syntax_nodes.num_rows)
    try:
        ast_nodes = _struct_field(syntax_nodes["extras_json"], "ast_nodes")
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError, TypeError, ValueError):
        return pa.nulls(syntax_nodes.num_rows)
    return ensure_array(ast_nodes)


def _ts_payloads_by_syntax_node(ts_nodes: pa.Table, xref: pa.Table) -> pa.Table:
    if ts_nodes.num_rows == 0 or xref.num_rows == 0:
        return pa.table({"syntax_node_id": [], "ts_nodes": []})
    if not {"ts_node_id", "syntax_node_id", "match_kind"}.issubset(set(xref.column_names)):
        return pa.table({"syntax_node_id": [], "ts_nodes": []})
    match_mask = invert_mask(equal_mask(xref["match_kind"], pa.scalar("NONE")))
    id_mask = is_valid_mask(xref["syntax_node_id"])
    filtered = safe_filter(xref, and_kleene(match_mask, id_mask))
    if filtered.num_rows == 0:
        return pa.table({"syntax_node_id": [], "ts_nodes": []})
    ts_selected = select_table_columns(
        ts_nodes,
        [
            "node_id",
            "node_type",
            "start_byte",
            "end_byte",
            "start_row",
            "start_col",
            "end_row",
            "end_col",
            "grammar_id",
            "kind_id",
            "parse_state",
            "next_parse_state",
            "is_named",
            "is_missing",
            "is_error",
            "has_error",
        ],
    )
    ts_selected = _rename_columns(ts_selected, {"node_id": "ts_node_id"})
    filtered = normalize_table_for_join(filtered)
    ts_selected = normalize_table_for_join(ts_selected)
    join_spec = ArrowJoinSpec(on=["ts_node_id"], how="left")
    join_options = build_join_options(filtered, ts_selected, normalize_inputs=False)
    joined = arrow_join_tables(
        filtered,
        ts_selected,
        spec=join_spec,
        options=join_options,
    )
    payload = pa.StructArray.from_arrays(
        [
            _column_or_null(joined, "ts_node_id"),
            _column_or_null(joined, "node_type"),
            _column_or_null(joined, "start_byte"),
            _column_or_null(joined, "end_byte"),
            _column_or_null(joined, "start_row"),
            _column_or_null(joined, "start_col"),
            _column_or_null(joined, "end_row"),
            _column_or_null(joined, "end_col"),
            _column_or_null(joined, "grammar_id"),
            _column_or_null(joined, "kind_id"),
            _column_or_null(joined, "parse_state"),
            _column_or_null(joined, "next_parse_state"),
            _column_or_null(joined, "is_named"),
            _column_or_null(joined, "is_missing"),
            _column_or_null(joined, "is_error"),
            _column_or_null(joined, "has_error"),
            _column_or_null(joined, "match_kind"),
        ],
        names=[
            "ts_node_id",
            "ts_node_type",
            "start_byte",
            "end_byte",
            "start_row",
            "start_col",
            "end_row",
            "end_col",
            "grammar_id",
            "kind_id",
            "parse_state",
            "next_parse_state",
            "is_named",
            "is_missing",
            "is_error",
            "has_error",
            "match_kind",
        ],
    )
    payload_table = pa.Table.from_arrays(
        [ensure_array(joined["syntax_node_id"]), payload],
        names=["syntax_node_id", "ts_payload"],
    )
    return _group_payloads_by_syntax_node(payload_table["syntax_node_id"], payload)


def _augment_syntax_nodes(syntax_nodes: pa.Table, ts_payloads: pa.Table) -> pa.Table:
    if syntax_nodes.num_rows == 0 or ts_payloads.num_rows == 0:
        return syntax_nodes
    if "node_id" not in syntax_nodes.column_names:
        return syntax_nodes
    if not {"syntax_node_id", "ts_nodes"}.issubset(set(ts_payloads.column_names)):
        return syntax_nodes
    node_ids = ensure_array(syntax_nodes["node_id"])
    payload_ids = ensure_array(ts_payloads["syntax_node_id"])
    indices = _index_lookup_indices(node_ids, payload_ids)
    ts_nodes = take_array(ensure_array(ts_payloads["ts_nodes"]), indices)
    ast_nodes = _ast_nodes_from_extras(syntax_nodes)
    extras = pa.StructArray.from_arrays(
        [ensure_array(ast_nodes), ensure_array(ts_nodes)],
        names=["ast_nodes", "ts_nodes"],
    )
    if "extras_json" in syntax_nodes.column_names:
        index = syntax_nodes.schema.get_field_index("extras_json")
        return syntax_nodes.set_column(index, "extras_json", extras)
    return syntax_nodes.append_column("extras_json", extras)


def _weld_coverage_table(
    ts_nodes: pa.Table,
    xref: pa.Table,
) -> pa.Table:
    if ts_nodes.num_rows == 0:
        return pa.Table.from_pylist([])
    key_cols = ["repo", "commit", "rel_path", "language"]
    if not set(key_cols).issubset(set(ts_nodes.column_names)):
        return pa.Table.from_pylist([])

    def _count_by(table: pa.Table, *, count_col: str, name: str) -> pa.Table:
        if table.num_rows == 0:
            empty = {col: [] for col in key_cols}
            empty[name] = []
            return pa.table(empty)
        grouped = (
            normalize_table_for_compute(table).group_by(key_cols).aggregate([(count_col, "count")])
        )
        rename: dict[str, str] = {}
        for column in grouped.column_names:
            if column.startswith(f"{count_col}_"):
                rename[column] = name
        return _rename_columns(grouped, rename)

    ts_counts = _count_by(ts_nodes, count_col="node_id", name="ts_node_count")
    if xref.num_rows == 0 or "ts_node_id" not in xref.column_names:
        mapped_counts = pa.table({**{col: [] for col in key_cols}, "mapped_count": []})
    else:
        match_mask = invert_mask(equal_mask(xref["match_kind"], pa.scalar("NONE")))
        mapped = safe_filter(xref, and_kleene(match_mask, is_valid_mask(xref["ts_node_id"])))
        mapped_counts = _count_by(mapped, count_col="ts_node_id", name="mapped_count")

    join_spec = ArrowJoinSpec(on=key_cols, how="left")
    join_options = build_join_options(ts_counts, mapped_counts)
    joined = arrow_join_tables(ts_counts, mapped_counts, spec=join_spec, options=join_options)
    mapped = pc.fill_null(_column_or_null(joined, "mapped_count"), pa.scalar(0))
    total = _column_or_null(joined, "ts_node_count")
    total_float = cast_array(total, pa.float64(), safe=True)
    mapped_float = cast_array(mapped, pa.float64(), safe=True)
    zero_mask = equal_mask(total, pa.scalar(0))
    ratio = _if_else(zero_mask, pa.scalar(0.0), _divide(mapped_float, total_float))
    return pa.table(
        {
            "repo": joined["repo"],
            "commit": joined["commit"],
            "rel_path": joined["rel_path"],
            "language": joined["language"],
            "ts_node_count": total,
            "mapped_count": mapped,
            "coverage_ratio": ratio,
        }
    )


def _reader_from_rows(table_key: str, rows: list[dict[str, object]]) -> pa.Table:
    try:
        table, _ = table_for_rows(table_key, rows)
    except (KeyError, RuntimeError):
        if not rows:
            return pa.Table.from_batches(pa.schema([]), [])
        table = pa.Table.from_pylist(rows)
    return dedupe_table_for_table(table_key, table)


def _empty_reader(table_key: str) -> pa.Table:
    try:
        return empty_table_for_table(table_key)
    except (KeyError, RuntimeError):
        return pa.Table.from_batches(pa.schema([]), [])


def _reader_from_table(table_key: str, table: pa.Table) -> pa.Table:
    if table.num_rows == 0:
        return _empty_reader(table_key)
    try:
        aligned = align_table_to_contract(
            table_key,
            table,
            target_name=SYNTAX_AUGMENT_TARGET_NAME,
            reporter=emit_alignment_report,
        )
    except (KeyError, RuntimeError):
        aligned = table
    return dedupe_table_for_table(table_key, aligned)


def _rename_columns(table: pa.Table, mapping: Mapping[str, str]) -> pa.Table:
    if not mapping:
        return table
    rename: dict[str, str] = {}
    for name in table.column_names:
        rename[name] = mapping.get(name, name)
    return table.rename_columns([rename[name] for name in table.column_names])


def _ts_nodes_to_syntax_nodes(ts_nodes: pa.Table) -> pa.Table:
    if ts_nodes.num_rows == 0:
        return pa.Table.from_pylist([])
    columns = [
        "repo",
        "commit",
        "rel_path",
        "language",
        "node_id",
        "node_type",
        "start_row",
        "start_col",
        "end_row",
        "end_col",
        "start_byte",
        "end_byte",
        "text_preview",
    ]
    existing = [name for name in columns if name in ts_nodes.column_names]
    selected = ts_nodes.select(existing)
    renamed = _rename_columns(
        selected,
        {
            "node_type": "node_kind",
            "start_row": "start_line",
            "end_row": "end_line",
        },
    )
    renamed = renamed.append_column("raw_kind", renamed.column("node_kind"))
    renamed = renamed.append_column("producer", constant_array(TS_PRODUCER, renamed.num_rows))
    renamed = renamed.append_column("extras_json", constant_array(None, renamed.num_rows))
    ordered = [
        "repo",
        "commit",
        "rel_path",
        "producer",
        "language",
        "node_id",
        "node_kind",
        "raw_kind",
        "start_line",
        "start_col",
        "end_line",
        "end_col",
        "start_byte",
        "end_byte",
        "text_preview",
        "extras_json",
    ]
    return renamed.select([name for name in ordered if name in renamed.column_names])


def _ts_edges_to_syntax_edges(ts_edges: pa.Table) -> pa.Table:
    if ts_edges.num_rows == 0:
        return pa.Table.from_pylist([])
    columns = [
        "repo",
        "commit",
        "rel_path",
        "parent_node_id",
        "child_node_id",
        "field_name",
        "child_ordinal",
    ]
    existing = [name for name in columns if name in ts_edges.column_names]
    selected = ts_edges.select(existing)
    selected = selected.append_column("producer", constant_array(TS_PRODUCER, selected.num_rows))
    selected = selected.append_column("edge_kind", constant_array(EDGE_KIND, selected.num_rows))
    ordered = [
        "repo",
        "commit",
        "rel_path",
        "producer",
        "parent_node_id",
        "child_node_id",
        "edge_kind",
        "field_name",
        "child_ordinal",
    ]
    return selected.select([name for name in ordered if name in selected.column_names])


def _filter_libcst_rows(
    frame: pa.Table,
    fallback_paths: pa.Array | pa.ChunkedArray,
) -> pa.Table:
    if not {"producer", "rel_path"}.issubset(frame.column_names):
        return frame
    fallback_values = _path_set(fallback_paths)
    if not fallback_values:
        return frame
    libcst_mask = and_kleene(
        equal_mask(frame["producer"], pa.scalar(SYNTAX_PRODUCER_LIBCST)),
        is_in_mask(frame["rel_path"], value_set=pa.array(sorted(fallback_values))),
    )
    return frame.filter(invert_mask(libcst_mask))


def _concat_if_non_empty(base: pa.Table, extra: pa.Table) -> pa.Table:
    if extra.num_rows == 0:
        return base
    return concat_tables_unified([base, extra])


def _filter_by_paths(table: pa.Table, paths: pa.Array | pa.ChunkedArray) -> pa.Table:
    if len(paths) == 0 or "rel_path" not in table.column_names:
        return table
    mask = is_in_mask(table["rel_path"], value_set=ensure_array(paths))
    return safe_filter(table, mask)


def _apply_fallback_paths(
    syntax_nodes: pa.Table,
    syntax_edges: pa.Table,
    ts_nodes: pa.Table,
    ts_edges: pa.Table,
    fallback_paths: pa.Array | pa.ChunkedArray,
) -> tuple[pa.Table, pa.Table]:
    if len(fallback_paths) == 0:
        return syntax_nodes, syntax_edges
    syntax_nodes = _filter_libcst_rows(syntax_nodes, fallback_paths)
    syntax_edges = _filter_libcst_rows(syntax_edges, fallback_paths)
    ts_fallback_nodes = _ts_nodes_to_syntax_nodes(_filter_by_paths(ts_nodes, fallback_paths))
    ts_fallback_edges = _ts_edges_to_syntax_edges(_filter_by_paths(ts_edges, fallback_paths))
    syntax_nodes = _concat_if_non_empty(syntax_nodes, ts_fallback_nodes)
    syntax_edges = _concat_if_non_empty(syntax_edges, ts_fallback_edges)
    return syntax_nodes, syntax_edges


def _resolve_fallback_paths(
    options: SyntaxAugmentOptions,
    parse_manifest: pa.Table,
) -> pa.Array | pa.ChunkedArray:
    if not options.fallback_on_libcst_failure:
        return pa.array([], type=pa.string())
    return _failure_paths(parse_manifest)


def _build_xref_table(ts_nodes: pa.Table, syntax_nodes: pa.Table) -> pa.Table:
    xref_exact = _xref_exact(ts_nodes, syntax_nodes)
    unmatched = _unmatched_ts_nodes(ts_nodes, xref_exact)
    producer_table = _producer_table(syntax_nodes)
    xref_fuzzy = _xref_fuzzy(unmatched, syntax_nodes, producer_table)
    return _xref_union(xref_exact, xref_fuzzy)


def _frame_or_empty(table_key: str, table: pa.Table) -> pa.Table:
    if table.num_rows == 0 or not table.column_names:
        return _empty_reader(table_key)
    return _reader_from_table(table_key, table)


def _frame_if_enabled(table_key: str, table: pa.Table, *, emit: bool) -> pa.Table:
    if not emit:
        return _empty_reader(table_key)
    return _frame_or_empty(table_key, table)


def syntax_augment__frames(
    env: BuildEnv,
    syntax_augment__inputs: _SyntaxAugmentInputs,
    syntax_augment__options: SyntaxAugmentOptions,
) -> SyntaxAugmentFrames:
    """Build canonical syntax nodes/edges and tree-sitter xref rows.

    Returns
    -------
    SyntaxAugmentFrames
        Canonical syntax nodes, edges, and optional tree-sitter xref rows.
    """
    scope = SnapshotScope.from_snapshot(env.snapshot)
    inputs = syntax_augment__inputs
    syntax_nodes = tabular_to_scoped_table(
        inputs.syntax_nodes,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    syntax_edges = tabular_to_scoped_table(
        inputs.syntax_edges,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    ts_nodes = tabular_to_scoped_table(
        inputs.ts_nodes,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    ts_edges = tabular_to_scoped_table(
        inputs.ts_edges,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    parse_manifest = tabular_to_scoped_table(
        inputs.parse_manifest,
        columns=None,
        scope=scope,
        require_scope_columns=True,
    )
    fallback_paths = _resolve_fallback_paths(syntax_augment__options, parse_manifest)
    syntax_nodes, syntax_edges = _apply_fallback_paths(
        syntax_nodes,
        syntax_edges,
        ts_nodes,
        ts_edges,
        fallback_paths,
    )
    xref_table = _build_xref_table(ts_nodes, syntax_nodes)
    syntax_nodes_augmented = _augment_syntax_nodes(
        syntax_nodes,
        _ts_payloads_by_syntax_node(inputs.ts_nodes, xref_table),
    )

    syntax_nodes_frame = _frame_or_empty(SYNTAX_NODES_AUGMENTED_TABLE_KEY, syntax_nodes_augmented)
    syntax_edges_frame = _frame_or_empty(SYNTAX_EDGES_AUGMENTED_TABLE_KEY, syntax_edges)
    xref_frame = _frame_if_enabled(
        TS_XREF_TABLE_KEY,
        xref_table,
        emit=syntax_augment__options.emit_ts_xref,
    )
    coverage_frame = _frame_or_empty(
        TS_WELD_COVERAGE_TABLE_KEY,
        _weld_coverage_table(inputs.ts_nodes, xref_table),
    )

    return SyntaxAugmentFrames(
        syntax_nodes=syntax_nodes_frame,
        syntax_edges=syntax_edges_frame,
        ts_syntax_node_xref=xref_frame,
        ts_weld_coverage=coverage_frame,
    )


def syntax_augment__syntax_nodes__base(
    syntax_augment__frames: SyntaxAugmentFrames,
) -> pa.Table:
    """Return canonical syntax nodes with tree-sitter augmentation.

    Returns
    -------
    pa.Table
        Canonical syntax node rows.
    """
    return syntax_augment__frames.syntax_nodes


def syntax_augment__syntax_edges__base(
    syntax_augment__frames: SyntaxAugmentFrames,
) -> pa.Table:
    """Return canonical syntax edges with tree-sitter fallback applied.

    Returns
    -------
    pa.Table
        Canonical syntax edge rows.
    """
    return syntax_augment__frames.syntax_edges


def syntax_augment__ts_syntax_node_xref__base(
    syntax_augment__frames: SyntaxAugmentFrames,
) -> pa.Table:
    """Return tree-sitter xref rows for canonical syntax nodes.

    Returns
    -------
    pa.Table
        Tree-sitter xref rows.
    """
    return syntax_augment__frames.ts_syntax_node_xref


def syntax_augment__ts_weld_coverage__base(
    syntax_augment__frames: SyntaxAugmentFrames,
) -> pa.Table:
    """Return per-file tree-sitter weld coverage rows.

    Returns
    -------
    pa.Table
        Weld coverage rows.
    """
    return syntax_augment__frames.ts_weld_coverage


_MODULE = sys.modules[__name__]
_SYNTAX_AUGMENT_TABLE_TARGET_SPEC = build_multi_table_target_spec_from_contexts(
    context=MultiTableTargetContext(
        domain="ingestion",
        target_name=SYNTAX_AUGMENT_TARGET_NAME,
        tables=(),
        table_materializations_node="syntax_augment__table_materializations",
        anchor_node_name="t__syntax_augment",
        save_spec_factory=RelationTableSaveSpec,
        default_input_type=pa.Table,
    ),
    table_contexts=(
        TableTargetTableContext(
            table_key=SYNTAX_NODES_AUGMENTED_TABLE_KEY,
            base_node="syntax_augment__syntax_nodes__base",
            save_spec=RelationTableSaveSpec(
                table_key=SYNTAX_NODES_AUGMENTED_TABLE_KEY,
                output_name=materialize_node(
                    f"{SYNTAX_NODES_AUGMENTED_TABLE_KEY}__{SYNTAX_AUGMENT_TARGET_NAME}"
                ),
            ),
            node_name="syntax_augment__syntax_nodes",
        ),
        TableTargetTableContext(
            table_key=SYNTAX_EDGES_AUGMENTED_TABLE_KEY,
            base_node="syntax_augment__syntax_edges__base",
            save_spec=RelationTableSaveSpec(
                table_key=SYNTAX_EDGES_AUGMENTED_TABLE_KEY,
                output_name=materialize_node(
                    f"{SYNTAX_EDGES_AUGMENTED_TABLE_KEY}__{SYNTAX_AUGMENT_TARGET_NAME}"
                ),
            ),
            node_name="syntax_augment__syntax_edges",
        ),
        TableTargetTableContext(
            table_key=TS_XREF_TABLE_KEY,
            base_node="syntax_augment__ts_syntax_node_xref__base",
            node_name="syntax_augment__ts_syntax_node_xref",
        ),
        TableTargetTableContext(
            table_key=TS_WELD_COVERAGE_TABLE_KEY,
            base_node="syntax_augment__ts_weld_coverage__base",
            node_name="syntax_augment__ts_weld_coverage",
        ),
    ),
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
