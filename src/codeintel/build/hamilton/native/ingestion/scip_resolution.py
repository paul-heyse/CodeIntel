"""SCIP resolution tables for deterministic symbol/GOID stitching."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl
from google.protobuf.struct_pb2 import NullValue, Struct

from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.patterns import (
    RelationTableSaveSpec,
    TableTargetSpec,
    TableTargetTableSpec,
    attach_table_target_template,
)
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.tabular.arrow_ops import arrow_join_lazyframes
from codeintel.build.tabular.conversion import tabular_to_frame, tabular_to_lazyframe
from codeintel.build.tabular.frames import JoinSpec, dedupe_frame_for_table, rows_to_frame
from codeintel.build.tabular.types import InferableTabularInput
from codeintel.core.intervals.span_resolver import SpanResolver

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    DagCatalog,
    TargetRunRecord,
    InferableTabularInput,
)

SCIP_RESOLUTION_TARGET_NAME = "scip_resolution"
SCIP_SYMBOL_GOID_XREF_TABLE_KEY = "core.scip_symbol_goid_xref"
SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY = "core.scip_occurrence_span_xref"
SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY = "core.scip_occurrence_syntax_xref"

_ROLE_DEFINITION = 0x1
_ROLE_IMPORT = 0x2
_ROLE_WRITE = 0x4
_ROLE_READ = 0x8


@dataclass(frozen=True)
class ScipResolutionFrames:
    """Derived frames for SCIP resolution outputs."""

    symbol_goid_xref: pl.LazyFrame
    occurrence_span_xref: pl.LazyFrame


@dataclass(slots=True)
class _SyntaxNodeIndex:
    resolver: SpanResolver[str]
    line_exact: dict[tuple[int, int, int, int], list[str]]


def _symbol_info_frame(symbol_info: InferableTabularInput) -> pl.LazyFrame:
    frame = tabular_to_lazyframe(symbol_info)
    return frame.select(
        "repo",
        "commit",
        "symbol",
        "enclosing_symbol",
    ).rename({"symbol": "scip_symbol"})


def _goids_frame(goids: InferableTabularInput) -> pl.LazyFrame:
    frame = tabular_to_lazyframe(goids)
    return (
        frame.select(
            "goid_h128",
            "rel_path",
            "start_line",
            "end_line",
        )
        .with_columns(
            pl.col("start_line").cast(pl.Int64),
            pl.col("end_line").cast(pl.Int64),
        )
        .filter(pl.col("start_line").is_not_null() & pl.col("end_line").is_not_null())
    )


def _occurrences_frame(occurrences: InferableTabularInput) -> pl.LazyFrame:
    frame = tabular_to_lazyframe(occurrences)
    return frame.rename({"symbol": "scip_symbol"}).with_columns(
        pl.col("start_line").cast(pl.Int64),
        pl.col("end_line").cast(pl.Int64),
    )


def _symbol_goid_xref_frame(
    *,
    occurrences: pl.LazyFrame,
    goids: pl.LazyFrame,
    created_at: datetime,
) -> pl.LazyFrame:
    definitions = occurrences.filter((pl.col("roles") & _ROLE_DEFINITION) != 0)
    # Contract: goids are unique per (rel_path, start_line, end_line).
    joined = arrow_join_lazyframes(
        definitions,
        goids,
        spec=JoinSpec(on=["rel_path", "start_line", "end_line"], how="left", validate="m:1"),
    )
    return joined.select(
        "repo",
        "commit",
        "scip_symbol",
        "goid_h128",
        pl.col("rel_path").alias("def_rel_path"),
        pl.col("start_line").alias("def_start_line"),
        pl.col("start_col").alias("def_start_col"),
        pl.col("end_line").alias("def_end_line"),
        pl.col("end_col").alias("def_end_col"),
        "position_encoding",
        "text_document_encoding",
        pl.lit(created_at).alias("created_at"),
    )


def _occurrence_span_xref_frame(
    *,
    occurrences: pl.LazyFrame,
    symbol_info: pl.LazyFrame,
    symbol_goid_xref: pl.LazyFrame,
    created_at: datetime,
) -> pl.LazyFrame:
    goid_lookup = symbol_goid_xref.select(
        "repo",
        "commit",
        "scip_symbol",
        "goid_h128",
    )
    # Contract: symbol_info/goid_lookup are unique per (repo, commit, scip_symbol).
    base = arrow_join_lazyframes(
        occurrences,
        symbol_info,
        spec=JoinSpec(on=["repo", "commit", "scip_symbol"], how="left", validate="m:1"),
    )
    base = arrow_join_lazyframes(
        base,
        goid_lookup,
        spec=JoinSpec(on=["repo", "commit", "scip_symbol"], how="left", validate="m:1"),
    )

    roles = pl.col("roles")
    return base.with_columns(
        ((roles & _ROLE_DEFINITION) != 0).alias("is_definition"),
        ((roles & _ROLE_IMPORT) != 0).alias("is_import"),
        ((roles & _ROLE_WRITE) != 0).alias("is_write"),
        ((roles & _ROLE_READ) != 0).alias("is_read"),
        ((roles & _ROLE_DEFINITION) == 0).alias("is_reference"),
        pl.lit(created_at).alias("created_at"),
    ).select(
        "repo",
        "commit",
        "rel_path",
        "scip_symbol",
        "roles",
        "is_definition",
        "is_reference",
        "is_import",
        "is_write",
        "is_read",
        "enclosing_symbol",
        "start_line",
        "start_col",
        "end_line",
        "end_col",
        "position_encoding",
        "text_document_encoding",
        "start_byte",
        "end_byte",
        "goid_h128",
        "created_at",
    )


def _stable_occurrence_id(row: dict[str, object]) -> str:
    msg = Struct()
    fields = (
        "rel_path",
        "scip_symbol",
        "occ_start_line",
        "occ_start_col",
        "occ_end_line",
        "occ_end_col",
        "occ_start_byte",
        "occ_end_byte",
    )
    for name in fields:
        value = row.get(name)
        if value is None:
            msg.fields[name].null_value = NullValue.NULL_VALUE
            continue
        msg.fields[name].string_value = str(value)
    payload = msg.SerializeToString(deterministic=True)
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def _build_syntax_node_indexes(
    nodes_frame: pl.DataFrame,
) -> dict[tuple[str, str], _SyntaxNodeIndex]:
    indexes: dict[tuple[str, str], _SyntaxNodeIndex] = {}
    for row in nodes_frame.iter_rows(named=True):
        rel_path = row.get("rel_path")
        producer = row.get("producer")
        node_id = row.get("node_id")
        if not isinstance(rel_path, str) or not isinstance(producer, str):
            continue
        if not isinstance(node_id, str):
            continue
        key = (rel_path, producer)
        index = indexes.get(key)
        if index is None:
            index = _SyntaxNodeIndex(
                resolver=SpanResolver.for_bytes(path_normalizer=lambda value: value),
                line_exact={},
            )
            indexes[key] = index
        start_line = row.get("start_line")
        start_col = row.get("start_col")
        end_line = row.get("end_line")
        end_col = row.get("end_col")
        if (
            isinstance(start_line, int)
            and isinstance(start_col, int)
            and isinstance(end_line, int)
            and isinstance(end_col, int)
        ):
            line_key = (start_line, start_col, end_line, end_col)
            index.line_exact.setdefault(line_key, []).append(node_id)
        start_byte = row.get("start_byte")
        end_byte = row.get("end_byte")
        if isinstance(start_byte, int) and isinstance(end_byte, int):
            index.resolver.add_span(rel_path, start_byte, end_byte, node_id)
    return indexes


def _match_occurrence_to_node(
    index: _SyntaxNodeIndex,
    rel_path: str,
    occ_row: dict[str, object],
) -> tuple[str | None, str, int]:
    start_byte = occ_row.get("occ_start_byte")
    end_byte = occ_row.get("occ_end_byte")
    if (
        isinstance(start_byte, int)
        and isinstance(end_byte, int)
        and start_byte >= 0
        and end_byte >= 0
    ):
        match = index.resolver.resolve(
            rel_path,
            start_byte,
            end_byte,
            allow_adjacent_point=True,
        )
        if match.match_kind != "NONE":
            return match.payload, match.match_kind, match.candidate_count

    start_line = occ_row.get("occ_start_line")
    start_col = occ_row.get("occ_start_col")
    end_line = occ_row.get("occ_end_line")
    end_col = occ_row.get("occ_end_col")
    if (
        isinstance(start_line, int)
        and isinstance(start_col, int)
        and isinstance(end_line, int)
        and isinstance(end_col, int)
    ):
        line_key = (start_line, start_col, end_line, end_col)
        exact = index.line_exact.get(line_key)
        if exact:
            return min(exact), "EXACT", len(exact)
    return None, "NONE", 0


def _occurrence_syntax_xref_rows(
    occurrences_frame: pl.DataFrame,
    nodes_frame: pl.DataFrame,
) -> list[dict[str, object]]:
    if occurrences_frame.is_empty() or nodes_frame.is_empty():
        return []
    indexes = _build_syntax_node_indexes(nodes_frame)
    occurrences_by_path: dict[str, list[dict[str, object]]] = {}
    for row in occurrences_frame.iter_rows(named=True):
        rel_path = row.get("rel_path")
        if not isinstance(rel_path, str):
            continue
        occurrences_by_path.setdefault(rel_path, []).append(row)

    rows: list[dict[str, object]] = []
    for (rel_path, producer), index in indexes.items():
        occ_rows = occurrences_by_path.get(rel_path)
        if not occ_rows:
            continue
        for occ in occ_rows:
            occ_row = {
                "occ_start_byte": occ.get("start_byte"),
                "occ_end_byte": occ.get("end_byte"),
                "occ_start_line": occ.get("start_line"),
                "occ_start_col": occ.get("start_col"),
                "occ_end_line": occ.get("end_line"),
                "occ_end_col": occ.get("end_col"),
            }
            syntax_node_id, match_kind, candidate_count = _match_occurrence_to_node(
                index,
                rel_path,
                occ_row,
            )
            rows.append(
                {
                    "repo": occ.get("repo"),
                    "commit": occ.get("commit"),
                    "rel_path": rel_path,
                    "producer": producer,
                    "scip_symbol": occ.get("scip_symbol"),
                    "scip_occurrence_id": _stable_occurrence_id(
                        {
                            **occ_row,
                            "rel_path": rel_path,
                            "scip_symbol": occ.get("scip_symbol"),
                        }
                    ),
                    "occ_start_byte": occ_row["occ_start_byte"],
                    "occ_end_byte": occ_row["occ_end_byte"],
                    "occ_start_line": occ_row["occ_start_line"],
                    "occ_start_col": occ_row["occ_start_col"],
                    "occ_end_line": occ_row["occ_end_line"],
                    "occ_end_col": occ_row["occ_end_col"],
                    "syntax_node_id": syntax_node_id,
                    "match_kind": match_kind,
                    "candidate_count": candidate_count,
                }
            )
    return rows


def scip_resolution__frames(
    q__core__scip_occurrences: InferableTabularInput,
    q__core__scip_symbol_information: InferableTabularInput,
    q__core__goids: InferableTabularInput,
) -> ScipResolutionFrames:
    """Build base SCIP resolution frames.

    Returns
    -------
    ScipResolutionFrames
        Frames for SCIP symbol and occurrence xref tables.
    """
    created_at = datetime.now(tz=UTC)
    occurrences = _occurrences_frame(q__core__scip_occurrences)
    symbol_info = _symbol_info_frame(q__core__scip_symbol_information)
    goids = _goids_frame(q__core__goids)
    symbol_goid_xref = _symbol_goid_xref_frame(
        occurrences=occurrences,
        goids=goids,
        created_at=created_at,
    )
    occurrence_span_xref = _occurrence_span_xref_frame(
        occurrences=occurrences,
        symbol_info=symbol_info,
        symbol_goid_xref=symbol_goid_xref,
        created_at=created_at,
    )
    return ScipResolutionFrames(
        symbol_goid_xref=symbol_goid_xref,
        occurrence_span_xref=occurrence_span_xref,
    )


def scip_resolution__symbol_goid_xref__base(
    scip_resolution__frames: ScipResolutionFrames,
) -> pl.LazyFrame:
    """Return rows for core.scip_symbol_goid_xref.

    Returns
    -------
    pl.LazyFrame
        Lazy frame for core.scip_symbol_goid_xref.
    """
    return dedupe_frame_for_table(
        scip_resolution__frames.symbol_goid_xref,
        table_key=SCIP_SYMBOL_GOID_XREF_TABLE_KEY,
    )


def scip_resolution__occurrence_span_xref__base(
    scip_resolution__frames: ScipResolutionFrames,
) -> pl.LazyFrame:
    """Return rows for core.scip_occurrence_span_xref.

    Returns
    -------
    pl.LazyFrame
        Lazy frame for core.scip_occurrence_span_xref.
    """
    return dedupe_frame_for_table(
        scip_resolution__frames.occurrence_span_xref,
        table_key=SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY,
    )


def scip_resolution__occurrence_syntax_xref__base(
    q__core__scip_occurrence_span_xref: InferableTabularInput,
    q__core__syntax_nodes: InferableTabularInput,
) -> pl.LazyFrame:
    """Return rows for core.scip_occurrence_syntax_xref.

    Returns
    -------
    pl.LazyFrame
        Lazy frame for core.scip_occurrence_syntax_xref.
    """
    occurrences_frame = tabular_to_frame(q__core__scip_occurrence_span_xref)
    nodes_frame = tabular_to_frame(q__core__syntax_nodes)
    rows = _occurrence_syntax_xref_rows(occurrences_frame, nodes_frame)
    frame = rows_to_frame(SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY, rows)
    return dedupe_frame_for_table(frame, table_key=SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY)


_MODULE = sys.modules[__name__]
_SCIP_RESOLUTION_TABLE_TARGET_SPEC = TableTargetSpec(
    domain="ingestion",
    target_name=SCIP_RESOLUTION_TARGET_NAME,
    tables=(
        TableTargetTableSpec(
            table_key=SCIP_SYMBOL_GOID_XREF_TABLE_KEY,
            base_node="scip_resolution__symbol_goid_xref__base",
            save_spec=RelationTableSaveSpec(table_key=SCIP_SYMBOL_GOID_XREF_TABLE_KEY),
            node_name="scip_resolution__symbol_goid_xref",
            input_type=pl.LazyFrame,
        ),
        TableTargetTableSpec(
            table_key=SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY,
            base_node="scip_resolution__occurrence_span_xref__base",
            save_spec=RelationTableSaveSpec(table_key=SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY),
            node_name="scip_resolution__occurrence_span_xref",
            input_type=pl.LazyFrame,
        ),
        TableTargetTableSpec(
            table_key=SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY,
            base_node="scip_resolution__occurrence_syntax_xref__base",
            save_spec=RelationTableSaveSpec(table_key=SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY),
            node_name="scip_resolution__occurrence_syntax_xref",
            input_type=pl.LazyFrame,
        ),
    ),
    table_materializations_node="scip_resolution__table_materializations",
    anchor_node_name="t__scip_resolution",
)
attach_table_target_template(_MODULE, spec=_SCIP_RESOLUTION_TABLE_TARGET_SPEC)
scip_resolution__symbol_goid_xref = _MODULE.scip_resolution__symbol_goid_xref
scip_resolution__occurrence_span_xref = _MODULE.scip_resolution__occurrence_span_xref
scip_resolution__occurrence_syntax_xref = _MODULE.scip_resolution__occurrence_syntax_xref
scip_resolution__table_materializations = _MODULE.scip_resolution__table_materializations
t__scip_resolution = _MODULE.t__scip_resolution


__all__ = [
    "SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY",
    "SCIP_OCCURRENCE_SYNTAX_XREF_TABLE_KEY",
    "SCIP_RESOLUTION_TARGET_NAME",
    "SCIP_SYMBOL_GOID_XREF_TABLE_KEY",
    "ScipResolutionFrames",
    "scip_resolution__frames",
    "scip_resolution__occurrence_span_xref",
    "scip_resolution__occurrence_syntax_xref",
    "scip_resolution__symbol_goid_xref",
    "scip_resolution__table_materializations",
    "t__scip_resolution",
]
