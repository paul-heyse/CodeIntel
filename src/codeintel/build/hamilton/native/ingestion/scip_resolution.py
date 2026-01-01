"""SCIP resolution tables for deterministic symbol/GOID stitching."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import polars as pl

from codeintel.build.hamilton.boundary_types import MaterializationResult
from codeintel.build.hamilton.dag_catalog import DagCatalog
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.native.ingestion.frame_utils import (
    dedupe_frame_for_table,
)
from codeintel.build.hamilton.native.materialization_records import (
    MaterializationRecordContext,
    record_from_materializations,
)
from codeintel.build.hamilton.native.patterns import (
    RelationTableSaveSpec,
    SaverContext,
    make_table_materializations_collector,
    save_relation_table,
)
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.hamilton.run_records import TargetRunRecord
from codeintel.build.hamilton.tagging import tag_compute
from codeintel.build.tabular.conversion import tabular_to_lazyframe
from codeintel.build.tabular.types import InferableTabularInput

_HAMILTON_TYPE_HINTS = (
    BuildEnv,
    DagCatalog,
    TargetRunRecord,
    InferableTabularInput,
)

SCIP_RESOLUTION_TARGET_NAME = "scip_resolution"
SCIP_SYMBOL_GOID_XREF_TABLE_KEY = "core.scip_symbol_goid_xref"
SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY = "core.scip_occurrence_span_xref"
SCIP_RESOLUTION_TABLE_KEYS = (
    SCIP_SYMBOL_GOID_XREF_TABLE_KEY,
    SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY,
)

SCIP_RESOLUTION_SAVE_CONTEXT = SaverContext(
    domain="ingestion",
    target=SCIP_RESOLUTION_TARGET_NAME,
)

_ROLE_DEFINITION = 0x1
_ROLE_IMPORT = 0x2
_ROLE_WRITE = 0x4
_ROLE_READ = 0x8


@dataclass(frozen=True)
class ScipResolutionFrames:
    """Derived frames for SCIP resolution outputs."""

    symbol_goid_xref: pl.LazyFrame
    occurrence_span_xref: pl.LazyFrame


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
    return frame.select(
        "goid_h128",
        "rel_path",
        "start_line",
        "end_line",
    ).filter(pl.col("start_line").is_not_null() & pl.col("end_line").is_not_null())


def _occurrences_frame(occurrences: InferableTabularInput) -> pl.LazyFrame:
    frame = tabular_to_lazyframe(occurrences)
    return frame.rename({"symbol": "scip_symbol"})


def _symbol_goid_xref_frame(
    *,
    occurrences: pl.LazyFrame,
    goids: pl.LazyFrame,
    created_at: datetime,
) -> pl.LazyFrame:
    definitions = occurrences.filter((pl.col("roles") & _ROLE_DEFINITION) != 0)
    joined = definitions.join(
        goids,
        on=["rel_path", "start_line", "end_line"],
        how="left",
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
    base = occurrences.join(
        symbol_info,
        on=["repo", "commit", "scip_symbol"],
        how="left",
    ).join(
        goid_lookup,
        on=["repo", "commit", "scip_symbol"],
        how="left",
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


@save_relation_table(
    context=SCIP_RESOLUTION_SAVE_CONTEXT,
    spec=RelationTableSaveSpec(table_key=SCIP_SYMBOL_GOID_XREF_TABLE_KEY),
)
@tag_compute(
    domain="ingestion",
    target=SCIP_RESOLUTION_TARGET_NAME,
    target_="scip_resolution__symbol_goid_xref",
)
def scip_resolution__symbol_goid_xref(
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


@save_relation_table(
    context=SCIP_RESOLUTION_SAVE_CONTEXT,
    spec=RelationTableSaveSpec(table_key=SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY),
)
@tag_compute(
    domain="ingestion",
    target=SCIP_RESOLUTION_TARGET_NAME,
    target_="scip_resolution__occurrence_span_xref",
)
def scip_resolution__occurrence_span_xref(
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


scip_resolution__table_materializations = make_table_materializations_collector(
    domain="ingestion",
    target=SCIP_RESOLUTION_TARGET_NAME,
    table_keys=SCIP_RESOLUTION_TABLE_KEYS,
    node_name="scip_resolution__table_materializations",
)


@codeintel_target(domain="ingestion", target=SCIP_RESOLUTION_TARGET_NAME)
def t__scip_resolution(
    env: BuildEnv,
    catalog: DagCatalog,
    scip_resolution__table_materializations: dict[str, MaterializationResult],
) -> TargetRunRecord:
    """Finalize scip_resolution target run record.

    Returns
    -------
    TargetRunRecord
        Run record for the scip_resolution target.
    """
    context = MaterializationRecordContext(
        env=env,
        catalog=catalog,
        target_name=SCIP_RESOLUTION_TARGET_NAME,
    )
    return record_from_materializations(
        context=context,
        artifact_materializations=None,
        table_materializations=scip_resolution__table_materializations,
    )


__all__ = [
    "SCIP_OCCURRENCE_SPAN_XREF_TABLE_KEY",
    "SCIP_RESOLUTION_TARGET_NAME",
    "SCIP_SYMBOL_GOID_XREF_TABLE_KEY",
    "ScipResolutionFrames",
    "scip_resolution__frames",
    "scip_resolution__occurrence_span_xref",
    "scip_resolution__symbol_goid_xref",
    "scip_resolution__table_materializations",
    "t__scip_resolution",
]
