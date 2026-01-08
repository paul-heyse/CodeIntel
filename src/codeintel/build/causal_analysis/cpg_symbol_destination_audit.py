"""Audit missing CPG SYMBOL destinations against symbol sources."""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pyarrow as pa

from codeintel.build.causal_analysis.scan_utils import (
    ScanConfig,
    TableScanResult,
    scan_table_with_fallback,
)
from codeintel.build.hamilton.native.graphs.cpg2.ids import cpg_node_id
from codeintel.core.columnar.iter import iter_array_values, iter_rows

LOG = logging.getLogger(__name__)

_OCCURRENCE_EDGE_KINDS = ("REFERS_TO", "DEFINES", "IMPORTS", "WRITES")
_RESOLUTION_EDGE_KIND = "RESOLVES_TO"


@dataclass(frozen=True, slots=True)
class SymbolDestinationAuditConfig:
    """Configuration for the CPG symbol destination audit."""

    dataset_root: Path
    snapshot_id: str
    output_path: Path
    repo: str | None = None
    commit: str | None = None
    sample_size: int = 20
    edge_sample_size: int = 20


@dataclass(frozen=True, slots=True)
class MatchSource:
    """Source definition for missing destination matching."""

    name: str
    table_key: str
    cpg_table_key: str | None = None
    pk_columns: tuple[str, ...] | None = None
    description: str | None = None


@dataclass(frozen=True, slots=True)
class MissingSymbolEdges:
    """Summary of missing SYMBOL-layer destinations."""

    missing_dst_ids: set[int]
    missing_edge_kind_counts: dict[str, int]
    missing_edge_kinds_by_dst: dict[int, dict[str, int]]
    missing_edge_samples: list[dict[str, object]]
    total_symbol_edges: int
    missing_symbol_edges: int


MATCH_SOURCES = (
    MatchSource(
        name="scip_symbol_information",
        table_key="core.scip_symbol_information",
        pk_columns=("repo", "commit", "symbol"),
        description="SCIP symbols that should map to CPG symbol nodes.",
    ),
    MatchSource(
        name="scip_symbol_goid_xref",
        table_key="core.scip_symbol_goid_xref",
        cpg_table_key="core.goids",
        pk_columns=("goid_h128",),
        description="GOIDs referenced by SCIP symbols.",
    ),
    MatchSource(
        name="goids",
        table_key="core.goids",
        pk_columns=("goid_h128",),
        description="All GOIDs for internal symbols.",
    ),
    MatchSource(
        name="py_sym_bindings",
        table_key="core.py_sym_bindings",
        pk_columns=("repo", "commit", "rel_path", "binding_id"),
        description="Python symtable bindings.",
    ),
    MatchSource(
        name="py_sym_symbols",
        table_key="core.py_sym_symbols",
        pk_columns=("repo", "commit", "rel_path", "symbol_row_id"),
        description="Python symtable symbols (no CPG node plane today).",
    ),
)


def _missing_ids_by_kind(
    missing_edge_kinds_by_dst: dict[int, dict[str, int]],
) -> dict[str, set[int]]:
    ids_by_kind: dict[str, set[int]] = defaultdict(set)
    for dst_id, kind_counts in missing_edge_kinds_by_dst.items():
        for kind in kind_counts:
            ids_by_kind[kind].add(dst_id)
    return {kind: set(ids) for kind, ids in ids_by_kind.items()}


def _load_run_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        LOG.warning("Failed to parse run summary: %s", exc)
        return None


def _normalize_id(value: object | None) -> int | None:
    if value is None:
        return None
    if not isinstance(value, (int, str, bytes, Decimal)):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _iter_rows(table: pa.Table | pa.RecordBatch) -> Iterable[dict[str, object]]:
    return iter_rows(table)


def _build_node_id_set(nodes: pa.Table) -> set[int]:
    node_ids: set[int] = set()
    index = nodes.schema.get_field_index("cpg_node_id")
    if index < 0:
        return node_ids
    for batch in nodes.to_batches():
        column = batch.column(index)
        for value in iter_array_values(column):
            normalized = _normalize_id(value)
            if normalized is not None:
                node_ids.add(normalized)
    return node_ids


def _missing_symbol_edges(
    edges: pa.Table,
    node_ids: set[int],
    sample_size: int,
) -> MissingSymbolEdges:
    missing_dst_ids: set[int] = set()
    missing_edge_kind_counts: dict[str, int] = defaultdict(int)
    missing_edge_kinds_by_dst: dict[int, dict[str, int]] = {}
    missing_edge_samples: list[dict[str, object]] = []
    total_symbol_edges = 0
    missing_symbol_edges = 0

    required = {"dst_cpg_node_id", "edge_layer", "edge_kind", "rel_path", "src_cpg_node_id"}
    if not required.issubset(set(edges.column_names)):
        return MissingSymbolEdges(
            missing_dst_ids=missing_dst_ids,
            missing_edge_kind_counts=dict(missing_edge_kind_counts),
            missing_edge_kinds_by_dst=missing_edge_kinds_by_dst,
            missing_edge_samples=missing_edge_samples,
            total_symbol_edges=0,
            missing_symbol_edges=0,
        )

    for batch in edges.to_batches():
        for row in _iter_rows(batch):
            edge_layer = row.get("edge_layer")
            if edge_layer != "SYMBOL":
                continue
            total_symbol_edges += 1
            dst_id = _normalize_id(row.get("dst_cpg_node_id"))
            if dst_id is None or dst_id in node_ids:
                continue
            missing_symbol_edges += 1
            missing_dst_ids.add(dst_id)
            edge_kind = row.get("edge_kind")
            if isinstance(edge_kind, str):
                missing_edge_kind_counts[edge_kind] += 1
                kind_counts = missing_edge_kinds_by_dst.setdefault(dst_id, {})
                kind_counts[edge_kind] = kind_counts.get(edge_kind, 0) + 1
            if len(missing_edge_samples) < sample_size:
                missing_edge_samples.append(
                    {
                        "src_cpg_node_id": _normalize_id(row.get("src_cpg_node_id")),
                        "dst_cpg_node_id": dst_id,
                        "edge_kind": edge_kind,
                        "edge_layer": edge_layer,
                        "rel_path": row.get("rel_path"),
                    }
                )

    return MissingSymbolEdges(
        missing_dst_ids=missing_dst_ids,
        missing_edge_kind_counts=dict(missing_edge_kind_counts),
        missing_edge_kinds_by_dst=missing_edge_kinds_by_dst,
        missing_edge_samples=missing_edge_samples,
        total_symbol_edges=total_symbol_edges,
        missing_symbol_edges=missing_symbol_edges,
    )


def _resolve_pk_columns(source: MatchSource) -> tuple[str, ...]:
    if source.pk_columns is not None:
        return source.pk_columns
    msg = f"Table {source.table_key} has no primary key configured for matching."
    raise ValueError(msg)


def _row_pk(
    row: dict[str, object],
    pk_columns: tuple[str, ...],
    *,
    repo: str | None,
    commit: str | None,
) -> dict[str, object] | None:
    pk: dict[str, object] = {}
    for column in pk_columns:
        if column == "repo" and repo is not None:
            value = repo
        elif column == "commit" and commit is not None:
            value = commit
        else:
            value = row.get(column)
        if value is None:
            return None
        pk[column] = value
    return pk


def _edge_kind_counts_for_ids(
    matched_ids: set[int],
    missing_edge_kinds_by_dst: dict[int, dict[str, int]],
) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for dst_id in matched_ids:
        for kind, count in missing_edge_kinds_by_dst.get(dst_id, {}).items():
            counts[kind] += count
    return dict(counts)


def _build_symbol_info_set(symbols: pa.Table) -> set[str]:
    if "symbol" not in symbols.column_names:
        return set()
    return {
        value for value in iter_array_values(symbols.column("symbol")) if isinstance(value, str)
    }


def _audit_occurrence_symbols(
    occurrences: pa.Table,
    missing_occurrence_ids: set[int],
    symbol_info_set: set[str],
    config: SymbolDestinationAuditConfig,
) -> dict[str, object]:
    matched_ids: set[int] = set()
    matched_symbols: set[str] = set()
    symbols_missing_info: set[str] = set()
    symbols_present_info: set[str] = set()
    seen_symbols: set[str] = set()
    rows_scanned = 0
    rows_missing_pk = 0

    for batch in occurrences.to_batches():
        for row in _iter_rows(batch):
            rows_scanned += 1
            pk = _row_pk(
                row,
                ("repo", "commit", "symbol"),
                repo=config.repo,
                commit=config.commit,
            )
            if pk is None:
                rows_missing_pk += 1
                continue
            symbol = pk.get("symbol")
            if not isinstance(symbol, str):
                rows_missing_pk += 1
                continue
            if symbol in seen_symbols:
                continue
            seen_symbols.add(symbol)
            cpg_id = cpg_node_id("core.scip_symbol_information", pk)
            if cpg_id not in missing_occurrence_ids:
                continue
            matched_ids.add(cpg_id)
            matched_symbols.add(symbol)
            if symbol in symbol_info_set:
                symbols_present_info.add(symbol)
            else:
                symbols_missing_info.add(symbol)

    summary: dict[str, object] = {
        "missing_dst_ids_total": len(missing_occurrence_ids),
        "missing_dst_ids_matched": len(matched_ids),
        "missing_dst_ids_unmatched": len(missing_occurrence_ids - matched_ids),
        "matched_symbols_unique": len(matched_symbols),
        "symbols_missing_info_count": len(symbols_missing_info),
        "symbols_present_info_count": len(symbols_present_info),
        "symbols_missing_info_sample": list(symbols_missing_info)[: config.sample_size],
        "symbols_present_info_sample": list(symbols_present_info)[: config.sample_size],
        "occurrence_rows_scanned": rows_scanned,
        "occurrence_rows_missing_pk": rows_missing_pk,
        "symbol_info_unique": len(symbol_info_set),
    }
    return summary


def _build_binding_id_set(
    bindings: pa.Table,
    config: SymbolDestinationAuditConfig,
) -> tuple[set[int], int, int]:
    ids: set[int] = set()
    rows_scanned = 0
    rows_missing_pk = 0
    for batch in bindings.to_batches():
        for row in _iter_rows(batch):
            rows_scanned += 1
            pk = _row_pk(
                row,
                ("repo", "commit", "rel_path", "binding_id"),
                repo=config.repo,
                commit=config.commit,
            )
            if pk is None:
                rows_missing_pk += 1
                continue
            ids.add(cpg_node_id("core.py_sym_bindings", pk))
    return ids, rows_scanned, rows_missing_pk


def _audit_resolution_edges(
    resolution_edges: pa.Table,
    missing_resolve_ids: set[int],
    binding_ids: set[int],
    config: SymbolDestinationAuditConfig,
) -> dict[str, object]:
    matched_ids: set[int] = set()
    matched_samples: list[dict[str, object]] = []
    rows_scanned = 0
    rows_missing_pk = 0

    for batch in resolution_edges.to_batches():
        for row in _iter_rows(batch):
            rows_scanned += 1
            repo = config.repo or row.get("repo")
            commit = config.commit or row.get("commit")
            rel_path = row.get("rel_path")
            dst_binding_id = row.get("dst_binding_id")
            if repo is None or commit is None or rel_path is None or dst_binding_id is None:
                rows_missing_pk += 1
                continue
            pk = {
                "repo": repo,
                "commit": commit,
                "rel_path": rel_path,
                "binding_id": dst_binding_id,
            }
            cpg_id = cpg_node_id("core.py_sym_bindings", pk)
            if cpg_id not in missing_resolve_ids:
                continue
            matched_ids.add(cpg_id)
            if len(matched_samples) < config.sample_size:
                matched_samples.append(
                    {
                        "dst_cpg_node_id": cpg_id,
                        "rel_path": rel_path,
                        "dst_binding_id": dst_binding_id,
                        "src_binding_id": row.get("src_binding_id"),
                        "edge_id": row.get("edge_id"),
                        "kind": row.get("kind"),
                        "confidence": row.get("confidence"),
                        "reason": row.get("reason"),
                        "binding_row_present": cpg_id in binding_ids,
                    }
                )

    matched_present = matched_ids & binding_ids
    matched_missing = matched_ids - binding_ids
    total_missing = max(len(missing_resolve_ids), 1)
    summary: dict[str, object] = {
        "missing_dst_ids_total": len(missing_resolve_ids),
        "missing_dst_ids_matched": len(matched_ids),
        "missing_dst_ids_unmatched": len(missing_resolve_ids - matched_ids),
        "matched_binding_rows_present": len(matched_present),
        "matched_binding_rows_missing": len(matched_missing),
        "matched_missing_dst_ratio": len(matched_ids) / total_missing,
        "resolution_rows_scanned": rows_scanned,
        "resolution_rows_missing_pk": rows_missing_pk,
        "matched_samples": matched_samples,
    }
    return summary


def _match_source(
    source: MatchSource,
    table: pa.Table,
    missing_dst_ids: set[int],
    missing_edge_kinds_by_dst: dict[int, dict[str, int]],
    config: SymbolDestinationAuditConfig,
) -> tuple[dict[str, object], set[int]]:
    pk_columns = _resolve_pk_columns(source)
    cpg_table_key = source.cpg_table_key or source.table_key
    matched_ids: set[int] = set()
    matched_samples: list[dict[str, object]] = []
    rows_scanned = 0
    rows_missing_pk = 0

    for batch in table.to_batches():
        for row in _iter_rows(batch):
            rows_scanned += 1
            pk = _row_pk(row, pk_columns, repo=config.repo, commit=config.commit)
            if pk is None:
                rows_missing_pk += 1
                continue
            cpg_id = cpg_node_id(cpg_table_key, pk)
            if cpg_id not in missing_dst_ids:
                continue
            matched_ids.add(cpg_id)
            if len(matched_samples) < config.sample_size:
                matched_samples.append(
                    {
                        "dst_cpg_node_id": cpg_id,
                        "pk": pk,
                        "edge_kinds": list(missing_edge_kinds_by_dst.get(cpg_id, {}).keys()),
                    }
                )

    edge_kind_counts = _edge_kind_counts_for_ids(matched_ids, missing_edge_kinds_by_dst)
    total_missing = max(len(missing_dst_ids), 1)
    summary: dict[str, object] = {
        "name": source.name,
        "table_key": source.table_key,
        "cpg_table_key": cpg_table_key,
        "pk_columns": list(pk_columns),
        "description": source.description,
        "rows_scanned": rows_scanned,
        "rows_missing_pk": rows_missing_pk,
        "matched_missing_dst_ids": len(matched_ids),
        "matched_missing_dst_ratio": len(matched_ids) / total_missing,
        "matched_edge_kind_counts": edge_kind_counts,
        "matched_samples": matched_samples,
    }
    return summary, matched_ids


def _match_sources(
    scan_config: ScanConfig,
    sources: tuple[MatchSource, ...],
    missing_edges: MissingSymbolEdges,
    config: SymbolDestinationAuditConfig,
) -> tuple[list[dict[str, object]], set[int], dict[str, TableScanResult]]:
    matches: list[dict[str, object]] = []
    matched_union: set[int] = set()
    source_scans: dict[str, TableScanResult] = {}
    for source in sources:
        scan_result = scan_table_with_fallback(
            scan_config,
            table_key=source.table_key,
            columns=_resolve_pk_columns(source),
        )
        source_scans[source.table_key] = scan_result
        source_summary, matched_ids = _match_source(
            source,
            scan_result.table,
            missing_edges.missing_dst_ids,
            missing_edges.missing_edge_kinds_by_dst,
            config,
        )
        source_summary["scan_row_counts"] = {
            "filtered_rows": scan_result.primary_row_count,
            "unfiltered_rows": scan_result.fallback_row_count,
            "used_fallback": scan_result.used_fallback,
        }
        matched_union.update(matched_ids)
        matches.append(source_summary)
    return matches, matched_union, source_scans


def _build_occurrence_summary(
    scan_config: ScanConfig,
    missing_ids_by_kind: dict[str, set[int]],
    symbol_info_scan: TableScanResult | None,
    config: SymbolDestinationAuditConfig,
) -> tuple[dict[str, object] | None, TableScanResult | None]:
    occurrence_missing_ids: set[int] = set()
    for kind in _OCCURRENCE_EDGE_KINDS:
        occurrence_missing_ids.update(missing_ids_by_kind.get(kind, set()))
    if not occurrence_missing_ids:
        return None, None
    symbol_info_set = (
        _build_symbol_info_set(symbol_info_scan.table) if symbol_info_scan is not None else set()
    )
    occurrence_scan = scan_table_with_fallback(
        scan_config,
        table_key="core.scip_occurrences",
        columns=("repo", "commit", "symbol"),
    )
    occurrence_summary = _audit_occurrence_symbols(
        occurrence_scan.table,
        occurrence_missing_ids,
        symbol_info_set,
        config,
    )
    occurrence_summary["scan_row_counts"] = {
        "filtered_rows": occurrence_scan.primary_row_count,
        "unfiltered_rows": occurrence_scan.fallback_row_count,
        "used_fallback": occurrence_scan.used_fallback,
    }
    return occurrence_summary, occurrence_scan


def _build_resolution_summary(
    scan_config: ScanConfig,
    missing_ids_by_kind: dict[str, set[int]],
    binding_scan: TableScanResult | None,
    config: SymbolDestinationAuditConfig,
) -> tuple[dict[str, object] | None, TableScanResult | None]:
    resolve_missing_ids = missing_ids_by_kind.get(_RESOLUTION_EDGE_KIND, set())
    if not resolve_missing_ids:
        return None, None
    if binding_scan is None:
        binding_scan = scan_table_with_fallback(
            scan_config,
            table_key="core.py_sym_bindings",
            columns=("repo", "commit", "rel_path", "binding_id"),
        )
    binding_ids, binding_rows_scanned, binding_rows_missing_pk = _build_binding_id_set(
        binding_scan.table,
        config,
    )
    resolution_scan = scan_table_with_fallback(
        scan_config,
        table_key="core.py_sym_resolution_edges",
        columns=(
            "repo",
            "commit",
            "rel_path",
            "edge_id",
            "src_binding_id",
            "dst_binding_id",
            "kind",
            "confidence",
            "reason",
        ),
    )
    resolution_summary = _audit_resolution_edges(
        resolution_scan.table,
        resolve_missing_ids,
        binding_ids,
        config,
    )
    resolution_summary["binding_rows_scanned"] = binding_rows_scanned
    resolution_summary["binding_rows_missing_pk"] = binding_rows_missing_pk
    resolution_summary["scan_row_counts"] = {
        "filtered_rows": resolution_scan.primary_row_count,
        "unfiltered_rows": resolution_scan.fallback_row_count,
        "used_fallback": resolution_scan.used_fallback,
    }
    return resolution_summary, resolution_scan


def audit_cpg_symbol_destinations(config: SymbolDestinationAuditConfig) -> dict[str, object]:
    """Audit missing SYMBOL-layer destinations against candidate sources.

    Returns
    -------
    dict[str, object]
        JSON-serializable summary of audit results.
    """
    scan_config = ScanConfig(
        dataset_root=config.dataset_root,
        snapshot_id=config.snapshot_id,
        repo=config.repo,
        commit=config.commit,
    )
    nodes_result = scan_table_with_fallback(
        scan_config,
        table_key="graph.cpg_nodes",
        columns=("cpg_node_id",),
    )
    edges_result = scan_table_with_fallback(
        scan_config,
        table_key="graph.cpg_edges",
        columns=(
            "src_cpg_node_id",
            "dst_cpg_node_id",
            "edge_kind",
            "edge_layer",
            "rel_path",
        ),
    )

    node_ids = _build_node_id_set(nodes_result.table)
    missing_edges = _missing_symbol_edges(
        edges_result.table,
        node_ids,
        sample_size=config.edge_sample_size,
    )
    missing_ids_by_kind = _missing_ids_by_kind(missing_edges.missing_edge_kinds_by_dst)
    matches, matched_union, source_scans = _match_sources(
        scan_config,
        MATCH_SOURCES,
        missing_edges,
        config,
    )
    occurrence_summary, occurrence_scan = _build_occurrence_summary(
        scan_config,
        missing_ids_by_kind,
        source_scans.get("core.scip_symbol_information"),
        config,
    )
    resolution_summary, resolution_scan = _build_resolution_summary(
        scan_config,
        missing_ids_by_kind,
        source_scans.get("core.py_sym_bindings"),
        config,
    )

    unmatched_count = len(missing_edges.missing_dst_ids - matched_union)
    summary: dict[str, object] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "snapshot_id": config.snapshot_id,
        "repo": config.repo,
        "commit": config.commit,
        "symbol_edges_total": missing_edges.total_symbol_edges,
        "symbol_edges_missing_dst": missing_edges.missing_symbol_edges,
        "missing_dst_unique": len(missing_edges.missing_dst_ids),
        "missing_edge_kind_counts": missing_edges.missing_edge_kind_counts,
        "missing_dst_ids_by_edge_kind": {
            kind: len(ids) for kind, ids in missing_ids_by_kind.items()
        },
        "missing_edge_samples": missing_edges.missing_edge_samples,
        "source_matches": matches,
        "occurrence_symbol_coverage": occurrence_summary,
        "resolution_edge_coverage": resolution_summary,
        "missing_dst_unmatched": unmatched_count,
        "scan_notes": [
            note
            for note in (
                "graph.cpg_nodes fallback used" if nodes_result.used_fallback else None,
                "graph.cpg_edges fallback used" if edges_result.used_fallback else None,
                "core.scip_occurrences fallback used"
                if occurrence_scan is not None and occurrence_scan.used_fallback
                else None,
                "core.py_sym_resolution_edges fallback used"
                if resolution_scan is not None and resolution_scan.used_fallback
                else None,
            )
            if note is not None
        ],
    }
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Wrote CPG symbol destination audit to %s", config.output_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit missing CPG symbol destinations.")
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("build/datasets"),
        help="Root directory for datasets.",
    )
    parser.add_argument(
        "--snapshot-id",
        type=str,
        default=None,
        help="Snapshot id to analyze (defaults to run summary commit).",
    )
    parser.add_argument(
        "--run-summary",
        type=Path,
        default=Path("build/diagnostics/run_summary.json"),
        help="Path to run summary json for default snapshot id.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("build/diagnostics/cpg_symbol_destination_audit.json"),
        help="Output path for the audit json.",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="Repo filter (used when dataset columns contain repo).",
    )
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="Commit filter (used when dataset columns contain commit).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of matched destination samples to retain per source.",
    )
    parser.add_argument(
        "--edge-sample-size",
        type=int,
        default=20,
        help="Number of missing edge samples to retain.",
    )
    return parser


def main() -> None:
    """Run the CLI for auditing CPG symbol destinations.

    Raises
    ------
    ValueError
        If snapshot_id is missing and run summary is unavailable.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = _build_parser()
    args = parser.parse_args()
    run_summary = _load_run_summary(args.run_summary)
    repo = args.repo or (run_summary.get("repo") if run_summary else None)
    commit = args.commit or (run_summary.get("commit") if run_summary else None)
    snapshot_id = args.snapshot_id or commit
    if snapshot_id is None:
        msg = "snapshot_id is required when run summary is unavailable"
        raise ValueError(msg)
    audit_cpg_symbol_destinations(
        SymbolDestinationAuditConfig(
            dataset_root=args.dataset_root,
            snapshot_id=snapshot_id,
            output_path=args.output_path,
            repo=repo,
            commit=commit,
            sample_size=args.sample_size,
            edge_sample_size=args.edge_sample_size,
        )
    )


if __name__ == "__main__":
    main()
