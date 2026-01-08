"""CPG edge integrity analysis helpers."""

from __future__ import annotations

import argparse
import json
import logging
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Any

from codeintel.build.causal_analysis.scan_utils import ScanConfig, scan_table_with_fallback
from codeintel.build.tabular.arrow_ops import iter_rows

LOG = logging.getLogger(__name__)

if TYPE_CHECKING:
    import pyarrow as pa


@dataclass(frozen=True, slots=True)
class CpgEdgeIntegrityConfig:
    """Configuration for CPG edge integrity analysis."""

    dataset_root: Path
    snapshot_id: str
    output_path: Path
    repo: str | None = None
    commit: str | None = None
    max_group_rows: int = 50
    missing_sample_size: int = 20
    edge_sample_size: int = 20


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


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _summarize_group(
    rows: Sequence[Mapping[str, object]],
    group_cols: list[str],
    max_rows: int | None,
) -> list[dict[str, Any]]:
    if not rows:
        return []
    stats_by_key: dict[tuple[object, ...], dict[str, int]] = {}
    for row in rows:
        key = tuple(row.get(column) for column in group_cols)
        stats = stats_by_key.setdefault(
            key,
            {
                "total": 0,
                "missing_src": 0,
                "missing_dst": 0,
                "missing_any": 0,
                "missing_both": 0,
            },
        )
        stats["total"] += 1
        if row.get("missing_src"):
            stats["missing_src"] += 1
        if row.get("missing_dst"):
            stats["missing_dst"] += 1
        if row.get("missing_any"):
            stats["missing_any"] += 1
        if row.get("missing_both"):
            stats["missing_both"] += 1
    records: list[dict[str, Any]] = []
    for key, stats in stats_by_key.items():
        record = {column: key[idx] for idx, column in enumerate(group_cols)}
        record.update(stats)
        total = max(stats["total"], 1)
        record["missing_src_rate"] = stats["missing_src"] / total
        record["missing_dst_rate"] = stats["missing_dst"] / total
        record["missing_any_rate"] = stats["missing_any"] / total
        records.append(record)
    records.sort(
        key=lambda item: (
            item["missing_any"],
            item["missing_dst"],
            item["total"],
        ),
        reverse=True,
    )
    if max_rows is not None:
        records = records[:max_rows]
    return [{str(key): value for key, value in row.items()} for row in records]


def _collect_samples(
    values: Iterable[object],
    sample_size: int,
) -> list[str]:
    if sample_size <= 0:
        return []
    samples: list[str] = []
    seen: set[object] = set()
    for value in values:
        if value is None or value in seen:
            continue
        seen.add(value)
        samples.append(str(value))
        if len(samples) >= sample_size:
            break
    return samples


def _build_node_maps(
    nodes: pa.Table,
) -> tuple[dict[int, str | None], dict[int, str | None], set[int]]:
    node_kind_map: dict[int, str | None] = {}
    source_table_map: dict[int, str | None] = {}
    node_ids: set[int] = set()
    if nodes.num_rows == 0:
        return node_kind_map, source_table_map, node_ids
    for row in iter_rows(nodes, ["cpg_node_id", "node_kind", "source_table_key"]):
        node_id = _normalize_id(row.get("cpg_node_id"))
        if node_id is None:
            continue
        node_ids.add(node_id)
        node_kind = row.get("node_kind")
        node_kind_map[node_id] = node_kind if isinstance(node_kind, str) else None
        source_table = row.get("source_table_key")
        source_table_map[node_id] = source_table if isinstance(source_table, str) else None
    return node_kind_map, source_table_map, node_ids


def _build_edge_rows(
    edges: pa.Table,
    node_ids: set[int],
    node_kind_map: dict[int, str | None],
    source_table_map: dict[int, str | None],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if edges.num_rows == 0:
        return rows
    for row in iter_rows(
        edges,
        ["src_cpg_node_id", "dst_cpg_node_id", "edge_kind", "edge_layer", "rel_path"],
    ):
        src_id = _normalize_id(row.get("src_cpg_node_id"))
        dst_id = _normalize_id(row.get("dst_cpg_node_id"))
        missing_src = src_id not in node_ids
        missing_dst = dst_id not in node_ids
        rows.append(
            {
                "src_cpg_node_id_int": src_id,
                "dst_cpg_node_id_int": dst_id,
                "edge_kind": row.get("edge_kind"),
                "edge_layer": row.get("edge_layer"),
                "rel_path": row.get("rel_path"),
                "missing_src": missing_src,
                "missing_dst": missing_dst,
                "missing_any": missing_src or missing_dst,
                "missing_both": missing_src and missing_dst,
                "src_node_kind": node_kind_map.get(src_id) if src_id is not None else None,
                "src_source_table_key": (
                    source_table_map.get(src_id) if src_id is not None else None
                ),
            }
        )
    return rows


def _build_summary(
    edge_rows: list[dict[str, object]],
    node_rows: int,
    config: CpgEdgeIntegrityConfig,
) -> dict[str, Any]:
    edge_count = len(edge_rows)
    missing_src_count = sum(1 for row in edge_rows if row.get("missing_src"))
    missing_dst_count = sum(1 for row in edge_rows if row.get("missing_dst"))
    missing_both_count = sum(1 for row in edge_rows if row.get("missing_both"))
    missing_any_count = sum(1 for row in edge_rows if row.get("missing_any"))
    rate_denominator = max(edge_count, 1)

    summary: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "snapshot_id": config.snapshot_id,
        "repo": config.repo,
        "commit": config.commit,
        "edge_rows": edge_count,
        "node_rows": node_rows,
        "missing_counts": {
            "src_missing": missing_src_count,
            "dst_missing": missing_dst_count,
            "missing_any": missing_any_count,
            "missing_both": missing_both_count,
            "missing_src_only": missing_src_count - missing_both_count,
            "missing_dst_only": missing_dst_count - missing_both_count,
        },
        "missing_rates": {
            "src_missing": missing_src_count / rate_denominator,
            "dst_missing": missing_dst_count / rate_denominator,
            "missing_any": missing_any_count / rate_denominator,
            "missing_both": missing_both_count / rate_denominator,
        },
        "breakdowns": {
            "edge_kind": _summarize_group(edge_rows, ["edge_kind"], None),
            "edge_layer": _summarize_group(edge_rows, ["edge_layer"], None),
            "edge_kind_layer": _summarize_group(
                edge_rows,
                ["edge_kind", "edge_layer"],
                config.max_group_rows,
            ),
            "rel_path": _summarize_group(edge_rows, ["rel_path"], config.max_group_rows),
            "src_node_kind": _summarize_group(edge_rows, ["src_node_kind"], None),
            "src_source_table_key": _summarize_group(
                edge_rows,
                ["src_source_table_key"],
                None,
            ),
        },
        "samples": {
            "missing_src_ids": _collect_samples(
                (row.get("src_cpg_node_id_int") for row in edge_rows if row.get("missing_src")),
                config.missing_sample_size,
            ),
            "missing_dst_ids": _collect_samples(
                (row.get("dst_cpg_node_id_int") for row in edge_rows if row.get("missing_dst")),
                config.missing_sample_size,
            ),
            "missing_edge_samples": (
                [
                    {
                        "src_cpg_node_id_int": row.get("src_cpg_node_id_int"),
                        "dst_cpg_node_id_int": row.get("dst_cpg_node_id_int"),
                        "edge_kind": row.get("edge_kind"),
                        "edge_layer": row.get("edge_layer"),
                        "rel_path": row.get("rel_path"),
                        "src_node_kind": row.get("src_node_kind"),
                        "src_source_table_key": row.get("src_source_table_key"),
                    }
                    for row in edge_rows
                    if row.get("missing_any")
                ][: config.edge_sample_size]
            ),
        },
    }
    return summary


def analyze_cpg_edge_integrity(config: CpgEdgeIntegrityConfig) -> dict[str, Any]:
    """Analyze CPG edges for missing node references.

    Returns
    -------
    dict[str, Any]
        JSON-serializable summary of missing node references.
    """
    scan_config = ScanConfig(
        dataset_root=config.dataset_root,
        snapshot_id=config.snapshot_id,
        repo=config.repo,
        commit=config.commit,
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
    nodes_result = scan_table_with_fallback(
        scan_config,
        table_key="graph.cpg_nodes",
        columns=(
            "cpg_node_id",
            "node_kind",
            "source_table_key",
        ),
    )
    node_kind_map, source_table_map, node_ids = _build_node_maps(nodes_result.table)
    edge_rows = _build_edge_rows(
        edges_result.table,
        node_ids,
        node_kind_map,
        source_table_map,
    )
    summary = _build_summary(
        edge_rows=edge_rows,
        node_rows=nodes_result.table.num_rows,
        config=config,
    )
    scan_notes: list[str] = []
    if edges_result.used_fallback:
        scan_notes.append(
            "graph.cpg_edges used unfiltered scan after filtered scan returned 0 rows"
        )
    if nodes_result.used_fallback:
        scan_notes.append(
            "graph.cpg_nodes used unfiltered scan after filtered scan returned 0 rows"
        )
    if scan_notes:
        summary["scan_notes"] = scan_notes
    summary["scan_row_counts"] = {
        "graph.cpg_edges": {
            "filtered_rows": edges_result.primary_row_count,
            "unfiltered_rows": edges_result.fallback_row_count,
        },
        "graph.cpg_nodes": {
            "filtered_rows": nodes_result.primary_row_count,
            "unfiltered_rows": nodes_result.fallback_row_count,
        },
    }
    _ensure_parent(config.output_path)
    config.output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    LOG.info("Wrote CPG edge integrity analysis to %s", config.output_path)
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze CPG edge integrity.")
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
        default=Path("build/diagnostics/cpg_edge_integrity_detailed.json"),
        help="Output path for the analysis json.",
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
        "--max-group-rows",
        type=int,
        default=50,
        help="Maximum rows to keep for large group breakdowns.",
    )
    parser.add_argument(
        "--missing-sample-size",
        type=int,
        default=20,
        help="Number of missing id samples to retain.",
    )
    parser.add_argument(
        "--edge-sample-size",
        type=int,
        default=20,
        help="Number of missing edge samples to retain.",
    )
    return parser


def main() -> None:
    """Run the CLI for CPG edge integrity analysis.

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

    analyze_cpg_edge_integrity(
        CpgEdgeIntegrityConfig(
            dataset_root=args.dataset_root,
            snapshot_id=snapshot_id,
            output_path=args.output_path,
            repo=repo,
            commit=commit,
            max_group_rows=args.max_group_rows,
            missing_sample_size=args.missing_sample_size,
            edge_sample_size=args.edge_sample_size,
        )
    )


if __name__ == "__main__":
    main()
