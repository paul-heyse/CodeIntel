# src/codeintel/docs_export/export_jsonl.py

from __future__ import annotations

"""
JSON/JSONL exporters for the CodeIntel metadata warehouse.

This module writes:

  - JSONL files for all tabular datasets described in README_METADATA
    (e.g. goids.jsonl, call_graph_edges.jsonl, coverage_functions.jsonl).
  - repo_map.json as a single JSON object.
  - An optional index.json manifest of exported files. :contentReference[oaicite:4]{index=4}
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import duckdb

log = logging.getLogger(__name__)


# Map DuckDB tables -> Document Output JSONL basenames
JSONL_DATASETS: Dict[str, str] = {
    # GOIDs / crosswalk
    "core.goids": "goids.jsonl",
    "core.goid_crosswalk": "goid_crosswalk.jsonl",

    # Call graph
    "graph.call_graph_nodes": "call_graph_nodes.jsonl",
    "graph.call_graph_edges": "call_graph_edges.jsonl",

    # CFG / DFG
    "graph.cfg_blocks": "cfg_blocks.jsonl",
    "graph.cfg_edges": "cfg_edges.jsonl",
    "graph.dfg_edges": "dfg_edges.jsonl",

    # Import / symbol uses
    "graph.import_graph_edges": "import_graph_edges.jsonl",
    "graph.symbol_use_edges": "symbol_use_edges.jsonl",

    # AST / CST
    "core.ast_nodes": "ast_nodes.jsonl",
    "core.ast_metrics": "ast_metrics.jsonl",
    "core.cst_nodes": "cst_nodes.jsonl",

    # Modules / config / diagnostics
    "core.modules": "modules.jsonl",
    "analytics.config_values": "config_values.jsonl",
    "analytics.static_diagnostics": "static_diagnostics.jsonl",

    # AST analytics / typing
    "analytics.hotspots": "hotspots.jsonl",
    "analytics.typedness": "typedness.jsonl",

    # Function analytics
    "analytics.function_metrics": "function_metrics.jsonl",
    "analytics.function_types": "function_types.jsonl",

    # Coverage + tests
    "analytics.coverage_lines": "coverage_lines.jsonl",
    "analytics.coverage_functions": "coverage_functions.jsonl",
    "analytics.test_catalog": "test_catalog.jsonl",
    "analytics.test_coverage_edges": "test_coverage_edges.jsonl",

    # Risk factors
    "analytics.goid_risk_factors": "goid_risk_factors.jsonl",
}


def export_jsonl_for_table(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    output_path: Path,
) -> None:
    """
    Export a single DuckDB table to JSONL using:

        COPY (SELECT * FROM <table>) TO 'path' (FORMAT JSON, ARRAY FALSE);

    DuckDB's JSON export with ARRAY FALSE produces one JSON object per line,
    matching the JSONL format described in the README. :contentReference[oaicite:5]{index=5}
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sql = f"COPY (SELECT * FROM {table_name}) TO ? (FORMAT JSON, ARRAY FALSE);"
    log.info("Exporting %s -> %s", table_name, output_path)
    con.execute(sql, [str(output_path)])


def export_all_jsonl(
    con: duckdb.DuckDBPyConnection,
    document_output_dir: Path,
) -> List[Path]:
    """
    Export all known datasets to JSONL files under `Document Output/`.

    Returns the list of JSON/JSONL paths written.
    """
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    written: List[Path] = []

    for table_name, filename in JSONL_DATASETS.items():
        output_path = document_output_dir / filename
        try:
            export_jsonl_for_table(con, table_name, output_path)
            written.append(output_path)
        except duckdb.Error as exc:
            # Log and continue: some tables may legitimately be empty or missing
            log.warning(
                "Failed to export %s to %s: %s",
                table_name,
                output_path,
                exc,
            )

    # repo_map.json is handled separately
    repo_map_path = export_repo_map_json(con, document_output_dir)
    if repo_map_path is not None:
        written.append(repo_map_path)

    # Optionally write a small manifest
    index_path = document_output_dir / "index.json"
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "files": [p.name for p in written],
    }
    index_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    written.append(index_path)

    return written


def export_repo_map_json(
    con: duckdb.DuckDBPyConnection,
    document_output_dir: Path,
) -> Path | None:
    """
    Export core.repo_map to a `repo_map.json` file matching the README
    structure:

      {
        "repo": "...",
        "commit": "...",
        "modules": { "pkg.mod": "path/to/file.py", ... },
        "overlays": {...},
        "generated_at": "2024-01-01T00:00:00Z"
      } :contentReference[oaicite:6]{index=6}
    """
    document_output_dir = document_output_dir.resolve()
    document_output_dir.mkdir(parents=True, exist_ok=True)

    df = con.execute("SELECT repo, commit, modules, overlays, generated_at FROM core.repo_map").fetch_df()
    if df.empty:
        log.warning("core.repo_map is empty; skipping repo_map.json export")
        return None

    # For now we export the first row; typical usage is one repo/commit per DB.
    row = df.iloc[0]
    payload = {
        "repo": row["repo"],
        "commit": row["commit"],
        "modules": row["modules"],
        "overlays": row.get("overlays") if "overlays" in df.columns else {},
        "generated_at": row["generated_at"].isoformat() if hasattr(row["generated_at"], "isoformat") else str(row["generated_at"]),
    }

    output_path = document_output_dir / "repo_map.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log.info("Exported repo_map.json -> %s", output_path)
    return output_path
