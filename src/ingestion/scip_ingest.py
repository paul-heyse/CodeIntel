# src/codeintel/ingestion/scip_ingest.py

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional

import duckdb

log = logging.getLogger(__name__)


def ingest_scip(
    con: duckdb.DuckDBPyConnection,
    repo_root: Path,
    repo: str,
    commit: str,
    *,
    build_dir: Path,
    document_output_dir: Path,
    scip_python_bin: str = "scip-python",
    scip_bin: str = "scip",
) -> Optional[Path]:
    """
    Run scip-python to build an index.scip and JSON view, then:

      - store them under `build_dir / "scip"/`
      - copy to `Document Output/` as index.scip / index.scip.json
      - register a DuckDB view `scip_index_view` over the JSON

    Returns the path to index.scip.json (or None if generation fails).
    """
    repo_root = repo_root.resolve()
    build_dir = build_dir.resolve()
    doc_dir = document_output_dir.resolve()

    scip_dir = build_dir / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)

    index_scip = scip_dir / "index.scip"
    index_json = scip_dir / "index.scip.json"

    # 1) Run scip-python index
    try:
        log.info("Running %s index in %s", scip_python_bin, repo_root)
        subprocess.run(
            [
                scip_python_bin,
                "index",
                str(repo_root),
                "--output",
                str(index_scip),
            ],
            check=True,
            cwd=str(repo_root),
        )
    except FileNotFoundError:
        log.warning("scip-python binary %r not found; skipping SCIP indexing", scip_python_bin)
        return None
    except subprocess.CalledProcessError as exc:
        log.warning("scip-python index failed with exit code %s", exc.returncode)
        return None

    # 2) Convert to JSON
    try:
        log.info("Running %s print --format json", scip_bin)
        proc = subprocess.run(
            [scip_bin, "print", "--format", "json", str(index_scip)],
            check=True,
            capture_output=True,
            text=True,
        )
        index_json.write_text(proc.stdout, encoding="utf8")
    except FileNotFoundError:
        log.warning("scip binary %r not found; skipping SCIP JSON export", scip_bin)
        return None
    except subprocess.CalledProcessError as exc:
        log.warning("scip print failed with exit code %s", exc.returncode)
        return None

    # 3) Copy into Document Output/
    doc_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(index_scip, doc_dir / "index.scip")
    shutil.copy2(index_json, doc_dir / "index.scip.json")

    # 4) Optional: register as a view in DuckDB for downstream joins
    con.execute(
        """
        CREATE OR REPLACE VIEW scip_index_view AS
        SELECT * FROM read_json(?);
        """,
        [str(index_json)],
    )

    log.info("SCIP index ingested and view scip_index_view created")
    return index_json
