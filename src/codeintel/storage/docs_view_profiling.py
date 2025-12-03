"""Profile docs view plans for subsystem profile/coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Final

from codeintel.storage.config import StorageConfig
from codeintel.storage.gateway import DuckDBConnection, DuckDBError, open_gateway

DOCS_VIEWS: Final[tuple[str, ...]] = (
    "docs.v_subsystem_profile",
    "docs.v_subsystem_coverage",
)

EXPLAIN_QUERIES: Final[dict[str, str]] = {
    "docs.v_subsystem_profile": "EXPLAIN SELECT * FROM docs.v_subsystem_profile LIMIT 50",
    "docs.v_subsystem_coverage": "EXPLAIN SELECT * FROM docs.v_subsystem_coverage LIMIT 50",
}

ANALYZE_QUERIES: Final[dict[str, str]] = {
    "docs.v_subsystem_profile": "EXPLAIN ANALYZE SELECT * FROM docs.v_subsystem_profile LIMIT 50",
    "docs.v_subsystem_coverage": "EXPLAIN ANALYZE SELECT * FROM docs.v_subsystem_coverage LIMIT 50",
}


def write_text(path: Path, content: str) -> None:
    """
    Write content to disk, ensuring the parent directory exists.

    Parameters
    ----------
    path
        Target file path.
    content
        Text content to write.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def explain(*, con: DuckDBConnection, view: str, analyze: bool) -> str:
    """
    Return EXPLAIN or EXPLAIN ANALYZE output for a docs view.

    Parameters
    ----------
    con
        Active DuckDB connection.
    view
        Fully qualified view name.
    analyze
        Whether to use EXPLAIN ANALYZE instead of EXPLAIN.

    Returns
    -------
    str
        Plan text emitted by DuckDB.
    """
    query = EXPLAIN_QUERIES[view] if not analyze else ANALYZE_QUERIES[view]
    rows = con.execute(query).fetchall()
    return "\n".join(str(row[0]) for row in rows)


def run_profile(*, db_path: Path, output_dir: Path, analyze: bool) -> None:
    """
    Generate profiling artifacts for the configured database.

    Parameters
    ----------
    db_path
        Path to the DuckDB database file.
    output_dir
        Directory to write profiling artifacts.
    analyze
        Whether to use EXPLAIN ANALYZE.

    Raises
    ------
    FileNotFoundError
        When the supplied db_path does not exist.
    """
    if not db_path.exists():
        message = f"DuckDB not found at {db_path}"
        raise FileNotFoundError(message)
    gateway = open_gateway(StorageConfig.for_readonly(db_path))
    try:
        con = gateway.con
        meta = {"db_path": str(db_path), "analyze": analyze, "views": DOCS_VIEWS}
        write_text(output_dir / "profile_meta.json", json.dumps(meta, indent=2))
        for view in DOCS_VIEWS:
            plan = explain(con=con, view=view, analyze=analyze)
            suffix = "analyze" if analyze else "explain"
            write_text(output_dir / f"{view.replace('.', '_')}.{suffix}.txt", plan)
    finally:
        gateway.close()


def main() -> int:
    """
    Entry point for generating docs view profiling artifacts.

    Returns
    -------
    int
        Process exit code (0 on success, 2 on argument parsing errors).
    """
    parser = argparse.ArgumentParser(
        description="Generate EXPLAIN/EXPLAIN ANALYZE artifacts for docs views."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("build/db/codeintel.duckdb"),
        help="Path to the DuckDB database (default: build/db/codeintel.duckdb).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("build/profiling"),
        help="Directory to write profiling artifacts (default: build/profiling).",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Emit EXPLAIN ANALYZE instead of EXPLAIN (may scan the DB).",
    )
    args = parser.parse_args()
    try:
        run_profile(db_path=args.db_path, output_dir=args.output_dir, analyze=bool(args.analyze))
    except (FileNotFoundError, DuckDBError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
