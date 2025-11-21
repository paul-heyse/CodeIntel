# src/codeintel/analytics/ast_metrics.py

from __future__ import annotations

import logging
import math
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import duckdb

log = logging.getLogger(__name__)


@dataclass
class HotspotsConfig:
    repo: str
    commit: str
    repo_root: Path
    max_commits: int = 2000  # limit git log depth for performance


def _collect_git_file_stats(cfg: HotspotsConfig) -> Dict[str, dict]:
    """
    Collect per-file commit/author/line-change stats from `git log --numstat`. 
    """
    repo_root = cfg.repo_root.resolve()
    cmd = [
        "git",
        "log",
        f"--max-count={cfg.max_commits}",
        "--numstat",
        "--date=short",
        "--pretty=format:COMMIT\t%H\t%an",
        "--no-renames",
    ]

    try:
        proc = subprocess.run(
            cmd,
            cwd=repo_root,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        log.warning("git not found; hotspot analytics will have zero churn data.")
        return {}

    if proc.returncode not in (0, 1):
        log.warning(
            "git log exited with code %s; stdout=%s stderr=%s",
            proc.returncode,
            proc.stdout[:500],
            proc.stderr[:500],
        )

    stats: Dict[str, dict] = defaultdict(
        lambda: {
            "commits": set(),
            "authors": set(),
            "lines_added": 0,
            "lines_deleted": 0,
        }
    )

    current_commit = None
    current_author = None

    for line in (proc.stdout or "").splitlines():
        if not line:
            continue
        if line.startswith("COMMIT\t"):
            _, commit_hash, author = line.split("\t", 2)
            current_commit = commit_hash
            current_author = author
            continue

        # numstat line: "<added>\t<deleted>\t<path>"
        parts = line.split("\t")
        if len(parts) != 3 or current_commit is None or current_author is None:
            continue
        added_s, deleted_s, path = parts
        try:
            added = int(added_s) if added_s != "-" else 0
            deleted = int(deleted_s) if deleted_s != "-" else 0
        except ValueError:
            added = deleted = 0

        rel_path = path.replace("\\", "/")
        s = stats[rel_path]
        s["commits"].add(current_commit)
        s["authors"].add(current_author)
        s["lines_added"] += added
        s["lines_deleted"] += deleted

    # Collapse sets to counts
    for rel_path, s in stats.items():
        s["commit_count"] = len(s["commits"])
        s["author_count"] = len(s["authors"])

    return stats


def build_hotspots(con: duckdb.DuckDBPyConnection, cfg: HotspotsConfig) -> None:
    """
    Populate analytics.hotspots from core.ast_metrics + git churn stats. 
    """
    df_ast = con.execute("SELECT rel_path, complexity FROM core.ast_metrics").fetch_df()
    if df_ast.empty:
        log.info("No rows in core.ast_metrics; skipping hotspots.")
        return

    git_stats = _collect_git_file_stats(cfg)

    con.execute("DELETE FROM analytics.hotspots")

    rows: List[tuple] = []
    for _, row in df_ast.iterrows():
        rel_path = str(row["rel_path"]).replace("\\", "/")
        complexity = float(row["complexity"]) if row["complexity"] is not None else 0.0
        s = git_stats.get(
            rel_path,
            {
                "commit_count": 0,
                "author_count": 0,
                "lines_added": 0,
                "lines_deleted": 0,
            },
        )
        commit_count = int(s.get("commit_count", 0))
        author_count = int(s.get("author_count", 0))
        lines_added = int(s.get("lines_added", 0))
        lines_deleted = int(s.get("lines_deleted", 0))

        # Simple composite hotspot score
        score = (
            0.4 * math.log1p(commit_count)
            + 0.3 * math.log1p(author_count)
            + 0.2 * math.log1p(lines_added + lines_deleted)
            + 0.1 * math.log1p(max(complexity, 0.0) + 1.0)
        )

        rows.append(
            (
                rel_path,
                commit_count,
                author_count,
                lines_added,
                lines_deleted,
                complexity,
                score,
            )
        )

    if rows:
        con.executemany(
            """
            INSERT INTO analytics.hotspots
              (rel_path, commit_count, author_count,
               lines_added, lines_deleted, complexity, score)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )

    log.info(
        "Hotspots build complete for repo=%s commit=%s: %d files",
        cfg.repo,
        cfg.commit,
        len(rows),
    )
