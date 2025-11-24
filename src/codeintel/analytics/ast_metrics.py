"""
Compute AST-derived metrics and git churn hotspots for a repository snapshot.

The routines here populate DuckDB analytics tables by combining LibCST-derived
complexity estimates with lightweight git log statistics. Hotspot scores help
prioritize files that change frequently and carry structural complexity.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from dataclasses import dataclass, field

from codeintel.config.models import HotspotsConfig
from codeintel.ingestion.common import run_batch
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.models.rows import HotspotRow, hotspot_row_to_tuple
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)
MAX_STDERR_CHARS = 500
NUMSTAT_FIELDS = 3
ChurnSummary = dict[str, int]


@dataclass
class FileChurn:
    """Aggregated churn details for a single file."""

    commits: set[str] = field(default_factory=set)
    authors: set[str] = field(default_factory=set)
    lines_added: int = 0
    lines_deleted: int = 0

    def to_summary(self) -> dict[str, int]:
        """
        Summarize churn counts for persistence.

        Returns
        -------
        dict[str, int]
            Counts keyed by commit/author/lines added/deleted.
        """
        return {
            "commit_count": len(self.commits),
            "author_count": len(self.authors),
            "lines_added": self.lines_added,
            "lines_deleted": self.lines_deleted,
        }


def _collect_git_file_stats(
    cfg: HotspotsConfig,
    runner: ToolRunner | None = None,
) -> dict[str, ChurnSummary]:
    """
    Collect per-file churn statistics from `git log --numstat`.

    The routine shells out to git to gather commit counts, author counts, and
    added/deleted line totals for every path touched within the configured
    commit window. It tolerates missing git binaries by returning an empty
    mapping so callers can still emit AST-only metrics.

    Parameters
    ----------
    cfg : HotspotsConfig
        Repository context including the root directory and log depth limit.
    runner : ToolRunner | None
        Optional shared ToolRunner for git invocations.

    Returns
    -------
    dict[str, dict]
        Mapping of repository-relative paths to churn statistics. Each value
        includes commit_count, author_count, lines_added, and lines_deleted.

    Notes
    -----
    The function logs git failures but never raises, enabling analytics
    pipelines to proceed even in shallow or detached checkouts. Complexity is
    proportional to the number of git log entries inspected.
    """
    if cfg.max_commits <= 0:
        return {}

    git_lines = _run_git_log(cfg, runner=runner)
    if git_lines is None:
        return {}
    return _parse_git_log_lines(git_lines)


def _run_git_log(cfg: HotspotsConfig, runner: ToolRunner | None = None) -> list[str] | None:
    repo_root = cfg.repo_root.resolve()
    active_runner = runner or ToolRunner(cache_dir=repo_root / "build" / ".tool_cache")
    args = [
        "git",
        "log",
        f"--max-count={cfg.max_commits}",
        "--numstat",
        "--date=short",
        "--pretty=format:COMMIT\t%H\t%an",
        "--no-renames",
    ]
    result = active_runner.run("git", args, cwd=repo_root)
    if result.returncode not in {0, 1}:
        log.warning(
            "git log exited with code %s; stdout=%s stderr=%s",
            result.returncode,
            result.stdout[:MAX_STDERR_CHARS],
            result.stderr[:MAX_STDERR_CHARS],
        )
    if result.returncode not in {0, 1}:
        return None
    return result.stdout.splitlines()


def _parse_git_log_lines(lines: Iterable[str]) -> dict[str, ChurnSummary]:
    stats: dict[str, FileChurn] = {}
    current_commit = None
    current_author = None

    for raw_line in lines:
        if not raw_line:
            continue
        if raw_line.startswith("COMMIT\t"):
            _, commit_hash, author = raw_line.split("\t", 2)
            current_commit = commit_hash
            current_author = author
            continue

        parts = raw_line.split("\t")
        if len(parts) != NUMSTAT_FIELDS or current_commit is None or current_author is None:
            continue
        added_s, deleted_s, path = parts
        added = int(added_s) if added_s.isdigit() else 0
        deleted = int(deleted_s) if deleted_s.isdigit() else 0

        churn = stats.setdefault(path.replace("\\", "/"), FileChurn())
        churn.commits.add(current_commit)
        churn.authors.add(current_author)
        churn.lines_added += added
        churn.lines_deleted += deleted

    return {path: churn.to_summary() for path, churn in stats.items()}


def build_hotspots(
    gateway: StorageGateway,
    cfg: HotspotsConfig,
    *,
    runner: ToolRunner | None = None,
) -> None:
    """
    Populate `analytics.hotspots` by merging AST complexity with git churn.

    Extended Summary
    ----------------
    The query walks all files present in `core.ast_metrics`, enriches them with
    git commit and author counts, and derives a composite hotspot score. Scores
    emphasize frequently modified files with higher structural complexity, which
    helps prioritize review and testing effort.

    Parameters
    ----------
    gateway :
        StorageGateway providing access to `core.ast_metrics` and `analytics.hotspots`
        tables.
    cfg : HotspotsConfig
        Repository metadata and git scan configuration used to scope the build.
    runner : ToolRunner | None
        Optional shared ToolRunner for git invocations (defaults to a local cache).

    Notes
    -----
    - Time complexity is O(n) over the number of files in `core.ast_metrics`
      plus the configured git log window.
    - The function is idempotent per repo/commit: it truncates
      `analytics.hotspots` before inserting new rows.
    - Git log failures degrade gracefully by treating churn metrics as zeros.

    Examples
    --------
    >>> from pathlib import Path
    >>> from codeintel.storage.gateway import open_memory_gateway
    >>> gateway = open_memory_gateway()
    >>> con = gateway.con
    >>> _ = con.execute("CREATE SCHEMA core")
    >>> _ = con.execute("CREATE SCHEMA analytics")
    >>> _ = con.execute("CREATE TABLE core.ast_metrics(rel_path VARCHAR, complexity DOUBLE)")
    >>> _ = con.execute(
    ...     "CREATE TABLE analytics.hotspots(rel_path VARCHAR, commit_count INTEGER,"
    ...     " author_count INTEGER, lines_added INTEGER, lines_deleted INTEGER,"
    ...     " complexity DOUBLE, score DOUBLE)"
    ... )
    >>> _ = con.execute("INSERT INTO core.ast_metrics VALUES ('sample.py', 3.0)")
    >>> cfg = HotspotsConfig(
    ...     repo="demo",
    ...     commit="abc123",
    ...     repo_root=Path("."),
    ...     max_commits=0,
    ... )
    >>> build_hotspots(gateway, cfg)  # consumes empty git stats when max_commits=0
    >>> con.execute("SELECT rel_path, score FROM analytics.hotspots").fetchall()
    [('sample.py', 0.17328679513998632)]
    """
    con = gateway.con
    df_ast = con.execute("SELECT rel_path, complexity FROM core.ast_metrics").fetch_df()
    if df_ast.empty:
        log.info("No rows in core.ast_metrics; skipping hotspots.")
        return

    git_stats = _collect_git_file_stats(cfg, runner=runner)

    con.execute("DELETE FROM analytics.hotspots")

    rows: list[HotspotRow] = []
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
            HotspotRow(
                rel_path=rel_path,
                commit_count=commit_count,
                author_count=author_count,
                lines_added=lines_added,
                lines_deleted=lines_deleted,
                complexity=complexity,
                score=score,
            )
        )

    run_batch(
        gateway,
        "analytics.hotspots",
        [hotspot_row_to_tuple(row) for row in rows],
        scope=f"{cfg.repo}@{cfg.commit}",
    )

    log.info(
        "Hotspots build complete for repo=%s commit=%s: %d files",
        cfg.repo,
        cfg.commit,
        len(rows),
    )
