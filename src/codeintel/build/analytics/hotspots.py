"""Hotspot churn parsing and scoring helpers."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

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
        """Summarize churn counts for persistence.

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


def _parse_git_log_lines(lines: Iterable[str]) -> dict[str, ChurnSummary]:
    stats: dict[str, FileChurn] = {}
    current_commit: str | None = None
    current_author: str | None = None

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

        normalized = path.replace("\\", "/")
        churn = stats.setdefault(normalized, FileChurn())
        churn.commits.add(current_commit)
        churn.authors.add(current_author)
        churn.lines_added += added
        churn.lines_deleted += deleted

    return {path: churn.to_summary() for path, churn in stats.items()}


def parse_git_log_lines(lines: Iterable[str]) -> dict[str, ChurnSummary]:
    """Parse git log output lines into churn summaries.

    Returns
    -------
    dict[str, ChurnSummary]
        Mapping of path to churn summary values.
    """
    return _parse_git_log_lines(lines)


def _compute_hotspot_score(
    *,
    commit_count: int,
    author_count: int,
    lines_added: int,
    lines_deleted: int,
    complexity: float,
) -> float:
    churn_lines = lines_added + lines_deleted
    return (
        0.4 * math.log1p(commit_count)
        + 0.3 * math.log1p(author_count)
        + 0.2 * math.log1p(churn_lines)
        + 0.1 * math.log1p(complexity + 1.0)
    )


def compute_hotspot_rows(
    ast_metrics: Iterable[tuple[str, float]],
    churn_stats: Mapping[str, ChurnSummary] | None = None,
) -> tuple[dict[str, object], ...]:
    """Compute hotspot rows from AST metrics and churn statistics.

    Parameters
    ----------
    ast_metrics
        Iterable of (rel_path, complexity) tuples.
    churn_stats
        Optional churn summaries keyed by normalized rel_path.

    Returns
    -------
    tuple[dict[str, object], ...]
        Hotspot rows keyed by analytics.hotspots columns.
    """
    stats = churn_stats or {}
    rows: list[dict[str, object]] = []

    for rel_path, complexity in ast_metrics:
        normalized = str(rel_path).replace("\\", "/")
        summary = stats.get(normalized)
        commit_count = int(summary.get("commit_count", 0)) if summary is not None else 0
        author_count = int(summary.get("author_count", 0)) if summary is not None else 0
        lines_added = int(summary.get("lines_added", 0)) if summary is not None else 0
        lines_deleted = int(summary.get("lines_deleted", 0)) if summary is not None else 0
        safe_complexity = max(float(complexity), 0.0)

        score = _compute_hotspot_score(
            commit_count=commit_count,
            author_count=author_count,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
            complexity=safe_complexity,
        )
        rows.append(
            {
                "rel_path": normalized,
                "commit_count": commit_count,
                "author_count": author_count,
                "lines_added": lines_added,
                "lines_deleted": lines_deleted,
                "complexity": safe_complexity,
                "score": score,
            }
        )

    return tuple(rows)


__all__ = [
    "ChurnSummary",
    "FileChurn",
    "compute_hotspot_rows",
    "parse_git_log_lines",
]
