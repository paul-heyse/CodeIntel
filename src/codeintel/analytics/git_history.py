"""Git history helpers for churn and per-function deltas."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from codeintel.ingestion.tool_runner import ToolRunner

log = logging.getLogger(__name__)

MAX_STDERR_CHARS = 500


@dataclass(frozen=True)
class FileCommitDelta:
    """Structured representation of a single commit touching a file."""

    commit_hash: str
    author_email: str
    author_ts: datetime | None
    old_path: str
    new_path: str
    added_spans: list[tuple[int, int]] = field(default_factory=list)
    deleted_spans: list[tuple[int, int]] = field(default_factory=list)
    lines_added: int = 0
    lines_deleted: int = 0


def _parse_hunk_header(header: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """
    Parse a unified diff hunk header.

    Parameters
    ----------
    header:
        Hunk header text starting with ``@@``.

    Returns
    -------
    tuple[tuple[int, int], tuple[int, int]] | None
        Old and new (start, length) ranges if parsing succeeds.
    """
    try:
        _, ranges, _ = header.split(" ", 2)
    except ValueError:
        return None
    old_part, new_part = ranges.split(" ")
    old_range = _parse_range(old_part.lstrip("-"))
    new_range = _parse_range(new_part.lstrip("+"))
    if old_range is None or new_range is None:
        return None
    return old_range, new_range


def _parse_range(raw: str) -> tuple[int, int] | None:
    if "," in raw:
        start_s, length_s = raw.split(",", 1)
    else:
        start_s, length_s = raw, "1"
    try:
        start = int(start_s)
        length = int(length_s)
    except ValueError:
        return None
    return start, length


def _parse_timestamp(raw: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _iter_git_log_lines(
    repo_root: Path,
    rel_path: str,
    *,
    max_history_days: int | None,
    default_branch: str,
    runner: ToolRunner | None,
) -> list[str] | None:
    since_arg: list[str] = []
    if max_history_days is not None:
        cutoff = datetime.now(tz=UTC) - timedelta(days=max_history_days)
        since_arg = [f"--since={cutoff.date().isoformat()}"]

    args = [
        "git",
        "log",
        default_branch,
        "--date=iso-strict",
        "--patch",
        "--unified=0",
        "--follow",
        "--no-renames",
        "--pretty=format:@@@%H\t%ae\t%ad",
        *since_arg,
        "--",
        rel_path,
    ]

    active_runner = runner or ToolRunner(cache_dir=repo_root / "build" / ".tool_cache")
    result = active_runner.run("git", args, cwd=repo_root)
    if result.returncode not in {0, 1}:
        log.warning(
            "git log failed for %s: code=%s stdout=%s stderr=%s",
            rel_path,
            result.returncode,
            result.stdout[:MAX_STDERR_CHARS],
            result.stderr[:MAX_STDERR_CHARS],
        )
        return None
    return result.stdout.splitlines()


def iter_file_history(
    repo_root: Path,
    rel_path: str,
    *,
    max_history_days: int | None,
    default_branch: str,
    runner: ToolRunner | None = None,
) -> Iterable[FileCommitDelta]:
    """
    Iterate structured git deltas for a file.

    Parameters
    ----------
    repo_root:
        Path to repository root for git invocations.
    rel_path:
        Repository-relative path to the file.
    max_history_days:
        Optional cutoff for git history. ``None`` scans the full history.
    default_branch:
        Branch or commit selector to anchor the log (e.g., ``HEAD`` or ``main``).
    runner:
        Optional shared ToolRunner for git invocations.
    """
    lines = _iter_git_log_lines(
        repo_root,
        rel_path,
        max_history_days=max_history_days,
        default_branch=default_branch,
        runner=runner,
    )
    if lines is None:
        return []

    return _parse_git_log(lines, rel_path)


def _parse_git_log(lines: list[str], rel_path: str) -> Iterator[FileCommitDelta]:
    current_commit: str | None = None
    current_author: str | None = None
    current_ts: datetime | None = None
    added_spans: list[tuple[int, int]] = []
    deleted_spans: list[tuple[int, int]] = []
    lines_added = 0
    lines_deleted = 0

    def flush() -> Iterator[FileCommitDelta]:
        nonlocal added_spans, deleted_spans, lines_added, lines_deleted
        if current_commit is None or current_author is None:
            return iter(())
        delta = FileCommitDelta(
            commit_hash=current_commit,
            author_email=current_author,
            author_ts=current_ts,
            old_path=rel_path,
            new_path=rel_path,
            added_spans=added_spans,
            deleted_spans=deleted_spans,
            lines_added=lines_added,
            lines_deleted=lines_deleted,
        )
        added_spans = []
        deleted_spans = []
        lines_added = 0
        lines_deleted = 0
        return iter((delta,))

    for line in lines:
        if not line:
            continue
        if line.startswith("@@@"):
            if current_commit is not None:
                yield from flush()
            _, meta = line.split("@@@", 1)
            parts = meta.split("\t")
            if len(parts) >= 3:
                current_commit, current_author, ts_raw = parts[:3]
                current_ts = _parse_timestamp(ts_raw)
            else:
                current_commit = parts[0] if parts else None
                current_author = parts[1] if len(parts) > 1 else None
                current_ts = None
            continue

        if line.startswith("@@ "):
            ranges = _parse_hunk_header(line)
            if ranges is None:
                continue
            (old_start, old_len), (new_start, new_len) = ranges
            if new_len > 0:
                added_spans.append((new_start, new_start + new_len - 1))
                lines_added += new_len
            if old_len > 0:
                deleted_spans.append((old_start, old_start + old_len - 1))
                lines_deleted += old_len
            continue

        if line.startswith("Binary files") or line.startswith("diff --git"):
            continue

    if current_commit is not None:
        yield from flush()
