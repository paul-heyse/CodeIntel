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
META_PARTS_REQUIRED = 3
_MIN_HUNK_PARTS = 2


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


@dataclass
class GitLogState:
    """Mutable state while iterating git log output."""

    current_commit: str | None = None
    current_author: str | None = None
    current_ts: datetime | None = None
    added_spans: list[tuple[int, int]] = field(default_factory=list)
    deleted_spans: list[tuple[int, int]] = field(default_factory=list)
    lines_added: int = 0
    lines_deleted: int = 0

    def reset_spans(self) -> None:
        """Reset accumulated span and line deltas."""
        self.added_spans = []
        self.deleted_spans = []
        self.lines_added = 0
        self.lines_deleted = 0

    def to_delta(self, rel_path: str) -> FileCommitDelta | None:
        """
        Convert the current state into a FileCommitDelta.

        Returns
        -------
        FileCommitDelta | None
            Delta for the active commit when metadata is available; otherwise ``None``.
        """
        if self.current_commit is None or self.current_author is None:
            return None
        delta = FileCommitDelta(
            commit_hash=self.current_commit,
            author_email=self.current_author,
            author_ts=self.current_ts,
            old_path=rel_path,
            new_path=rel_path,
            added_spans=self.added_spans,
            deleted_spans=self.deleted_spans,
            lines_added=self.lines_added,
            lines_deleted=self.lines_deleted,
        )
        self.reset_spans()
        return delta


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
    if not header.startswith("@@"):
        return None
    body = header.strip().removeprefix("@@").removesuffix("@@").strip()
    parts = body.split()
    if len(parts) < _MIN_HUNK_PARTS:
        return None
    old_range = _parse_range(parts[0].lstrip("-"))
    new_range = _parse_range(parts[1].lstrip("+"))
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

    Returns
    -------
    Iterable[FileCommitDelta]
        Parsed git deltas for the target file.
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
    state = GitLogState()

    for line in lines:
        commit_delta = _commit_delta_from_line(line, state, rel_path)
        if commit_delta is not None:
            yield commit_delta
            continue
        if _update_from_hunk(line, state):
            continue
        if _should_skip_diff_line(line):
            continue

    delta = state.to_delta(rel_path)
    if delta is not None:
        yield delta


def _commit_delta_from_line(line: str, state: GitLogState, rel_path: str) -> FileCommitDelta | None:
    if not line or not line.startswith("@@@"):
        return None
    return _start_commit(line, state, rel_path)


def _update_from_hunk(line: str, state: GitLogState) -> bool:
    if not line or not line.startswith("@@ "):
        return False
    ranges = _parse_hunk_header(line)
    if ranges is None:
        return True
    (old_start, old_len), (new_start, new_len) = ranges
    if new_len > 0:
        state.added_spans.append((new_start, new_start + new_len - 1))
        state.lines_added += new_len
    if old_len > 0:
        state.deleted_spans.append((old_start, old_start + old_len - 1))
        state.lines_deleted += old_len
    return True


def _should_skip_diff_line(line: str) -> bool:
    return not line or line.startswith(("Binary files", "diff --git"))


def _start_commit(line: str, state: GitLogState, rel_path: str) -> FileCommitDelta | None:
    delta = state.to_delta(rel_path)
    _, meta = line.split("@@@", 1)
    parts = meta.split("\t")
    if len(parts) >= META_PARTS_REQUIRED:
        state.current_commit, state.current_author, ts_raw = parts[:3]
        state.current_ts = _parse_timestamp(ts_raw)
    else:
        state.current_commit = parts[0] if parts else None
        state.current_author = parts[1] if len(parts) > 1 else None
        state.current_ts = None
    return delta
