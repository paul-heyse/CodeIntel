"""Snapshot helpers for Hamilton Phase 2 CLI tests.

Provides JSON and text normalization utilities for testing CLI outputs
against golden reference files. Dynamic fields like timestamps and durations
are removed to enable deterministic snapshot comparisons.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

# Fields that vary between runs and should be removed for comparison
DEFAULT_DYNAMIC_KEYS: frozenset[str] = frozenset({
    "run_id",
    "duration_ms",
    "duration_seconds",
    "started_at",
    "completed_at",
    "recorded_at",
    "computed_at",
    "timestamp",
    "total_duration_ms",
    "now",
})


@dataclass(frozen=True)
class TextReplace:
    """Regex-based text replacement for snapshot normalization.

    Use this to normalize paths, IDs, timestamps, or other dynamic
    content in text-based snapshots (DOT, Mermaid, etc.).

    Attributes
    ----------
    pattern
        Regular expression pattern to match.
    repl
        Replacement string (may contain backreferences).

    Examples
    --------
    >>> tr = TextReplace(pattern=r"/tmp/[^\\s]+", repl="<TMP>")
    >>> tr.pattern
    '/tmp/[^\\\\s]+'
    """

    pattern: str
    repl: str


def normalize_json_obj(obj: object, *, strip_keys: frozenset[str]) -> object:
    """Remove dynamic fields from an object for snapshot comparison.

    Recursively processes dicts and lists, removing any keys that are
    in strip_keys to enable deterministic comparisons.

    Parameters
    ----------
    obj
        Object to normalize (dict, list, or scalar).
    strip_keys
        Set of keys to remove from dictionaries.

    Returns
    -------
    object
        Normalized object with dynamic fields removed.

    Examples
    --------
    >>> normalize_json_obj({"target": "modules", "duration_ms": 123.4}, strip_keys=frozenset({"duration_ms"}))
    {'target': 'modules'}
    """
    if isinstance(obj, dict):
        return {
            k: normalize_json_obj(v, strip_keys=strip_keys)
            for k, v in obj.items()
            if k not in strip_keys
        }
    if isinstance(obj, list):
        return [normalize_json_obj(x, strip_keys=strip_keys) for x in obj]
    return obj


def normalize_text(text: str, *, replaces: Iterable[TextReplace]) -> str:
    """Normalize text output for snapshot comparison.

    Standardizes line endings, trims trailing whitespace, and applies
    optional regex replacements for dynamic content.

    Parameters
    ----------
    text
        Raw text output from CLI command.
    replaces
        Iterable of TextReplace objects for regex substitution.

    Returns
    -------
    str
        Normalized text ending with a single newline.

    Examples
    --------
    >>> normalize_text("line1  \\r\\nline2\\r", replaces=[])
    'line1\\nline2\\n'
    """
    # Normalize line endings
    t = text.replace("\r\n", "\n").replace("\r", "\n")

    # Trim trailing whitespace per line
    t = "\n".join(line.rstrip() for line in t.split("\n"))

    # Strip leading/trailing blank lines and ensure trailing newline
    t = t.strip() + "\n"

    # Apply regex replacements
    for r in replaces:
        t = re.sub(r.pattern, r.repl, t)

    return t


def load_json(text: str) -> object:
    """Parse JSON text to Python object.

    Parameters
    ----------
    text
        JSON string to parse.

    Returns
    -------
    object
        Parsed Python object.

    Raises
    ------
    json.JSONDecodeError
        If text is not valid JSON.
    """
    return json.loads(text)


def dump_json(obj: object) -> str:
    """Serialize Python object to formatted JSON string.

    Produces deterministic output with sorted keys and consistent
    indentation for reliable snapshot comparison.

    Parameters
    ----------
    obj
        Python object to serialize.

    Returns
    -------
    str
        Formatted JSON string ending with newline.
    """
    return json.dumps(obj, indent=2, sort_keys=True) + "\n"


def assert_or_update_snapshot(
    *,
    actual: str,
    snapshot_path: Path,
    update: bool,
) -> None:
    """Assert content matches snapshot or update the snapshot file.

    When update is True, writes actual content to snapshot_path.
    When update is False, compares actual to existing snapshot.

    Parameters
    ----------
    actual
        Actual content from CLI output (normalized).
    snapshot_path
        Path to snapshot file.
    update
        If True, update snapshot instead of comparing.

    Raises
    ------
    AssertionError
        If update is False and actual doesn't match expected.
    FileNotFoundError
        If update is False and snapshot file doesn't exist.
    """
    if update:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        snapshot_path.write_text(actual, encoding="utf-8")
        return

    if not snapshot_path.exists():
        msg = (
            f"Snapshot file not found: {snapshot_path}\n"
            f"Run with --update-cli-snapshots to create it."
        )
        raise FileNotFoundError(msg)

    expected = snapshot_path.read_text(encoding="utf-8")
    if actual != expected:
        # Create diff-friendly error message
        msg = (
            f"Snapshot mismatch: {snapshot_path}\n"
            f"--- Expected ---\n{expected}\n"
            f"--- Actual ---\n{actual}\n"
            f"Run with --update-cli-snapshots to update."
        )
        raise AssertionError(msg)


def normalize_and_format_json(
    text: str,
    *,
    strip_keys: frozenset[str] | None = None,
) -> str:
    """Parse, normalize, and re-serialize JSON for snapshot comparison.

    Parameters
    ----------
    text
        JSON string from CLI output.
    strip_keys
        Keys to remove (defaults to DEFAULT_DYNAMIC_KEYS).

    Returns
    -------
    str
        Normalized, formatted JSON string.
    """
    keys = strip_keys if strip_keys is not None else DEFAULT_DYNAMIC_KEYS
    data = load_json(text)
    normalized = normalize_json_obj(data, strip_keys=keys)
    return dump_json(normalized)


__all__ = [
    "DEFAULT_DYNAMIC_KEYS",
    "TextReplace",
    "assert_or_update_snapshot",
    "dump_json",
    "load_json",
    "normalize_and_format_json",
    "normalize_json_obj",
    "normalize_text",
]

