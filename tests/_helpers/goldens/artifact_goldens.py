"""Golden helpers for JSON artifacts."""

from __future__ import annotations

import difflib
import json
import os
from pathlib import Path

type JSONPrimitive = str | int | float | bool | None
type JSONValue = JSONPrimitive | list[JSONValue] | dict[str, JSONValue]


def load_json_artifact(path: Path) -> JSONValue:
    """Load a JSON artifact from disk.

    Parameters
    ----------
    path
        Path to the JSON artifact.

    Returns
    -------
    JSONValue
        Parsed JSON payload.
    """
    return json.loads(path.read_text(encoding="utf-8"))


def assert_json_artifact_matches_golden(
    *,
    actual_path: Path,
    golden_path: Path,
    update_mode: bool | None = None,
) -> JSONValue:
    """Assert a JSON artifact matches a golden file.

    Parameters
    ----------
    actual_path
        Path to the JSON artifact to compare.
    golden_path
        Expected golden JSON file path.
    update_mode
        When True, overwrite golden files with current output.

    Returns
    -------
    JSONValue
        Parsed JSON payload for the actual artifact.

    Raises
    ------
    AssertionError
        If the golden file is missing or differs from current output.
    """
    payload = load_json_artifact(actual_path)
    actual = _format_json(payload)
    should_update = update_mode if update_mode is not None else _update_enabled()

    if should_update:
        golden_path.parent.mkdir(parents=True, exist_ok=True)
        golden_path.write_text(actual, encoding="utf-8")
        return payload

    if not golden_path.exists():
        message = (
            f"Golden file not found: {golden_path}\n"
            "Run with UPDATE_GOLDEN=1 or --update-golden to create it.\n"
            f"Actual output:\n{actual}"
        )
        raise AssertionError(message)

    expected = golden_path.read_text(encoding="utf-8")
    if actual != expected:
        diff = _format_diff(expected, actual)
        message = f"Artifact output differs from golden: {golden_path}\n{diff}"
        raise AssertionError(message)
    return payload


def _format_json(payload: JSONValue) -> str:
    return json.dumps(payload, indent=2, sort_keys=True).strip() + "\n"


def _format_diff(expected: str, actual: str) -> str:
    diff = difflib.unified_diff(
        expected.splitlines(),
        actual.splitlines(),
        fromfile="expected",
        tofile="actual",
        lineterm="",
    )
    return "\n".join(diff)


def _update_enabled() -> bool:
    return os.environ.get("UPDATE_GOLDEN", "").lower() in {"1", "true"}


__all__ = ["assert_json_artifact_matches_golden", "load_json_artifact"]
