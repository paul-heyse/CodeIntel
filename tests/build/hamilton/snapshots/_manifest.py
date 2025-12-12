"""Typed manifest loader for CLI snapshot tests.

Provides structured loading of snapshot test manifests from JSON or YAML
files with full type validation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from tests.build.hamilton.snapshots._snapshot import TextReplace

if TYPE_CHECKING:
    from collections.abc import Mapping

SnapshotKind = Literal["json", "text"]
OutputSelect = Literal["stdout", "stderr", "both"]


@dataclass(frozen=True)
class SnapshotDefaults:
    """Default values for snapshot test cases.

    Attributes
    ----------
    kind
        Default output kind (json or text).
    output
        Default output stream selection.
    exit_code
        Default expected exit code.
    env
        Default environment variable overrides.
    """

    kind: SnapshotKind = "json"
    output: OutputSelect = "stdout"
    exit_code: int = 0
    env: Mapping[str, str] | None = None


@dataclass(frozen=True)
class SnapshotCase:
    """Individual snapshot test case definition.

    Attributes
    ----------
    name
        Unique case identifier used for test naming.
    args
        CLI arguments to pass (excluding program name).
    kind
        Output format: "json" or "text".
    output
        Output stream: "stdout", "stderr", or "both".
    exit_code
        Expected exit code (typically 0).
    env
        Environment variable overrides for this case.
    snapshot
        Snapshot filename (defaults to name + appropriate extension).
    strip_keys
        Additional JSON keys to strip beyond defaults.
    replace
        Text replacement patterns for normalization.
    tags
        Tags for filtering (e.g., ["pr14", "graph"]).
    """

    name: str
    args: tuple[str, ...]
    kind: SnapshotKind
    output: OutputSelect
    exit_code: int
    env: Mapping[str, str] | None
    snapshot: str
    strip_keys: tuple[str, ...]
    replace: tuple[TextReplace, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True)
class SnapshotManifest:
    """Complete snapshot test manifest.

    Attributes
    ----------
    app_import
        Import path for the CLI app (e.g., "codeintel.cli.app:app").
    defaults
        Default values for case fields.
    cases
        Tuple of test case definitions.
    """

    app_import: str
    defaults: SnapshotDefaults
    cases: tuple[SnapshotCase, ...]


def _get_str(d: Mapping[str, Any], key: str, *, default: str | None = None) -> str:
    """Extract string value from mapping with optional default.

    Parameters
    ----------
    d
        Mapping to extract from.
    key
        Key to look up.
    default
        Default value if key is missing.

    Returns
    -------
    str
        Extracted string value.

    Raises
    ------
    TypeError
        If value is not a string.
    KeyError
        If key is missing and no default provided.
    """
    if default is not None:
        v = d.get(key, default)
    else:
        v = d[key]
    if not isinstance(v, str):
        msg = f"Expected string for '{key}', got {type(v).__name__}"
        raise TypeError(msg)
    return v


def _get_int(d: Mapping[str, Any], key: str, *, default: int) -> int:
    """Extract integer value from mapping.

    Parameters
    ----------
    d
        Mapping to extract from.
    key
        Key to look up.
    default
        Default value if key is missing.

    Returns
    -------
    int
        Extracted integer value.

    Raises
    ------
    TypeError
        If value is not an integer.
    """
    v = d.get(key, default)
    if not isinstance(v, int):
        msg = f"Expected int for '{key}', got {type(v).__name__}"
        raise TypeError(msg)
    return v


def _get_kind(d: Mapping[str, Any], key: str, *, default: SnapshotKind) -> SnapshotKind:
    """Extract snapshot kind from mapping.

    Parameters
    ----------
    d
        Mapping to extract from.
    key
        Key to look up.
    default
        Default kind value.

    Returns
    -------
    SnapshotKind
        "json" or "text".

    Raises
    ------
    ValueError
        If value is not a valid kind.
    """
    v = d.get(key, default)
    if v not in ("json", "text"):
        msg = f"Invalid kind: {v!r}"
        raise ValueError(msg)
    return v  # type: ignore[return-value]


def _get_output(
    d: Mapping[str, Any], key: str, *, default: OutputSelect
) -> OutputSelect:
    """Extract output selection from mapping.

    Parameters
    ----------
    d
        Mapping to extract from.
    key
        Key to look up.
    default
        Default output selection.

    Returns
    -------
    OutputSelect
        "stdout", "stderr", or "both".

    Raises
    ------
    ValueError
        If value is not a valid output selection.
    """
    v = d.get(key, default)
    if v not in ("stdout", "stderr", "both"):
        msg = f"Invalid output: {v!r}"
        raise ValueError(msg)
    return v  # type: ignore[return-value]


def _get_env(d: Mapping[str, Any], key: str) -> Mapping[str, str] | None:
    """Extract environment mapping from dictionary.

    Parameters
    ----------
    d
        Mapping to extract from.
    key
        Key to look up.

    Returns
    -------
    Mapping[str, str] | None
        Environment mapping or None.

    Raises
    ------
    TypeError
        If value is not a valid env mapping.
    """
    v = d.get(key)
    if v is None:
        return None
    if not isinstance(v, dict):
        msg = f"Expected dict[str,str] for '{key}', got {type(v).__name__}"
        raise TypeError(msg)
    for k, val in v.items():
        if not isinstance(k, str) or not isinstance(val, str):
            msg = f"Expected dict[str,str] for '{key}'"
            raise TypeError(msg)
    return v


def _get_args(d: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract CLI arguments list from mapping.

    Parameters
    ----------
    d
        Mapping to extract from.

    Returns
    -------
    tuple[str, ...]
        CLI arguments as tuple.

    Raises
    ------
    TypeError
        If value is not a list of strings.
    """
    v = d.get("args")
    if not isinstance(v, list):
        msg = "Expected list[str] for 'args'"
        raise TypeError(msg)
    for item in v:
        if not isinstance(item, str):
            msg = "Expected list[str] for 'args'"
            raise TypeError(msg)
    return tuple(v)


def _get_replace(d: Mapping[str, Any]) -> tuple[TextReplace, ...]:
    """Extract text replacement patterns from mapping.

    Parameters
    ----------
    d
        Mapping to extract from.

    Returns
    -------
    tuple[TextReplace, ...]
        Text replacement patterns.

    Raises
    ------
    TypeError
        If value is not a valid replace list.
    """
    raw = d.get("replace") or []
    if not isinstance(raw, list):
        msg = "Expected list for 'replace'"
        raise TypeError(msg)
    out: list[TextReplace] = []
    for item in raw:
        if not isinstance(item, dict):
            msg = "Expected dict items in 'replace'"
            raise TypeError(msg)
        out.append(
            TextReplace(
                pattern=_get_str(item, "pattern"),
                repl=_get_str(item, "repl"),
            )
        )
    return tuple(out)


def _get_strip_keys(d: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract strip_keys list from mapping.

    Parameters
    ----------
    d
        Mapping to extract from.

    Returns
    -------
    tuple[str, ...]
        Keys to strip from JSON output.

    Raises
    ------
    TypeError
        If value is not a list of strings.
    """
    raw = d.get("strip_keys") or []
    if not isinstance(raw, list):
        msg = "Expected list[str] for 'strip_keys'"
        raise TypeError(msg)
    for item in raw:
        if not isinstance(item, str):
            msg = "Expected list[str] for 'strip_keys'"
            raise TypeError(msg)
    return tuple(raw)


def _get_tags(d: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract tags list from mapping.

    Parameters
    ----------
    d
        Mapping to extract from.

    Returns
    -------
    tuple[str, ...]
        Tags for filtering.

    Raises
    ------
    TypeError
        If value is not a list of strings.
    """
    raw = d.get("tags") or []
    if not isinstance(raw, list):
        msg = "Expected list[str] for 'tags'"
        raise TypeError(msg)
    for item in raw:
        if not isinstance(item, str):
            msg = "Expected list[str] for 'tags'"
            raise TypeError(msg)
    return tuple(raw)


def _load_manifest_data(path: Path) -> dict[str, Any]:
    """Load manifest data from JSON or YAML file.

    Parameters
    ----------
    path
        Path to manifest file.

    Returns
    -------
    dict[str, Any]
        Parsed manifest data.

    Raises
    ------
    ValueError
        If file extension is not supported.
    RuntimeError
        If YAML is requested but pyyaml is not installed.
    TypeError
        If root is not a dictionary.
    """
    suffix = path.suffix.lower()
    text = path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:
            msg = (
                "PyYAML is required to load YAML manifests. "
                "Install with: pip install pyyaml"
            )
            raise RuntimeError(msg) from e
        data = yaml.safe_load(text)
    else:
        msg = f"Unsupported manifest extension: {suffix} (use .json, .yaml, or .yml)"
        raise ValueError(msg)

    if not isinstance(data, dict):
        msg = "Manifest root must be an object"
        raise TypeError(msg)
    return data


def load_snapshot_manifest(path: Path) -> SnapshotManifest:
    """Load and validate a snapshot test manifest.

    Supports both JSON and YAML formats based on file extension.

    Parameters
    ----------
    path
        Path to manifest file.

    Returns
    -------
    SnapshotManifest
        Parsed and validated manifest.

    Raises
    ------
    TypeError
        If manifest structure is invalid.
    ValueError
        If values are out of range.
    KeyError
        If required fields are missing.
    """
    data = _load_manifest_data(path)

    app_import = _get_str(data, "app_import")
    defaults_raw = data.get("defaults") or {}
    if not isinstance(defaults_raw, dict):
        msg = "'defaults' must be an object"
        raise TypeError(msg)

    defaults = SnapshotDefaults(
        kind=_get_kind(defaults_raw, "kind", default="json"),
        output=_get_output(defaults_raw, "output", default="stdout"),
        exit_code=_get_int(defaults_raw, "exit_code", default=0),
        env=_get_env(defaults_raw, "env"),
    )

    cases_raw = data.get("cases")
    if not isinstance(cases_raw, list):
        msg = "'cases' must be a list"
        raise TypeError(msg)

    cases: list[SnapshotCase] = []
    for c in cases_raw:
        if not isinstance(c, dict):
            msg = "Each case must be an object"
            raise TypeError(msg)

        name = _get_str(c, "name")
        args = _get_args(c)
        kind = _get_kind(c, "kind", default=defaults.kind)
        output = _get_output(c, "output", default=defaults.output)
        exit_code = _get_int(c, "exit_code", default=defaults.exit_code)

        # Merge env (defaults overridden by case)
        env_default = dict(defaults.env or {})
        env_case = dict(_get_env(c, "env") or {})
        env = {**env_default, **env_case} if (env_default or env_case) else None

        # Snapshot path inference
        snapshot = c.get("snapshot")
        if snapshot is None:
            snapshot = f"{name}.json" if kind == "json" else f"{name}.txt"
        if not isinstance(snapshot, str):
            msg = "Expected string for 'snapshot'"
            raise TypeError(msg)

        cases.append(
            SnapshotCase(
                name=name,
                args=args,
                kind=kind,
                output=output,
                exit_code=exit_code,
                env=env,
                snapshot=snapshot,
                strip_keys=_get_strip_keys(c),
                replace=_get_replace(c),
                tags=_get_tags(c),
            )
        )

    return SnapshotManifest(app_import=app_import, defaults=defaults, cases=tuple(cases))


__all__ = [
    "OutputSelect",
    "SnapshotCase",
    "SnapshotDefaults",
    "SnapshotKind",
    "SnapshotManifest",
    "load_snapshot_manifest",
]

