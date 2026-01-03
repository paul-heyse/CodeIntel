"""IO helpers for dependency analysis orchestration."""

from __future__ import annotations

import ast
import json
import logging
from typing import TYPE_CHECKING

import yaml

from codeintel.build.analytics.compute.dependencies.classification import (
    DependencyModePattern,
    LibraryPattern,
)
from codeintel.build.analytics.compute.dependencies.compute import load_config_key_map
from codeintel.build.analytics.compute.dependencies.detection import build_alias_map

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)


def load_dependency_patterns(
    repo_root: Path,
    dependency_patterns_path: Path | None,
) -> dict[str, LibraryPattern]:
    """Load dependency patterns from the repository configuration.

    Returns
    -------
    dict[str, LibraryPattern]
        Mapping of library names to dependency patterns.
    """
    path = dependency_patterns_path
    if path is None:
        path = repo_root / "config" / "dependency_patterns.yml"
    if not path.is_file():
        log.warning("Dependency patterns file not found at %s", path)
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf8")) or {}
    except yaml.YAMLError as exc:
        log.warning("Failed to parse dependency patterns at %s: %s", path, exc)
        return {}

    libs = raw.get("libs", {}) if isinstance(raw, dict) else {}
    patterns: dict[str, LibraryPattern] = {}
    for library, payload in libs.items():
        if not isinstance(payload, dict):
            continue
        matchers: list[DependencyModePattern] = []
        for entry in payload.get("patterns", []) or []:
            matcher = _pattern_from_entry(entry)
            if matcher is not None:
                matchers.append(matcher)
        patterns[str(library)] = LibraryPattern(
            library=str(library),
            service_name=payload.get("service_name"),
            category=payload.get("category"),
            matchers=matchers,
        )
    return patterns


def _pattern_from_entry(entry: object) -> DependencyModePattern | None:
    if not isinstance(entry, dict):
        return None
    modes = entry.get("mode") or entry.get("modes")
    mode_list = _ensure_str_list(modes)
    if not mode_list:
        return None
    return DependencyModePattern(
        modes=mode_list,
        method=entry.get("method"),
        method_prefix=entry.get("method_prefix"),
        match=entry.get("match"),
    )


def _ensure_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return [value]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
        return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def build_alias_maps(
    repo_root: Path,
    module_map: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Build import alias maps for each module by parsing source files.

    Returns
    -------
    dict[str, dict[str, str]]
        Mapping of relative paths to their alias maps.
    """
    alias_maps: dict[str, dict[str, str]] = {}
    for rel_path in module_map:
        abs_path = (repo_root / rel_path).resolve()
        try:
            source = abs_path.read_text(encoding="utf8")
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(source, filename=str(abs_path))
        except (SyntaxError, ValueError):
            continue
        alias_maps[rel_path] = build_alias_map(tree)
    return alias_maps


__all__ = [
    "build_alias_maps",
    "load_config_key_map",
    "load_dependency_patterns",
]
