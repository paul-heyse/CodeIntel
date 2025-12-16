"""Detect external dependency usage and populate analytics tables.

Column definitions and internal helper functions for dependency analysis.

The pure compute functions are available in ``codeintel.analytics.dependencies.compute``:
- ``compute_dependency_calls_pure`` returns ``DependencyCallsResult``
- ``compute_external_dependencies_pure`` returns ``ExternalDependenciesResult``

The Hamilton native module is at:
``codeintel.build.hamilton.native.analytics.dependencies``
"""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import yaml

from codeintel.analytics.compute.dependencies.classification import (
    CALLSITE_MEDIUM_THRESHOLD,
    SEVERITY_SCORES,
    DependencyModePattern,
    LibraryPattern,
)
from codeintel.analytics.compute.dependencies.detection import DependencyCall
from codeintel.analytics.compute.evidence.collection import EvidenceCollector
from codeintel.analytics.utilities.ast import resolve_call_target, safe_unparse, snippet_from_lines
from codeintel.core.paths import normalize_path

EXTERNAL_DEPENDENCY_CALLS_COLS = [
    "repo",
    "commit",
    "dep_id",
    "library",
    "service_name",
    "function_goid_h128",
    "function_urn",
    "rel_path",
    "module",
    "qualname",
    "callsite_count",
    "modes",
    "evidence_json",
    "created_at",
]
EXTERNAL_DEPENDENCIES_COLS = [
    "repo",
    "commit",
    "dep_id",
    "library",
    "service_name",
    "category",
    "language",
    "severity",
    "criticality",
    "risk_score",
    "function_count",
    "callsite_count",
    "modules_json",
    "usage_modes",
    "config_keys",
    "risk_level",
    "created_at",
]

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from codeintel.analytics.ast_features.model import FunctionAstFeatures
    from codeintel.analytics.parsing.ast_cache import FunctionAst
    from codeintel.config.primitives import SnapshotRef
    from codeintel.core.catalog import FunctionCatalogProvider
    from codeintel.storage.gateway import DuckDBConnection

log = logging.getLogger(__name__)


@dataclass
class DependencyAggregate:
    """Aggregated usage for a dependency."""

    library: str
    service_name: str | None
    category: str | None
    severity: str | None = None
    criticality: float | None = None
    risk_score: float | None = None
    modules: set[str] = field(default_factory=set)
    functions: set[int] = field(default_factory=set)
    callsite_count: int = 0
    modes: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class DependencyContext:
    """Shared context for dependency call classification."""

    repo: str
    commit: str
    alias_maps: dict[str, dict[str, str]]
    patterns: dict[str, LibraryPattern]
    module_map: dict[str, str]
    catalog: FunctionCatalogProvider
    now: datetime
    features: dict[int, FunctionAstFeatures]


class DependencyCallVisitor(ast.NodeVisitor):
    """Walk a function AST and collect dependency calls."""

    def __init__(
        self,
        alias_map: dict[str, str],
        patterns: dict[str, LibraryPattern],
        rel_path: str,
        lines: Sequence[str],
    ) -> None:
        self.alias_map = alias_map
        self.patterns = patterns
        self._rel_path = rel_path
        self._lines = lines
        self.calls: list[DependencyCall] = []

    def visit_Call(self, node: ast.Call) -> None:
        target = resolve_call_target(node.func, self.alias_map)
        library = target.library
        method = target.attribute or target.base
        if library is None or library not in self.patterns:
            self.generic_visit(node)
            return
        pattern = self.patterns[library]
        target_text = safe_unparse(node) or ""
        modes, matcher = _classify_modes(pattern, method, target_text)
        severity = (matcher.severity if matcher else None) or pattern.severity
        criticality = (matcher.criticality if matcher else None) or pattern.criticality
        risk_score = _risk_score(severity, criticality)
        matched_pattern = matcher.name if matcher is not None else method
        lineno = getattr(node, "lineno", None)
        end_lineno = getattr(node, "end_lineno", lineno)
        snippet = snippet_from_lines(self._lines, lineno, end_lineno)
        self.calls.append(
            DependencyCall(
                library=library,
                target=target_text or (method or ""),
                modes=modes,
                severity=severity,
                criticality=criticality,
                matched_pattern=matched_pattern,
                risk_score=risk_score,
                lineno=lineno,
                end_lineno=end_lineno,
                snippet=snippet,
            )
        )
        self.generic_visit(node)


@dataclass(frozen=True)
class ExternalDependencyInputs:
    """Inputs for external dependency call analysis."""

    catalog_provider: FunctionCatalogProvider
    module_map: dict[str, str]
    ast_by_goid: dict[int, FunctionAst]
    features_map: dict[int, FunctionAstFeatures]
    missing_goids: set[int] | None = None


def _function_call_rows(
    *,
    goid: int,
    func_ast: FunctionAst,
    context: DependencyContext,
) -> list[tuple[object, ...]]:
    feature_vector = context.features.get(goid)
    if feature_vector is not None and not (
        feature_vector.io_flags.uses_network
        or feature_vector.db_libs
        or feature_vector.http_client_libs
        or feature_vector.message_libs
    ):
        return []
    alias_map = context.alias_maps.get(func_ast.rel_path, {})
    visitor = DependencyCallVisitor(
        alias_map,
        context.patterns,
        func_ast.rel_path,
        func_ast.lines,
    )
    visitor.visit(func_ast.node)
    grouped = _group_calls(visitor.calls)
    if not grouped:
        return []

    module = context.module_map.get(func_ast.rel_path)
    if module is None:
        return []
    urn = context.catalog.urn_for_goid(goid) or ""
    rows: list[tuple[object, ...]] = []
    for library, calls in grouped.items():
        pattern = context.patterns[library]
        dep_id = _dep_id(context.repo, context.commit, library)
        modes = sorted({mode for call in calls for mode in call.modes})
        collector = EvidenceCollector()
        for call in calls:
            collector.add_sample(
                path=func_ast.rel_path,
                line_span=(call.lineno, call.end_lineno),
                snippet=call.snippet,
                details={
                    "target": call.target,
                    "modes": call.modes,
                    "matched_pattern": call.matched_pattern,
                    "severity": call.severity,
                    "criticality": call.criticality,
                },
                tags=(library,),
            )
        evidence = collector.to_dicts()
        rows.append(
            (
                context.repo,
                context.commit,
                dep_id,
                library,
                pattern.service_name or library,
                _decimal(goid),
                urn,
                func_ast.rel_path,
                module,
                func_ast.qualname,
                len(calls),
                modes,
                evidence,
                context.now,
            )
        )
    return rows


def _fetch_dependency_call_rows(
    con: DuckDBConnection, snapshot: SnapshotRef
) -> list[tuple[object, ...]]:
    return con.execute(
        """
        SELECT dep_id, library, function_goid_h128, module,
               callsite_count, modes, severity, criticality, risk_score
        FROM analytics.external_dependency_calls
        WHERE repo = ? AND commit = ?
        """,
        [snapshot.repo, snapshot.commit],
    ).fetchall()


def _aggregate_dependency_calls(
    rows: list[tuple[object, ...]], patterns: dict[str, LibraryPattern]
) -> dict[str, DependencyAggregate]:
    aggregates: dict[str, DependencyAggregate] = {}
    for (
        dep_id,
        library,
        function_goid,
        module,
        callsite_count,
        modes_obj,
        severity,
        criticality,
        risk_score,
    ) in rows:
        if library is None or dep_id is None:
            continue
        lib_key = str(library)
        pattern = patterns.get(lib_key)
        severity_value = _as_str(severity) or (pattern.severity if pattern else None)
        criticality_value = _as_float(criticality) if criticality is not None else None
        aggregate = aggregates.setdefault(
            str(dep_id),
            DependencyAggregate(
                library=lib_key,
                service_name=(pattern.service_name if pattern else None) or lib_key,
                category=pattern.category if pattern else None,
                severity=severity_value,
                criticality=(
                    criticality_value
                    if criticality_value is not None
                    else (pattern.criticality if pattern else None)
                ),
                risk_score=None,
            ),
        )
        if module:
            aggregate.modules.add(str(module))
        function_goid_value = _as_int(function_goid)
        if function_goid_value is not None:
            aggregate.functions.add(function_goid_value)
        aggregate.callsite_count += _as_int(callsite_count) or 0
        aggregate.modes.update(_ensure_str_list(modes_obj))
        agg_score = (
            _as_float(risk_score)
            if risk_score is not None
            else _risk_score(severity_value, criticality_value)
        )
        if agg_score is not None:
            prev_score = aggregate.risk_score or 0.0
            if agg_score > prev_score:
                aggregate.risk_score = agg_score
        if severity_value and aggregate.severity is None:
            aggregate.severity = severity_value
        if criticality_value is not None and aggregate.criticality is None:
            aggregate.criticality = criticality_value
    return aggregates


def _serialize_dependency_rows(
    aggregates: dict[str, DependencyAggregate],
    config_keys_by_module: dict[str, set[str]],
    snapshot: SnapshotRef,
    *,
    language: str = "python",
) -> list[tuple[object, ...]]:
    dep_rows: list[tuple[object, ...]] = []
    now = datetime.now(tz=UTC)
    for dep_id, aggregate in aggregates.items():
        config_keys: set[str] = set()
        for module in aggregate.modules:
            config_keys.update(config_keys_by_module.get(module, set()))
        risk_level = aggregate.severity or _risk_level(aggregate.modes, aggregate.callsite_count)
        dep_rows.append(
            (
                snapshot.repo,
                snapshot.commit,
                dep_id,
                aggregate.library,
                aggregate.service_name,
                aggregate.category,
                language,
                aggregate.severity,
                aggregate.criticality,
                aggregate.risk_score,
                len(aggregate.functions),
                aggregate.callsite_count,
                sorted(aggregate.modules),
                sorted(aggregate.modes),
                sorted(config_keys) if config_keys else None,
                risk_level,
                now,
            )
        )
    return dep_rows


def _as_int(value: object | None) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, (float, Decimal)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_float(value: object | None) -> float | None:
    if isinstance(value, (int, float, Decimal)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _as_str(value: object | None) -> str | None:
    return str(value) if isinstance(value, str) else None


def _load_dependency_patterns(
    repo_root: Path, dependency_patterns_path: Path | None
) -> dict[str, LibraryPattern]:
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


def _build_alias_maps(repo_root: Path, module_map: dict[str, str]) -> dict[str, dict[str, str]]:
    alias_maps: dict[str, dict[str, str]] = {}
    for rel_path in module_map:
        abs_path = (repo_root / rel_path).resolve()
        try:
            source = abs_path.read_text(encoding="utf8")
        except (FileNotFoundError, UnicodeDecodeError):
            continue
        try:
            tree = ast.parse(source, filename=str(abs_path))
        except SyntaxError:
            continue
        alias_maps[normalize_path(rel_path)] = _build_alias_map(tree)
    return alias_maps


def _build_alias_map(tree: ast.AST) -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                alias_map[alias.asname or alias.name] = root
        elif isinstance(node, ast.ImportFrom):
            if node.module is None:
                continue
            root = node.module.split(".")[0]
            for alias in node.names:
                alias_map[alias.asname or alias.name] = root
    return alias_map


def _group_calls(calls: list[DependencyCall]) -> dict[str, list[DependencyCall]]:
    grouped: dict[str, list[DependencyCall]] = defaultdict(list)
    for call in calls:
        grouped[call.library].append(call)
    return grouped


def _classify_modes(
    pattern: LibraryPattern, method: str | None, target: str
) -> tuple[list[str], DependencyModePattern | None]:
    modes: set[str] = set()
    for matcher in pattern.matchers:
        if matcher.method and method == matcher.method:
            modes.update(matcher.modes)
            return sorted(modes), matcher
        if matcher.method_prefix and target.startswith(str(matcher.method_prefix)):
            modes.update(matcher.modes)
            return sorted(modes), matcher
        if matcher.match and matcher.match in target:
            modes.update(matcher.modes)
            return sorted(modes), matcher
    return (["unknown"], None)


def _load_config_keys(con: DuckDBConnection, repo: str, commit: str) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    rows = con.execute(
        """
        SELECT reference_modules, key
        FROM analytics.config_values
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    for ref_modules, key in rows:
        if key is None or ref_modules is None:
            continue
        modules = _ensure_str_list(ref_modules)
        for module in modules:
            mapping[module].add(str(key))
    return mapping


def load_config_key_map(con: DuckDBConnection, repo: str, commit: str) -> dict[str, set[str]]:
    """
    Load config keys keyed by module for a repo snapshot.

    Returns
    -------
    dict[str, set[str]]
        Mapping of module name to referenced config keys.
    """
    return _load_config_keys(con, repo, commit)


def _risk_level(modes: set[str], callsite_count: int) -> str:
    if "admin" in modes or "write" in modes:
        return "high"
    if callsite_count > CALLSITE_MEDIUM_THRESHOLD or "read" in modes:
        return "medium"
    return "low"


def _severity_score(severity: str | None) -> float | None:
    if severity is None:
        return None
    return SEVERITY_SCORES.get(severity.lower())


def _risk_score(severity: str | None, criticality: float | None) -> float | None:
    base = _severity_score(severity)
    if base is None:
        return None
    multiplier = criticality if criticality is not None else 1.0
    return base * float(multiplier)


def _ensure_str_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            return [value]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    return []


def _dep_id(repo: str, commit: str, library: str) -> str:
    raw = f"{repo}:{commit}:{library}"
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _decimal(value: int) -> Decimal:
    return Decimal(value)


__all__ = [
    "load_config_key_map",
]
