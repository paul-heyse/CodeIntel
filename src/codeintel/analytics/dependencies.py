"""Detect external dependency usage and populate analytics tables."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import duckdb
import yaml

from codeintel.analytics.function_ast_cache import FunctionAst, load_function_asts
from codeintel.config.models import ExternalDependenciesConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.function_catalog_service import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.ingestion.common import load_module_map
from codeintel.storage.gateway import StorageGateway
from codeintel.utils.paths import normalize_rel_path

log = logging.getLogger(__name__)

CALLSITE_MEDIUM_THRESHOLD = 10
SEVERITY_SCORES = {
    "critical": 4.0,
    "high": 3.0,
    "medium": 2.0,
    "low": 1.0,
    "info": 0.5,
}


@dataclass(frozen=True)
class DependencyModePattern:
    """Classification rule for a dependency call."""

    modes: list[str]
    method: str | None = None
    method_prefix: str | None = None
    match: str | None = None
    severity: str | None = None
    criticality: float | None = None
    name: str | None = None


@dataclass(frozen=True)
class LibraryPattern:
    """Pattern bundle for a specific library."""

    library: str
    service_name: str | None
    category: str | None
    matchers: list[DependencyModePattern]
    severity: str | None = None
    criticality: float | None = None
    language: str = "python"


@dataclass(frozen=True)
class DependencyCall:
    """A single call into an external library."""

    library: str
    target: str
    modes: list[str]
    severity: str | None
    criticality: float | None
    matched_pattern: str | None = None
    risk_score: float | None = None


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


class DependencyCallVisitor(ast.NodeVisitor):
    """Walk a function AST and collect dependency calls."""

    def __init__(
        self,
        alias_map: dict[str, str],
        patterns: dict[str, LibraryPattern],
    ) -> None:
        self.alias_map = alias_map
        self.patterns = patterns
        self.calls: list[DependencyCall] = []

    def visit_Call(self, node: ast.Call) -> None:
        library, method = _resolve_library(node.func, self.alias_map)
        if library is None or library not in self.patterns:
            self.generic_visit(node)
            return
        pattern = self.patterns[library]
        target = _safe_unparse(node)
        modes, matcher = _classify_modes(pattern, method, target)
        severity = (matcher.severity if matcher else None) or pattern.severity
        criticality = (matcher.criticality if matcher else None) or pattern.criticality
        risk_score = _risk_score(severity, criticality)
        matched_pattern = matcher.name if matcher is not None else method
        self.calls.append(
            DependencyCall(
                library=library,
                target=target,
                modes=modes,
                severity=severity,
                criticality=criticality,
                matched_pattern=matched_pattern,
                risk_score=risk_score,
            )
        )
        self.generic_visit(node)


def build_external_dependency_calls(
    gateway: StorageGateway,
    cfg: ExternalDependenciesConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> None:
    """
    Populate analytics.external_dependency_calls from AST traversal.

    Parameters
    ----------
    gateway
        Storage gateway with live DuckDB connection.
    cfg
        External dependency configuration (repo context, patterns).
    catalog_provider
        Optional function catalog to reuse across steps.
    """
    patterns = _load_dependency_patterns(cfg)
    if not patterns:
        log.warning("No dependency patterns loaded; skipping dependency call analysis")
        return

    con = gateway.con
    ensure_schema(con, "analytics.external_dependency_calls")
    con.execute(
        "DELETE FROM analytics.external_dependency_calls WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    catalog = catalog_provider or FunctionCatalogService.from_db(
        gateway, repo=cfg.repo, commit=cfg.commit
    )
    module_map = load_module_map(gateway, cfg.repo, cfg.commit)
    ast_by_goid, missing = load_function_asts(
        gateway,
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        catalog_provider=catalog,
    )
    if missing:
        log.debug(
            "Skipping %d functions without AST spans during dependency analysis", len(missing)
        )
    alias_maps = _build_alias_maps(cfg.repo_root, module_map)
    now = datetime.now(tz=UTC)
    context = DependencyContext(
        repo=cfg.repo,
        commit=cfg.commit,
        alias_maps=alias_maps,
        patterns=patterns,
        module_map=module_map,
        catalog=catalog,
        now=now,
    )

    rows: list[tuple[object, ...]] = []

    for goid, func_ast in ast_by_goid.items():
        rows.extend(
            _function_call_rows(
                goid=goid,
                func_ast=func_ast,
                context=context,
            )
        )

    if rows:
        con.executemany(
            """
            INSERT INTO analytics.external_dependency_calls (
                repo, commit, dep_id, library, service_name,
                function_goid_h128, function_urn, rel_path, module, qualname,
                callsite_count, modes, evidence_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    log.info(
        "external_dependency_calls populated: %d rows for %s@%s",
        len(rows),
        cfg.repo,
        cfg.commit,
    )


def _function_call_rows(
    *,
    goid: int,
    func_ast: FunctionAst,
    context: DependencyContext,
) -> list[tuple[object, ...]]:
    alias_map = context.alias_maps.get(func_ast.rel_path, {})
    visitor = DependencyCallVisitor(alias_map, context.patterns)
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
        evidence = {"examples": [call.target for call in calls[:3]]}
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


def build_external_dependencies(
    gateway: StorageGateway,
    cfg: ExternalDependenciesConfig,
) -> None:
    """
    Aggregate dependency usage into analytics.external_dependencies.

    Parameters
    ----------
    gateway
        Storage gateway with live DuckDB connection.
    cfg
        External dependency configuration.
    """
    patterns = _load_dependency_patterns(cfg)
    if not patterns:
        log.warning("No dependency patterns loaded; skipping dependency aggregation")
        return

    con = gateway.con
    _prepare_external_dependencies(con, cfg)

    config_keys_by_module = _load_config_keys(con, cfg.repo, cfg.commit)
    rows = _fetch_dependency_call_rows(con, cfg)
    aggregates = _aggregate_dependency_calls(rows, patterns)
    dep_rows = _serialize_dependency_rows(aggregates, config_keys_by_module, cfg)

    if dep_rows:
        con.executemany(
            """
            INSERT INTO analytics.external_dependencies (
                repo, commit, dep_id, library, service_name, category, language,
                severity, criticality, risk_score,
                function_count, callsite_count, modules_json, usage_modes,
                config_keys, risk_level, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            dep_rows,
        )
    log.info(
        "external_dependencies populated: %d rows for %s@%s",
        len(dep_rows),
        cfg.repo,
        cfg.commit,
    )


def _prepare_external_dependencies(con: duckdb.DuckDBPyConnection, cfg: ExternalDependenciesConfig) -> None:
    ensure_schema(con, "analytics.external_dependencies")
    con.execute(
        "DELETE FROM analytics.external_dependencies WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )


def _fetch_dependency_call_rows(
    con: duckdb.DuckDBPyConnection, cfg: ExternalDependenciesConfig
) -> list[tuple[object, ...]]:
    return con.execute(
        """
        SELECT dep_id, library, function_goid_h128, module,
               callsite_count, modes, severity, criticality, risk_score
        FROM analytics.external_dependency_calls
        WHERE repo = ? AND commit = ?
        """,
        [cfg.repo, cfg.commit],
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
        aggregate = aggregates.setdefault(
            str(dep_id),
            DependencyAggregate(
                library=lib_key,
                service_name=(pattern.service_name if pattern else None) or lib_key,
                category=pattern.category if pattern else None,
                severity=pattern.severity if pattern else None,
                criticality=pattern.criticality if pattern else None,
                risk_score=None,
            ),
        )
        if module:
            aggregate.modules.add(str(module))
        if function_goid is not None:
            aggregate.functions.add(int(function_goid))
        aggregate.callsite_count += int(callsite_count or 0)
        aggregate.modes.update(_ensure_str_list(modes_obj))
        agg_score = risk_score if risk_score is not None else _risk_score(severity, criticality)
        if agg_score is not None:
            prev_score = aggregate.risk_score or 0.0
            if agg_score > prev_score:
                aggregate.risk_score = agg_score
        if severity and aggregate.severity is None:
            aggregate.severity = severity
        if criticality is not None and aggregate.criticality is None:
            aggregate.criticality = float(criticality)
    return aggregates


def _serialize_dependency_rows(
    aggregates: dict[str, DependencyAggregate],
    config_keys_by_module: dict[str, set[str]],
    cfg: ExternalDependenciesConfig,
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
                cfg.repo,
                cfg.commit,
                dep_id,
                aggregate.library,
                aggregate.service_name,
                aggregate.category,
                cfg.language,
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


def _load_dependency_patterns(cfg: ExternalDependenciesConfig) -> dict[str, LibraryPattern]:
    path = cfg.dependency_patterns_path
    if path is None:
        path = cfg.repo_root / "config" / "dependency_patterns.yml"
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
        alias_maps[normalize_rel_path(rel_path)] = _build_alias_map(tree)
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


def _resolve_library(func: ast.AST, alias_map: dict[str, str]) -> tuple[str | None, str | None]:
    base_name = _base_name(func)
    library = alias_map.get(base_name) if base_name is not None else None
    method = None
    if isinstance(func, ast.Attribute):
        method = func.attr
    elif isinstance(func, ast.Name):
        method = func.id
    return library, method


def _base_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _base_name(node.value)
    if isinstance(node, ast.Call):
        return _base_name(node.func)
    return None


def _load_config_keys(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> dict[str, set[str]]:
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


def load_config_key_map(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> dict[str, set[str]]:
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


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except (AttributeError, ValueError, TypeError):
        return node.__class__.__name__


def _dep_id(repo: str, commit: str, library: str) -> str:
    raw = f"{repo}:{commit}:{library}"
    return hashlib.sha1(raw.encode("utf-8"), usedforsecurity=False).hexdigest()[:16]


def _decimal(value: int) -> Decimal:
    return Decimal(value)


__all__ = [
    "build_external_dependencies",
    "build_external_dependency_calls",
    "load_config_key_map",
]
