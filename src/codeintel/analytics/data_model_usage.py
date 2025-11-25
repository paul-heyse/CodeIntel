"""Classify how functions use extracted data models."""

from __future__ import annotations

import ast
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime

import duckdb

from codeintel.analytics.function_ast_cache import FunctionAst, load_function_asts
from codeintel.config.models import DataModelUsageConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.graphs.function_catalog_service import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.ingestion.common import load_module_map
from codeintel.storage.gateway import StorageGateway
from codeintel.utils.paths import normalize_rel_path

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInfo:
    """Minimal model identity used during usage classification."""

    model_id: str
    name: str
    module: str


@dataclass
class ModelUsageResult:
    """Usage kinds and evidence collected for a single function."""

    usage_kinds: dict[str, set[str]] = field(default_factory=dict)
    evidence: dict[tuple[str, str], list[dict[str, object]]] = field(default_factory=dict)


@dataclass(frozen=True)
class ModelUsageArtifacts:
    """Reusable lookup tables for model usage classification."""

    ast_by_goid: dict[int, FunctionAst]
    module_map: dict[str, str]
    param_types: dict[int, dict[str, str]]
    model_simple: dict[str, list[ModelInfo]]
    model_qualified: dict[str, ModelInfo]
    subsystem_map: dict[str, tuple[str, str]]


def _parse_param_types(raw: str | dict[str, object] | None) -> dict[str, str]:
    if raw is None:
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
    else:
        parsed = raw
    if not isinstance(parsed, dict):
        return {}
    result: dict[str, str] = {}
    for key, value in parsed.items():
        if value is None:
            continue
        result[str(key)] = str(value)
    return result


def _build_model_index(
    models: list[ModelInfo],
) -> tuple[dict[str, list[ModelInfo]], dict[str, ModelInfo]]:
    simple: dict[str, list[ModelInfo]] = {}
    qualified: dict[str, ModelInfo] = {}
    for model in models:
        simple.setdefault(model.name, []).append(model)
        qualified[f"{model.module}.{model.name}"] = model
    return simple, qualified


def _resolve_model(
    name: str,
    simple_index: dict[str, list[ModelInfo]],
    qualified_index: dict[str, ModelInfo],
) -> ModelInfo | None:
    normalized = name
    if normalized in qualified_index:
        return qualified_index[normalized]
    candidate_name = normalized.split(".")[-1]
    candidates = simple_index.get(candidate_name, [])
    return candidates[0] if candidates else None


def _resolve_model_from_hint(
    type_hint: str | None,
    simple_index: dict[str, list[ModelInfo]],
    qualified_index: dict[str, ModelInfo],
) -> ModelInfo | None:
    if not type_hint:
        return None
    normalized = re.sub(r"[^A-Za-z0-9_\.]", " ", type_hint)
    tokens = normalized.split()
    for token in tokens:
        resolved = _resolve_model(token, simple_index, qualified_index)
        if resolved is not None:
            return resolved
    return None


def _call_target_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _call_target_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None


class ModelUsageVisitor(ast.NodeVisitor):
    """Collect per-model usage kinds and evidence from a function AST."""

    def __init__(
        self,
        *,
        models_by_name: dict[str, list[ModelInfo]],
        models_by_qualified: dict[str, ModelInfo],
        lines: list[str],
        max_examples: int,
    ) -> None:
        self._models_by_name = models_by_name
        self._models_by_qualified = models_by_qualified
        self._lines = lines
        self._max_examples = max_examples
        self._model_vars: dict[str, ModelInfo] = {}
        self.result = ModelUsageResult()

    def seed_parameters(self, param_types: dict[str, str]) -> None:
        """Prime the visitor with models inferred from function parameters."""
        for name, type_hint in param_types.items():
            resolved = _resolve_model_from_hint(
                type_hint, self._models_by_name, self._models_by_qualified
            )
            if resolved is not None:
                self._model_vars[name] = resolved

    def visit_Assign(self, node: ast.Assign) -> None:
        """Capture assignments that construct or bind models."""
        self._maybe_capture_assignment(node.targets, node.value)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handle annotated assignments for potential model creation."""
        targets = [node.target] if node.target is not None else []
        self._maybe_capture_assignment(targets, node.value)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Track creation, serialization, and validation calls on models."""
        target_name = _call_target_name(node.func)
        resolved_class = _resolve_model(
            target_name or "",
            self._models_by_name,
            self._models_by_qualified,
        )
        if resolved_class is not None:
            self._record_usage(resolved_class.model_id, "create", node.lineno)
        if isinstance(node.func, ast.Attribute):
            base = node.func.value
            attr_name = node.func.attr
            if isinstance(base, ast.Name):
                model = self._model_vars.get(base.id)
                if model is not None:
                    if attr_name in {
                        "dict",
                        "json",
                        "model_dump",
                        "model_dump_json",
                        "asdict",
                        "astuple",
                    }:
                        self._record_usage(model.model_id, "serialize", node.lineno)
                    elif attr_name in {"update", "append", "extend", "setdefault"}:
                        self._record_usage(model.model_id, "update", node.lineno)
                    elif attr_name in {"parse_obj", "validate", "model_validate"}:
                        self._record_usage(model.model_id, "validate", node.lineno)
        for arg in node.args:
            if isinstance(arg, ast.Name) and arg.id in self._model_vars:
                self._record_usage(self._model_vars[arg.id].model_id, "serialize", arg.lineno)
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        """Identify attribute reads and writes on tracked models."""
        if isinstance(node.value, ast.Name):
            model = self._model_vars.get(node.value.id)
            if model is not None:
                if isinstance(node.ctx, ast.Store):
                    self._record_usage(model.model_id, "update", node.lineno)
                else:
                    self._record_usage(model.model_id, "read", node.lineno)
        self.generic_visit(node)

    def visit_Delete(self, node: ast.Delete) -> None:
        """Capture model deletions."""
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id in self._model_vars:
                self._record_usage(self._model_vars[target.id].model_id, "delete", target.lineno)
            if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                base = target.value.id
                if base in self._model_vars:
                    self._record_usage(self._model_vars[base].model_id, "delete", target.lineno)
        self.generic_visit(node)

    def _maybe_capture_assignment(
        self, targets: Sequence[ast.expr], value: ast.AST | None
    ) -> None:
        if isinstance(value, ast.Call):
            call_target = _call_target_name(value.func) or ""
            resolved = _resolve_model(call_target, self._models_by_name, self._models_by_qualified)
            if resolved is not None:
                for target in targets:
                    if isinstance(target, ast.Name):
                        self._model_vars[target.id] = resolved
                self._record_usage(resolved.model_id, "create", getattr(value, "lineno", 0))
        if isinstance(value, ast.AST):
            self.generic_visit(value)

    def _record_usage(self, model_id: str, kind: str, lineno: int | None) -> None:
        if lineno is None:
            lineno = 0
        usage = self.result.usage_kinds.setdefault(model_id, set())
        usage.add(kind)
        evidence_key = (model_id, kind)
        samples = self.result.evidence.setdefault(evidence_key, [])
        if len(samples) >= self._max_examples:
            return
        snippet = ""
        if 1 <= lineno <= len(self._lines):
            snippet = self._lines[lineno - 1].strip()
        samples.append({"line": lineno, "snippet": snippet})


def _load_models(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> list[ModelInfo]:
    rows = con.execute(
        """
        SELECT model_id, model_name, module, rel_path
        FROM analytics.data_models
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    return [ModelInfo(model_id=row[0], name=row[1], module=row[2]) for row in rows]


def _subsystem_by_module(
    con: duckdb.DuckDBPyConnection, repo: str, commit: str
) -> dict[str, tuple[str, str]]:
    rows = con.execute(
        """
        SELECT sm.module, sm.subsystem_id, s.name
        FROM analytics.subsystem_modules sm
        LEFT JOIN analytics.subsystems s
          ON s.repo = sm.repo AND s.commit = sm.commit AND s.subsystem_id = sm.subsystem_id
        WHERE sm.repo = ? AND sm.commit = ?
        """,
        [repo, commit],
    ).fetchall()
    mapping: dict[str, tuple[str, str]] = {}
    for module, subsystem_id, name in rows:
        mapping[str(module)] = (str(subsystem_id), str(name) if name is not None else "")
    return mapping


def _context_for_module(
    module: str | None, subsystem_map: dict[str, tuple[str, str]]
) -> dict[str, object]:
    context: dict[str, object] = {}
    if module:
        context["module"] = module
        subsystem = subsystem_map.get(module)
        if subsystem is not None:
            subsystem_id, subsystem_name = subsystem
            context["subsystem_id"] = subsystem_id
            if subsystem_name:
                context["subsystem_name"] = subsystem_name
    return context


def compute_data_model_usage(
    gateway: StorageGateway,
    cfg: DataModelUsageConfig,
    *,
    catalog_provider: FunctionCatalogProvider | None = None,
) -> None:
    """
    Populate analytics.data_model_usage with per-function model usage classifications.

    Parameters
    ----------
    gateway
        Storage gateway for the active DuckDB database.
    cfg
        Data model usage configuration.
    catalog_provider
        Optional function catalog to reuse across analytics steps.
    """
    con = gateway.con
    ensure_schema(con, "analytics.data_model_usage")
    con.execute(
        "DELETE FROM analytics.data_model_usage WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    models = _load_models(con, cfg.repo, cfg.commit)
    if not models:
        log.info("No data models found for %s@%s; skipping usage analysis", cfg.repo, cfg.commit)
        return
    model_simple, model_qualified = _build_model_index(models)

    module_map = load_module_map(gateway, cfg.repo, cfg.commit)
    subsystem_map = _subsystem_by_module(con, cfg.repo, cfg.commit)
    ast_by_goid, missing = load_function_asts(
        gateway,
        repo=cfg.repo,
        commit=cfg.commit,
        repo_root=cfg.repo_root,
        catalog_provider=(
            catalog_provider
            or FunctionCatalogService.from_db(gateway, repo=cfg.repo, commit=cfg.commit)
        ),
    )
    if missing:
        log.debug(
            "Skipping %d functions without AST spans during model usage analysis",
            len(missing),
        )

    param_types: dict[int, dict[str, str]] = {
        int(goid): _parse_param_types(raw_param_types)
        for goid, raw_param_types in con.execute(
            """
            SELECT function_goid_h128, param_types
            FROM analytics.function_types
            WHERE repo = ? AND commit = ?
            """,
            [cfg.repo, cfg.commit],
        ).fetchall()
    }

    artifacts = ModelUsageArtifacts(
        ast_by_goid=ast_by_goid,
        module_map=module_map,
        param_types=param_types,
        model_simple=model_simple,
        model_qualified=model_qualified,
        subsystem_map=subsystem_map,
    )
    rows_to_insert = _build_usage_rows(artifacts=artifacts, cfg=cfg)

    if rows_to_insert:
        con.executemany(
            """
            INSERT INTO analytics.data_model_usage (
                repo, commit, model_id, function_goid_h128,
                usage_kinds_json, evidence_json, context_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows_to_insert,
        )
    log.info(
        "data_model_usage populated: %d rows for %s@%s",
        len(rows_to_insert),
        cfg.repo,
        cfg.commit,
    )


def _build_usage_rows(
    *,
    artifacts: ModelUsageArtifacts,
    cfg: DataModelUsageConfig,
) -> list[tuple[object, ...]]:
    now = datetime.now(tz=UTC)
    rows_to_insert: list[tuple[object, ...]] = []
    for goid, func_ast in artifacts.ast_by_goid.items():
        rel_path = normalize_rel_path(func_ast.rel_path)
        module = artifacts.module_map.get(rel_path)
        visitor = ModelUsageVisitor(
            models_by_name=artifacts.model_simple,
            models_by_qualified=artifacts.model_qualified,
            lines=func_ast.lines,
            max_examples=cfg.max_examples_per_usage,
        )
        visitor.seed_parameters(artifacts.param_types.get(goid, {}))
        visitor.visit(func_ast.node)
        for model_id, kinds in visitor.result.usage_kinds.items():
            evidence_map = {
                usage: visitor.result.evidence.get((model_id, usage), []) for usage in kinds
            }
            context = _context_for_module(module, artifacts.subsystem_map)
            rows_to_insert.append(
                (
                    cfg.repo,
                    cfg.commit,
                    model_id,
                    goid,
                    json.dumps(sorted(kinds)),
                    json.dumps(evidence_map),
                    json.dumps(context) if context else None,
                    now,
                )
            )
    return rows_to_insert
