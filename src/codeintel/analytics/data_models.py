"""Extract structured data models from class definitions."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import duckdb

from codeintel.config.models import DataModelsConfig
from codeintel.config.schemas.sql_builder import ensure_schema
from codeintel.ingestion.ast_utils import parse_python_module
from codeintel.utils.paths import normalize_rel_path, relpath_to_module

log = logging.getLogger(__name__)

FIELD_CONSTRAINT_KEYS: tuple[str, ...] = (
    "gt",
    "ge",
    "lt",
    "le",
    "min_length",
    "max_length",
    "regex",
)


@dataclass(frozen=True)
class ClassMeta:
    """Minimal metadata needed to align AST classes with GOIDs."""

    goid: int | None
    rel_path: str
    qualname: str
    start_line: int
    end_line: int
    module: str


@dataclass(frozen=True)
class FieldSpec:
    """Structured representation of a model field."""

    name: str
    type: str | None
    required: bool
    has_default: bool
    default_expr: str | None
    constraints: dict[str, object]
    source: str


@dataclass
class ModelRecord:
    """In-memory representation of a detected model prior to persistence."""

    model_id: str
    goid: int | None
    model_name: str
    module: str
    rel_path: str
    model_kind: str
    base_classes: list[dict[str, str]]
    fields: list[FieldSpec]
    relationships: list[dict[str, object]]
    doc_short: str | None
    doc_long: str | None


def _safe_unparse(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    try:
        return ast.unparse(node)
    except (AttributeError, TypeError, ValueError):
        return None


def _annotation_to_str(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    text = _safe_unparse(node)
    if text is not None:
        return text
    if isinstance(node, ast.Name):
        return node.id
    return node.__class__.__name__


def _literal_value(node: ast.AST | None) -> object:
    if node is None:
        return None
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, (ast.Num, ast.Str)):
        return getattr(node, "n", None) or getattr(node, "s", None)
    if isinstance(node, ast.NameConstant):
        return node.value
    text = _safe_unparse(node)
    return text if text is not None else None


def _call_name(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        value_name = _call_name(node.value)
        return f"{value_name}.{node.attr}" if value_name else node.attr
    return _safe_unparse(node)


def _compute_model_id(repo: str, commit: str, module: str, qualname: str, model_kind: str) -> str:
    raw = f"{repo}:{commit}:{module}:{qualname}:{model_kind}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _is_dataclass(decorators: list[str]) -> bool:
    return any(dec.endswith("dataclass") for dec in decorators)


def _class_kind(decorators: list[str], bases: list[str]) -> str:
    lowered_bases = {base.split(".")[-1].lower() for base in bases}
    if _is_dataclass(decorators):
        return "dataclass"
    if "basemodel" in lowered_bases:
        return "pydantic_model"
    if "typeddict" in lowered_bases:
        return "typeddict"
    if "protocol" in lowered_bases:
        return "protocol"
    if "model" in lowered_bases:
        return "orm_model"
    return "generic_class"


def _constraints_from_call(value: ast.AST | None) -> dict[str, object]:
    constraints: dict[str, object | None] = dict.fromkeys(FIELD_CONSTRAINT_KEYS, None)
    if not isinstance(value, ast.Call):
        return constraints
    for kw in value.keywords:
        if kw.arg is None:
            continue
        if kw.arg in constraints:
            constraints[kw.arg] = _literal_value(kw.value)
    return constraints


def _is_required_marker(value: ast.AST | None) -> bool:
    if isinstance(value, ast.Constant):
        return value.value is Ellipsis
    if not isinstance(value, ast.Call):
        return False
    func_name = _call_name(value.func) or ""
    if not func_name.endswith(("Field", "field")):
        return False
    if not value.args:
        return False
    arg0 = value.args[0]
    return isinstance(arg0, ast.Constant) and arg0.value is Ellipsis


def _field_source(value: ast.AST | None) -> str:
    if isinstance(value, ast.Call):
        func_name = _call_name(value.func) or ""
        if func_name.endswith("Field"):
            return "pydantic_field"
        if func_name.endswith("field"):
            return "dataclass_field"
    return "annotation"


def _total_flag(node: ast.ClassDef) -> bool:
    total_flag = True
    for keyword in node.keywords:
        if keyword.arg == "total":
            total_value = _literal_value(keyword.value)
            if isinstance(total_value, bool):
                total_flag = total_value
    for base in node.bases:
        if (
            isinstance(base, ast.Subscript)
            and _safe_unparse(base.value) == "TypedDict"
            and isinstance(base.slice, ast.keyword)
            and base.slice.arg == "total"
        ):
            total_value = _literal_value(base.slice.value)
            if isinstance(total_value, bool):
                total_flag = total_value
    return total_flag


def _field_spec_from_assign(
    stmt: ast.AnnAssign,
    *,
    model_kind: str,
    total_flag: bool,
) -> FieldSpec | None:
    target = stmt.target
    if not isinstance(target, ast.Name):
        return None
    field_name = target.id
    annotation = _annotation_to_str(stmt.annotation)
    constraints = _constraints_from_call(stmt.value)
    required = True
    has_default = False
    default_expr = None
    if stmt.value is not None:
        required = not _is_required_marker(stmt.value)
        has_default = not required
        default_expr = _safe_unparse(stmt.value)
    if model_kind == "typeddict":
        required = required and total_flag
        has_default = not required
    return FieldSpec(
        name=field_name,
        type=annotation,
        required=required,
        has_default=has_default,
        default_expr=default_expr,
        constraints=constraints,
        source=_field_source(stmt.value),
    )


def _build_field_specs(
    node: ast.ClassDef,
    *,
    model_kind: str,
) -> list[FieldSpec]:
    fields: list[FieldSpec] = []
    total_flag = _total_flag(node)
    for stmt in node.body:
        if not isinstance(stmt, ast.AnnAssign):
            continue
        field_spec = _field_spec_from_assign(stmt, model_kind=model_kind, total_flag=total_flag)
        if field_spec is not None:
            fields.append(field_spec)
    return fields


def _load_class_metadata(con: duckdb.DuckDBPyConnection, repo: str, commit: str) -> list[ClassMeta]:
    rows = con.execute(
        """
        SELECT g.goid_h128, g.rel_path, g.qualname, g.start_line, g.end_line, m.module
        FROM core.goids g
        LEFT JOIN core.modules m
          ON m.path = g.rel_path
        WHERE g.repo = ? AND g.commit = ? AND g.kind = 'class'
        """,
        [repo, commit],
    ).fetchall()
    metas: list[ClassMeta] = []
    for goid_h128, rel_path, qualname, start_line, end_line, module in rows:
        if rel_path is None or start_line is None:
            continue
        metas.append(
            ClassMeta(
                goid=int(goid_h128) if goid_h128 is not None else None,
                rel_path=normalize_rel_path(rel_path),
                qualname=str(qualname),
                start_line=int(start_line),
                end_line=int(end_line) if end_line is not None else int(start_line),
                module=str(module) if module is not None else relpath_to_module(rel_path),
            )
        )
    return metas


def _doc_map(
    con: duckdb.DuckDBPyConnection,
    *,
    repo: str,
    commit: str,
) -> dict[tuple[str, str], tuple[str | None, str | None]]:
    rows = con.execute(
        """
        SELECT rel_path, qualname, short_desc, long_desc
        FROM core.docstrings
        WHERE repo = ? AND commit = ? AND kind = 'class'
        """,
        [repo, commit],
    ).fetchall()
    mapping: dict[tuple[str, str], tuple[str | None, str | None]] = {}
    for rel_path, qualname, short_desc, long_desc in rows:
        mapping[normalize_rel_path(rel_path), str(qualname)] = (
            short_desc,
            long_desc,
        )
    return mapping


def _class_decorators(node: ast.ClassDef) -> list[str]:
    decorators: list[str] = []
    for dec in node.decorator_list:
        text = _safe_unparse(dec)
        if text is not None:
            decorators.append(text)
    return decorators


def _base_classes(node: ast.ClassDef) -> list[dict[str, str]]:
    bases: list[dict[str, str]] = []
    for base in node.bases:
        text = _safe_unparse(base)
        if text is None:
            continue
        name = text.split(".")[-1]
        bases.append({"name": name, "qualname": text})
    return bases


def _match_class_meta(meta_by_line: dict[int, ClassMeta], node: ast.ClassDef) -> ClassMeta | None:
    lineno = getattr(node, "lineno", None)
    if lineno is None:
        return None
    return meta_by_line.get(int(lineno))


def _all_class_defs(tree: ast.AST) -> Iterator[ast.ClassDef]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            yield node


def _gather_models_for_path(
    rel_path: str,
    abs_path: Path,
    metas: list[ClassMeta],
    docstrings: dict[tuple[str, str], tuple[str | None, str | None]],
    cfg: DataModelsConfig,
) -> list[ModelRecord]:
    parsed = parse_python_module(abs_path)
    if parsed is None:
        log.debug("Skipping %s; unable to parse module", abs_path)
        return []
    _, tree = parsed
    meta_by_line = {meta.start_line: meta for meta in metas}
    models: list[ModelRecord] = []
    for cls_node in _all_class_defs(tree):
        meta = _match_class_meta(meta_by_line, cls_node)
        if meta is None:
            continue
        decorators = _class_decorators(cls_node)
        base_classes = _base_classes(cls_node)
        base_names = [base["qualname"] for base in base_classes]
        model_kind = _class_kind(decorators, base_names)
        fields = _build_field_specs(cls_node, model_kind=model_kind)
        doc_short, doc_long = docstrings.get((rel_path, meta.qualname), (None, None))
        model_id = _compute_model_id(cfg.repo, cfg.commit, meta.module, meta.qualname, model_kind)
        models.append(
            ModelRecord(
                model_id=model_id,
                goid=meta.goid,
                model_name=cls_node.name,
                module=meta.module,
                rel_path=rel_path,
                model_kind=model_kind,
                base_classes=base_classes,
                fields=fields,
                relationships=[],
                doc_short=doc_short,
                doc_long=doc_long,
            )
        )
    return models


def _normalize_snake(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\.]", "", text)


def _resolve_target(
    type_hint: str | None,
    *,
    qualified_index: dict[str, ModelRecord],
    name_index: dict[str, list[ModelRecord]],
) -> ModelRecord | None:
    if type_hint is None:
        return None
    normalized = _normalize_snake(type_hint)
    for qualified_name, candidate in qualified_index.items():
        if re.search(rf"\b{re.escape(qualified_name)}\b", normalized):
            return candidate
    for name, candidates in name_index.items():
        if re.search(rf"\b{re.escape(name)}\b", normalized):
            return candidates[0]
    return None


def _relationships_for_model(
    model: ModelRecord,
    *,
    qualified_index: dict[str, ModelRecord],
    name_index: dict[str, list[ModelRecord]],
) -> list[dict[str, object]]:
    relationships: list[dict[str, object]] = []
    for field in model.fields:
        target = _resolve_target(field.type, qualified_index=qualified_index, name_index=name_index)
        if target is None:
            continue
        multiplicity = (
            "many"
            if field.type and re.search(r"\b(list|set|tuple|Sequence)\b", field.type)
            else "one"
        )
        relationships.append(
            {
                "field": field.name,
                "target_model_id": target.model_id,
                "target_model_name": target.model_name,
                "multiplicity": multiplicity,
                "kind": "reference",
                "via": "annotation",
            }
        )
    return relationships


def _attach_relationships(models: list[ModelRecord]) -> None:
    if not models:
        return
    name_index: dict[str, list[ModelRecord]] = {}
    qualified_index: dict[str, ModelRecord] = {}
    for model in models:
        name_index.setdefault(model.model_name, []).append(model)
        qualified = f"{model.module}.{model.model_name}"
        qualified_index[qualified] = model

    for model in models:
        model.relationships = _relationships_for_model(
            model, qualified_index=qualified_index, name_index=name_index
        )


def compute_data_models(con: duckdb.DuckDBPyConnection, cfg: DataModelsConfig) -> None:
    """
    Populate analytics.data_models with extracted model definitions.

    Parameters
    ----------
    con
        DuckDB connection scoped to the target repository.
    cfg
        Data model extraction configuration.
    """
    ensure_schema(con, "analytics.data_models")
    con.execute(
        "DELETE FROM analytics.data_models WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )

    class_metas = _load_class_metadata(con, cfg.repo, cfg.commit)
    if not class_metas:
        log.info(
            "No class metadata found for %s@%s; skipping data model extraction",
            cfg.repo,
            cfg.commit,
        )
        return

    metas_by_path: dict[str, list[ClassMeta]] = {}
    for meta in class_metas:
        metas_by_path.setdefault(meta.rel_path, []).append(meta)
    docs = _doc_map(con, repo=cfg.repo, commit=cfg.commit)

    models: list[ModelRecord] = []
    for rel_path, metas in metas_by_path.items():
        abs_path = (Path(cfg.repo_root) / rel_path).resolve()
        if not abs_path.is_file():
            log.debug("Skipping %s; file missing on disk", abs_path)
            continue
        models.extend(
            _gather_models_for_path(
                rel_path,
                abs_path,
                metas,
                docs,
                cfg,
            )
        )

    _attach_relationships(models)
    now = datetime.now(tz=UTC)
    rows: list[tuple[object, ...]] = [
        (
            cfg.repo,
            cfg.commit,
            model.model_id,
            model.goid,
            model.model_name,
            model.module,
            model.rel_path,
            model.model_kind,
            json.dumps(model.base_classes),
            json.dumps(
                [
                    {
                        "name": field.name,
                        "type": field.type,
                        "required": field.required,
                        "has_default": field.has_default,
                        "default_expr": field.default_expr,
                        "constraints": field.constraints,
                        "source": field.source,
                    }
                    for field in model.fields
                ]
            ),
            json.dumps(model.relationships),
            model.doc_short,
            model.doc_long,
            now,
        )
        for model in models
    ]

    if rows:
        con.executemany(
            """
            INSERT INTO analytics.data_models (
                repo, commit, model_id, goid_h128,
                model_name, module, rel_path, model_kind,
                base_classes_json, fields_json, relationships_json,
                doc_short, doc_long, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    log.info("data_models populated: %d models for %s@%s", len(rows), cfg.repo, cfg.commit)
