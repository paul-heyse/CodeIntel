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
    "nullable",
    "unique",
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
    relationship_hints: list[dict[str, object]]
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
    lowered_full = {base.lower() for base in bases}
    kind = "generic_class"
    if _is_dataclass(decorators):
        kind = "dataclass"
    elif "basemodel" in lowered_bases:
        kind = "pydantic_model"
    elif "typeddict" in lowered_bases:
        kind = "typeddict"
    elif "protocol" in lowered_bases:
        kind = "protocol"
    elif any("django" in base and base.endswith("model") for base in lowered_full):
        kind = "django_model"
    elif "model" in lowered_bases or any(
        base.endswith("base") or "declarative" in base for base in lowered_bases
    ):
        kind = "orm_model"
    return kind


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
        if func_name.lower().endswith("column"):
            return "orm_column"
        if func_name.lower().endswith("relationship"):
            return "orm_relationship"
    return "annotation"


def _constraints_from_keywords(keywords: list[ast.keyword]) -> dict[str, object]:
    constraints: dict[str, object | None] = dict.fromkeys(FIELD_CONSTRAINT_KEYS, None)
    for kw in keywords:
        if kw.arg is None:
            continue
        if kw.arg in constraints:
            constraints[kw.arg] = _literal_value(kw.value)
        if kw.arg == "max_length":
            constraints["max_length"] = _literal_value(kw.value)
        if kw.arg == "nullable":
            constraints["nullable"] = _literal_value(kw.value)
        if kw.arg == "unique":
            constraints["unique"] = _literal_value(kw.value)
    return constraints


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


def _relationship_hint(
    *,
    field_name: str,
    target: str,
    multiplicity: str,
    kind: str,
    via: str,
) -> dict[str, object]:
    return {
        "field": field_name,
        "target": target,
        "multiplicity": multiplicity,
        "kind": kind,
        "via": via,
    }


def _foreign_key_target(args: list[ast.AST], keywords: list[ast.keyword]) -> str | None:
    for arg in args:
        if isinstance(arg, ast.Call):
            call_name = _call_name(arg.func) or ""
            if "foreignkey" in call_name.lower() and arg.args:
                return str(_literal_value(arg.args[0]))
        if isinstance(arg, (ast.Constant, ast.Name, ast.Attribute)):
            call_name = _call_name(arg)
            if call_name is not None and "foreignkey" in call_name.lower():
                return call_name
    for kw in keywords:
        if kw.arg and "foreignkey" in kw.arg.lower():
            return str(_literal_value(kw.value))
    return None


def _relationship_hints_from_call(field_name: str, value: ast.Call) -> list[dict[str, object]]:
    func_name = (_call_name(value.func) or "").lower()
    hints: list[dict[str, object]] = []
    target_value: str | None = None
    multiplicity = "many"
    if "relationship" in func_name:
        if value.args:
            target_value = str(_literal_value(value.args[0]))
        uselist_kw = next((kw for kw in value.keywords if kw.arg == "uselist"), None)
        if uselist_kw is not None:
            use_list_val = _literal_value(uselist_kw.value)
            if isinstance(use_list_val, bool) and not use_list_val:
                multiplicity = "one"
        hints.append(
            _relationship_hint(
                field_name=field_name,
                target=target_value or "",
                multiplicity=multiplicity,
                kind="relationship",
                via="sqlalchemy_relationship",
            )
        )
    fk_target = _foreign_key_target(list(value.args), value.keywords)
    if fk_target:
        hints.append(
            _relationship_hint(
                field_name=field_name,
                target=fk_target,
                multiplicity="many",
                kind="foreign_key",
                via="sqlalchemy_column",
            )
        )
    if "foreignkey" in func_name:
        if value.args:
            target_value = str(_literal_value(value.args[0]))
        hints.append(
            _relationship_hint(
                field_name=field_name,
                target=target_value or "",
                multiplicity="many",
                kind="foreign_key",
                via="django_field",
            )
        )
    if any(name in func_name for name in ("onetoonefield", "manytomanyfield")):
        if value.args:
            target_value = str(_literal_value(value.args[0]))
        multiplicity = "one" if "onetoonefield" in func_name else "many"
        hints.append(
            _relationship_hint(
                field_name=field_name,
                target=target_value or "",
                multiplicity=multiplicity,
                kind="relationship",
                via="django_field",
            )
        )
    return hints


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


def _field_spec_from_call(
    field_name: str,
    value: ast.Call,
    *,
    model_kind: str,
) -> tuple[FieldSpec | None, list[dict[str, object]]]:
    func_name = (_call_name(value.func) or "").lower()
    constraints = _constraints_from_keywords(value.keywords)
    required = True
    has_default = False
    default_expr = None
    field_type = None
    relationship_hints = _relationship_hints_from_call(field_name, value)
    if "column" in func_name or "field" in func_name or "relationship" in func_name:
        if value.args:
            arg0 = value.args[0]
            if isinstance(arg0, ast.Call) and "foreignkey" in (_call_name(arg0.func) or "").lower():
                field_type = "ForeignKey"
            else:
                field_type = _annotation_to_str(arg0)
        required = constraints.get("nullable") is not True
        has_default = any(kw.arg in {"default", "server_default"} for kw in value.keywords)
        default_kw = next((kw for kw in value.keywords if kw.arg == "default"), None)
        if default_kw is not None:
            default_expr = _safe_unparse(default_kw.value)
        if model_kind == "django_model":
            null_kw = next((kw for kw in value.keywords if kw.arg == "null"), None)
            blank_kw = next((kw for kw in value.keywords if kw.arg == "blank"), None)
            if null_kw is not None:
                required = not bool(_literal_value(null_kw.value))
            if blank_kw is not None and bool(_literal_value(blank_kw.value)):
                required = False
            if default_kw is not None:
                has_default = True
        source = _field_source(value)
        field_spec = FieldSpec(
            name=field_name,
            type=field_type,
            required=required,
            has_default=has_default,
            default_expr=default_expr,
            constraints=constraints,
            source=source,
        )
        return field_spec, relationship_hints
    return None, relationship_hints


def _build_field_specs(
    node: ast.ClassDef,
    *,
    model_kind: str,
) -> tuple[list[FieldSpec], list[dict[str, object]]]:
    fields: list[FieldSpec] = []
    rel_hints: list[dict[str, object]] = []
    total_flag = _total_flag(node)
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign):
            field_spec = _field_spec_from_assign(stmt, model_kind=model_kind, total_flag=total_flag)
            if field_spec is not None:
                fields.append(field_spec)
                if isinstance(stmt.value, ast.Call):
                    rel_hints.extend(_relationship_hints_from_call(field_spec.name, stmt.value))
        if isinstance(stmt, ast.Assign):
            if not stmt.targets:
                continue
            target = stmt.targets[0]
            if not isinstance(target, ast.Name):
                continue
            if isinstance(stmt.value, ast.Call):
                spec, hints = _field_spec_from_call(target.id, stmt.value, model_kind=model_kind)
                rel_hints.extend(hints)
                if spec is not None:
                    fields.append(spec)
    return fields, rel_hints


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
        fields, rel_hints = _build_field_specs(cls_node, model_kind=model_kind)
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
                relationship_hints=rel_hints,
                doc_short=doc_short,
                doc_long=doc_long,
            )
        )
    return models


def _normalize_snake(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\.]", "", text)


def _candidate_tokens(value: str) -> list[str]:
    separators = re.split(r"[^A-Za-z0-9_]+", value)
    tokens = [value]
    tokens.extend(separators)
    for part in separators:
        if "." in part:
            tokens.extend(part.split("."))
    return [token for token in tokens if token]


def _match_qualified_pattern(
    normalized: str, qualified_index: dict[str, ModelRecord]
) -> ModelRecord | None:
    for qualified_name, candidate in qualified_index.items():
        if re.search(rf"\b{re.escape(qualified_name)}\b", normalized):
            return candidate
    return None


def _match_name_pattern(
    normalized: str, name_index: dict[str, list[ModelRecord]]
) -> ModelRecord | None:
    for name, candidates in name_index.items():
        if re.search(rf"\b{re.escape(name)}\b", normalized):
            return candidates[0]
    return None


def _match_tokenized(
    type_hint: str,
    *,
    qualified_index: dict[str, ModelRecord],
    name_index: dict[str, list[ModelRecord]],
    qualified_index_lower: dict[str, ModelRecord],
    name_index_lower: dict[str, list[ModelRecord]],
) -> ModelRecord | None:
    for token in _candidate_tokens(type_hint):
        lowered = token.lower()
        if token in qualified_index:
            return qualified_index[token]
        if lowered in qualified_index_lower:
            return qualified_index_lower[lowered]
        if token in name_index:
            return name_index[token][0]
        if lowered in name_index_lower:
            return name_index_lower[lowered][0]
    return None


def _resolve_target(
    type_hint: str | None,
    *,
    qualified_index: dict[str, ModelRecord],
    name_index: dict[str, list[ModelRecord]],
    qualified_index_lower: dict[str, ModelRecord],
    name_index_lower: dict[str, list[ModelRecord]],
) -> ModelRecord | None:
    if type_hint is None:
        return None
    normalized = _normalize_snake(type_hint)
    candidate = _match_qualified_pattern(normalized, qualified_index)
    if candidate is not None:
        return candidate
    candidate = _match_name_pattern(normalized, name_index)
    if candidate is not None:
        return candidate
    return _match_tokenized(
        type_hint,
        qualified_index=qualified_index,
        name_index=name_index,
        qualified_index_lower=qualified_index_lower,
        name_index_lower=name_index_lower,
    )


def _relationships_for_model(
    model: ModelRecord,
    *,
    qualified_index: dict[str, ModelRecord],
    name_index: dict[str, list[ModelRecord]],
    qualified_index_lower: dict[str, ModelRecord],
    name_index_lower: dict[str, list[ModelRecord]],
) -> list[dict[str, object]]:
    relationships: list[dict[str, object]] = []
    for field in model.fields:
        target = _resolve_target(
            field.type,
            qualified_index=qualified_index,
            name_index=name_index,
            qualified_index_lower=qualified_index_lower,
            name_index_lower=name_index_lower,
        )
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
    for hint in model.relationship_hints:
        target_text = str(hint.get("target") or "")
        target = _resolve_target(
            target_text,
            qualified_index=qualified_index,
            name_index=name_index,
            qualified_index_lower=qualified_index_lower,
            name_index_lower=name_index_lower,
        )
        if target is None:
            continue
        relationships.append(
            {
                "field": str(hint.get("field")),
                "target_model_id": target.model_id,
                "target_model_name": target.model_name,
                "multiplicity": str(hint.get("multiplicity", "many")),
                "kind": str(hint.get("kind", "reference")),
                "via": str(hint.get("via", "hint")),
            }
        )
    return relationships


def _attach_relationships(models: list[ModelRecord]) -> None:
    if not models:
        return
    name_index: dict[str, list[ModelRecord]] = {}
    name_index_lower: dict[str, list[ModelRecord]] = {}
    qualified_index: dict[str, ModelRecord] = {}
    qualified_index_lower: dict[str, ModelRecord] = {}
    for model in models:
        name_index.setdefault(model.model_name, []).append(model)
        name_index_lower.setdefault(model.model_name.lower(), []).append(model)
        qualified = f"{model.module}.{model.model_name}"
        qualified_index[qualified] = model
        qualified_index_lower[qualified.lower()] = model

    for model in models:
        model.relationships = _relationships_for_model(
            model,
            qualified_index=qualified_index,
            name_index=name_index,
            qualified_index_lower=qualified_index_lower,
            name_index_lower=name_index_lower,
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
