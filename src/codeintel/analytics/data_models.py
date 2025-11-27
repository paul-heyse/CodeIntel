"""Extract structured data models from class definitions."""

from __future__ import annotations

import ast
import hashlib
import json
import logging
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from pathlib import Path

from codeintel.analytics.ast_utils import call_name, literal_value, safe_unparse, snippet_from_lines
from codeintel.analytics.evidence import EvidenceCollector
from codeintel.config import DataModelsStepConfig
from codeintel.ingestion.ast_utils import parse_python_module
from codeintel.ingestion.paths import normalize_rel_path, relpath_to_module
from codeintel.storage.gateway import DuckDBConnection, StorageGateway
from codeintel.storage.sql_helpers import ensure_schema

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
    lineno: int | None


@dataclass(frozen=True)
class RelationshipHint:
    """Relationship clue captured from field declarations."""

    field_name: str
    target: str
    multiplicity: str
    kind: str
    via: str
    lineno: int | None


@dataclass(frozen=True)
class RelationshipSpec:
    """Resolved relationship between two models."""

    field_name: str | None
    target_model_id: str | None
    target_module: str | None
    target_model_name: str | None
    kind: str
    multiplicity: str | None
    via: str | None
    rel_path: str
    lineno: int | None
    evidence: list[dict[str, object]] = dataclass_field(default_factory=list)


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
    import_context: ImportContext
    fields: list[FieldSpec]
    relationships: list[RelationshipSpec]
    relationship_hints: list[RelationshipHint]
    doc_short: str | None
    doc_long: str | None
    lines: list[str]


@dataclass(frozen=True)
class ImportContext:
    """Import aliases used to resolve relationship targets."""

    module: str
    name_to_module: dict[str, str]
    module_aliases: dict[str, str]


@dataclass(frozen=True)
class ModelLookup:
    """Index of parsed models for relationship resolution."""

    by_module_and_name: dict[tuple[str, str], ModelRecord]
    by_name: dict[str, list[ModelRecord]]
    by_name_lower: dict[str, list[ModelRecord]]


_safe_unparse = safe_unparse
_literal_value = literal_value
_call_name = call_name


def _annotation_to_str(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    text = safe_unparse(node)
    if text:
        return text
    if isinstance(node, ast.Name):
        return node.id
    return node.__class__.__name__


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


def _resolve_relative_module(current_module: str, level: int, module: str | None) -> str:
    parts = current_module.split(".")
    if parts:
        parts = parts[:-1]
    hops = max(level - 1, 0)
    if hops:
        parts = parts[:-hops]
    base = ".".join(parts)
    if module:
        return f"{base}.{module}" if base else module
    return base


def _build_import_context(module: str, tree: ast.AST) -> ImportContext:
    name_to_module: dict[str, str] = {}
    module_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            resolved = _resolve_relative_module(module, node.level, node.module)
            for alias in node.names:
                target = alias.asname or alias.name
                name_to_module[target] = resolved
        if isinstance(node, ast.Import):
            for alias in node.names:
                target = alias.asname or alias.name
                module_aliases[target] = alias.name
    return ImportContext(
        module=module, name_to_module=name_to_module, module_aliases=module_aliases
    )


def _attr_to_str(node: ast.Attribute) -> str | None:
    parts: list[str] = []
    current: ast.AST | None = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    else:
        return None
    return ".".join(reversed(parts))


def _symbol_candidates_from_type(type_text: str | None) -> list[str]:
    if not type_text:
        return []
    try:
        expr = ast.parse(type_text, mode="eval").body
    except SyntaxError:
        return []
    symbols: list[str] = []
    _collect_type_symbols(expr, symbols)
    return symbols


def _collect_type_symbols(node: ast.AST, symbols: list[str]) -> None:
    if isinstance(node, ast.Name):
        symbols.append(node.id)
        return
    if isinstance(node, ast.Attribute):
        attr_text = _attr_to_str(node)
        if attr_text:
            symbols.append(attr_text)
        return
    if isinstance(node, ast.Subscript):
        _collect_type_symbols(node.value, symbols)
        if isinstance(node.slice, ast.AST):
            _collect_type_symbols(node.slice, symbols)
        return
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        _collect_type_symbols(node.left, symbols)
        _collect_type_symbols(node.right, symbols)
        return
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        for elt in node.elts:
            _collect_type_symbols(elt, symbols)
        return
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        symbols.extend(_symbol_candidates_from_type(node.value))


def _candidate_qualified_symbols(symbol: str, ctx: ImportContext) -> list[tuple[str | None, str]]:
    if not symbol:
        return []
    parts = symbol.split(".")
    if len(parts) == 1:
        module = ctx.name_to_module.get(symbol)
        return [(module, symbol), (ctx.module, symbol), (None, symbol)]
    head, *rest = parts
    tail_name = rest[-1]
    module_tail = ".".join(rest[:-1]) if len(rest) > 1 else ""
    base_module = ctx.module_aliases.get(head) or ctx.name_to_module.get(head)
    guesses: list[tuple[str | None, str]] = []
    if base_module:
        module_guess = base_module
        if module_tail:
            module_guess = f"{base_module}.{module_tail}"
        guesses.append((module_guess, tail_name))
    module_guess = ".".join(parts[:-1])
    guesses.append((module_guess, tail_name))
    guesses.append((ctx.module, tail_name))
    guesses.append((None, tail_name))
    deduped: list[tuple[str | None, str]] = []
    seen: set[tuple[str | None, str]] = set()
    for guess in guesses:
        if guess not in seen:
            deduped.append(guess)
            seen.add(guess)
    return deduped


def _build_model_lookup(models: Sequence[ModelRecord]) -> ModelLookup:
    by_module_and_name: dict[tuple[str, str], ModelRecord] = {}
    by_name: dict[str, list[ModelRecord]] = {}
    by_name_lower: dict[str, list[ModelRecord]] = {}
    for model in models:
        key = (model.module, model.model_name)
        by_module_and_name[key] = model
        by_name.setdefault(model.model_name, []).append(model)
        by_name_lower.setdefault(model.model_name.lower(), []).append(model)
    return ModelLookup(
        by_module_and_name=by_module_and_name,
        by_name=by_name,
        by_name_lower=by_name_lower,
    )


def _resolve_target_model(
    symbol: str,
    ctx: ImportContext,
    lookup: ModelLookup,
) -> ModelRecord | None:
    for module_guess, name in _candidate_qualified_symbols(symbol, ctx):
        if module_guess is not None and (module_guess, name) in lookup.by_module_and_name:
            return lookup.by_module_and_name[module_guess, name]
        lower_name = name.lower()
        if module_guess is None:
            candidates = lookup.by_name.get(name) or lookup.by_name_lower.get(lower_name) or []
            if len(candidates) == 1:
                return candidates[0]
    return None


def _resolve_target_from_text(
    type_text: str | None,
    ctx: ImportContext,
    lookup: ModelLookup,
) -> tuple[ModelRecord, str] | None:
    for symbol in _symbol_candidates_from_type(type_text):
        target = _resolve_target_model(symbol, ctx, lookup)
        if target is not None:
            return target, symbol
    return None


def _multiplicity_from_type(type_text: str | None) -> str:
    if not type_text:
        return "one"
    lowered = type_text.lower()
    for marker in ("list", "set", "tuple", "sequence"):
        if marker in lowered:
            return "many"
    return "one"


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


def _relationship_hints_from_call(field_name: str, value: ast.Call) -> list[RelationshipHint]:
    func_name = (_call_name(value.func) or "").lower()
    hints: list[RelationshipHint] = []
    target_value: str | None = None
    multiplicity = "many"
    lineno = getattr(value, "lineno", None)
    if "relationship" in func_name:
        if value.args:
            target_value = str(_literal_value(value.args[0]))
        uselist_kw = next((kw for kw in value.keywords if kw.arg == "uselist"), None)
        if uselist_kw is not None:
            use_list_val = _literal_value(uselist_kw.value)
            if isinstance(use_list_val, bool) and not use_list_val:
                multiplicity = "one"
        hints.append(
            RelationshipHint(
                field_name=field_name,
                target=target_value or "",
                multiplicity=multiplicity,
                kind="relationship",
                via="sqlalchemy_relationship",
                lineno=lineno,
            ),
        )
    fk_target = _foreign_key_target(list(value.args), value.keywords)
    if fk_target:
        hints.append(
            RelationshipHint(
                field_name=field_name,
                target=fk_target,
                multiplicity="many",
                kind="foreign_key",
                via="sqlalchemy_column",
                lineno=lineno,
            ),
        )
    if "foreignkey" in func_name:
        if value.args:
            target_value = str(_literal_value(value.args[0]))
        hints.append(
            RelationshipHint(
                field_name=field_name,
                target=target_value or "",
                multiplicity="many",
                kind="foreign_key",
                via="django_field",
                lineno=lineno,
            ),
        )
    if any(name in func_name for name in ("onetoonefield", "manytomanyfield")):
        if value.args:
            target_value = str(_literal_value(value.args[0]))
        multiplicity = "one" if "onetoonefield" in func_name else "many"
        hints.append(
            RelationshipHint(
                field_name=field_name,
                target=target_value or "",
                multiplicity=multiplicity,
                kind="relationship",
                via="django_field",
                lineno=lineno,
            ),
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
        lineno=getattr(stmt, "lineno", None),
    )


def _field_spec_from_call(
    field_name: str,
    value: ast.Call,
    *,
    model_kind: str,
) -> tuple[FieldSpec | None, list[RelationshipHint]]:
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
            lineno=getattr(value, "lineno", None),
        )
        return field_spec, relationship_hints
    return None, relationship_hints


def _build_field_specs(
    node: ast.ClassDef,
    *,
    model_kind: str,
) -> tuple[list[FieldSpec], list[RelationshipHint]]:
    fields: list[FieldSpec] = []
    rel_hints: list[RelationshipHint] = []
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


def _load_class_metadata(con: DuckDBConnection, repo: str, commit: str) -> list[ClassMeta]:
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
    con: DuckDBConnection,
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
    cfg: DataModelsStepConfig,
) -> list[ModelRecord]:
    parsed = parse_python_module(abs_path)
    if parsed is None:
        log.debug("Skipping %s; unable to parse module", abs_path)
        return []
    lines, tree = parsed
    meta_by_line = {meta.start_line: meta for meta in metas}
    module_name = metas[0].module if metas else relpath_to_module(rel_path)
    import_ctx = _build_import_context(module_name, tree)
    models: list[ModelRecord] = []
    for cls_node in _all_class_defs(tree):
        meta = _match_class_meta(meta_by_line, cls_node)
        if meta is None:
            continue
        decorators = _class_decorators(cls_node)
        base_classes = _base_classes(cls_node)
        model_kind = _class_kind(decorators, [base["qualname"] for base in base_classes])
        fields, rel_hints = _build_field_specs(cls_node, model_kind=model_kind)
        doc_pair = docstrings.get((rel_path, meta.qualname), (None, None))
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
                import_context=import_ctx,
                fields=fields,
                relationships=[],
                relationship_hints=rel_hints,
                doc_short=doc_pair[0],
                doc_long=doc_pair[1],
                lines=lines,
            )
        )
    return models


def _relationships_for_model(
    model: ModelRecord,
    *,
    lookup: ModelLookup,
) -> list[RelationshipSpec]:
    relationships: list[RelationshipSpec] = []
    for field in model.fields:
        resolved = _resolve_target_from_text(field.type, model.import_context, lookup)
        if resolved is None:
            continue
        target, symbol = resolved
        evidence = EvidenceCollector()
        evidence.add_sample(
            path=model.rel_path,
            line_span=(field.lineno, field.lineno),
            snippet=snippet_from_lines(model.lines, field.lineno, field.lineno),
            details={
                "type_hint": field.type or "",
                "resolved_symbol": symbol,
                "via": "annotation",
            },
            tags=("relationship",),
        )
        relationships.append(
            RelationshipSpec(
                field_name=field.name,
                target_model_id=target.model_id,
                target_module=target.module,
                target_model_name=target.model_name,
                kind="reference",
                multiplicity=_multiplicity_from_type(field.type),
                via="annotation",
                rel_path=model.rel_path,
                lineno=field.lineno,
                evidence=evidence.to_dicts(),
            )
        )
    for hint in model.relationship_hints:
        resolved = _resolve_target_from_text(hint.target, model.import_context, lookup)
        if resolved is None:
            continue
        target, symbol = resolved
        evidence = EvidenceCollector()
        evidence.add_sample(
            path=model.rel_path,
            line_span=(hint.lineno, hint.lineno),
            snippet=snippet_from_lines(model.lines, hint.lineno, hint.lineno),
            details={
                "target_text": hint.target,
                "resolved_symbol": symbol,
                "via": hint.via,
            },
            tags=("relationship_hint",),
        )
        relationships.append(
            RelationshipSpec(
                field_name=hint.field_name,
                target_model_id=target.model_id,
                target_module=target.module,
                target_model_name=target.model_name,
                kind=hint.kind,
                multiplicity=hint.multiplicity,
                via=hint.via,
                rel_path=model.rel_path,
                lineno=hint.lineno,
                evidence=evidence.to_dicts(),
            )
        )
    return relationships


def _attach_relationships(models: list[ModelRecord]) -> None:
    if not models:
        return
    lookup = _build_model_lookup(models)
    for model in models:
        model.relationships = _relationships_for_model(
            model,
            lookup=lookup,
        )


def _persist_models(
    con: DuckDBConnection,
    cfg: DataModelsStepConfig,
    models: list[ModelRecord],
    now: datetime,
) -> None:
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
                        "lineno": field.lineno,
                    }
                    for field in model.fields
                ]
            ),
            json.dumps(
                [
                    {
                        "field": rel.field_name,
                        "target_model_id": rel.target_model_id,
                        "target_model_name": rel.target_model_name,
                        "target_module": rel.target_module,
                        "multiplicity": rel.multiplicity,
                        "kind": rel.kind,
                        "via": rel.via,
                        "evidence": rel.evidence,
                        "rel_path": rel.rel_path,
                        "lineno": rel.lineno,
                    }
                    for rel in model.relationships
                ]
            ),
            model.doc_short,
            model.doc_long,
            now,
        )
        for model in models
    ]
    field_rows: list[tuple[object, ...]] = []
    relationship_rows: list[tuple[object, ...]] = []
    for model in models:
        field_rows.extend(
            [
                (
                    cfg.repo,
                    cfg.commit,
                    model.model_id,
                    field_spec.name,
                    field_spec.type,
                    field_spec.required,
                    field_spec.has_default,
                    field_spec.default_expr,
                    json.dumps(field_spec.constraints),
                    field_spec.source,
                    model.rel_path,
                    field_spec.lineno,
                    now,
                )
                for field_spec in model.fields
            ]
        )
        relationship_rows.extend(
            [
                (
                    cfg.repo,
                    cfg.commit,
                    model.model_id,
                    rel.target_model_id,
                    rel.target_module,
                    rel.target_model_name,
                    rel.field_name,
                    rel.kind,
                    rel.multiplicity,
                    rel.via,
                    json.dumps(rel.evidence) if rel.evidence else None,
                    rel.rel_path,
                    rel.lineno,
                    now,
                )
                for rel in model.relationships
            ]
        )

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
    if field_rows:
        con.executemany(
            """
            INSERT INTO analytics.data_model_fields (
                repo, commit, model_id, field_name, field_type, required,
                has_default, default_expr, constraints_json, source, rel_path,
                lineno, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            field_rows,
        )
    if relationship_rows:
        con.executemany(
            """
            INSERT INTO analytics.data_model_relationships (
                repo, commit, source_model_id, target_model_id, target_module,
                target_model_name, field_name, relationship_kind, multiplicity,
                via, evidence_json, rel_path, lineno, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            relationship_rows,
        )


def compute_data_models(
    gateway: StorageGateway,
    cfg: DataModelsStepConfig,
) -> None:
    """
    Populate analytics.data_models with extracted model definitions.

    Parameters
    ----------
    gateway
        Storage gateway scoped to the target repository.
    cfg
        Data model extraction configuration.
    """
    con = gateway.con
    ensure_schema(con, "analytics.data_models")
    ensure_schema(con, "analytics.data_model_fields")
    ensure_schema(con, "analytics.data_model_relationships")
    con.execute(
        "DELETE FROM analytics.data_models WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    con.execute(
        "DELETE FROM analytics.data_model_fields WHERE repo = ? AND commit = ?",
        [cfg.repo, cfg.commit],
    )
    con.execute(
        "DELETE FROM analytics.data_model_relationships WHERE repo = ? AND commit = ?",
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
    _persist_models(con, cfg, models, datetime.now(tz=UTC))
    log.info("data_models populated: %d models for %s@%s", len(models), cfg.repo, cfg.commit)
