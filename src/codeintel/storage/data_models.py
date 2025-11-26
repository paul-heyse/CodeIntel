"""Typed accessors for data model tables and docs views."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal

from codeintel.storage.gateway import StorageGateway


def _as_int(value: Decimal | int | None) -> int | None:
    if value is None:
        return None
    return int(value)


def _decode_list(value: object) -> list[object]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return []


def _decode_dict(value: object) -> dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return value
    return {}


def _decode_json(value: object) -> object:
    if value is None:
        return []
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return []
    return []


def _decode_base_classes(value: object) -> list[dict[str, str]]:
    raw = _decode_json(value)
    if not isinstance(raw, list):
        return []
    base_classes: list[dict[str, str]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", ""))
        qualname = str(item.get("qualname", ""))
        if not name and not qualname:
            continue
        base_classes.append({"name": name, "qualname": qualname})
    return base_classes


def _decode_constraints(value: object) -> dict[str, object]:
    raw = _decode_json(value)
    return raw if isinstance(raw, dict) else {}


def _decode_evidence(value: object) -> dict[str, object]:
    raw = _decode_json(value)
    return raw if isinstance(raw, dict) else {}


def _normalize_created_at(value: object, default: datetime) -> datetime:
    if isinstance(value, datetime):
        return value
    return default


@dataclass(frozen=True)
class DataModelRow:
    """Base metadata for a detected data model."""

    repo: str
    commit: str
    model_id: str
    goid_h128: int | None
    model_name: str
    module: str
    rel_path: str
    model_kind: str
    base_classes: list[dict[str, str]]
    doc_short: str | None
    doc_long: str | None
    created_at: datetime


@dataclass(frozen=True)
class DataModelFieldRow:
    """Normalized data model field definition."""

    repo: str
    commit: str
    model_id: str
    name: str
    field_type: str | None
    required: bool
    has_default: bool
    default_expr: str | None
    constraints: dict[str, object]
    source: str
    rel_path: str
    lineno: int | None
    created_at: datetime


@dataclass(frozen=True)
class DataModelRelationshipRow:
    """Normalized relationship between two data models."""

    repo: str
    commit: str
    source_model_id: str
    target_model_id: str
    target_module: str | None
    target_model_name: str | None
    field_name: str | None
    relationship_kind: str
    multiplicity: str | None
    via: str | None
    evidence: dict[str, object]
    rel_path: str
    lineno: int | None
    created_at: datetime


@dataclass(frozen=True)
class NormalizedDataModel:
    """Fully expanded data model with normalized fields and relationships."""

    repo: str
    commit: str
    model_id: str
    goid_h128: int | None
    model_name: str
    module: str
    rel_path: str
    model_kind: str
    base_classes: list[dict[str, str]]
    fields: list[DataModelFieldRow]
    relationships: list[DataModelRelationshipRow]
    doc_short: str | None
    doc_long: str | None
    created_at: datetime


def fetch_models(gateway: StorageGateway, repo: str, commit: str) -> list[DataModelRow]:
    """
    Return data model rows for a repo/commit.

    Parameters
    ----------
    gateway
        Storage gateway bound to the target DuckDB database.
    repo
        Repository slug.
    commit
        Commit SHA.

    Returns
    -------
    list[DataModelRow]
        Parsed data model rows with base classes decoded.
    """
    rows = gateway.con.execute(
        """
        SELECT
            repo, commit, model_id, goid_h128, model_name, module, rel_path, model_kind,
            base_classes_json, doc_short, doc_long, created_at
        FROM analytics.data_models
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    result: list[DataModelRow] = []
    for (
        row_repo,
        row_commit,
        model_id,
        goid_h128,
        model_name,
        module,
        rel_path,
        model_kind,
        base_classes_json,
        doc_short,
        doc_long,
        created_at,
    ) in rows:
        result.append(
            DataModelRow(
                repo=str(row_repo),
                commit=str(row_commit),
                model_id=str(model_id),
                goid_h128=_as_int(goid_h128),
                model_name=str(model_name),
                module=str(module),
                rel_path=str(rel_path),
                model_kind=str(model_kind),
                base_classes=_decode_base_classes(base_classes_json),
                doc_short=str(doc_short) if doc_short is not None else None,
                doc_long=str(doc_long) if doc_long is not None else None,
                created_at=created_at,
            )
        )
    return result


def fetch_fields(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    model_ids: Sequence[str] | None = None,
) -> list[DataModelFieldRow]:
    """
    Return normalized field rows for the provided models.

    Parameters
    ----------
    gateway
        Storage gateway bound to the target DuckDB database.
    repo
        Repository slug.
    commit
        Commit SHA.
    model_ids
        Optional whitelist of model_ids to include.

    Returns
    -------
    list[DataModelFieldRow]
        Normalized fields for the requested models.
    """
    rows = gateway.con.execute(
        """
        SELECT
            repo, commit, model_id, field_name, field_type, required, has_default,
            default_expr, constraints_json, source, rel_path, lineno, created_at
        FROM analytics.data_model_fields
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    allowed = set(model_ids) if model_ids else None
    result: list[DataModelFieldRow] = []
    for (
        row_repo,
        row_commit,
        model_id,
        field_name,
        field_type,
        required,
        has_default,
        default_expr,
        constraints_json,
        source,
        rel_path,
        lineno,
        created_at,
    ) in rows:
        if allowed is not None and model_id not in allowed:
            continue
        result.append(
            DataModelFieldRow(
                repo=str(row_repo),
                commit=str(row_commit),
                model_id=str(model_id),
                name=str(field_name),
                field_type=str(field_type) if field_type is not None else None,
                required=bool(required),
                has_default=bool(has_default),
                default_expr=str(default_expr) if default_expr is not None else None,
                constraints=_decode_constraints(constraints_json),
                source=str(source),
                rel_path=str(rel_path),
                lineno=int(lineno) if lineno is not None else None,
                created_at=created_at,
            )
        )
    return result


def fetch_relationships(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    model_ids: Sequence[str] | None = None,
) -> list[DataModelRelationshipRow]:
    """
    Return normalized relationships for the provided models.

    Parameters
    ----------
    gateway
        Storage gateway bound to the target DuckDB database.
    repo
        Repository slug.
    commit
        Commit SHA.
    model_ids
        Optional whitelist of source model_ids to include.

    Returns
    -------
    list[DataModelRelationshipRow]
        Normalized relationships for the requested models.
    """
    rows = gateway.con.execute(
        """
        SELECT
            repo, commit, source_model_id, target_model_id, target_module,
            target_model_name, field_name, relationship_kind, multiplicity, via,
            evidence_json, rel_path, lineno, created_at
        FROM analytics.data_model_relationships
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    allowed = set(model_ids) if model_ids else None
    result: list[DataModelRelationshipRow] = []
    for (
        row_repo,
        row_commit,
        source_model_id,
        target_model_id,
        target_module,
        target_model_name,
        field_name,
        relationship_kind,
        multiplicity,
        via,
        evidence_json,
        rel_path,
        lineno,
        created_at,
    ) in rows:
        if allowed is not None and source_model_id not in allowed:
            continue
        result.append(
            DataModelRelationshipRow(
                repo=str(row_repo),
                commit=str(row_commit),
                source_model_id=str(source_model_id),
                target_model_id=str(target_model_id),
                target_module=str(target_module) if target_module is not None else None,
                target_model_name=str(target_model_name) if target_model_name is not None else None,
                field_name=str(field_name) if field_name is not None else None,
                relationship_kind=str(relationship_kind),
                multiplicity=str(multiplicity) if multiplicity is not None else None,
                via=str(via) if via is not None else None,
                evidence=_decode_evidence(evidence_json),
                rel_path=str(rel_path),
                lineno=int(lineno) if lineno is not None else None,
                created_at=created_at,
            )
        )
    return result


def fetch_models_normalized(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    *,
    model_ids: Sequence[str] | None = None,
) -> list[NormalizedDataModel]:
    """
    Return normalized data models, including fields and relationships.

    Parameters
    ----------
    gateway
        Storage gateway bound to the target DuckDB database.
    repo
        Repository slug.
    commit
        Commit SHA.
    model_ids
        Optional whitelist of model_ids to include.

    Returns
    -------
    list[NormalizedDataModel]
        Normalized models sourced from docs.v_data_models_normalized.
    """
    allowed = set(model_ids) if model_ids else None
    return _fetch_models_from_view(gateway, repo, commit, allowed)


def _decode_field_structs(
    fields: object,
    *,
    repo: str,
    commit: str,
    model_id: str,
    default_created_at: datetime,
) -> list[DataModelFieldRow]:
    decoded = _decode_json(fields)
    if not isinstance(decoded, list):
        return []
    parsed: list[DataModelFieldRow] = []
    for item in decoded:
        if not isinstance(item, dict):
            continue
        parsed.append(
            DataModelFieldRow(
                repo=repo,
                commit=commit,
                model_id=model_id,
                name=str(item.get("field_name") or item.get("name") or ""),
                field_type=str(item.get("field_type") or item.get("type"))
                if item.get("field_type") is not None or item.get("type") is not None
                else None,
                required=bool(item.get("required", False)),
                has_default=bool(item.get("has_default", False)),
                default_expr=str(item.get("default_expr"))
                if item.get("default_expr") is not None
                else None,
                constraints=_decode_constraints(item.get("constraints")),
                source=str(item.get("source") or ""),
                rel_path=str(item.get("rel_path") or ""),
                lineno=int(item["lineno"])
                if "lineno" in item and item["lineno"] is not None
                else None,
                created_at=_normalize_created_at(item.get("created_at"), default_created_at),
            )
        )
    return parsed


def _decode_relationship_structs(
    relationships: object,
    *,
    repo: str,
    commit: str,
    model_id: str,
    default_created_at: datetime,
) -> list[DataModelRelationshipRow]:
    decoded = _decode_json(relationships)
    if not isinstance(decoded, list):
        return []
    parsed: list[DataModelRelationshipRow] = []
    for item in decoded:
        if not isinstance(item, dict):
            continue
        parsed.append(
            DataModelRelationshipRow(
                repo=repo,
                commit=commit,
                source_model_id=model_id,
                target_model_id=str(item.get("target_model_id") or ""),
                target_module=str(item.get("target_module"))
                if item.get("target_module") is not None
                else None,
                target_model_name=str(item.get("target_model_name"))
                if item.get("target_model_name") is not None
                else None,
                field_name=str(item.get("field") or item.get("field_name") or ""),
                relationship_kind=str(item.get("kind") or item.get("relationship_kind") or ""),
                multiplicity=str(item.get("multiplicity"))
                if item.get("multiplicity") is not None
                else None,
                via=str(item.get("via")) if item.get("via") is not None else None,
                evidence=_decode_evidence(item.get("evidence")),
                rel_path=str(item.get("rel_path") or ""),
                lineno=int(item["lineno"])
                if "lineno" in item and item["lineno"] is not None
                else None,
                created_at=_normalize_created_at(item.get("created_at"), default_created_at),
            )
        )
    return parsed


def _fetch_models_from_view(
    gateway: StorageGateway,
    repo: str,
    commit: str,
    allowed: set[str] | None,
) -> list[NormalizedDataModel]:
    rows = gateway.con.execute(
        """
        SELECT
            repo, commit, model_id, goid_h128, model_name, module, rel_path, model_kind,
            base_classes_json, fields, relationships, doc_short, doc_long, created_at
        FROM docs.v_data_models_normalized
        WHERE repo = ? AND commit = ?
        """,
        [repo, commit],
    ).fetchall()
    result: list[NormalizedDataModel] = []
    for (
        row_repo,
        row_commit,
        model_id,
        goid_h128,
        model_name,
        module,
        rel_path,
        model_kind,
        base_classes_json,
        fields,
        relationships,
        doc_short,
        doc_long,
        created_at,
    ) in rows:
        if allowed is not None and model_id not in allowed:
            continue
        field_rows = _decode_field_structs(
            fields,
            repo=str(row_repo),
            commit=str(row_commit),
            model_id=str(model_id),
            default_created_at=created_at,
        )
        relationship_rows = _decode_relationship_structs(
            relationships,
            repo=str(row_repo),
            commit=str(row_commit),
            model_id=str(model_id),
            default_created_at=created_at,
        )
        result.append(
            NormalizedDataModel(
                repo=str(row_repo),
                commit=str(row_commit),
                model_id=str(model_id),
                goid_h128=_as_int(goid_h128),
                model_name=str(model_name),
                module=str(module),
                rel_path=str(rel_path),
                model_kind=str(model_kind),
                base_classes=_decode_base_classes(base_classes_json),
                fields=field_rows,
                relationships=relationship_rows,
                doc_short=str(doc_short) if doc_short is not None else None,
                doc_long=str(doc_long) if doc_long is not None else None,
                created_at=created_at,
            )
        )
    return result
