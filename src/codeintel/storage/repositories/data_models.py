"""Typed accessors for data model tables and docs views."""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, SupportsInt, cast

import pandas as pd

from codeintel.core.ibis_typing import and_predicates, isin_values
from codeintel.storage.gateway import ibis_facade
from codeintel.storage.helpers.json import decode_json, decode_json_dict

if TYPE_CHECKING:
    from collections.abc import Sequence

    from codeintel.storage.gateway import StorageGateway


_DEFAULT_CREATED_AT = datetime.fromtimestamp(0, tz=UTC)


def _as_int(value: object) -> int | None:
    result: int | None = None
    if value is None or isinstance(value, bool):
        result = None
    elif isinstance(value, int):
        result = value
    elif isinstance(value, float):
        result = int(value) if value.is_integer() else None
    elif isinstance(value, str):
        stripped = value.strip()
        if stripped and (stripped.isdigit() or (stripped[0] == "-" and stripped[1:].isdigit())):
            result = int(stripped)
    elif hasattr(value, "__int__"):
        with contextlib.suppress(TypeError, ValueError):
            result = int(cast("SupportsInt", value))
    return result


def _decode_base_classes(value: object) -> list[dict[str, str]]:
    raw = decode_json(value)
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
    tbl = ibis_facade.table(gateway, "analytics.data_models")
    expr = tbl.filter(and_predicates(tbl.repo == repo, tbl.commit == commit)).select(
        "repo",
        "commit",
        "model_id",
        "goid_h128",
        "model_name",
        "module",
        "rel_path",
        "model_kind",
        "base_classes_json",
        "doc_short",
        "doc_long",
        "created_at",
    )
    df = pd.DataFrame(expr.execute())

    result: list[DataModelRow] = []
    for record in df.to_dict(orient="records"):
        row: dict[str, object] = record
        created_at = _normalize_created_at(row["created_at"], default=_DEFAULT_CREATED_AT)
        result.append(
            DataModelRow(
                repo=str(row["repo"]),
                commit=str(row["commit"]),
                model_id=str(row["model_id"]),
                goid_h128=_as_int(row["goid_h128"]),
                model_name=str(row["model_name"]),
                module=str(row["module"]),
                rel_path=str(row["rel_path"]),
                model_kind=str(row["model_kind"]),
                base_classes=_decode_base_classes(row["base_classes_json"]),
                doc_short=str(row["doc_short"]) if row["doc_short"] is not None else None,
                doc_long=str(row["doc_long"]) if row["doc_long"] is not None else None,
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
    tbl = ibis_facade.table(gateway, "analytics.data_model_fields")
    expr = tbl.filter(and_predicates(tbl.repo == repo, tbl.commit == commit))

    if model_ids is not None:
        expr = expr.filter(isin_values(tbl.model_id, model_ids))

    expr = expr.select(
        "repo",
        "commit",
        "model_id",
        "field_name",
        "field_type",
        "required",
        "has_default",
        "default_expr",
        "constraints_json",
        "source",
        "rel_path",
        "lineno",
        "created_at",
    )
    df = pd.DataFrame(expr.execute())

    result: list[DataModelFieldRow] = []
    for record in df.to_dict(orient="records"):
        row: dict[str, object] = record
        created_at = _normalize_created_at(row["created_at"], default=_DEFAULT_CREATED_AT)
        result.append(
            DataModelFieldRow(
                repo=str(row["repo"]),
                commit=str(row["commit"]),
                model_id=str(row["model_id"]),
                name=str(row["field_name"]),
                field_type=str(row["field_type"]) if row["field_type"] is not None else None,
                required=bool(row["required"]),
                has_default=bool(row["has_default"]),
                default_expr=str(row["default_expr"]) if row["default_expr"] is not None else None,
                constraints=decode_json_dict(row["constraints_json"]),
                source=str(row["source"]),
                rel_path=str(row["rel_path"]),
                lineno=_as_int(row["lineno"]),
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
    tbl = ibis_facade.table(gateway, "analytics.data_model_relationships")
    expr = tbl.filter(and_predicates(tbl.repo == repo, tbl.commit == commit))

    if model_ids is not None:
        expr = expr.filter(isin_values(tbl.source_model_id, model_ids))

    expr = expr.select(
        "repo",
        "commit",
        "source_model_id",
        "target_model_id",
        "target_module",
        "target_model_name",
        "field_name",
        "relationship_kind",
        "multiplicity",
        "via",
        "evidence_json",
        "rel_path",
        "lineno",
        "created_at",
    )
    df = pd.DataFrame(expr.execute())

    result: list[DataModelRelationshipRow] = []
    for record in df.to_dict(orient="records"):
        row: dict[str, object] = record
        created_at = _normalize_created_at(row["created_at"], default=_DEFAULT_CREATED_AT)
        result.append(
            DataModelRelationshipRow(
                repo=str(row["repo"]),
                commit=str(row["commit"]),
                source_model_id=str(row["source_model_id"]),
                target_model_id=str(row["target_model_id"]),
                target_module=str(row["target_module"])
                if row["target_module"] is not None
                else None,
                target_model_name=str(row["target_model_name"])
                if row["target_model_name"] is not None
                else None,
                field_name=str(row["field_name"]) if row["field_name"] is not None else None,
                relationship_kind=str(row["relationship_kind"]),
                multiplicity=str(row["multiplicity"]) if row["multiplicity"] is not None else None,
                via=str(row["via"]) if row["via"] is not None else None,
                evidence=decode_json_dict(row["evidence_json"]),
                rel_path=str(row["rel_path"]),
                lineno=_as_int(row["lineno"]),
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
    decoded = decode_json(fields)
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
                constraints=decode_json_dict(item.get("constraints")),
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
    decoded = decode_json(relationships)
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
                evidence=decode_json_dict(item.get("evidence")),
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
    """
    Fetch normalized data models from the view using Ibis.

    Parameters
    ----------
    gateway
        Storage gateway bound to the target DuckDB database.
    repo
        Repository slug.
    commit
        Commit SHA.
    allowed
        Optional set of model_ids to include; None means all.

    Returns
    -------
    list[NormalizedDataModel]
        Normalized models with decoded fields and relationships.
    """
    tbl = ibis_facade.table(gateway, "docs.v_data_models_normalized")
    expr = tbl.filter(and_predicates(tbl.repo == repo, tbl.commit == commit))

    if allowed is not None:
        expr = expr.filter(isin_values(tbl.model_id, allowed))

    expr = expr.select(
        "repo",
        "commit",
        "model_id",
        "goid_h128",
        "model_name",
        "module",
        "rel_path",
        "model_kind",
        "base_classes_json",
        "fields",
        "relationships",
        "doc_short",
        "doc_long",
        "created_at",
    )
    df = pd.DataFrame(expr.execute())

    result: list[NormalizedDataModel] = []
    for record in df.to_dict(orient="records"):
        row: dict[str, object] = record
        created_at = _normalize_created_at(row["created_at"], default=_DEFAULT_CREATED_AT)
        field_rows = _decode_field_structs(
            row["fields"],
            repo=str(row["repo"]),
            commit=str(row["commit"]),
            model_id=str(row["model_id"]),
            default_created_at=created_at,
        )
        relationship_rows = _decode_relationship_structs(
            row["relationships"],
            repo=str(row["repo"]),
            commit=str(row["commit"]),
            model_id=str(row["model_id"]),
            default_created_at=created_at,
        )
        result.append(
            NormalizedDataModel(
                repo=str(row["repo"]),
                commit=str(row["commit"]),
                model_id=str(row["model_id"]),
                goid_h128=_as_int(row["goid_h128"]),
                model_name=str(row["model_name"]),
                module=str(row["module"]),
                rel_path=str(row["rel_path"]),
                model_kind=str(row["model_kind"]),
                base_classes=_decode_base_classes(row["base_classes_json"]),
                fields=field_rows,
                relationships=relationship_rows,
                doc_short=str(row["doc_short"]) if row["doc_short"] is not None else None,
                doc_long=str(row["doc_long"]) if row["doc_long"] is not None else None,
                created_at=created_at,
            )
        )
    return result
