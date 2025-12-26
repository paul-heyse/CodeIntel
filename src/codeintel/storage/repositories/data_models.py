"""Typed accessors for data model tables and docs views."""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import SupportsInt, cast

from codeintel.core.ibis_typing import and_predicates, isin_values
from codeintel.storage.helpers.json import decode_json, decode_json_dict
from codeintel.storage.repositories.base import BaseRepository

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


@dataclass(frozen=True)
class DataModelsRepository(BaseRepository):
    """Read-only access to data model metadata tables and normalized views."""

    def list_class_metadata_rows(self) -> list[dict[str, object]]:
        """List class metadata needed for data model extraction.

        Returns
        -------
        list[dict[str, object]]
            Raw metadata rows for class GOIDs and module mapping.
        """
        goids = self._ibis_table("core.goids")
        modules = self._ibis_table("core.modules")
        joined = goids.left_join(
            modules,
            [
                goids.rel_path == modules.path,
                goids.repo == modules.repo,
                goids.commit == modules.commit,
            ],
        )
        expr = joined.filter(
            and_predicates(
                goids.repo == self.repo,
                goids.commit == self.commit,
                goids.kind == "class",
            )
        ).select(
            goids.goid_h128.name("goid_h128"),
            goids.rel_path,
            goids.qualname,
            goids.start_line,
            goids.end_line,
            modules.module.name("module"),
        )
        return self._ibis_to_dicts(expr)

    def list_class_docstrings_rows(self) -> list[dict[str, object]]:
        """List docstrings needed for data model extraction.

        Returns
        -------
        list[dict[str, object]]
            Raw docstring rows keyed by path + qualname.
        """
        docstrings = self._ibis_table("core.docstrings")
        expr = docstrings.filter(
            and_predicates(
                docstrings.repo == self.repo,
                docstrings.commit == self.commit,
                docstrings.kind == "class",
            )
        ).select(
            docstrings.rel_path,
            docstrings.qualname,
            docstrings.short_desc,
            docstrings.long_desc,
        )
        return self._ibis_to_dicts(expr)

    def list_models(self) -> list[DataModelRow]:
        """List data model rows for the bound snapshot.

        Returns
        -------
        list[DataModelRow]
            Data model rows for the repository snapshot.
        """
        tbl = self._ibis_table("analytics.data_models")
        expr = tbl.select(
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
        rows = self._ibis_to_dicts(expr, table_key="analytics.data_models")

        result: list[DataModelRow] = []
        for row in rows:
            created_at = _normalize_created_at(row.get("created_at"), default=_DEFAULT_CREATED_AT)
            result.append(
                DataModelRow(
                    repo=str(row.get("repo") or self.repo),
                    commit=str(row.get("commit") or self.commit),
                    model_id=str(row.get("model_id") or ""),
                    goid_h128=_as_int(row.get("goid_h128")),
                    model_name=str(row.get("model_name") or ""),
                    module=str(row.get("module") or ""),
                    rel_path=str(row.get("rel_path") or ""),
                    model_kind=str(row.get("model_kind") or ""),
                    base_classes=_decode_base_classes(row.get("base_classes_json")),
                    doc_short=str(row["doc_short"]) if row.get("doc_short") is not None else None,
                    doc_long=str(row["doc_long"]) if row.get("doc_long") is not None else None,
                    created_at=created_at,
                )
            )
        return result

    def list_fields(self, *, model_ids: Sequence[str] | None = None) -> list[DataModelFieldRow]:
        """List normalized field rows for the requested model_ids (or all).

        Parameters
        ----------
        model_ids
            Optional filter restricting results to these model ids.

        Returns
        -------
        list[DataModelFieldRow]
            Field rows for the selected models.
        """
        tbl = self._ibis_table("analytics.data_model_fields")
        expr = tbl
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
        rows = self._ibis_to_dicts(expr, table_key="analytics.data_model_fields")

        result: list[DataModelFieldRow] = []
        for row in rows:
            created_at = _normalize_created_at(row.get("created_at"), default=_DEFAULT_CREATED_AT)
            result.append(
                DataModelFieldRow(
                    repo=str(row.get("repo") or self.repo),
                    commit=str(row.get("commit") or self.commit),
                    model_id=str(row.get("model_id") or ""),
                    name=str(row.get("field_name") or ""),
                    field_type=str(row["field_type"])
                    if row.get("field_type") is not None
                    else None,
                    required=bool(row.get("required", False)),
                    has_default=bool(row.get("has_default", False)),
                    default_expr=str(row["default_expr"])
                    if row.get("default_expr") is not None
                    else None,
                    constraints=decode_json_dict(row.get("constraints_json")),
                    source=str(row.get("source") or ""),
                    rel_path=str(row.get("rel_path") or ""),
                    lineno=_as_int(row.get("lineno")),
                    created_at=created_at,
                )
            )
        return result

    def list_relationships(
        self,
        *,
        model_ids: Sequence[str] | None = None,
    ) -> list[DataModelRelationshipRow]:
        """List normalized relationship rows for the requested model_ids (or all).

        Parameters
        ----------
        model_ids
            Optional filter restricting results to relationships sourced from these model ids.

        Returns
        -------
        list[DataModelRelationshipRow]
            Relationship rows for the selected models.
        """
        tbl = self._ibis_table("analytics.data_model_relationships")
        expr = tbl
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
        rows = self._ibis_to_dicts(expr, table_key="analytics.data_model_relationships")

        result: list[DataModelRelationshipRow] = []
        for row in rows:
            created_at = _normalize_created_at(row.get("created_at"), default=_DEFAULT_CREATED_AT)
            result.append(
                DataModelRelationshipRow(
                    repo=str(row.get("repo") or self.repo),
                    commit=str(row.get("commit") or self.commit),
                    source_model_id=str(row.get("source_model_id") or ""),
                    target_model_id=str(row.get("target_model_id") or ""),
                    target_module=str(row["target_module"])
                    if row.get("target_module") is not None
                    else None,
                    target_model_name=str(row["target_model_name"])
                    if row.get("target_model_name") is not None
                    else None,
                    field_name=str(row["field_name"])
                    if row.get("field_name") is not None
                    else None,
                    relationship_kind=str(row.get("relationship_kind") or ""),
                    multiplicity=str(row["multiplicity"])
                    if row.get("multiplicity") is not None
                    else None,
                    via=str(row["via"]) if row.get("via") is not None else None,
                    evidence=decode_json_dict(row.get("evidence_json")),
                    rel_path=str(row.get("rel_path") or ""),
                    lineno=_as_int(row.get("lineno")),
                    created_at=created_at,
                )
            )
        return result

    def list_models_normalized(
        self,
        *,
        model_ids: Sequence[str] | None = None,
    ) -> list[NormalizedDataModel]:
        """List normalized data models with decoded fields and relationships.

        Parameters
        ----------
        model_ids
            Optional filter restricting results to these model ids.

        Returns
        -------
        list[NormalizedDataModel]
            Fully decoded models with embedded field and relationship details.
        """
        allowed = set(model_ids) if model_ids else None
        tbl = self._ibis_table("docs.v_data_models_normalized")
        expr = tbl
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
        rows = self._ibis_to_dicts(expr, table_key="docs.v_data_models_normalized")

        result: list[NormalizedDataModel] = []
        for row in rows:
            created_at = _normalize_created_at(row.get("created_at"), default=_DEFAULT_CREATED_AT)
            repo = str(row.get("repo") or self.repo)
            commit = str(row.get("commit") or self.commit)
            model_id = str(row.get("model_id") or "")
            result.append(
                NormalizedDataModel(
                    repo=repo,
                    commit=commit,
                    model_id=model_id,
                    goid_h128=_as_int(row.get("goid_h128")),
                    model_name=str(row.get("model_name") or ""),
                    module=str(row.get("module") or ""),
                    rel_path=str(row.get("rel_path") or ""),
                    model_kind=str(row.get("model_kind") or ""),
                    base_classes=_decode_base_classes(row.get("base_classes_json")),
                    fields=_decode_field_structs(
                        row.get("fields"),
                        repo=repo,
                        commit=commit,
                        model_id=model_id,
                        default_created_at=created_at,
                    ),
                    relationships=_decode_relationship_structs(
                        row.get("relationships"),
                        repo=repo,
                        commit=commit,
                        model_id=model_id,
                        default_created_at=created_at,
                    ),
                    doc_short=str(row["doc_short"]) if row.get("doc_short") is not None else None,
                    doc_long=str(row["doc_long"]) if row.get("doc_long") is not None else None,
                    created_at=created_at,
                )
            )
        return result


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
