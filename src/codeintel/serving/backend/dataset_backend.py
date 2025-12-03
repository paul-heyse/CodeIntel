"""Dataset-focused backend for DuckDB-backed query services."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from codeintel.config.datasets import DatasetContract
from codeintel.serving import domain_models as dm
from codeintel.serving.backend.core import BackendContext, DuckDBConnection, DuckDBRepositories
from codeintel.serving.backend.domain_builders import DatasetSchemaInput, build_dataset_schema
from codeintel.serving.backend.pagination import clamp_limit, clamp_offset
from codeintel.serving.backend.query_api import DatasetQueriesApi
from codeintel.serving.mcp import errors
from codeintel.serving.mcp.models import DatasetSchemaColumn, DatasetSpecDescriptor
from codeintel.storage.datasets import dataset_for_name, list_dataset_specs, load_dataset_registry
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.repositories import DatasetReadRepository


def _fetch_duckdb_schema(con: DuckDBConnection, table_key: str) -> list[DatasetSchemaColumn]:
    """
    Return column descriptors for a DuckDB table/view.

    Parameters
    ----------
    con
        DuckDB connection.
    table_key
        Fully qualified table/view name.

    Returns
    -------
    list[DatasetSchemaColumn]
        Column descriptors derived from information_schema.
    """
    if "." not in table_key:
        return []
    schema_name, table_name = table_key.split(".", maxsplit=1)
    rows = con.execute(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = ? AND table_name = ?
        ORDER BY ordinal_position
        """,
        [schema_name, table_name],
    ).fetchall()
    return [
        DatasetSchemaColumn(
            name=str(col_name),
            type=str(col_type),
            nullable=str(nullable).upper() == "YES",
        )
        for col_name, col_type, nullable in rows
    ]


def _schema_path(schema_id: str) -> Path:
    """
    Return the on-disk path for a dataset JSON Schema identifier.

    Parameters
    ----------
    schema_id:
        Identifier without the ``.json`` suffix.

    Returns
    -------
    Path
        Filesystem path to the JSON Schema document.
    """
    root = Path("src/codeintel/config/schemas/export")
    return root / f"{schema_id}.json"


def _load_json_schema(ds: DatasetContract) -> dict[str, object] | None:
    """
    Load a JSON Schema document for a dataset if present on disk.

    Parameters
    ----------
    ds
        Dataset metadata entry from the registry.

    Returns
    -------
    dict[str, object] | None
        Parsed JSON Schema when available.
    """
    if ds.json_schema_id is None:
        return None
    schema_path = _schema_path(ds.json_schema_id)
    if not schema_path.exists():
        return None
    return json.loads(schema_path.read_text(encoding="utf-8"))


def _normalize_validation_profile(
    value: str | None,
) -> Literal["strict", "lenient"] | None:
    """
    Restrict validation profile to supported literals.

    Returns
    -------
    Literal["strict", "lenient"] | None
        Normalized validation profile when valid.
    """
    if value == "strict":
        return "strict"
    if value == "lenient":
        return "lenient"
    return None


@dataclass
class DatasetBackend(DatasetQueriesApi):
    """DuckDB-backed implementation of DatasetQueriesApi."""

    context: BackendContext
    repositories: DuckDBRepositories

    @property
    def datasets(self) -> DatasetReadRepository:
        """
        Return the lazily constructed dataset repository.

        Returns
        -------
        DatasetReadRepository
            Dataset repository scoped to the configured repo/commit.
        """
        return self.repositories.datasets

    @property
    def gateway(self) -> StorageGateway:
        """Return the backing storage gateway."""
        return self.context.gateway

    @property
    def con(self) -> DuckDBConnection:
        """Return the active DuckDB connection."""
        return self.context.gateway.con

    def list_datasets(self) -> list[dm.DatasetDescriptorDomain]:
        """
        List datasets from the registry.

        Returns
        -------
        list[dm.DatasetDescriptorDomain]
            Dataset descriptors derived from registered specs.
        """
        registry = load_dataset_registry(self.gateway.con)
        specs = list_dataset_specs(registry)
        descriptors: list[dm.DatasetDescriptorDomain] = []
        for spec in specs:
            capabilities = cast("Mapping[str, bool]", spec.get("capabilities", {}))
            descriptors.append(
                dm.DatasetDescriptorDomain(
                    name=cast("str", spec["name"]),
                    table=cast("str", spec.get("table") or spec.get("table_key")),
                    family=cast("str | None", spec.get("family")),
                    description=cast("str", spec.get("description") or ""),
                    owner=cast("str | None", spec.get("owner")),
                    schema_version=cast("str | None", spec.get("schema_version")),
                    stable_id=cast("str | None", spec.get("stable_id")),
                    is_docs_view=bool(
                        spec.get("is_view")
                        or capabilities.get("docs_view")
                        or capabilities.get("is_docs_view")
                    ),
                    is_read_only=bool(capabilities.get("read_only")),
                )
            )
        return descriptors

    def dataset_specs(self) -> list[DatasetSpecDescriptor]:
        """
        Return dataset specs ordered by name.

        Returns
        -------
        list[DatasetSpecDescriptor]
            Dataset specs sorted by name.
        """
        registry = load_dataset_registry(self.gateway.con)
        specs = list_dataset_specs(registry)
        sorted_specs = sorted(specs, key=lambda spec: cast("str", spec["name"]))
        results: list[DatasetSpecDescriptor] = []
        for spec in sorted_specs:
            normalized: dict[str, object] = dict(spec)
            if spec.get("schema_columns"):
                normalized["schema_columns"] = list(
                    cast("list[str]", spec.get("schema_columns", []))
                )
            else:
                ds = registry.by_name.get(cast("str", spec.get("name")))
                normalized["schema_columns"] = (
                    ds.schema.column_names() if ds is not None and ds.schema is not None else []
                )
            normalized["upstream_dependencies"] = list(
                cast("list[str]", spec.get("upstream_dependencies", []))
            )
            normalized["capabilities"] = dict(cast("dict[str, bool]", spec.get("capabilities", {})))
            normalized["validation_profile"] = _normalize_validation_profile(
                cast("str | None", spec.get("validation_profile"))
            )
            results.append(DatasetSpecDescriptor.model_validate(normalized))
        return results

    def read_dataset_rows(
        self,
        *,
        dataset_name: str,
        limit: int | None = None,
        offset: int = 0,
    ) -> Sequence[Mapping[str, object]]:
        """
        Read dataset rows from the configured dataset registry.

        Parameters
        ----------
        dataset_name
            Logical dataset identifier.
        limit
            Maximum number of rows to return; clamped to backend limits when provided.
        offset
            Starting offset for the dataset slice.

        Returns
        -------
        Sequence[Mapping[str, object]]
            Dataset rows as dictionaries.

        Raises
        ------
        errors.invalid_argument
            If the offset is invalid.
        errors.not_found
            If the dataset is unknown.
        """
        registry = load_dataset_registry(self.gateway.con)
        try:
            ds = dataset_for_name(registry, dataset_name)
        except KeyError as exc:
            message = f"Unknown dataset: {dataset_name}"
            raise errors.not_found(message) from exc
        offset_clamp = clamp_offset(offset)
        if offset_clamp.has_error:
            detail = offset_clamp.messages[0].detail if offset_clamp.messages else None
            detail_text = detail or "offset must be non-negative"
            raise errors.invalid_argument(detail_text)
        clamp = clamp_limit(
            limit,
            default=self.context.limits.default_limit,
            max_limit=self.context.limits.max_rows_per_call,
        )
        return self.datasets.read_dataset_rows(
            table_key=ds.table_key,
            limit=clamp.limit_or_default(self.context.limits.default_limit),
            offset=offset_clamp.applied,
        )

    def dataset_schema(self, *, dataset_name: str, sample_limit: int = 5) -> dm.DatasetSchema:
        """
        Return DuckDB schema, JSON Schema, and samples for a dataset.

        Parameters
        ----------
        dataset_name
            Logical dataset identifier.
        sample_limit
            Maximum number of sample rows to include.

        Returns
        -------
        dm.DatasetSchema
            Dataset schema payload including samples and metadata.

        Raises
        ------
        errors.not_found
            If the dataset is unknown.
        """
        registry = load_dataset_registry(self.gateway.con)
        try:
            ds = dataset_for_name(registry, dataset_name)
        except KeyError as exc:
            message = f"Unknown dataset: {dataset_name}"
            raise errors.not_found(message) from exc
        duckdb_schema = _fetch_duckdb_schema(self.gateway.con, ds.table_key)
        sample_rows = self.datasets.read_dataset_rows(
            table_key=ds.table_key,
            limit=sample_limit,
            offset=0,
        )
        schema_input = DatasetSchemaInput(
            dataset_name=dataset_name,
            table_key=ds.table_key,
            duckdb_schema=duckdb_schema,
            json_schema=_load_json_schema(ds),
            sample_rows=sample_rows,
            capabilities=ds.capabilities(),
            owner=ds.owner,
            freshness_sla=ds.freshness_sla,
            retention_policy=ds.retention_policy,
            schema_version=ds.schema_version,
            stable_id=ds.stable_id,
            validation_profile=_normalize_validation_profile(ds.validation_profile),
            meta=dm.ResponseMeta(),
        )
        return build_dataset_schema(schema_input)


__all__ = ["DatasetBackend"]
