"""Factories for dataset descriptor payloads used in tests."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from codeintel.serving import domain_models as dm
from codeintel.serving.mcp.models import DatasetDescriptor, DatasetSpecDescriptor


@dataclass
class DescriptorOptions:
    """Optional fields for dataset descriptors."""

    family: str | None = None
    owner: str | None = None
    schema_version: str | None = None
    stable_id: str | None = None
    is_docs_view: bool = False
    is_read_only: bool = False
    meta: dm.ResponseMeta | None = None
    applied_limit: int | None = None


def make_descriptor(
    *,
    name: str,
    table: str,
    description: str,
    options: DescriptorOptions | None = None,
) -> tuple[dm.DatasetDescriptorDomain, DatasetDescriptor, dict[str, object], dm.ResponseMeta]:
    """
    Build transport/domain variants for a dataset descriptor plus metadata.

    Parameters
    ----------
    name
        Dataset name.
    table
        Physical table or view backing the dataset.
    description
        Human-readable description of the dataset.
    options
        Optional descriptor configuration (family, owner, flags, metadata).

    Returns
    -------
    tuple
        (domain, transport_model, payload_dict, meta) variants.
    """
    opts = options or DescriptorOptions()
    meta_obj = opts.meta or dm.ResponseMeta(applied_limit=opts.applied_limit)
    domain = dm.DatasetDescriptorDomain(
        name=name,
        table=table,
        description=description,
        family=opts.family,
        owner=opts.owner,
        schema_version=opts.schema_version,
        stable_id=opts.stable_id,
        is_docs_view=opts.is_docs_view,
        is_read_only=opts.is_read_only,
    )
    model = DatasetDescriptor.model_validate(asdict(domain))
    return domain, model, model.model_dump(), meta_obj


def dataset_descriptor_variants(
    *,
    name: str,
    table: str,
    description: str,
) -> tuple[dm.DatasetDescriptorDomain, DatasetDescriptor, dict[str, object]]:
    """
    Return domain, Pydantic model, and dict variants for normalization tests.

    Returns
    -------
    tuple
        (domain, pydantic_model, payload_dict) variants.
    """
    domain, model, payload, _meta = make_descriptor(
        name=name,
        table=table,
        description=description,
    )
    return domain, model, payload


@dataclass
class SpecOptions:
    """Optional fields for dataset spec descriptors."""

    family: str | None = None
    is_view: bool = True
    schema_columns: list[str] | None = None
    description: str | None = None
    owner: str | None = None
    has_row_binding: bool = True
    json_schema_id: str | None = None
    validation_profile: str | None = None
    stable_id: str | None = None
    schema_version: str | None = None
    capabilities: dict[str, bool] | None = None
    meta: dm.ResponseMeta | None = None
    applied_limit: int | None = None


def make_spec(
    *,
    name: str,
    table_key: str,
    options: SpecOptions | None = None,
) -> tuple[DatasetSpecDescriptor, dict[str, object], dm.ResponseMeta]:
    """
    Build transport variants for dataset specs with optional metadata.

    Parameters
    ----------
    name
        Dataset name.
    table_key
        Storage key for the dataset (table or view).
    options
        Optional spec configuration (schema, ownership, flags, metadata).

    Returns
    -------
    tuple
        (pydantic_model, payload_dict, meta) variants.
    """
    opts = SpecOptions(**asdict(options)) if options is not None else SpecOptions()
    payload = {
        "name": name,
        "table_key": table_key,
        "family": opts.family,
        "is_view": opts.is_view,
        "schema_columns": opts.schema_columns or [],
        "description": opts.description,
        "owner": opts.owner,
        "has_row_binding": opts.has_row_binding,
        "json_schema_id": opts.json_schema_id,
        "validation_profile": opts.validation_profile,
        "stable_id": opts.stable_id,
        "schema_version": opts.schema_version,
        "capabilities": opts.capabilities or {},
    }
    model = DatasetSpecDescriptor.model_validate(payload)
    meta_obj = opts.meta or dm.ResponseMeta(applied_limit=opts.applied_limit)
    return model, model.model_dump(), meta_obj


def dataset_spec_variants(
    *,
    name: str,
    table_key: str,
    options: SpecOptions | None = None,
) -> tuple[DatasetSpecDescriptor, dict[str, object]]:
    """
    Return DatasetSpecDescriptor model and dict payload variants.

    Returns
    -------
    tuple
        (pydantic_model, payload_dict) variants.
    """
    model, payload, _meta = make_spec(name=name, table_key=table_key, options=options)
    return model, payload


def sample_dataset_specs() -> list[DatasetSpecDescriptor]:
    """
    Build representative dataset specs for MCP tests (unicode/null/flags).

    Returns
    -------
    list[DatasetSpecDescriptor]
        Multiple specs covering docs views, read-only flags, and unicode names.
    """
    return [
        make_spec(
            name="Analytics Metrics Δ",
            table_key="analytics.fn_metrics",
            options=SpecOptions(
                family="analytics",
                is_view=False,
                description="Primary metrics table Δ",
                owner=None,
                schema_columns=["col1", "col2"],
                has_row_binding=True,
            ),
        )[0],
        make_spec(
            name="Docs View",
            table_key="docs.fn_metrics_δ",
            options=SpecOptions(
                family="docs",
                is_view=True,
                description="Docs view for metrics",
                owner="team-docs",
                schema_columns=["col1", "col2"],
                has_row_binding=False,
                schema_version=None,
                json_schema_id=None,
                validation_profile="strict",
            ),
        )[0],
    ]


__all__ = [
    "DescriptorOptions",
    "SpecOptions",
    "dataset_descriptor_variants",
    "dataset_spec_variants",
    "make_descriptor",
    "make_spec",
    "sample_dataset_specs",
]
