"""Shared Arrow dataset scanning helpers."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds

if TYPE_CHECKING:
    from pyarrow.dataset import Scanner


@dataclass(frozen=True, slots=True)
class DatasetScanOptions:
    """Options for Arrow dataset scanning."""

    batch_size: int
    batch_readahead: int | None = None
    fragment_readahead: int | None = None
    filter_expression: ds.Expression | None = None
    use_threads: bool | None = None
    memory_pool: pa.MemoryPool | None = None
    schema: pa.Schema | None = None
    columns: Sequence[str] | None = None
    unify_schemas: bool = False
    metrics_enabled: bool = False


def build_scanner(dataset: ds.Dataset, *, options: DatasetScanOptions) -> Scanner:
    """Build a dataset scanner using shared scan options.

    Returns
    -------
    Scanner
        Configured scanner for the dataset.
    """
    schema = _resolve_scan_schema(dataset, options)
    scan_kwargs = _build_scan_kwargs(options, schema)
    filter_expression = options.filter_expression
    if filter_expression is None:
        return _scanner_with_schema(dataset, scan_kwargs)

    fragments = _fragments_for_filter(dataset, filter_expression)
    resolved_schema = schema or dataset.schema
    fragment_scanner = _scanner_from_fragments(
        fragments,
        resolved_schema,
        scan_kwargs,
    )
    if fragment_scanner is not None:
        return fragment_scanner

    scan_kwargs["filter"] = filter_expression
    return _scanner_with_schema(dataset, scan_kwargs)


def unify_dataset_schema(dataset: ds.Dataset) -> pa.Schema | None:
    """Return a unified schema for a dataset when fragments diverge.

    Returns
    -------
    pa.Schema | None
        Unified schema if available; otherwise dataset schema.
    """
    fragments = _dataset_fragments(dataset)
    if fragments is None:
        return dataset.schema
    schemas: list[pa.Schema] = []
    for fragment in fragments:
        schema = getattr(fragment, "physical_schema", None)
        if isinstance(schema, pa.Schema):
            schemas.append(schema)
    if not schemas:
        return dataset.schema
    if len(schemas) == 1:
        return schemas[0]
    try:
        return pa.unify_schemas(schemas)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return dataset.schema


def _dataset_fragments(dataset: ds.Dataset) -> Iterable[ds.Fragment] | None:
    get_fragments = getattr(dataset, "get_fragments", None)
    if not callable(get_fragments):
        return None
    try:
        fragments = get_fragments()
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None
    if not isinstance(fragments, Iterable):
        return None
    return fragments


def _scanner_with_schema(dataset: ds.Dataset, scan_kwargs: dict[str, object]) -> Scanner:
    try:
        return dataset.scanner(**scan_kwargs)
    except TypeError:
        scan_kwargs.pop("schema", None)
        return dataset.scanner(**scan_kwargs)


def _resolve_scan_schema(
    dataset: ds.Dataset,
    options: DatasetScanOptions,
) -> pa.Schema | None:
    schema = options.schema
    if schema is None and options.unify_schemas:
        return unify_dataset_schema(dataset)
    return schema


def _build_scan_kwargs(
    options: DatasetScanOptions,
    schema: pa.Schema | None,
) -> dict[str, object]:
    scan_kwargs: dict[str, object] = {"batch_size": options.batch_size}
    if options.batch_readahead is not None:
        scan_kwargs["batch_readahead"] = options.batch_readahead
    if options.fragment_readahead is not None:
        scan_kwargs["fragment_readahead"] = options.fragment_readahead
    if options.use_threads is not None:
        scan_kwargs["use_threads"] = options.use_threads
    if options.memory_pool is not None:
        scan_kwargs["memory_pool"] = options.memory_pool
    if options.columns is not None:
        scan_kwargs["columns"] = list(options.columns)
    if schema is not None:
        scan_kwargs["schema"] = schema
    return scan_kwargs


def _scanner_from_fragments(
    fragments: tuple[ds.Fragment, ...] | None,
    schema: pa.Schema,
    scan_kwargs: dict[str, object],
) -> Scanner | None:
    if not fragments:
        return None
    from_fragments = getattr(ds.Scanner, "from_fragments", None)
    if callable(from_fragments):
        try:
            return from_fragments(fragments, schema=schema, **scan_kwargs)
        except (TypeError, ValueError, pa.ArrowInvalid):
            return None
    from_fragment = getattr(ds.Scanner, "from_fragment", None)
    if callable(from_fragment) and len(fragments) == 1:
        try:
            return from_fragment(fragments[0], schema=schema, **scan_kwargs)
        except (TypeError, ValueError, pa.ArrowInvalid):
            return None
    return None


def _fragments_for_filter(
    dataset: ds.Dataset,
    filter_expression: ds.Expression,
) -> tuple[ds.Fragment, ...] | None:
    get_fragments = getattr(dataset, "get_fragments", None)
    if not callable(get_fragments):
        return None
    try:
        fragments = get_fragments(filter=filter_expression)
        if not isinstance(fragments, Iterable):
            return None
        return tuple(fragments)
    except (TypeError, ValueError, pa.ArrowInvalid):
        return None


__all__ = ["DatasetScanOptions", "build_scanner", "unify_dataset_schema"]
