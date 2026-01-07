"""Normalization utilities for ingestion Arrow outputs."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import TypedDict, Unpack

import pyarrow as pa

from codeintel.build.tabular.arrow_ops import (
    AlignmentReporter,
    align_table_to_contract,
    dedupe_table_for_table,
)
from codeintel.build.tabular.conversion import tabular_to_arrow_table
from codeintel.build.tabular.types import InferableTabularInput


class NormalizationOverrides(TypedDict, total=False):
    """Keyword overrides for ingestion normalization."""

    target_name: str | None
    add_missing: bool
    keep_extras: bool | None
    reporter: AlignmentReporter | None


@dataclass(frozen=True, slots=True)
class NormalizationOptions:
    """Options for ingestion normalization."""

    target_name: str | None = None
    add_missing: bool = True
    keep_extras: bool | None = None
    reporter: AlignmentReporter | None = None


def _merge_normalization_options(
    options: NormalizationOptions | None,
    overrides: NormalizationOverrides,
) -> NormalizationOptions:
    resolved = options or NormalizationOptions()
    if overrides:
        return replace(resolved, **overrides)
    return resolved


def normalize_ingest_frame(
    frame: InferableTabularInput | None,
    *,
    table_key: str,
    options: NormalizationOptions | None = None,
    **overrides: Unpack[NormalizationOverrides],
) -> pa.Table | None:
    """Normalize ingestion frames for schema alignment and deduping.

    Parameters
    ----------
    frame
        Tabular input to normalize (None means skip).
    table_key
        Target table key for schema alignment.
    options
        Optional normalization options for alignment and deduping.
    overrides
        Keyword overrides for normalization options.

    Returns
    -------
    pa.Table | None
        Normalized Arrow table or None if input is None.
    """
    if frame is None:
        return None
    table = tabular_to_arrow_table(frame)
    if table.num_rows == 0:
        return pa.Table.from_batches([], schema=table.schema)
    resolved_options = _merge_normalization_options(options, overrides)
    extras_policy = None
    if resolved_options.keep_extras is True:
        extras_policy = "retain"
    elif resolved_options.keep_extras is False:
        extras_policy = "drop"
    aligned = (
        align_table_to_contract(
            table_key,
            table,
            target_name=resolved_options.target_name,
            extras_policy=extras_policy,
            reporter=resolved_options.reporter,
        )
        if resolved_options.add_missing or extras_policy is not None
        else table
    )
    return dedupe_table_for_table(table_key, aligned)


__all__ = ["normalize_ingest_frame"]
