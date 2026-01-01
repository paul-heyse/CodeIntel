"""Typed query-result coercion helpers."""

from __future__ import annotations

from codeintel.core.query_results import (
    ScalarCoercionError,
    coerce_datetime,
    coerce_float,
    coerce_int,
    coerce_literal,
    coerce_optional_datetime,
    coerce_optional_float,
    coerce_optional_int,
    coerce_optional_str,
    coerce_str,
    iter_records_from_arrow_reader,
    iter_records_from_relation,
    iter_tuples_from_arrow_reader,
    iter_tuples_from_relation,
    records_from_arrow_batch,
    records_from_arrow_reader,
    records_from_arrow_table,
    records_from_relation,
)

__all__ = [
    "ScalarCoercionError",
    "coerce_datetime",
    "coerce_float",
    "coerce_int",
    "coerce_literal",
    "coerce_optional_datetime",
    "coerce_optional_float",
    "coerce_optional_int",
    "coerce_optional_str",
    "coerce_str",
    "iter_records_from_arrow_reader",
    "iter_records_from_relation",
    "iter_tuples_from_arrow_reader",
    "iter_tuples_from_relation",
    "records_from_arrow_batch",
    "records_from_arrow_reader",
    "records_from_arrow_table",
    "records_from_relation",
]
