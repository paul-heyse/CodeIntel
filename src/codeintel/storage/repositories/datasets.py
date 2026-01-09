"""Repository for dataset access helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.core.columnar.compute_helpers import combine_table_chunks
from codeintel.core.columnar.finalize_ops import finalize_spec_for_table, finalize_table
from codeintel.core.columnar.nested_ops import deep_cast_table_to_contract
from codeintel.core.schemas.arrow_gen import arrow_contract_for_table_schema
from codeintel.core.schemas.service import get_schema_service
from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict

MAX_ROW_LIMIT = 9_223_372_036_854_775_807


def _finalize_dataset_table(table_key: str, table: pa.Table) -> pa.Table:
    schema_service = get_schema_service()
    table_schema = schema_service.require_table_schema(table_key)
    contract = arrow_contract_for_table_schema(table_schema=table_schema)
    compact = combine_table_chunks(table)
    casted = deep_cast_table_to_contract(compact, contract)
    finalized = finalize_table(
        casted,
        spec=finalize_spec_for_table(
            table_key,
            mode="tolerant",
            emit_artifacts=True,
        ),
    )
    return finalized.good


@dataclass(frozen=True)
class DatasetReadRepository(BaseRepository):
    """Read dataset rows via DuckDB relations."""

    def read_dataset_table(
        self,
        table_key: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        finalize: bool = True,
    ) -> pa.Table:
        """
        Return dataset rows as an Arrow Table validated by Arrow checks.

        Parameters
        ----------
        table_key
            Fully qualified table name (e.g., "core.modules").
        limit
            Maximum number of rows to return.
        offset
            Number of rows to skip.
        finalize
            Whether to apply finalize/deep-cast alignment before returning.

        Returns
        -------
        pyarrow.Table
            Validated dataset slice.
        """
        relation = self._relation(table_key)

        if limit is not None or offset:
            limit_value = limit if limit is not None else MAX_ROW_LIMIT
            relation = relation.limit(limit_value, offset=offset)

        reader = self._relation_to_reader(
            relation,
            table_key=table_key,
            batch_size=DEFAULT_ARROW_BATCH_SIZE,
        )
        table = pa.Table.from_batches(reader, schema=reader.schema)
        if not finalize:
            return table
        return _finalize_dataset_table(table_key, table)

    def read_dataset_dataframe(
        self,
        table_key: str,
        *,
        limit: int | None = None,
        offset: int = 0,
        finalize: bool = True,
    ) -> pa.Table:
        """Return dataset rows as an Arrow Table (legacy name).

        Returns
        -------
        pyarrow.Table
            Arrow table containing the dataset rows.
        """
        return self.read_dataset_table(
            table_key,
            limit=limit,
            offset=offset,
            finalize=finalize,
        )

    def read_dataset_rows(self, table_key: str, *, limit: int, offset: int) -> list[RowDict]:
        """
        Return dataset rows with limit/offset.

        Parameters
        ----------
        table_key
            Fully qualified table name.
        limit
            Maximum number of rows.
        offset
            Number of rows to skip.

        Returns
        -------
        list[RowDict]
            Slice of rows from the requested dataset.
        """
        relation = self._relation(table_key)
        if limit is not None or offset:
            limit_value = limit if limit is not None else MAX_ROW_LIMIT
            relation = relation.limit(limit_value, offset=offset)
        return self._relation_to_dicts(relation, table_key)
