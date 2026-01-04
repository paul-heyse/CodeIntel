"""Repository for dataset access helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pyarrow as pa

from codeintel.storage.constants import DEFAULT_ARROW_BATCH_SIZE
from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict

MAX_ROW_LIMIT = 9_223_372_036_854_775_807


@dataclass(frozen=True)
class DatasetReadRepository(BaseRepository):
    """Read dataset rows via DuckDB relations."""

    def read_dataset_table(
        self,
        table_key: str,
        *,
        limit: int | None = None,
        offset: int = 0,
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
        return pa.Table.from_batches(reader, schema=reader.schema)

    def read_dataset_dataframe(
        self,
        table_key: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> pa.Table:
        """Return dataset rows as an Arrow Table (legacy name).

        Returns
        -------
        pyarrow.Table
            Arrow table containing the dataset rows.
        """
        return self.read_dataset_table(table_key, limit=limit, offset=offset)

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
