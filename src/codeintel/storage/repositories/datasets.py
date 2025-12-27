"""Repository for dataset access helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from codeintel.storage.repositories.base import BaseRepository
from codeintel.storage.validation.pandera_df import validate_df

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict

MAX_ROW_LIMIT = 9_223_372_036_854_775_807


@dataclass(frozen=True)
class DatasetReadRepository(BaseRepository):
    """Read dataset rows via DuckDB relations."""

    def read_dataset_dataframe(
        self,
        table_key: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> pd.DataFrame:
        """
        Return dataset rows as a Pandas DataFrame validated by Pandera.

        Parameters
        ----------
        table_key
            Fully qualified table name (e.g., "analytics.function_metrics").
        limit
            Maximum number of rows to return.
        offset
            Number of rows to skip.

        Returns
        -------
        pandas.DataFrame
            Validated dataset slice.
        """
        relation = self._relation(table_key)

        if limit is not None or offset:
            limit_value = limit if limit is not None else MAX_ROW_LIMIT
            relation = relation.limit(limit_value, offset=offset)

        table = relation.fetch_arrow_table()
        df = table.to_pandas()
        return validate_df(table_key, df)

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
