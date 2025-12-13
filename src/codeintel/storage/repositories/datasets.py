"""Repository for dataset access helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

from codeintel.config.datasets.validation import validate_df
from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict

MAX_ROW_LIMIT = 9_223_372_036_854_775_807


@dataclass(frozen=True)
class DatasetReadRepository(BaseRepository):
    """Read dataset rows via Ibis table access."""

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
        expr = self._ibis_table(table_key)

        if limit is not None or offset:
            limit_value = limit if limit is not None else MAX_ROW_LIMIT
            expr = expr.limit(limit_value, offset=offset)

        df = pd.DataFrame(expr.execute())
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
        df = self.read_dataset_dataframe(table_key, limit=limit, offset=offset)
        return df.to_dict(orient="records")
