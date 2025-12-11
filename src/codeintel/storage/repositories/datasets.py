"""Repository for dataset access helpers."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from ibis.common.exceptions import IbisError

from codeintel.storage.pandera_schemas import validate_dataset_df
from codeintel.storage.repositories.base import BaseRepository, RowDict


@dataclass(frozen=True)
class DatasetReadRepository(BaseRepository):
    """Read dataset rows via metadata.dataset_rows."""

    def read_dataset_dataframe(
        self,
        table_key: str,
        *,
        limit: int | None = None,
        offset: int = 0,
    ) -> pd.DataFrame:
        """
        Return dataset rows as a Pandas DataFrame validated by Pandera.

        Falls back to the dataset_rows macro when the table cannot be resolved
        through the Ibis gateway.
        """
        try:
            expr = self.gateway.ibis.table(table_key)
            if offset:
                expr = expr.offset(offset)
            if limit is not None:
                expr = expr.limit(limit)
            df = expr.execute()
        except IbisError:
            row_limit = limit if limit is not None else 9_223_372_036_854_775_807
            result = self.con.execute(
                "SELECT * FROM metadata.dataset_rows(?, ?, ?)",
                [table_key, row_limit, offset],
            )
            df = result.fetch_df()
        return validate_dataset_df(table_key, df)

    def read_dataset_rows(self, table_key: str, *, limit: int, offset: int) -> list[RowDict]:
        """
        Return dataset rows with limit/offset.

        Returns
        -------
        list[RowDict]
            Slice of rows from the requested dataset.
        """
        df = self.read_dataset_dataframe(table_key, limit=limit, offset=offset)
        return df.to_dict(orient="records")
