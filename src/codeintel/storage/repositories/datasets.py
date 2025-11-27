"""Repository for dataset access helpers."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.storage.repositories.base import BaseRepository, RowDict, fetch_all_dicts


@dataclass(frozen=True)
class DatasetReadRepository(BaseRepository):
    """Read dataset rows via metadata.dataset_rows."""

    def read_dataset_rows(self, table_key: str, *, limit: int, offset: int) -> list[RowDict]:
        """
        Return dataset rows with limit/offset.

        Returns
        -------
        list[RowDict]
            Slice of rows from the requested dataset.
        """
        sql = "SELECT * FROM metadata.dataset_rows(?, ?, ?)"
        return fetch_all_dicts(self.con, sql, [table_key, limit, offset])
