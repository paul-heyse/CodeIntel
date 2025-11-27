"""Repository for file and module queries."""

from __future__ import annotations

from dataclasses import dataclass

from codeintel.storage.repositories.base import (
    BaseRepository,
    RowDict,
    fetch_all_dicts,
    fetch_one_dict,
)


@dataclass(frozen=True)
class ModuleRepository(BaseRepository):
    """Read module and file metadata from docs views."""

    def get_file_summary(self, rel_path: str) -> RowDict | None:
        """
        Return file summary row for a relative path.

        Returns
        -------
        RowDict | None
            File summary row when present.
        """
        sql = """
            SELECT *
            FROM docs.v_file_summary
            WHERE rel_path = ?
              AND repo = ?
              AND commit = ?
            LIMIT 1
        """
        return fetch_one_dict(self.con, sql, [rel_path, self.repo, self.commit])

    def get_module_architecture(self, module: str) -> RowDict | None:
        """
        Return module architecture row.

        Returns
        -------
        RowDict | None
            Module architecture when found.
        """
        sql = """
            SELECT *
            FROM docs.v_module_architecture
            WHERE repo = ?
              AND commit = ?
              AND module = ?
            LIMIT 1
        """
        return fetch_one_dict(self.con, sql, [self.repo, self.commit, module])

    def get_module_profile(self, module: str) -> RowDict | None:
        """
        Return module profile row.

        Returns
        -------
        RowDict | None
            Module profile when found.
        """
        sql = """
            SELECT *
            FROM docs.v_module_profile
            WHERE repo = ?
              AND commit = ?
              AND module = ?
            LIMIT 1
        """
        return fetch_one_dict(self.con, sql, [self.repo, self.commit, module])

    def get_file_profile(self, rel_path: str) -> RowDict | None:
        """
        Return file profile row.

        Returns
        -------
        RowDict | None
            File profile when present.
        """
        sql = """
            SELECT *
            FROM docs.v_file_profile
            WHERE repo = ?
              AND commit = ?
              AND rel_path = ?
            LIMIT 1
        """
        return fetch_one_dict(self.con, sql, [self.repo, self.commit, rel_path])

    def get_file_hints(self, rel_path: str) -> list[RowDict]:
        """
        Return IDE hints for a given file path.

        Returns
        -------
        list[RowDict]
            Hint rows for the requested file.
        """
        sql = """
            SELECT *
            FROM docs.v_ide_hints
            WHERE repo = ?
              AND commit = ?
              AND rel_path = ?
        """
        return fetch_all_dicts(self.con, sql, [self.repo, self.commit, rel_path])
