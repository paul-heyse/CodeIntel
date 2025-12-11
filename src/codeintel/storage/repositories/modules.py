"""Repository for file and module queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.ibis_types import and_predicates
from codeintel.storage.repositories.base import BaseRepository, RowDict

if TYPE_CHECKING:
    import ibis.expr.types as it


@dataclass(frozen=True)
class ModuleRepository(BaseRepository):
    """Read module and file metadata from docs views."""

    def list_modules(self) -> list[str]:
        """
        List module identifiers for the repo/commit.

        Uses Ibis for query construction with SQL fallback.

        Returns
        -------
        list[str]
            Sorted module names for the current snapshot.
        """

        def ibis_query() -> it.Table:
            tbl = self._ibis_table("core.modules")
            return (
                tbl.filter(and_predicates(tbl.repo == self.repo, tbl.commit == self.commit))
                .select("module")
                .order_by("module")
            )

        sql = """
            SELECT module
            FROM core.modules
            WHERE repo = ?
              AND commit = ?
            ORDER BY module
        """
        # NOTE: Skip Pandera validation (table_key=None) because we're only
        # selecting a single column. Full schema validation would fail since
        # the Pandera schema expects all columns.
        rows = self._ibis_with_fallback(ibis_query, sql, [self.repo, self.commit])
        return [str(row["module"]) for row in rows]

    def get_file_summary(self, rel_path: str) -> RowDict | None:
        """
        Return file summary row for a relative path.

        Uses Ibis with SQL fallback.

        Returns
        -------
        RowDict | None
            File summary row when present.
        """

        def ibis_query() -> it.Table:
            tbl = self._ibis_table("docs.v_file_summary")
            return tbl.filter(
                and_predicates(
                    tbl.rel_path == rel_path,
                    tbl.repo == self.repo,
                    tbl.commit == self.commit,
                )
            )

        sql = """
            SELECT *
            FROM docs.v_file_summary
            WHERE rel_path = ?
              AND repo = ?
              AND commit = ?
            LIMIT 1
        """
        return self._ibis_one_with_fallback(ibis_query, sql, [rel_path, self.repo, self.commit])

    def get_module_architecture(self, module: str) -> RowDict | None:
        """
        Return module architecture row.

        Uses Ibis with SQL fallback.

        Returns
        -------
        RowDict | None
            Module architecture when found.
        """

        def ibis_query() -> it.Table:
            tbl = self._ibis_table("docs.v_module_architecture")
            return tbl.filter(
                and_predicates(
                    tbl.repo == self.repo,
                    tbl.commit == self.commit,
                    tbl.module == module,
                )
            )

        sql = """
            SELECT *
            FROM docs.v_module_architecture
            WHERE repo = ?
              AND commit = ?
              AND module = ?
            LIMIT 1
        """
        return self._ibis_one_with_fallback(ibis_query, sql, [self.repo, self.commit, module])

    def get_module_profile(self, module: str) -> RowDict | None:
        """
        Return module profile row.

        Uses Ibis with SQL fallback.

        Returns
        -------
        RowDict | None
            Module profile when found.
        """

        def ibis_query() -> it.Table:
            tbl = self._ibis_table("analytics.module_profile")
            return tbl.filter(
                and_predicates(
                    tbl.repo == self.repo,
                    tbl.commit == self.commit,
                    tbl.module == module,
                )
            )

        sql = """
            SELECT *
            FROM analytics.module_profile
            WHERE repo = ?
              AND commit = ?
              AND module = ?
            LIMIT 1
        """
        return self._ibis_one_with_fallback(
            ibis_query,
            sql,
            [self.repo, self.commit, module],
            table_key="analytics.module_profile",
        )

    def get_file_profile(self, rel_path: str) -> RowDict | None:
        """
        Return file profile row.

        Uses Ibis with SQL fallback.

        Returns
        -------
        RowDict | None
            File profile when present.
        """

        def ibis_query() -> it.Table:
            tbl = self._ibis_table("analytics.file_profile")
            return tbl.filter(
                and_predicates(
                    tbl.repo == self.repo,
                    tbl.commit == self.commit,
                    tbl.rel_path == rel_path,
                )
            )

        sql = """
            SELECT *
            FROM analytics.file_profile
            WHERE repo = ?
              AND commit = ?
              AND rel_path = ?
            LIMIT 1
        """
        return self._ibis_one_with_fallback(
            ibis_query,
            sql,
            [self.repo, self.commit, rel_path],
            table_key="analytics.file_profile",
        )

    def get_file_hints(self, rel_path: str) -> list[RowDict]:
        """
        Return IDE hints for a given file path.

        Uses Ibis with SQL fallback.

        Returns
        -------
        list[RowDict]
            Hint rows for the requested file.
        """

        def ibis_query() -> it.Table:
            tbl = self._ibis_table("docs.v_ide_hints")
            return tbl.filter(
                and_predicates(
                    tbl.repo == self.repo,
                    tbl.commit == self.commit,
                    tbl.rel_path == rel_path,
                )
            )

        sql = """
            SELECT *
            FROM docs.v_ide_hints
            WHERE repo = ?
              AND commit = ?
              AND rel_path = ?
        """
        return self._ibis_with_fallback(ibis_query, sql, [self.repo, self.commit, rel_path])
