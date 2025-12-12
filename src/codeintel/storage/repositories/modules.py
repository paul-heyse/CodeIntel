"""Repository for file and module queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.storage.ibis_types import and_predicates
from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict


@dataclass(frozen=True)
class ModuleRepository(BaseRepository):
    """Read module and file metadata from docs views."""

    def list_modules(self) -> list[str]:
        """
        List module identifiers for the repo/commit.

        Returns
        -------
        list[str]
            Sorted module names for the current snapshot.
        """
        tbl = self._ibis_table("core.modules")
        expr = (
            tbl.filter(and_predicates(tbl.repo == self.repo, tbl.commit == self.commit))
            .select("module")
            .order_by("module")
        )

        rows = self._ibis_to_dicts(expr)
        return [str(row["module"]) for row in rows]

    def get_file_summary(self, rel_path: str) -> RowDict | None:
        """
        Return file summary row for a relative path.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        RowDict | None
            File summary row when present.
        """
        tbl = self._ibis_table("docs.v_file_summary")
        expr = tbl.filter(
            and_predicates(
                tbl.rel_path == rel_path,
                tbl.repo == self.repo,
                tbl.commit == self.commit,
            )
        ).limit(1)
        return self._ibis_to_one(expr)

    def get_module_architecture(self, module: str) -> RowDict | None:
        """
        Return module architecture row.

        Parameters
        ----------
        module
            Module name to look up.

        Returns
        -------
        RowDict | None
            Module architecture when found.
        """
        tbl = self._ibis_table("docs.v_module_architecture")
        expr = tbl.filter(
            and_predicates(
                tbl.repo == self.repo,
                tbl.commit == self.commit,
                tbl.module == module,
            )
        ).limit(1)
        return self._ibis_to_one(expr)

    def get_module_profile(self, module: str) -> RowDict | None:
        """
        Return module profile row.

        Parameters
        ----------
        module
            Module name to look up.

        Returns
        -------
        RowDict | None
            Module profile when found.
        """
        tbl = self._ibis_table("analytics.module_profile")
        expr = tbl.filter(
            and_predicates(
                tbl.repo == self.repo,
                tbl.commit == self.commit,
                tbl.module == module,
            )
        ).limit(1)
        return self._ibis_to_one(expr, table_key="analytics.module_profile")

    def get_file_profile(self, rel_path: str) -> RowDict | None:
        """
        Return file profile row.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        RowDict | None
            File profile when present.
        """
        tbl = self._ibis_table("analytics.file_profile")
        expr = tbl.filter(
            and_predicates(
                tbl.repo == self.repo,
                tbl.commit == self.commit,
                tbl.rel_path == rel_path,
            )
        ).limit(1)
        return self._ibis_to_one(expr, table_key="analytics.file_profile")

    def get_file_hints(self, rel_path: str) -> list[RowDict]:
        """
        Return IDE hints for a given file path.

        Parameters
        ----------
        rel_path
            Relative file path.

        Returns
        -------
        list[RowDict]
            Hint rows for the requested file.
        """
        tbl = self._ibis_table("docs.v_ide_hints")
        expr = tbl.filter(
            and_predicates(
                tbl.repo == self.repo,
                tbl.commit == self.commit,
                tbl.rel_path == rel_path,
            )
        )
        return self._ibis_to_dicts(expr)
