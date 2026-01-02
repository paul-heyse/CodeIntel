"""Repository for file and module queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        relation = self._relation("core.modules").select("module").order("module")
        rows = self._relation_to_dicts(relation)
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
        relation = self._relation("docs.v_file_summary")
        relation = relation.filter(self._predicate_eq("rel_path", rel_path)).limit(1)
        return self._relation_to_one(relation)

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
        relation = self._relation("docs.v_module_architecture")
        relation = relation.filter(self._predicate_eq("module", module)).limit(1)
        return self._relation_to_one(relation)

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
        relation = self._relation("docs.v_ide_hints")
        relation = relation.filter(self._predicate_eq("rel_path", rel_path))
        return self._relation_to_dicts(relation)
