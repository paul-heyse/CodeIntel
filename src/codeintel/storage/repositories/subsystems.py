"""Repository for subsystem-related queries."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from duckdb import SQLExpression

from codeintel.storage.repositories.base import BaseRepository

if TYPE_CHECKING:
    from codeintel.storage.repositories.base import RowDict


@dataclass(frozen=True)
class SubsystemRepository(BaseRepository):
    """Read subsystem summaries and memberships."""

    def list_subsystems(
        self,
        *,
        limit: int,
        role: str | None = None,
        query: str | None = None,
    ) -> list[RowDict]:
        """
        List subsystem summaries with optional role and search filters.

        Parameters
        ----------
        limit
            Maximum number of results.
        role
            Optional role filter.
        query
            Optional search query for name/description.

        Returns
        -------
        list[RowDict]
            Subsystem summary rows ordered by module count.
        """
        relation = self._relation("docs.v_subsystem_summary")

        if role:
            modules = self._relation("analytics.subsystem_modules")
            role_subsystems = (
                modules.filter(self._predicate_eq("role", role))
                .select("subsystem_id")
                .distinct()
            )
            relation = relation.join(role_subsystems, "subsystem_id")

        if query:
            pattern = f"%{query.replace(\"'\", \"''\")}%"
            predicate = SQLExpression(
                f"name ILIKE '{pattern}' OR description ILIKE '{pattern}'"
            )
            relation = relation.filter(predicate)

        relation = relation.order("module_count DESC, subsystem_id").limit(limit)
        return self._relation_to_dicts(relation, "docs.v_subsystem_summary")

    def get_subsystem_summary(self, subsystem_id: str) -> RowDict | None:
        """
        Return a single subsystem summary by identifier.

        Parameters
        ----------
        subsystem_id
            The subsystem identifier.

        Returns
        -------
        RowDict | None
            Subsystem summary row when present.
        """
        relation = self._relation("docs.v_subsystem_summary")
        relation = relation.filter(self._predicate_eq("subsystem_id", subsystem_id)).limit(1)
        rows = self._relation_to_dicts(relation, "docs.v_subsystem_summary")
        return rows[0] if rows else None

    def search_subsystems(
        self,
        *,
        limit: int,
        role: str | None = None,
        query: str | None = None,
    ) -> list[RowDict]:
        """
        Alias for list_subsystems to make intent explicit.

        Parameters
        ----------
        limit
            Maximum number of results.
        role
            Optional role filter.
        query
            Optional search query.

        Returns
        -------
        list[RowDict]
            Subsystem rows matching the search parameters.
        """
        return self.list_subsystems(limit=limit, role=role, query=query)

    def list_subsystem_modules(self, subsystem_id: str) -> list[RowDict]:
        """
        Return module memberships for a subsystem.

        Parameters
        ----------
        subsystem_id
            The subsystem identifier.

        Returns
        -------
        list[RowDict]
            Module membership rows ordered by module.
        """
        relation = self._relation("docs.v_module_with_subsystem")
        relation = relation.filter(self._predicate_eq("subsystem_id", subsystem_id))
        relation = relation.order("module")
        return self._relation_to_dicts(relation, "docs.v_module_with_subsystem")

    def list_subsystem_memberships(self) -> list[RowDict]:
        """
        Return all subsystem-module memberships for the repo/commit.

        Returns
        -------
        list[RowDict]
            Membership rows keyed by subsystem and module.
        """
        relation = self._relation("analytics.subsystem_modules").select("subsystem_id", "module")
        return self._relation_to_dicts(relation)

    def list_subsystems_for_module(self, module: str) -> list[RowDict]:
        """
        Return subsystem memberships for a module.

        Parameters
        ----------
        module
            The module name.

        Returns
        -------
        list[RowDict]
            Subsystem membership rows for the module.
        """
        relation = self._relation("docs.v_module_with_subsystem")
        relation = relation.filter(self._predicate_eq("module", module))
        return self._relation_to_dicts(relation, "docs.v_module_with_subsystem")

    def _has_cache(self, cache_table: str) -> bool:
        """
        Check if a cache table has data for this repo/commit.

        Parameters
        ----------
        cache_table
            Fully qualified table name for the cache.

        Returns
        -------
        bool
            True if cache has at least one row for this repo/commit.
        """
        relation = self._relation(cache_table).limit(1)
        return self._relation_exists(relation)

    def list_subsystem_profiles(self, *, limit: int) -> list[RowDict]:
        """
        Return subsystem profile rows from docs views.

        Parameters
        ----------
        limit
            Maximum number of results.

        Returns
        -------
        list[RowDict]
            Profile rows ordered by module count then subsystem_id.
        """
        cache_table = "analytics.subsystem_profile_cache"
        if self._has_cache(cache_table):
            relation = self._relation(cache_table)
            relation = relation.order("module_count DESC, subsystem_id").limit(limit)
            return self._relation_to_dicts(relation, cache_table)

        relation = self._relation("docs.v_subsystem_profile")
        relation = relation.order("module_count DESC, subsystem_id").limit(limit)
        return self._relation_to_dicts(relation, "docs.v_subsystem_profile")

    def list_subsystem_coverage(self, *, limit: int) -> list[RowDict]:
        """
        Return subsystem coverage rollups from docs views.

        Parameters
        ----------
        limit
            Maximum number of results.

        Returns
        -------
        list[RowDict]
            Coverage rows ordered by test count then subsystem_id.
        """
        cache_table = "analytics.subsystem_coverage_cache"
        if self._has_cache(cache_table):
            relation = self._relation(cache_table)
            relation = relation.order("test_count DESC NULLS LAST, subsystem_id").limit(limit)
            return self._relation_to_dicts(relation, cache_table)

        relation = self._relation("docs.v_subsystem_coverage")
        relation = relation.order("test_count DESC NULLS LAST, subsystem_id").limit(limit)
        return self._relation_to_dicts(relation, "docs.v_subsystem_coverage")
