"""Repository wrapper for data model accessors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from codeintel.storage.data_models import (
    DataModelFieldRow,
    DataModelRelationshipRow,
    DataModelRow,
    NormalizedDataModel,
    fetch_fields,
    fetch_models,
    fetch_models_normalized,
    fetch_relationships,
)
from codeintel.storage.gateway import StorageGateway


@dataclass(frozen=True)
class DataModelRepository:
    """Expose typed data model helpers via the gateway."""

    gateway: StorageGateway

    def models(self, repo: str, commit: str) -> list[DataModelRow]:
        """
        Return raw data models.

        Returns
        -------
        list[DataModelRow]
            Parsed rows for the requested revision.
        """
        return fetch_models(self.gateway, repo, commit)

    def models_normalized(self, repo: str, commit: str) -> list[NormalizedDataModel]:
        """
        Return normalized data models with fields and relationships.

        Returns
        -------
        list[NormalizedDataModel]
            Normalized data models including nested fields and relationships.
        """
        return fetch_models_normalized(self.gateway, repo, commit)

    def fields(
        self,
        repo: str,
        commit: str,
        *,
        model_ids: Sequence[str] | None = None,
    ) -> list[DataModelFieldRow]:
        """
        Return data model fields for optional subset of model ids.

        Returns
        -------
        list[DataModelFieldRow]
            Field rows filtered by model identifiers when provided.
        """
        return fetch_fields(self.gateway, repo, commit, model_ids=model_ids)

    def relationships(
        self,
        repo: str,
        commit: str,
        *,
        model_ids: Sequence[str] | None = None,
    ) -> list[DataModelRelationshipRow]:
        """
        Return data model relationships for optional subset of model ids.

        Returns
        -------
        list[DataModelRelationshipRow]
            Relationship rows filtered by model identifiers when provided.
        """
        return fetch_relationships(self.gateway, repo, commit, model_ids=model_ids)
