"""Semantic view registry loaded from published artifacts.

The registry is the serving-side representation of semantic views, compiled into
`semantic_registry.json` during the build phase.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.serving.semantic.models import SemanticViewSpec

if TYPE_CHECKING:
    from pathlib import Path


@dataclass(frozen=True)
class SemanticRegistry:
    """Registry of semantic views.

    Parameters
    ----------
    version
        Registry schema version.
    views
        Tuple of semantic view specifications.
    """

    version: str
    views: tuple[SemanticViewSpec, ...]

    @classmethod
    def load(cls, path: Path) -> SemanticRegistry:
        """Load registry from JSON file.

        Parameters
        ----------
        path
            Path to semantic_registry.json.

        Returns
        -------
        SemanticRegistry
            Loaded registry instance.

        Raises
        ------
        KeyError
            If the registry payload does not include a version field.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        if "version" not in payload:
            msg = "Semantic registry missing version"
            raise KeyError(msg)
        views_raw = payload.get("views", [])
        views = tuple(SemanticViewSpec.model_validate(v) for v in views_raw)
        return cls(version=str(payload["version"]), views=views)

    def by_id(self, view_id: str) -> SemanticViewSpec:
        """Look up view by semantic ID.

        Parameters
        ----------
        view_id
            Semantic view identifier.

        Returns
        -------
        SemanticViewSpec
            Matching view specification.

        Raises
        ------
        KeyError
            If view_id not found.
        """
        for view in self.views:
            if view.id == view_id:
                return view
        msg = f"Unknown semantic view: {view_id}"
        raise KeyError(msg)

    def list_view_ids(self) -> list[str]:
        """Return all registered view IDs.

        Returns
        -------
        list[str]
            Registered semantic view identifiers.
        """
        return [v.id for v in self.views]

    def to_json(self) -> str:
        """Serialize registry to JSON string.

        Returns
        -------
        str
            JSON string representation of this registry.
        """
        return json.dumps(
            {
                "version": self.version,
                "views": [v.model_dump(mode="json") for v in self.views],
            },
            indent=2,
            sort_keys=True,
        )


__all__ = ["SemanticRegistry"]
