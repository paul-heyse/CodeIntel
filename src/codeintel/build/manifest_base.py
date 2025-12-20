"""Shared base class for manifest-like payloads."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from codeintel.build.manifest_utils import write_manifest_json

if TYPE_CHECKING:
    from pathlib import Path


class ManifestBase(ABC):
    """Base class for deterministic manifest serialization."""

    @abstractmethod
    def to_json_obj(self) -> dict[str, object]:
        """Return a JSON-serializable representation.

        Returns
        -------
        dict[str, object]
            JSON-serializable manifest payload.
        """

    def to_json(self) -> str:
        """Serialize the manifest to a deterministic JSON string.

        Returns
        -------
        str
            JSON string representation of this manifest.
        """
        return json.dumps(self.to_json_obj(), indent=2, sort_keys=True)

    def write_json(self, path: Path) -> Path:
        """Write the manifest to disk with deterministic formatting.

        Parameters
        ----------
        path
            Destination path for the manifest file.

        Returns
        -------
        Path
            Path to the written manifest file.
        """
        write_manifest_json(path, self.to_json_obj())
        return path


__all__ = ["ManifestBase"]
