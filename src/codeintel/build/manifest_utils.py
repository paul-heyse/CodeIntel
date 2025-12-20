"""Shared JSON manifest read/write helpers."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pathlib import Path


def write_manifest_json(path: Path, payload: dict[str, Any]) -> None:
    """Write a JSON manifest with deterministic formatting.

    Parameters
    ----------
    path
        Destination path for the manifest file.
    payload
        JSON-serializable manifest payload.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_manifest_json(path: Path) -> dict[str, Any]:
    """Read a JSON manifest file.

    Parameters
    ----------
    path
        Path to the manifest file.

    Returns
    -------
    dict[str, Any]
        Parsed JSON payload.
    """
    return json.loads(path.read_text(encoding="utf-8"))


__all__ = ["read_manifest_json", "write_manifest_json"]
