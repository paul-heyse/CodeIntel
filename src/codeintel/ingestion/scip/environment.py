"""Helpers for scip-python environment discovery."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path


@dataclass(frozen=True)
class ScipEnvironmentResolution:
    """Resolved environment inputs for scip-python."""

    environment_json: Path | None
    source: str | None


def pip_available() -> bool:
    """Return True when pip or pip3 is discoverable on PATH."""
    return shutil.which("pip") is not None or shutil.which("pip3") is not None


def build_environment_entries() -> list[dict[str, object]]:
    """Collect installed distributions for scip-python environment JSON."""
    entries: list[dict[str, object]] = []
    for dist in metadata.distributions():
        name = dist.metadata.get("Name") or dist.metadata.get("Summary") or dist.name
        files = [str(path) for path in (dist.files or ())]
        entries.append(
            {
                "name": name,
                "version": dist.version,
                "files": sorted(set(files)),
            }
        )
    entries.sort(key=lambda entry: str(entry["name"]))
    return entries


def write_environment_json(output_path: Path) -> None:
    """Write scip-python environment JSON to the requested path."""
    entries = build_environment_entries()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(entries, indent=2, sort_keys=True)
    output_path.write_text(f"{payload}\n", encoding="utf-8")


def resolve_environment_json(
    *,
    environment_json: Path | None,
    scip_dir: Path,
) -> ScipEnvironmentResolution:
    """Resolve which environment discovery mode to use for scip-python."""
    if environment_json is not None:
        if not environment_json.is_file():
            message = f"SCIP environment JSON not found: {environment_json}"
            raise ValueError(message)
        return ScipEnvironmentResolution(environment_json=environment_json, source="json")
    if pip_available():
        return ScipEnvironmentResolution(environment_json=None, source="pip")
    env_path = scip_dir / "env.json"
    write_environment_json(env_path)
    return ScipEnvironmentResolution(environment_json=env_path, source="json")


__all__ = [
    "ScipEnvironmentResolution",
    "build_environment_entries",
    "pip_available",
    "resolve_environment_json",
    "write_environment_json",
]
