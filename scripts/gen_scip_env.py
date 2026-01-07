#!/usr/bin/env python3
"""Generate a scip-python environment JSON from installed distributions."""

from __future__ import annotations

import json
import sys
from importlib import metadata


def _build_environment_entries() -> list[dict[str, object]]:
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


def main() -> int:
    """Write the scip-python environment JSON to stdout.

    Returns
    -------
    int
        Exit code.
    """
    entries = _build_environment_entries()
    sys.stdout.write(json.dumps(entries, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
