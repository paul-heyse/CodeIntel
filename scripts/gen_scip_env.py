#!/usr/bin/env python3
"""Generate a scip-python environment JSON from installed distributions."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

_environment_module = importlib.import_module("codeintel.ingestion.scip.environment")
build_environment_entries = _environment_module.build_environment_entries


def main() -> int:
    """Write the scip-python environment JSON to stdout.

    Returns
    -------
    int
        Exit code.
    """
    entries = build_environment_entries()
    sys.stdout.write(json.dumps(entries, indent=2, sort_keys=True))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
