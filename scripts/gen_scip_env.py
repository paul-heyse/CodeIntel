#!/usr/bin/env python3
"""Generate a scip-python environment JSON from installed distributions."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from codeintel.ingestion.scip.environment import build_environment_entries


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
