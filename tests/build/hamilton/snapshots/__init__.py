"""CLI golden snapshot testing infrastructure for Hamilton Phase 2.

This package provides a manifest-driven snapshot testing framework for
validating CLI command outputs against golden reference files.

Key components:
- _snapshot.py: JSON/text normalization and assertion helpers
- _manifest.py: Typed manifest loader supporting JSON and YAML
- _runner.py: CLI execution and snapshot comparison

Usage:
    pytest -m cli_snapshot
    pytest -m cli_snapshot --update-cli-snapshots
    pytest -m cli_snapshot --cli-snapshot-tags pr14,graph
    pytest -m cli_snapshot --list-cli-snapshots
"""

from __future__ import annotations

__all__: list[str] = []
