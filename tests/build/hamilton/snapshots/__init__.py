"""CLI golden snapshot testing infrastructure for Hamilton Phase 2.

This package provides a manifest-driven snapshot testing framework for
validating CLI command outputs against golden reference files.

Key components:
- _snapshot.py: JSON/text normalization and assertion helpers
- _manifest.py: Typed manifest loader supporting JSON and YAML
- _runner.py: CLI execution and snapshot comparison

Usage:
    pytest -m cli_snapshot                         # Run all snapshot tests
    pytest -m cli_snapshot --update-cli-snapshots  # Update snapshots
    pytest -m cli_snapshot --cli-snapshot-tags pr14,graph  # Filter by tags
    pytest -m cli_snapshot --list-cli-snapshots    # List cases and exit
"""

from __future__ import annotations

__all__: list[str] = []

