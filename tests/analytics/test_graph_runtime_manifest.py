"""Unit tests for graph runtime manifest helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from codeintel.analytics.graphs.plugins import GraphMetricPlugin
from codeintel.analytics.graphs.runtime.manifest import (
    InputHashPayload,
    ManifestState,
    compute_input_hash,
    compute_options_hash,
    is_unchanged,
    load_prior_manifest,
)
from codeintel.config.steps_graphs import GraphRunScope


def test_compute_input_hash_includes_scope_and_options() -> None:
    """Input hash should change when scope or options hash changes."""
    scope_a = GraphRunScope(paths=("a.py",))
    scope_b = GraphRunScope(paths=("b.py",))
    plugin_name = "manifest_plugin"
    options_hash = compute_options_hash(
        GraphMetricPlugin(
            name=plugin_name,
            description="hash test",
            stage="core",
            enabled_by_default=False,
            run=lambda _ctx: None,
        ),
        {"flag": True},
    )
    hash_a = compute_input_hash(
        InputHashPayload(
            repo="demo/repo",
            commit="deadbeef",
            plugin_name=plugin_name,
            version_hash="v1",
            scope=scope_a,
            options_hash=options_hash,
        )
    )
    hash_b = compute_input_hash(
        InputHashPayload(
            repo="demo/repo",
            commit="deadbeef",
            plugin_name=plugin_name,
            version_hash="v1",
            scope=scope_b,
            options_hash=options_hash,
        )
    )
    if hash_a == hash_b:
        pytest.fail("Input hash should change when scope changes")


def test_load_prior_manifest_merges_meta(tmp_path: Path) -> None:
    """load_prior_manifest should merge meta fields into the record."""
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "name": "demo",
                        "status": "succeeded",
                        "meta": {
                            "input_hash": "abc",
                            "options_hash": "def",
                            "row_counts": {"t": 1},
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    prior = load_prior_manifest(manifest_path)
    if prior is None:
        pytest.fail("Manifest should load records")
    record = prior["demo"]
    if record.get("input_hash") != "abc":
        pytest.fail("Input hash should be merged from meta")
    if record.get("options_hash") != "def":
        pytest.fail("Options hash should be merged from meta")
    if record.get("row_counts") != {"t": 1}:
        pytest.fail("Row counts should be merged from meta")


def test_is_unchanged_matches_hashes_without_row_counts() -> None:
    """is_unchanged should rely on hashes when row_count_tables is empty."""
    manifest = {
        "demo": {
            "status": "succeeded",
            "input_hash": "abc",
            "options_hash": None,
        }
    }
    state_unchanged = ManifestState(
        plugin_name="demo",
        row_count_tables=(),
        gateway=None,
        repo="demo/repo",
        commit="deadbeef",
        input_hash="abc",
        options_hash=None,
    )
    state_changed = ManifestState(
        plugin_name="demo",
        row_count_tables=(),
        gateway=None,
        repo="demo/repo",
        commit="deadbeef",
        input_hash="xyz",
        options_hash=None,
    )
    unchanged = is_unchanged(prior_manifest=manifest, state=state_unchanged)
    changed = is_unchanged(prior_manifest=manifest, state=state_changed)
    if not unchanged:
        pytest.fail("Matching hashes should be treated as unchanged")
    if changed:
        pytest.fail("Mismatched hashes should force re-execution")
