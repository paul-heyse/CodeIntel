"""Tests for graph validation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Final

import pytest
from _pytest.logging import LogCaptureFixture

from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.validation import GraphValidationOptions, run_graph_validations
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.schemas import apply_all_schemas
from tests._helpers.fixtures import seed_graph_validation_gaps


def test_run_graph_validations_emits_warnings(
    caplog: LogCaptureFixture, fresh_gateway: StorageGateway
) -> None:
    """
    Graph validations should warn for common integrity gaps.

    Raises
    ------
    AssertionError
        If expected warning text is absent.
    """
    gateway = fresh_gateway
    repo: Final = "demo/repo"
    commit: Final = "deadbeef"
    apply_all_schemas(gateway.con)
    seed_graph_validation_gaps(gateway, repo=repo, commit=commit)
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=Path())

    with caplog.at_level("WARNING"):
        run_graph_validations(
            gateway,
            snapshot=snapshot,
            runtime=GraphRuntimeOptions(snapshot=snapshot),
        )

    messages = " ".join(record.message for record in caplog.records)
    expected = ["outside caller spans", "module(s) have no GOIDs"]
    for needle in expected:
        if needle not in messages:
            message = f"Expected warning containing '{needle}' but messages were: {messages}"
            raise AssertionError(message)


def test_run_graph_validations_snapshot_mismatch_raises(
    fresh_gateway: StorageGateway,
) -> None:
    """Graph runtime snapshot must align with validation snapshot."""
    gateway = fresh_gateway
    apply_all_schemas(gateway.con)
    snapshot = SnapshotRef(repo="demo/repo", commit="deadbeef", repo_root=Path())
    mismatched_runtime = GraphRuntimeOptions(
        snapshot=SnapshotRef(repo="other/repo", commit="cafebabe", repo_root=Path())
    )

    with pytest.raises(ValueError, match="GraphRuntime snapshot mismatch"):
        run_graph_validations(
            gateway,
            snapshot=snapshot,
            runtime=mismatched_runtime,
        )


def test_run_graph_validations_hard_fail_on_error(
    fresh_gateway: StorageGateway,
) -> None:
    """Hard-fail mode should raise when error-level findings exist."""
    gateway = fresh_gateway
    apply_all_schemas(gateway.con)
    repo = "demo/repo"
    commit = "deadbeef"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=Path())
    seed_graph_validation_gaps(gateway, repo=repo, commit=commit)
    runtime = GraphRuntimeOptions(snapshot=snapshot)

    with pytest.raises(RuntimeError, match="error-level findings"):
        run_graph_validations(
            gateway,
            snapshot=snapshot,
            runtime=runtime,
            options=GraphValidationOptions(
                severity_overrides={
                    "missing_function_goids": "error",
                    "callsite_span_mismatch": "error",
                },
                hard_fail=True,
            ),
        )


def test_run_graph_validations_caps_findings(
    fresh_gateway: StorageGateway,
) -> None:
    """
    Per-rule caps should limit persisted validation rows.

    Raises
    ------
    AssertionError
        When a rule exceeds the configured cap.
    """
    gateway = fresh_gateway
    apply_all_schemas(gateway.con)
    repo = "demo/repo"
    commit = "deadbeef"
    snapshot = SnapshotRef(repo=repo, commit=commit, repo_root=Path())
    seed_graph_validation_gaps(gateway, repo=repo, commit=commit)
    runtime = GraphRuntimeOptions(snapshot=snapshot)

    run_graph_validations(
        gateway,
        snapshot=snapshot,
        runtime=runtime,
        options=GraphValidationOptions(max_findings_per_rule=1),
    )
    rows = gateway.con.execute(
        "SELECT graph_name, COUNT(*) FROM analytics.graph_validation GROUP BY graph_name"
    ).fetchall()
    for _, count in rows:
        if int(count) > 1:
            message = f"Expected cap to apply, found {count} rows"
            raise AssertionError(message)
