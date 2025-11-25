"""Tests for graph validation helpers."""

from __future__ import annotations

from typing import Final

from _pytest.logging import LogCaptureFixture

from codeintel.graphs.validation import run_graph_validations
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

    with caplog.at_level("WARNING"):
        run_graph_validations(gateway, repo=repo, commit=commit)

    messages = " ".join(record.message for record in caplog.records)
    expected = ["outside caller spans", "module(s) have no GOIDs"]
    for needle in expected:
        if needle not in messages:
            message = f"Expected warning containing '{needle}' but messages were: {messages}"
            raise AssertionError(message)
