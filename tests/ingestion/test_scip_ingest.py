"""Integration tests for the Hamilton scip target."""

from __future__ import annotations

import json
import shutil

import pytest

from tests._helpers.assertions import assert_row_count, assert_target_ok
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness
from tests._helpers.tool_payloads import scip_json_payload
from tests._helpers.tool_sandbox import ToolSandbox, ToolStubSpec


def test_scip_target_writes_tables(build_harness: HamiltonBuildHarness) -> None:
    """Ensure scip target writes scip tables when SCIP binaries are available."""
    if shutil.which("scip-python") is None or shutil.which("scip") is None:
        pytest.skip("scip-python or scip not available on PATH")

    result = build_harness.run_targets(["scip"])
    record = build_harness.record("scip", result=result)
    assert_target_ok(record)
    assert_row_count(record.row_counts, "core.scip_symbols", min_rows=1)
    assert_row_count(record.row_counts, "core.scip_occurrences", min_rows=1)

    scip_dir = build_harness.artifacts.paths.scip_dir
    index_scip = scip_dir / "index.scip"
    index_json = scip_dir / "index.json"
    if not index_scip.is_file():
        pytest.fail("index.scip was not created under build/scip")
    if not index_json.is_file():
        pytest.fail("index.json was not created under build/scip")


def test_scip_target_with_stubbed_artifacts(build_harness: HamiltonBuildHarness) -> None:
    """Ensure scip target can ingest from pre-seeded artifacts without tool binaries."""
    artifacts = build_harness.artifacts
    index_scip, index_json = artifacts.write_dummy_scip_artifacts()

    result = build_harness.run_targets(["scip"])
    record = build_harness.record("scip", result=result)
    assert_target_ok(record)
    assert_row_count(record.row_counts, "core.scip_symbols", min_rows=1)
    assert_row_count(record.row_counts, "core.scip_occurrences", min_rows=1)

    if not index_scip.is_file():
        pytest.fail(f"Expected scip index file to exist: {index_scip}")
    if not index_json.is_file():
        pytest.fail(f"Expected scip json file to exist: {index_json}")


def test_scip_target_via_harness_real_tools(
    build_harness: HamiltonBuildHarness,
    tool_sandbox: ToolSandbox,
) -> None:
    """Ensure scip target can run through the harness with stubbed tools."""
    scip_payload = scip_json_payload()
    tool_sandbox.install_stub(
        "scip-python",
        spec=ToolStubSpec(
            creates="--output",
            creates_payload="scip-binary",
        ),
    )
    tool_sandbox.install_stub(
        "scip",
        spec=ToolStubSpec(
            stdout=json.dumps(scip_payload),
        ),
    )
    with tool_sandbox.prepend_path():
        result = build_harness.run_targets(["scip"])
        record = build_harness.record("scip", result=result)
    assert_target_ok(record)
    assert_row_count(record.row_counts, "core.scip_symbols", min_rows=1)
    assert_row_count(record.row_counts, "core.scip_occurrences", min_rows=1)
