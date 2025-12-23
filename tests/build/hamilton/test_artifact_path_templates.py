"""Tests for artifact path template resolution."""

from __future__ import annotations

from pathlib import Path

from codeintel.build.hamilton.materializers import FileArtifactSaver
from codeintel.build.targets import OutputTarget, TargetGraph
from tests._helpers.assertions.expectation_assertions import expect_equal, expect_true
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness


def test_file_artifact_saver_resolves_path_from_template(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Resolve artifact path from a DAG-provided template."""
    env = build_harness.build_env()
    template = "{build_dir}/artifacts/test.json"
    expected = env.paths.build_dir / "artifacts" / "test.json"

    graph = TargetGraph()
    graph.register(OutputTarget(name="tool_target", module="ingestion"))
    saver = FileArtifactSaver(
        env=env,
        graph=graph,
        target_name="tool_target",
        artifact_name="tool_artifact",
        path_template=template,
    )

    meta = saver.save_data(b"payload")

    expect_equal(meta["status"], "succeeded", label="status")
    expect_equal(meta["path"], str(expected), label="resolved_path")
    path_value = meta.get("path")
    expect_true(isinstance(path_value, str), message="Expected artifact path string")
    if isinstance(path_value, str):
        expect_true(Path(path_value).exists(), message="Expected artifact file to exist")


def test_file_artifact_saver_requires_template(
    build_harness: HamiltonBuildHarness,
) -> None:
    """Missing path templates should fail fast."""
    env = build_harness.build_env()

    graph = TargetGraph()
    graph.register(OutputTarget(name="tool_target", module="ingestion"))
    saver = FileArtifactSaver(
        env=env,
        graph=graph,
        target_name="tool_target",
        artifact_name="tool_artifact",
        path_template=None,
    )

    meta = saver.save_data(b"payload")

    expect_equal(meta["status"], "failed", label="status")
    error_value = meta.get("error")
    expect_true(
        isinstance(error_value, str) and "Missing artifact path_template" in error_value,
        message="Expected missing path_template error",
    )
