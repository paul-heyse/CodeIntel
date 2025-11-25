"""Unit tests for test profile analytics heuristics."""

from __future__ import annotations

from pathlib import Path

import pytest

from codeintel.analytics.test_profiles import (
    ImportanceInputs,
    IoFlags,
    TestAstInfo,
    TestRecord,
    build_test_ast_index_for_tests,
    compute_flakiness_score,
    compute_importance_score,
    infer_behavior_tags,
)

IMPORTANCE_LOWER_BOUND = 0.55
IMPORTANCE_UPPER_BOUND = 0.7


def test_flakiness_score_caps_at_one() -> None:
    """Ensure the flakiness heuristic is capped to 1.0."""
    score = compute_flakiness_score(
        status="xfail",
        markers=["flaky", "slow"],
        duration_ms=2500.0,
        io_flags=IoFlags(
            uses_network=True,
            uses_db=True,
            uses_filesystem=True,
            uses_subprocess=False,
        ),
        slow_test_threshold_ms=2000.0,
    )
    if score != 1.0:
        pytest.fail(f"Expected flakiness score of 1.0, got {score}")


def test_behavior_tags_include_ast_and_io_hints() -> None:
    """Behavior tags should reflect name, markers, IO flags, and AST cues."""
    flags = IoFlags(uses_network=True, uses_db=False, uses_filesystem=False, uses_subprocess=True)
    ast_info = TestAstInfo(
        assert_count=1,
        raise_count=1,
        uses_pytest_raises=True,
        uses_concurrency_lib=True,
        has_boundary_asserts=True,
        io_flags=flags,
    )
    tags = infer_behavior_tags(
        name="test_concurrent_error_path",
        markers=["xfail", "network"],
        io_flags=flags,
        ast_info=ast_info,
    )
    required = {"concurrency", "error_paths", "known_bug", "network_interaction"}
    missing = required.difference(tags)
    if missing:
        pytest.fail(f"Missing expected tags: {missing}")
    for tag in ("edge_cases", "process_interaction", "io_heavy"):
        if tag not in tags:
            pytest.fail(f"Expected tag {tag} to be present")


def test_importance_score_includes_subsystem_risk() -> None:
    """Importance should blend coverage breadth, graph weight, and subsystem risk."""
    inputs = ImportanceInputs(
        functions_covered_count=2,
        weighted_degree=5.0,
        max_function_count=4,
        max_weighted_degree=10.0,
        subsystem_risk=0.8,
        max_subsystem_risk=1.0,
    )
    score = compute_importance_score(inputs)
    if score is None:
        pytest.fail("Importance score should not be None for populated inputs")
    if not (IMPORTANCE_LOWER_BOUND <= score <= IMPORTANCE_UPPER_BOUND):
        pytest.fail(f"Importance score {score} outside expected range")


def test_ast_index_detects_io_and_asserts(tmp_path: Path) -> None:
    """AST indexing should capture IO flags, raises, and boundary asserts."""
    test_file = tmp_path / "test_sample.py"
    test_file.write_text(
        "\n".join(
            [
                "import requests",
                "import pytest",
                "import subprocess",
                "",
                "def test_error_case(tmp_path):",
                '    resp = requests.get("http://example.com")',
                '    subprocess.run(["echo", "hi"])',
                "    with pytest.raises(ValueError):",
                '        raise ValueError("bad")',
                "    value = 1",
                "    assert value >= 1",
            ]
        ),
        encoding="utf-8",
    )
    record = TestRecord(
        test_id="test_error_case",
        test_goid_h128=None,
        urn=None,
        rel_path=str(test_file.relative_to(tmp_path)),
        module=None,
        qualname="test_error_case",
        language="python",
        kind="function",
        status="passed",
        duration_ms=10.0,
        markers=[],
        flaky=False,
        start_line=5,
        end_line=11,
    )
    index = build_test_ast_index_for_tests(tmp_path, [record])
    info = index[record.test_id]
    checks = [
        (info.assert_count == 1, "Expected one assert in span"),
        (info.raise_count == 1, "Expected one raise in span"),
        (info.io_flags.uses_network, "Expected network usage flag"),
        (info.io_flags.uses_subprocess, "Expected subprocess usage flag"),
        (info.has_boundary_asserts, "Expected boundary assert detection"),
        (info.uses_pytest_raises, "Expected pytest.raises detection"),
    ]
    failures = [message for condition, message in checks if not condition]
    if failures:
        pytest.fail(f"AST index failures: {failures}")
