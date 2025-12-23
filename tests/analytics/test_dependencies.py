"""Consolidated dependency analytics tests with shared seeds and helpers."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pytest

from codeintel.analytics.compute.dependencies.classification import (
    CALLSITE_MEDIUM_THRESHOLD,
    SEVERITY_SCORES,
    DependencyModePattern,
    classify_modes,
    risk_level,
    risk_score,
    severity_score,
)
from codeintel.analytics.compute.dependencies.detection import (
    DependencyCallVisitor,
    build_alias_map,
    build_alias_maps,
    group_calls_by_library,
)
from codeintel.analytics.dependencies import load_config_key_map
from tests._helpers.analytics_samples import (
    dependency_alias_sources,
    dependency_calls_sample,
    dependency_library_patterns,
    dependency_patterns_yaml,
)
from tests._helpers.assertions import (
    MissingExtraOptions,
    assert_edge_count,
    assert_no_cycles,
    build_dependency_graph,
    expect_equal,
    expect_false,
    expect_in,
    expect_is_none,
    expect_is_not_none,
    expect_length,
    expect_not_in,
    expect_true,
    format_missing_extra,
)
from tests._helpers.scenarios import TestScenario

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from codeintel.analytics.compute.dependencies.classification import (
        LibraryPattern,
    )
    from tests._helpers.context import TestContext

EXPECTED_SEVERITY_COUNT = 5
EXPECTED_REQUESTS_CALLS = 2
EXPECTED_GROUPED_LIBRARIES = 2
EXPECTED_GROUPED_EDGES = 3


@dataclass(frozen=True)
class DependenciesFixture:
    """Shared dependency context and patterns."""

    ctx: TestContext
    patterns: dict[str, LibraryPattern]


@pytest.fixture
def dependencies_ctx(tmp_path: Path) -> Iterator[DependenciesFixture]:
    """Provide seeded config values and dependency patterns.

    Yields
    ------
    Iterator[DependenciesFixture]
        Test context seeded with config values and parsed patterns.
    """
    ctx = TestScenario.with_dependencies().build(tmp_path)
    patterns = dependency_library_patterns()
    patterns_path = ctx.repo_root / "config" / "dependency_patterns.yml"
    patterns_path.parent.mkdir(parents=True, exist_ok=True)
    patterns_path.write_text(dependency_patterns_yaml(patterns), encoding="utf-8")
    try:
        yield DependenciesFixture(ctx=ctx, patterns=patterns)
    finally:
        ctx.close()


def test_load_config_keys_filters_repo(dependencies_ctx: DependenciesFixture) -> None:
    """Ensure config_values rows are filtered by repo/commit."""
    ctx = dependencies_ctx.ctx
    con = ctx.gateway.con
    con.executemany(
        """
        INSERT INTO analytics.config_values (
            repo, commit, config_path, format, key, reference_paths,
            reference_modules, reference_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "other/repo",
                "deadbeef",
                "cfg/other.yaml",
                "yaml",
                "feature.flag",
                json.dumps(["cfg/other.yaml"]),
                json.dumps(["other.mod"]),
                1,
            )
        ],
    )

    mapping = load_config_key_map(con, ctx.repo, ctx.commit)

    expect_in("pkg.mod_a", mapping)
    expect_in("pkg.mod_b", mapping)
    expect_not_in("other.mod", mapping)
    expect_in("database.host", mapping["pkg.mod_a"])
    expect_in("cache.ttl", mapping["pkg.mod_b"])


def test_classify_modes_prioritizes_specific_matchers(
    dependencies_ctx: DependenciesFixture,
) -> None:
    """Ensure matcher ordering and fallback semantics behave as expected."""
    pattern = dependencies_ctx.patterns["requests"]

    modes, matched = classify_modes(pattern, "get", "requests.get")
    expect_equal(modes, ["read"])
    expect_is_not_none(matched)
    if matched is not None:
        expect_equal(matched.modes, ["read"])

    modes_with_prefix, matched_prefix = classify_modes(pattern, "post_json", "requests.post_json")
    expect_in("write", modes_with_prefix)
    expect_is_not_none(matched_prefix)

    modes_unknown, matched_unknown = classify_modes(pattern, "head", "requests.head")
    expect_equal(modes_unknown, ["unknown"])
    expect_is_none(matched_unknown)


def test_severity_and_risk_scores() -> None:
    """Map severities to scores and derive risk scores."""
    expect_equal(severity_score("high"), 3.0)
    expect_is_none(severity_score("unknown"))
    expect_equal(risk_score("high", 2.0), 6.0)
    expect_is_none(risk_score(None, 2.0))


def test_risk_level_balances_modes_and_frequency() -> None:
    """Derive risk level from usage modes and callsite frequency."""
    expect_equal(risk_level({"write"}, 1), "high")
    expect_equal(risk_level({"read"}, CALLSITE_MEDIUM_THRESHOLD), "medium")
    expect_equal(risk_level({"read"}, 5), "low")


def test_severity_scores_constant() -> None:
    """Verify SEVERITY_SCORES constant includes expected severities."""
    expect_equal(len(SEVERITY_SCORES), EXPECTED_SEVERITY_COUNT)
    expect_in("critical", SEVERITY_SCORES)


def test_build_alias_maps_handles_dotted_imports(
    dependencies_ctx: DependenciesFixture,
) -> None:
    """Verify alias maps are built for multiple files.

    Raises
    ------
    AssertionError
        If the alias map paths do not match the expected sources.
    """
    repo_root = dependencies_ctx.ctx.repo_root
    sources = dependency_alias_sources()
    module_map = {name: f"pkg.{name.removesuffix('.py')}" for name in sources}
    for name, source in sources.items():
        (repo_root / name).write_text(source, encoding="utf-8")

    alias_maps = build_alias_maps(repo_root, module_map)
    expected_paths = sorted(sources)
    actual_paths = sorted(alias_maps)
    if actual_paths != expected_paths:
        raise AssertionError(
            format_missing_extra(
                expected_paths,
                actual_paths,
                options=MissingExtraOptions(
                    noun="alias map paths",
                    context="dependency alias maps",
                ),
            )
        )
    expect_equal(alias_maps["a.py"], {"rq": "requests"})
    expect_equal(alias_maps["b.py"], {"create_engine": "sqlalchemy"})


def test_build_alias_map_variants() -> None:
    """Verify alias map construction covers import styles."""
    source = """
import requests
import pandas as pd
from sqlalchemy import create_engine
from os.path import join
"""
    tree = ast.parse(source)
    alias_map = build_alias_map(tree)
    expected = {
        "requests": "requests",
        "pd": "pandas",
        "create_engine": "sqlalchemy",
        "join": "os",
    }
    expect_equal(alias_map, expected)


def test_dependency_call_visitor_detects_calls(dependencies_ctx: DependenciesFixture) -> None:
    """Verify DependencyCallVisitor captures dependency calls with snippets."""
    patterns = dependencies_ctx.patterns
    source = """
import requests
requests.get("http://example.com/users")
requests.post("http://example.com/users", data={})
"""
    tree = ast.parse(source)
    lines = source.splitlines()
    alias_map = {"requests": "requests"}

    visitor = DependencyCallVisitor(
        alias_map=alias_map,
        patterns=patterns,
        rel_path="test.py",
        lines=lines,
    )
    visitor.visit(tree)

    expect_length(visitor.calls, EXPECTED_REQUESTS_CALLS)
    modes = [call.modes for call in visitor.calls]
    expect_in(["read"], modes)
    expect_in(["write"], modes)
    expect_true(all(call.snippet for call in visitor.calls))


def test_dependency_call_visitor_ignores_unknown_libraries(
    dependencies_ctx: DependenciesFixture,
) -> None:
    """Verify calls to unknown libraries are ignored."""
    source = """
import unknown_lib
unknown_lib.do_something()
"""
    tree = ast.parse(source)
    lines = source.splitlines()
    alias_map = {"unknown_lib": "unknown_lib"}

    visitor = DependencyCallVisitor(
        alias_map=alias_map,
        patterns=dependencies_ctx.patterns,
        rel_path="test.py",
        lines=lines,
    )
    visitor.visit(tree)

    expect_length(visitor.calls, 0)


def test_group_calls_by_library_builds_graph(dependencies_ctx: DependenciesFixture) -> None:
    """Verify calls are grouped by library and edges are cycle free."""
    expect_in("requests", dependencies_ctx.patterns)
    calls = dependency_calls_sample()
    grouped = group_calls_by_library(calls)
    expect_length(grouped, EXPECTED_GROUPED_LIBRARIES)

    edges = [
        (library, call.target)
        for library, library_calls in grouped.items()
        for call in library_calls
    ]
    graph = build_dependency_graph(edges)
    assert_edge_count(graph, EXPECTED_GROUPED_EDGES)
    assert_no_cycles(graph)


def test_dependency_mode_pattern_matching() -> None:
    """Verify pattern matching across multiple criteria."""
    pattern = DependencyModePattern(
        modes=["admin"],
        method="admin_execute",
        match="DROP TABLE",
    )
    expect_true(pattern.matches("admin_execute", "db.admin_execute()"))
    expect_true(pattern.matches("execute", "db.execute('DROP TABLE users')"))
    expect_false(pattern.matches("execute", "db.execute('SELECT * FROM users')"))
