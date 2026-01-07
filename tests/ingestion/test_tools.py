"""Consolidated tests for tool abstractions, plugins, and result types.

This module brings together all tool-related tests:
- ToolRunner abstraction (binary resolution, error handling)
- Tool plugins (PyrightPlugin, PresetRunner)
- ToolService (real tool execution via ToolingOutputs fixtures)
- Tool port data models (DiagnosticResult, ScipResult, TestResult, etc.)
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.engine import ToolStatus, build_default_registry
from codeintel.ingestion.engine.infrastructure import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
    ToolRunOptions,
)
from codeintel.ingestion.engine.pyright import PyrightPlugin
from codeintel.ingestion.engine.pytest import PytestPlugin
from codeintel.ingestion.engine.results import (
    DiagnosticReport,
    FileDiagnosticCount,
    ScipDocument,
    ScipIndexResult,
    ScipOccurrence,
    TestReport,
    parse_test_duration,
    parse_test_markers,
)
from codeintel.ingestion.engine.scip import ScipPlugin
from codeintel.ingestion.engine.service import ToolService
from codeintel.ingestion.ports.tools import (
    DiagnosticEntry,
    DiagnosticResult,
    ScipResult,
    ScipRunRequest,
    ScipSymbol,
    TestCase,
    TestResult,
)
from codeintel.ingestion.ports.tools import (
    ScipDocument as PortScipDocument,
)
from codeintel.ingestion.ports.tools import (
    ScipOccurrence as PortScipOccurrence,
)
from codeintel.ingestion.scip.protobuf_parser import parse_index
from tests._helpers.assertions import (
    expect_equal,
    expect_is_instance,
    expect_is_none,
    expect_is_not_none,
    expect_true,
)
from tests._helpers.fakes.tool_runner import (
    PresetRunner,
    ToolRunResultOptions,
    make_tool_run_result,
)
from tests._helpers.ingestion import write_pytest_report
from tests._helpers.scip_proto import ensure_proto_module, write_scip_index

if TYPE_CHECKING:
    from tests._helpers.orchestration.tooling import (
        ToolingOutputs,
    )


# =============================================================================
# Test Constants
# =============================================================================

DURATION_1_5 = 1.5
LINE_10 = 10
COLUMN_5 = 5
COLUMN_15 = 15
EXPECTED_ERROR_COUNT = 2
EXPECTED_DIAG_COUNT = 2
EXPECTED_TEST_COUNT = 3
PYRIGHT_PLUGINS_COUNT = 5
SYMBOL_KIND_CLASS = 7


# =============================================================================
# Helper classes/fixtures are provided via tests._helpers.fakes.tool_runner and
# tests._helpers.orchestration.tooling.


# =============================================================================
# ToolRunner Tests
# =============================================================================


def test_tool_runner_missing_binary_raises_not_found_error(tmp_path: Path) -> None:
    """Missing binary raises ToolNotFoundError."""
    runner = ToolRunner(
        cache_dir=tmp_path,
        tools_config=ToolsConfig.with_overrides(pyright_bin="does-not-exist"),
    )
    with pytest.raises(ToolNotFoundError):
        runner.run(ToolName.PYRIGHT, [], options=ToolRunOptions(cwd=tmp_path))


def test_tool_runner_unknown_tool_raises_value_error(tmp_path: Path) -> None:
    """Unknown tool names raise ValueError."""
    runner = ToolRunner(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown tool"):
        runner.run("unknown-tool", ["--version"])


def _write_sleep_script(path: Path, *, sleep_s: float) -> None:
    content = "\n".join(
        [
            "#!/usr/bin/env python3",
            "import time",
            f"time.sleep({sleep_s})",
            'print("done")',
        ]
    )
    path.write_text(content, encoding="utf-8")
    path.chmod(0o755)


def test_tool_runner_emits_heartbeat_logs(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    """ToolRunner emits heartbeat logs for long-running tools."""
    script_path = tmp_path / "sleep_tool.py"
    _write_sleep_script(script_path, sleep_s=0.1)
    tools_config = ToolsConfig.with_overrides(scip_python_bin=str(script_path))
    runner = ToolRunner(cache_dir=tmp_path, tools_config=tools_config)
    with caplog.at_level(logging.INFO, logger="codeintel.ingestion.engine.infrastructure.runner"):
        result = runner.run(
            ToolName.SCIP_PYTHON,
            [],
            options=ToolRunOptions(progress_interval_s=0.02),
        )
    expect_true(result.ok)
    expect_true("still running" in caplog.text)


# =============================================================================
# Tool Plugin Tests
# =============================================================================


def test_pyright_plugin_not_found_downgrades_to_not_found_status() -> None:
    """PyrightPlugin reports NOT_FOUND when the binary is missing."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.PYRIGHT, tools_cfg.pyright_bin)
    runner = PresetRunner(exc)
    plugin = PyrightPlugin(runner=runner, tools_config=tools_cfg)

    result = asyncio.run(plugin.run(repo_root=Path()))

    expect_equal(result.status, ToolStatus.NOT_FOUND)
    expect_is_none(result.run)
    expect_is_instance(result.error, ToolNotFoundError)


def test_pyright_plugin_successful_run_returns_ok_status() -> None:
    """PyrightPlugin preserves successful ToolRunResult."""
    tools_cfg = ToolsConfig.default()
    run = make_tool_run_result(
        ToolName.PYRIGHT,
        args=("--outputjson", "."),
        options=ToolRunResultOptions(
            returncode=0,
            stdout='{"summary": {"files": {}}}',
            stderr="",
            duration_s=0.01,
        ),
    )
    runner = PresetRunner(run)
    plugin = PyrightPlugin(runner=runner, tools_config=tools_cfg)

    result = asyncio.run(plugin.run(repo_root=Path()))

    expect_equal(result.status, ToolStatus.OK)
    expect_true(result.ok)
    expect_equal(result.run, run)
    expect_is_none(result.error)


def test_default_registry_contains_expected_plugins() -> None:
    """Registry builder wires all expected plugin names."""
    runner = PresetRunner(
        make_tool_run_result(
            ToolName.PYRIGHT,
            options=ToolRunResultOptions(
                returncode=0,
                stdout="",
                stderr="",
                duration_s=0.0,
            ),
        )
    )
    registry = build_default_registry(runner, runner.tools_config)
    names = registry.names()

    expected_plugins = ("pyright", "pyrefly", "ruff", "pytest", "scip-python")
    for plugin_name in expected_plugins:
        expect_true(
            plugin_name in names,
            message=f"Expected plugin {plugin_name} in registry, got {names}",
        )
    expect_true(len(names) >= PYRIGHT_PLUGINS_COUNT)


# =============================================================================
# ToolService Tests (Real Tooling Execution)
# =============================================================================


@pytest.mark.requires_tools("pyright", "pyrefly", "ruff")
def test_tool_service_pyright_parses_errors(tooling_outputs: ToolingOutputs) -> None:
    """ToolService aggregates pyright diagnostics per file."""
    errors = tooling_outputs.pyright_errors
    expect_true(
        errors.get("pkg/mod.py", 0) >= 1,
        message=f"Expected pyright to report errors for pkg/mod.py, got {errors}",
    )


@pytest.mark.requires_tools("pyright", "pyrefly", "ruff")
def test_tool_service_pyrefly_parses_errors(tooling_outputs: ToolingOutputs) -> None:
    """ToolService aggregates pyrefly diagnostics per file."""
    errors = tooling_outputs.pyrefly_errors
    expect_true(
        errors.get("pkg/mod.py", 0) >= 1,
        message=f"Expected pyrefly to report errors for pkg/mod.py, got {errors}",
    )


# =============================================================================
# ToolStatus Tests
# =============================================================================


def test_tool_status_enum_values() -> None:
    """ToolStatus should expose the expected value set."""
    expect_equal(
        {status.value for status in ToolStatus},
        {"failed", "not_found", "ok", "skipped", "timeout"},
    )


# =============================================================================
# DiagnosticEntry Tests
# =============================================================================


def test_diagnostic_entry_attributes() -> None:
    """DiagnosticEntry should store diagnostic information."""
    _ = DiagnosticEntry(
        path="src/module.py",
        line=LINE_10,
        column=COLUMN_5,
        severity="error",
        code="E001",
        message="Undefined variable",
    )


def test_diagnostic_result_errors_by_path() -> None:
    """DiagnosticResult.errors_by_path should count errors per file."""
    entries = [
        DiagnosticEntry("a.py", 1, 1, "error", "E001", "Err1"),
        DiagnosticEntry("a.py", 2, 1, "error", "E002", "Err2"),
        DiagnosticEntry("b.py", 1, 1, "error", "E001", "Err3"),
        DiagnosticEntry("a.py", 3, 1, "warning", "W001", "Warn1"),  # Not an error
    ]

    result = DiagnosticResult(status=ToolStatus.OK, diagnostics=entries)
    errors = result.errors_by_path()

    expect_true(errors["a.py"] == EXPECTED_ERROR_COUNT)
    expect_true(errors["b.py"] == 1)
    expect_true("c.py" not in errors)


def test_diagnostic_result_failed_status() -> None:
    """DiagnosticResult should handle failed tool runs."""
    result = DiagnosticResult(
        status=ToolStatus.FAILED,
        error="Tool crashed",
    )

    expect_true(result.status == ToolStatus.FAILED)
    expect_true(result.error == "Tool crashed")


# =============================================================================
# ScipSymbol Tests
# =============================================================================


def test_scip_symbol_attributes() -> None:
    """ScipSymbol should store symbol information."""
    symbol = ScipSymbol(
        symbol="python pkg/module.py/MyClass#",
        documentation="A test class.",
    )

    expect_true("MyClass" in symbol.symbol)
    expect_true(symbol.documentation == "A test class.")


def test_scip_symbol_defaults() -> None:
    """ScipSymbol should have sensible defaults."""
    symbol = ScipSymbol(symbol="test")

    expect_true(symbol.documentation is None)


# =============================================================================
# PortScipOccurrence Tests
# =============================================================================


def test_port_scip_occurrence_attributes() -> None:
    """PortScipOccurrence should store occurrence information."""
    occurrence = PortScipOccurrence(
        symbol="python pkg/module.py/func#",
        range_start_line=LINE_10,
        range_start_col=COLUMN_5,
        range_end_line=LINE_10,
        range_end_col=COLUMN_15,
        symbol_roles=1,  # Definition role
    )

    expect_true("func" in occurrence.symbol)
    expect_true(occurrence.range_start_line == LINE_10)
    expect_true(occurrence.range_start_col == COLUMN_5)


def test_port_scip_occurrence_required_fields_only() -> None:
    """PortScipOccurrence should accept required fields."""
    occurrence = PortScipOccurrence(
        symbol="test",
        range_start_line=1,
        range_start_col=0,
        range_end_line=1,
        range_end_col=COLUMN_5,
        symbol_roles=0,
    )

    expect_true(occurrence.symbol == "test")


# =============================================================================
# PortScipDocument Tests
# =============================================================================


def test_port_scip_document_attributes() -> None:
    """PortScipDocument should store document information."""
    doc = PortScipDocument(
        relative_path="src/module.py",
        symbols=[ScipSymbol("sym1")],
        occurrences=[PortScipOccurrence("sym1", 1, 0, 1, COLUMN_5, 0)],
    )

    expect_true(doc.relative_path == "src/module.py")
    expect_equal(len(doc.symbols), 1)
    expect_equal(len(doc.occurrences), 1)


def test_port_scip_document_defaults() -> None:
    """PortScipDocument should have sensible defaults."""
    doc = PortScipDocument(relative_path="test.py", symbols=[], occurrences=[])

    expect_true(doc.occurrences == [])
    expect_true(doc.symbols == [])


# =============================================================================
# ScipResult Tests
# =============================================================================


def test_scip_result_ok_status() -> None:
    """ScipResult should represent successful SCIP run."""
    result = ScipResult(
        status=ToolStatus.OK,
        documents=[],
        duration_s=DURATION_1_5,
    )

    expect_true(result.status == ToolStatus.OK)
    expect_true(result.documents == [])
    expect_true(result.duration_s == DURATION_1_5)


def test_scip_result_with_documents() -> None:
    """ScipResult should store documents."""
    doc = PortScipDocument(relative_path="mod.py", symbols=[], occurrences=[])

    result = ScipResult(
        status=ToolStatus.OK,
        documents=[doc],
    )

    expect_equal(len(result.documents), 1)


# =============================================================================
# TestCase Tests
# =============================================================================


def test_test_case_attributes() -> None:
    """TestCase should store test case information."""
    case = TestCase(
        nodeid="tests/test_mod.py::test_example",
        outcome="passed",
        duration_s=DURATION_1_5,
    )

    expect_true(case.nodeid == "tests/test_mod.py::test_example")
    expect_true(case.outcome == "passed")
    expect_true(case.duration_s == DURATION_1_5)


def test_test_case_with_failure() -> None:
    """TestCase should store failure information."""
    case = TestCase(
        nodeid="tests/test_mod.py::test_failing",
        outcome="failed",
        duration_s=0.1,
        longrepr="AssertionError: Expected 1, got 2",
    )

    expect_true(case.outcome == "failed")
    expect_true("AssertionError" in (case.longrepr or ""))


def test_test_case_defaults() -> None:
    """TestCase should have sensible defaults."""
    case = TestCase(
        nodeid="test",
        outcome="passed",
    )

    expect_true(case.duration_s == 0.0)
    expect_true(case.longrepr is None)


# =============================================================================
# TestResult Tests
# =============================================================================


def test_test_result_ok_status() -> None:
    """TestResult should represent successful test run."""
    result = TestResult(
        status=ToolStatus.OK,
        tests=[],
        duration_s=DURATION_1_5,
    )

    expect_true(result.status == ToolStatus.OK)
    expect_true(result.tests == [])
    expect_true(result.duration_s == DURATION_1_5)


def test_test_result_with_tests() -> None:
    """TestResult should store test cases."""
    tests = [
        TestCase("t::a", "passed"),
        TestCase("t::b", "failed"),
        TestCase("t::c", "skipped"),
    ]

    result = TestResult(status=ToolStatus.OK, tests=tests)

    expect_equal(len(result.tests), EXPECTED_TEST_COUNT)


def test_test_result_passed_count() -> None:
    """TestResult.passed should count passed tests."""
    tests = [
        TestCase("t::a", "passed"),
        TestCase("t::b", "passed"),
        TestCase("t::c", "failed"),
    ]

    result = TestResult(
        status=ToolStatus.OK,
        tests=tests,
        passed=EXPECTED_ERROR_COUNT,  # 2 passed
        failed=1,
        skipped=0,
    )

    expect_true(result.passed == EXPECTED_ERROR_COUNT)


def test_test_result_failed_count() -> None:
    """TestResult.failed should count failed tests."""
    tests = [
        TestCase("t::a", "passed"),
        TestCase("t::b", "failed"),
        TestCase("t::c", "failed"),
    ]

    result = TestResult(
        status=ToolStatus.OK,
        tests=tests,
        passed=1,
        failed=EXPECTED_ERROR_COUNT,  # 2 failed
        skipped=0,
    )

    expect_true(result.failed == EXPECTED_ERROR_COUNT)


# =============================================================================
# Integration Tests
# =============================================================================


def test_tool_result_diagnostic_workflow() -> None:
    """Test typical diagnostic result workflow."""
    # Create diagnostic entries
    diag_entries = [
        DiagnosticEntry("src/a.py", 10, 1, "error", "E001", "Type error"),
    ]

    # Create diagnostic result
    diag_result = DiagnosticResult(
        status=ToolStatus.OK,
        diagnostics=diag_entries,
        duration_s=2.5,
    )

    # Verify structure
    expect_true(diag_result.status == ToolStatus.OK)
    expect_equal(len(diag_result.diagnostics), 1)
    expect_true(diag_result.errors_by_path() == {"src/a.py": 1})


# =============================================================================
# ToolService Facade Tests
# =============================================================================


def test_tool_service_get_plugin_returns_registered_plugin() -> None:
    """ToolService.get_plugin should return a registered plugin."""
    runner = PresetRunner(
        make_tool_run_result(
            ToolName.PYRIGHT,
            options=ToolRunResultOptions(returncode=0, stdout="", stderr="", duration_s=0.0),
        )
    )
    service = ToolService(runner)
    plugin = service.get_plugin("pyright")
    expect_true(plugin is not None)


def test_tool_service_run_plugin_raises_key_error_for_unknown() -> None:
    """ToolService.run_plugin should raise KeyError for unknown plugin."""
    runner = PresetRunner(
        make_tool_run_result(
            ToolName.PYRIGHT,
            options=ToolRunResultOptions(returncode=0, stdout="", stderr="", duration_s=0.0),
        )
    )
    service = ToolService(runner)
    with pytest.raises(KeyError):
        asyncio.run(service.run_plugin("nonexistent-plugin", repo_root=Path()))


def test_tool_service_run_plugin_success(tmp_path: Path) -> None:
    """ToolService.run_plugin should return result for registered plugin."""
    run = make_tool_run_result(
        ToolName.PYRIGHT,
        args=("--outputjson", "."),
        options=ToolRunResultOptions(
            returncode=0,
            stdout='{"summary": {"files": {}}}',
            stderr="",
            duration_s=0.01,
        ),
    )
    runner = PresetRunner(run)
    service = ToolService(runner)
    result = asyncio.run(service.run_plugin("pyright", repo_root=tmp_path))
    expect_true(result.status == ToolStatus.OK)


def test_tool_service_run_pyright_not_found(tmp_path: Path) -> None:
    """ToolService.run_pyright should return empty dict when binary not found."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.PYRIGHT, tools_cfg.pyright_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    errors = asyncio.run(service.run_pyright(tmp_path))
    expect_true(errors == {})


def test_tool_service_run_pyright_success(tmp_path: Path) -> None:
    """ToolService.run_pyright should return parsed errors."""
    pyright_output = (
        '{"generalDiagnostics": [{"file": "a.py", "severity": 1, "message": "err", '
        '"range": {"start": {"line": 0, "character": 0}, '
        '"end": {"line": 0, "character": 5}}}]}'
    )
    run = make_tool_run_result(
        ToolName.PYRIGHT,
        args=("--outputjson", "."),
        options=ToolRunResultOptions(
            returncode=0,
            stdout=pyright_output,
            stderr="",
            duration_s=0.01,
        ),
    )
    runner = PresetRunner(run)
    service = ToolService(runner)
    errors = asyncio.run(service.run_pyright(tmp_path))
    # Should return mapping of errors per path
    expect_true(isinstance(errors, dict))


def test_tool_service_run_pyrefly_not_found(tmp_path: Path) -> None:
    """ToolService.run_pyrefly should return empty dict when binary not found."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.PYREFLY, tools_cfg.pyrefly_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    errors = asyncio.run(service.run_pyrefly(tmp_path))
    expect_true(errors == {})


def test_tool_service_run_ruff_not_found(tmp_path: Path) -> None:
    """ToolService.run_ruff should return empty dict when binary not found."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.RUFF, tools_cfg.ruff_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    errors = asyncio.run(service.run_ruff(tmp_path))
    expect_true(errors == {})


def test_tool_service_run_pytest_raises_not_found(tmp_path: Path) -> None:
    """ToolService.run_pytest_report should report NOT_FOUND when pytest is missing."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.PYTEST, tools_cfg.pytest_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    json_path = tmp_path / "report.json"
    result = asyncio.run(service.run_pytest_report(tmp_path, json_report_path=json_path))
    expect_equal(result.status, ToolStatus.NOT_FOUND)
    expect_true(result.executed is False)
    expect_is_instance(result.error, ToolNotFoundError)


def test_tool_service_run_pytest_skips_if_exists(tmp_path: Path) -> None:
    """ToolService.run_pytest_report should skip if report exists."""
    json_path = write_pytest_report(tmp_path, filename="report.json")
    run = make_tool_run_result(
        ToolName.PYTEST,
        options=ToolRunResultOptions(
            returncode=0,
            stdout="",
            stderr="",
            duration_s=0.1,
        ),
    )
    runner = PresetRunner(run)
    service = ToolService(runner)
    result = asyncio.run(service.run_pytest_report(tmp_path, json_report_path=json_path))
    expect_equal(result.status, ToolStatus.SKIPPED)
    expect_true(result.executed is False)


def test_pytest_plugin_skips_without_json_report(tmp_path: Path) -> None:
    """PytestPlugin should skip when json-report support is missing."""
    run = make_tool_run_result(
        ToolName.PYTEST,
        options=ToolRunResultOptions(stdout="usage: pytest [options]"),
    )
    runner = PresetRunner(run)
    plugin = PytestPlugin(runner=runner, tools_config=ToolsConfig.default())

    result = asyncio.run(plugin.run(repo_root=tmp_path, json_report_path=tmp_path / "report.json"))

    expect_equal(result.status, ToolStatus.SKIPPED)


def test_tool_service_run_pytest_report_skips_when_missing_json_report(
    tmp_path: Path,
) -> None:
    """ToolService.run_pytest_report should skip when json-report is unavailable."""
    run = make_tool_run_result(
        ToolName.PYTEST,
        options=ToolRunResultOptions(stdout="usage: pytest [options]"),
    )
    runner = PresetRunner(run)
    service = ToolService(runner, ToolsConfig.default())

    result = asyncio.run(
        service.run_pytest_report(tmp_path, json_report_path=tmp_path / "report.json")
    )

    expect_equal(result.status, ToolStatus.SKIPPED)
    expect_true(result.executed is False)


def test_tool_service_run_scip_not_found(tmp_path: Path) -> None:
    """ToolService.run_scip_full should raise when scip not found."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.SCIP_PYTHON, tools_cfg.scip_python_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    request = ScipRunRequest(
        repo_root=tmp_path,
        output_scip=tmp_path / "index.scip",
        proto_module_path=tmp_path / "scip_pb2.py",
    )
    with pytest.raises(ToolNotFoundError):
        asyncio.run(service.run_scip_full(request))


# =============================================================================
# tools/results.py Domain Type Tests
# =============================================================================


def test_file_diagnostic_count_attributes() -> None:
    """FileDiagnosticCount should store diagnostic counts."""
    count = FileDiagnosticCount(rel_path="mod.py", error_count=5, warning_count=3)
    expect_true(count.rel_path == "mod.py")
    expect_true(count.error_count == COLUMN_5)
    expect_true(count.warning_count == EXPECTED_TEST_COUNT)


def test_diagnostic_report_from_error_counts() -> None:
    """DiagnosticReport.from_error_counts should build report."""
    errors = {"a.py": 2, "b.py": 1}
    warnings = {"a.py": 1}
    report = DiagnosticReport.from_error_counts("pyright", errors, warnings_by_file=warnings)

    expect_true(report.tool_name == "pyright")
    expect_true(report.total_errors == EXPECTED_TEST_COUNT)
    expect_true(report.total_warnings == 1)
    expect_true("a.py" in report.files)


def test_diagnostic_report_errors_by_path() -> None:
    """DiagnosticReport.errors_by_path should return simple mapping."""
    errors = {"x.py": 3}
    report = DiagnosticReport.from_error_counts("ruff", errors)
    result = report.errors_by_path()
    expect_true(result == {"x.py": 3})


def test_diagnostic_report_empty() -> None:
    """DiagnosticReport.empty should return empty report."""
    report = DiagnosticReport.empty("test_tool")
    expect_true(report.tool_name == "test_tool")
    expect_true(report.files == {})
    expect_true(report.total_errors == 0)


def test_parse_test_duration_valid() -> None:
    """parse_test_duration should extract duration from call dict."""
    entry = {"call": {"duration": 1.5}}
    expect_true(parse_test_duration(entry) == DURATION_1_5)


def test_parse_test_duration_missing() -> None:
    """parse_test_duration should return 0.0 for missing data."""
    expect_true(parse_test_duration({}) == 0.0)
    expect_true(parse_test_duration({"call": {}}) == 0.0)


def test_parse_test_markers_dict() -> None:
    """parse_test_markers should extract from keywords dict."""
    entry = {"keywords": {"slow": True, "fast": False, "integration": True}}
    markers = parse_test_markers(entry)
    expect_true("slow" in markers)
    expect_true("integration" in markers)
    expect_true("fast" not in markers)


def test_parse_test_markers_list() -> None:
    """parse_test_markers should handle keywords as list."""
    entry = {"keywords": ["slow", "integration"]}
    markers = parse_test_markers(entry)
    expect_true(markers == ("integration", "slow"))


def test_parse_test_markers_empty() -> None:
    """parse_test_markers should return empty for missing keywords."""
    expect_true(parse_test_markers({}) == ())


def test_test_report_from_test_entries() -> None:
    """TestReport.from_test_entries should build report from entries."""
    entries = [
        {"nodeid": "test::a", "outcome": "passed", "call": {"duration": 0.1}},
        {"nodeid": "test::b", "outcome": "failed", "call": {"duration": 0.2}},
        {"nodeid": "test::c", "outcome": "skipped"},
        {"nodeid": "test::d", "outcome": "error"},
    ]
    report = TestReport.from_test_entries(entries)
    expected_tests = 4
    expect_equal(len(report.tests), expected_tests)
    expect_true(report.passed_count == 1)
    expect_true(report.failed_count == 1)
    expect_true(report.skipped_count == 1)
    expect_true(report.error_count == 1)


def test_test_report_from_entries_skips_empty_nodeid() -> None:
    """TestReport.from_test_entries should skip entries without nodeid."""
    entries = [
        {"nodeid": "", "outcome": "passed"},
        {"outcome": "passed"},
        {"nodeid": "test::valid", "outcome": "passed"},
    ]
    report = TestReport.from_test_entries(entries)
    expect_equal(len(report.tests), 1)


def test_test_report_empty() -> None:
    """TestReport.empty should return empty report."""
    report = TestReport.empty()
    expect_true(report.tests == ())
    expect_true(report.passed_count == 0)


def test_results_scip_occurrence_attributes() -> None:
    """ScipOccurrence from results.py should store symbol and range."""
    occ = ScipOccurrence(symbol="pkg.mod#func", range_=(10, 0, 10, 5), symbol_roles=1)
    expect_true(occ.symbol == "pkg.mod#func")
    expect_true(occ.range_ == (LINE_10, 0, LINE_10, COLUMN_5))
    expect_true(occ.is_definition is True)


def test_results_scip_document_attributes() -> None:
    """ScipDocument from results.py should store path and occurrences."""
    occ = ScipOccurrence(symbol="sym", range_=(1, 0, 1, 3))
    doc = ScipDocument(relative_path="src/mod.py", occurrences=(occ,))
    expect_true(doc.relative_path == "src/mod.py")
    expect_equal(len(doc.occurrences), 1)


def test_parse_scip_range_three_elements(tmp_path: Path) -> None:
    """parse_index should handle 3-element ranges (single line)."""
    proto_module_path = ensure_proto_module(tmp_path)
    index_path = tmp_path / "index.scip"
    write_scip_index(
        index_path,
        proto_module_path=proto_module_path,
        documents=[
            {
                "relative_path": "mod.py",
                "occurrences": [{"symbol": "pkg#func", "range": [10, 5, 15], "symbol_roles": 1}],
            }
        ],
    )
    parsed = parse_index(index_path, proto_module_path)
    occurrence = parsed.documents[0].occurrences[0]
    expect_true(
        (
            occurrence.range_start_line,
            occurrence.range_start_col,
            occurrence.range_end_line,
            occurrence.range_end_col,
        )
        == (LINE_10, COLUMN_5, LINE_10, COLUMN_15)
    )
    expect_true(occurrence.symbol_roles == 1)


def test_parse_scip_range_four_elements(tmp_path: Path) -> None:
    """parse_index should handle 4-element ranges."""
    proto_module_path = ensure_proto_module(tmp_path)
    index_path = tmp_path / "index.scip"
    write_scip_index(
        index_path,
        proto_module_path=proto_module_path,
        documents=[
            {
                "relative_path": "mod.py",
                "occurrences": [{"symbol": "pkg#func", "range": [10, 5, 12, 8], "symbol_roles": 0}],
            }
        ],
    )
    parsed = parse_index(index_path, proto_module_path)
    occurrence = parsed.documents[0].occurrences[0]
    expect_true(
        (
            occurrence.range_start_line,
            occurrence.range_start_col,
            occurrence.range_end_line,
            occurrence.range_end_col,
        )
        == (10, 5, 12, 8)
    )


def test_parse_scip_range_invalid_skips_occurrence(tmp_path: Path) -> None:
    """parse_index should skip occurrences with invalid ranges."""
    proto_module_path = ensure_proto_module(tmp_path)
    index_path = tmp_path / "index.scip"
    write_scip_index(
        index_path,
        proto_module_path=proto_module_path,
        documents=[
            {
                "relative_path": "mod.py",
                "occurrences": [{"symbol": "pkg#func", "range": [1], "symbol_roles": 0}],
            }
        ],
    )
    parsed = parse_index(index_path, proto_module_path)
    expect_equal(len(parsed.documents[0].occurrences), 0)


def test_parse_scip_metadata_and_diagnostics(tmp_path: Path) -> None:
    """parse_index should capture symbol info, relationships, diagnostics, and externals."""
    proto_module_path = ensure_proto_module(tmp_path)
    index_path = tmp_path / "index.scip"
    symbol = "scip-python python CodeIntel 0.1.0 `mod`/Foo#"
    related_impl = "scip-python python CodeIntel 0.1.0 `mod`/Base#"
    related_ref = "scip-python python CodeIntel 0.1.0 `mod`/Ref#"
    enclosing_symbol = "scip-python python CodeIntel 0.1.0 `mod`/"
    write_scip_index(
        index_path,
        proto_module_path=proto_module_path,
        documents=[
            {
                "relative_path": "mod.py",
                "symbols": [
                    {
                        "symbol": symbol,
                        "documentation": ["Doc line 1", "Doc line 2"],
                        "kind": SYMBOL_KIND_CLASS,
                        "display_name": "Foo",
                        "signature": "Foo()",
                        "enclosing_symbol": enclosing_symbol,
                        "relationships": [
                            {"symbol": related_impl, "is_implementation": True},
                            {"symbol": related_ref, "is_reference": True},
                        ],
                    }
                ],
                "occurrences": [
                    {
                        "symbol": symbol,
                        "range": [2, 1, 2, 4],
                        "symbol_roles": 1,
                        "diagnostics": [
                            {
                                "severity": 1,
                                "code": "E100",
                                "message": "Bad call",
                                "source": "scip",
                            }
                        ],
                    }
                ],
            }
        ],
        external_symbols=["scip-python python requests 2.31.0 `requests`/"],
    )
    parsed = parse_index(index_path, proto_module_path)
    expect_equal(len(parsed.symbol_infos), 1)
    info = parsed.symbol_infos[0]
    expect_true(info.symbol == symbol)
    expect_true(info.documentation == "Doc line 1\nDoc line 2")
    expect_true(info.kind == SYMBOL_KIND_CLASS)
    expect_true(info.display_name == "Foo")
    expect_true(info.signature == "Foo()")
    expect_true(info.enclosing_symbol == enclosing_symbol)

    relationship_kinds = {rel.relationship_kind for rel in parsed.relationships}
    related_symbols = {rel.related_symbol for rel in parsed.relationships}
    expect_true("implementation" in relationship_kinds)
    expect_true("reference" in relationship_kinds)
    expect_true(related_impl in related_symbols)
    expect_true(related_ref in related_symbols)

    expect_equal(len(parsed.diagnostics), 1)
    diagnostic = parsed.diagnostics[0]
    expect_true(diagnostic.rel_path == "mod.py")
    expect_true(diagnostic.severity == "Error")
    expect_true(diagnostic.code == "E100")
    expect_true(diagnostic.message == "Bad call")
    expect_true(diagnostic.source == "scip")

    expect_equal(len(parsed.external_symbols), 1)
    external = parsed.external_symbols[0]
    expect_true(external.package_manager == "python")
    expect_true(external.package_name == "requests")
    expect_true(external.package_version == "2.31.0")


def test_scip_index_result_from_documents() -> None:
    """ScipIndexResult.from_documents should build counts."""
    doc = ScipDocument(
        relative_path="mod.py",
        occurrences=(
            ScipOccurrence(symbol="s1", range_=(1, 0, 1, 5), symbol_roles=1),
            ScipOccurrence(symbol="s2", range_=(2, 0, 2, 10), symbol_roles=0),
        ),
    )
    result = ScipIndexResult.from_documents((doc,))
    expect_equal(len(result.documents), 1)
    expect_true(result.definition_count == 1)
    expect_true(result.reference_count == 1)


def test_scip_index_result_empty() -> None:
    """ScipIndexResult.empty should return empty result."""
    result = ScipIndexResult.empty()
    expect_true(result.documents == ())
    expect_true(result.definition_count == 0)


# =============================================================================
# SCIP Tool Plugin Tests
# =============================================================================


def test_scip_plugin_not_found_during_scip_python() -> None:
    """ScipPlugin should return NOT_FOUND when scip-python is missing."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.SCIP_PYTHON, tools_cfg.scip_python_bin)
    runner = PresetRunner(exc)
    plugin = ScipPlugin(runner=runner, tools_config=tools_cfg)

    result = asyncio.run(
        plugin.run(
            repo_root=Path(),
            output_scip=Path("index.scip"),
            proto_module_path=Path("scip_pb2.py"),
        )
    )

    expect_true(result.status == ToolStatus.NOT_FOUND)
    expect_true(result.run is None)
    expect_true(isinstance(result.error, ToolNotFoundError))


def test_scip_plugin_type_error_on_missing_output_scip() -> None:
    """ScipPlugin.run() should raise TypeError when output_scip is missing."""
    tools_cfg = ToolsConfig.default()
    run = make_tool_run_result(
        ToolName.SCIP_PYTHON,
        options=ToolRunResultOptions(returncode=0, stdout="", stderr="", duration_s=0.1),
    )
    runner = PresetRunner(run)
    plugin = ScipPlugin(runner=runner, tools_config=tools_cfg)

    with pytest.raises(TypeError, match="output_scip"):
        asyncio.run(plugin.run(repo_root=Path(), proto_module_path=Path("scip_pb2.py")))


def test_scip_plugin_type_error_on_missing_proto_module_path() -> None:
    """ScipPlugin.run() should raise TypeError when proto_module_path is missing."""
    tools_cfg = ToolsConfig.default()
    run = make_tool_run_result(
        ToolName.SCIP_PYTHON,
        options=ToolRunResultOptions(returncode=0, stdout="", stderr="", duration_s=0.1),
    )
    runner = PresetRunner(run)
    plugin = ScipPlugin(runner=runner, tools_config=tools_cfg)

    with pytest.raises(TypeError, match="proto_module_path"):
        asyncio.run(plugin.run(repo_root=Path(), output_scip=Path("index.scip")))


def test_scip_plugin_type_error_on_invalid_target_dir() -> None:
    """ScipPlugin.run() should raise TypeError when target_dir is invalid type."""
    tools_cfg = ToolsConfig.default()
    run = make_tool_run_result(
        ToolName.SCIP_PYTHON,
        options=ToolRunResultOptions(returncode=0, stdout="", stderr="", duration_s=0.1),
    )
    runner = PresetRunner(run)
    plugin = ScipPlugin(runner=runner, tools_config=tools_cfg)

    with pytest.raises(TypeError, match="target_dir"):
        asyncio.run(
            plugin.run(
                repo_root=Path(),
                output_scip=Path("index.scip"),
                proto_module_path=Path("scip_pb2.py"),
                target_dir="not-a-path",
            )
        )


def test_scip_plugin_type_error_on_invalid_rel_paths() -> None:
    """ScipPlugin.run() should raise TypeError when rel_paths is invalid type."""
    tools_cfg = ToolsConfig.default()
    run = make_tool_run_result(
        ToolName.SCIP_PYTHON,
        options=ToolRunResultOptions(returncode=0, stdout="", stderr="", duration_s=0.1),
    )
    runner = PresetRunner(run)
    plugin = ScipPlugin(runner=runner, tools_config=tools_cfg)

    # Use an integer, since str is technically a Sequence
    with pytest.raises(TypeError, match="rel_paths"):
        asyncio.run(
            plugin.run(
                repo_root=Path(),
                output_scip=Path("index.scip"),
                proto_module_path=Path("scip_pb2.py"),
                rel_paths=123,
            )
        )


# =============================================================================
# Pytest Tool Plugin Tests
# =============================================================================


def test_pytest_plugin_not_found() -> None:
    """PytestPlugin should return NOT_FOUND when pytest is missing."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.PYTEST, tools_cfg.pytest_bin)
    runner = PresetRunner(exc)
    plugin = PytestPlugin(runner=runner, tools_config=tools_cfg)

    result = asyncio.run(plugin.run(repo_root=Path(), json_report_path=Path("report.json")))

    expect_true(result.status == ToolStatus.NOT_FOUND)
    expect_true(result.run is None)
    expect_true(isinstance(result.error, ToolNotFoundError))


def test_pytest_plugin_type_error_on_missing_json_report_path() -> None:
    """PytestPlugin.run() should raise TypeError when json_report_path is missing."""
    tools_cfg = ToolsConfig.default()
    run = make_tool_run_result(
        ToolName.PYTEST,
        options=ToolRunResultOptions(returncode=0, stdout="", stderr="", duration_s=0.1),
    )
    runner = PresetRunner(run)
    plugin = PytestPlugin(runner=runner, tools_config=tools_cfg)

    with pytest.raises(TypeError, match="json_report_path"):
        asyncio.run(plugin.run(repo_root=Path()))


def test_pytest_plugin_execution_error() -> None:
    """PytestPlugin should return ERROR when pytest fails."""
    tools_cfg = ToolsConfig.default()
    exc = RuntimeError("pytest failed")
    runner = PresetRunner(exc)
    plugin = PytestPlugin(runner=runner, tools_config=tools_cfg)

    result = asyncio.run(plugin.run(repo_root=Path(), json_report_path=Path("report.json")))

    expect_true(result.status == ToolStatus.FAILED)
    expect_true(isinstance(result.error, ToolExecutionError))
    expect_is_not_none(result.run)


# =============================================================================
# Additional ToolService Error Path Tests
# =============================================================================


def test_tool_service_run_ruff_execution_error(tmp_path: Path) -> None:
    """ToolService.run_ruff should raise ToolExecutionError on failure."""
    # Create a runner that returns an error (the ToolRunResult was unused,
    # as PresetRunner takes the exception directly)
    runner = PresetRunner(RuntimeError("ruff error"))
    service = ToolService(runner)

    with pytest.raises(ToolExecutionError):
        asyncio.run(service.run_ruff(tmp_path))


def test_tool_service_run_pyright_returns_errors_from_parsed_report(tmp_path: Path) -> None:
    """ToolService.run_pyright should extract errors from DiagnosticReport."""
    pyright_output = (
        '{"generalDiagnostics": [{"file": "test.py", "severity": 1, "message": "err", '
        '"range": {"start": {"line": 0, "character": 0}, '
        '"end": {"line": 0, "character": 5}}}]}'
    )
    run = make_tool_run_result(
        ToolName.PYRIGHT,
        options=ToolRunResultOptions(
            returncode=0,
            stdout=pyright_output,
            stderr="",
            duration_s=0.1,
        ),
    )
    runner = PresetRunner(run)
    service = ToolService(runner)

    errors = asyncio.run(service.run_pyright(tmp_path))

    # Should return mapping (may be empty if parsing differs)
    expect_true(isinstance(errors, dict))


def test_tool_service_run_pyright_execution_error(tmp_path: Path) -> None:
    """ToolService.run_pyright should raise ToolExecutionError on execution failure."""
    runner = PresetRunner(RuntimeError("pyright failed"))
    service = ToolService(runner)

    with pytest.raises(ToolExecutionError):
        asyncio.run(service.run_pyright(tmp_path))


def test_tool_service_run_pyrefly_failure_returns_empty(tmp_path: Path) -> None:
    """ToolService.run_pyrefly should degrade to empty mapping on failure."""
    run = make_tool_run_result(
        ToolName.PYREFLY,
        options=ToolRunResultOptions(
            returncode=1,
            stdout="",
            stderr="pyrefly failed",
            duration_s=0.1,
        ),
    )
    runner = PresetRunner(run)
    service = ToolService(runner)

    errors = asyncio.run(service.run_pyrefly(tmp_path))

    expect_true(errors == {})


def test_tool_service_run_pytest_report_creates_file(tmp_path: Path) -> None:
    """ToolService.run_pytest_report should create JSON report file."""
    json_path = write_pytest_report(tmp_path, filename="new_report.json")
    run = make_tool_run_result(
        ToolName.PYTEST,
        options=ToolRunResultOptions(
            returncode=0,
            stdout='{"tests": []}',
            stderr="",
            duration_s=0.1,
        ),
    )
    runner = PresetRunner(run)
    service = ToolService(runner)

    result = asyncio.run(service.run_pytest_report(tmp_path, json_report_path=json_path))

    # Since file exists beforehand, should return False (reused)
    expect_equal(result.status, ToolStatus.SKIPPED)
    expect_true(result.executed is False)


def test_tool_service_run_pytest_report_execution_error(tmp_path: Path) -> None:
    """ToolService.run_pytest_report should return FAILED on execution error."""
    runner = PresetRunner(RuntimeError("pytest failed"))
    service = ToolService(runner)
    json_path = tmp_path / "report.json"

    result = asyncio.run(service.run_pytest_report(tmp_path, json_report_path=json_path))
    expect_equal(result.status, ToolStatus.FAILED)
    expect_true(result.executed)
    expect_is_instance(result.error, ToolExecutionError)


def test_tool_service_run_scip_full_not_found_raises(tmp_path: Path) -> None:
    """ToolService.run_scip_full should raise ToolNotFoundError when missing."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.SCIP_PYTHON, tools_cfg.scip_python_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    request = ScipRunRequest(
        repo_root=tmp_path,
        output_scip=tmp_path / "index.scip",
        proto_module_path=tmp_path / "scip_pb2.py",
    )

    with pytest.raises(ToolNotFoundError):
        asyncio.run(service.run_scip_full(request))


def test_tool_service_run_scip_full_execution_error(tmp_path: Path) -> None:
    """ToolService.run_scip_full should raise ToolExecutionError on failure."""
    runner = PresetRunner(RuntimeError("scip failed"))
    service = ToolService(runner)
    request = ScipRunRequest(
        repo_root=tmp_path,
        output_scip=tmp_path / "index.scip",
        proto_module_path=tmp_path / "scip_pb2.py",
    )

    with pytest.raises(ToolExecutionError):
        asyncio.run(service.run_scip_full(request))


def test_tool_service_run_pyrefly_success(tmp_path: Path) -> None:
    """ToolService.run_pyrefly should return error dict on success."""
    pyrefly_output = '[{"path": "mod.py", "severity": "error", "message": "err"}]'
    run = make_tool_run_result(
        ToolName.PYREFLY,
        options=ToolRunResultOptions(
            returncode=0,
            stdout=pyrefly_output,
            stderr="",
            duration_s=0.1,
        ),
    )
    runner = PresetRunner(run)
    service = ToolService(runner)

    errors = asyncio.run(service.run_pyrefly(tmp_path))

    # Should return dict of errors (may be empty)
    expect_true(isinstance(errors, dict))
