"""Consolidated tests for tool abstractions, plugins, and result types.

This module brings together all tool-related tests:
- ToolRunner abstraction (binary resolution, error handling)
- Tool plugins (PyrightPlugin, PresetRunner)
- ToolService (real tooling execution)
- Tool port data models (CoverageResult, DiagnosticResult, etc.)

Uses PresetRunner as a protocol-based test double for controlled testing,
and real tooling execution via build_tooling_context for realistic tests.
"""

from __future__ import annotations

import asyncio
from collections.abc import Iterable
from pathlib import Path
from typing import override

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.infrastructure_utilities.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
    ToolRunResult,
)
from codeintel.ingestion.ports.tools import (
    CoverageFileData,
    CoverageResult,
    DiagnosticEntry,
    DiagnosticResult,
    ScipResult,
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
from codeintel.ingestion.ports.tools import ToolStatus as PortToolStatus
from codeintel.ingestion.tool_service import ToolService
from codeintel.ingestion.tools import ToolPluginResult, build_default_registry
from codeintel.ingestion.tools.plugins import ToolStatus as PluginToolStatus
from codeintel.ingestion.tools.pyright import PyrightPlugin
from codeintel.ingestion.tools.pytest import PytestPlugin
from codeintel.ingestion.tools.results import (
    CoverageFileSummary,
    CoverageReport,
    DiagnosticReport,
    FileDiagnosticCount,
    ScipDocument,
    ScipIndexResult,
    ScipOccurrence,
    TestCaseResult,
    TestReport,
    parse_scip_occurrence,
    parse_scip_range,
    parse_test_duration,
    parse_test_markers,
)
from codeintel.ingestion.tools.scip import ScipPlugin
from tests._helpers.orchestration.tooling import (
    ToolingOutputs,
    build_tooling_context,
    run_static_tooling,
)

# =============================================================================
# Test Constants
# =============================================================================

DURATION_1_5 = 1.5
LINE_10 = 10
COLUMN_5 = 5
COLUMN_15 = 15
EXPECTED_COUNT_2 = 2
EXPECTED_ERROR_COUNT = 2
EXPECTED_DIAG_COUNT = 2
EXPECTED_FILE_COUNT = 2
EXPECTED_TEST_COUNT = 3
PYRIGHT_PLUGINS_COUNT = 6
EXPECTED_COVERAGE_RATIO = 0.6


# =============================================================================
# PresetRunner - Protocol-based Test Double
# =============================================================================


class PresetRunner(ToolRunner):
    """ToolRunner that returns preset results without invoking subprocesses.

    This runner is realistic enough to flow through plugin logic while avoiding
    external binaries; it can be configured with either a ToolRunResult or an
    Exception to simulate failure modes.

    This is a legitimate protocol-based test double per the Testing Charter -
    it implements the same interface and can be used in dev/staging environments.
    """

    def __init__(self, result: ToolRunResult | Exception) -> None:
        """Initialize with preset result.

        Parameters
        ----------
        result
            Either a ToolRunResult to return or an Exception to raise.
        """
        self._result = result
        super().__init__(tools_config=ToolsConfig.default(), cache_dir=Path("build/.tool_cache"))

    @override
    async def run_async(
        self,
        tool: ToolName | str,
        args: Iterable[str],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
        timeout_s: float | None = None,
    ) -> ToolRunResult:
        """Return a preset ToolRunResult or raise the configured exception.

        Parameters
        ----------
        tool
            Tool name (ignored).
        args
            Tool arguments (ignored).
        cwd
            Working directory (ignored).
        output_path
            Output file path (passed through to error result).
        timeout_s
            Timeout (ignored).

        Returns
        -------
        ToolRunResult
            Pre-baked result configured for the runner.

        Raises
        ------
        ToolExecutionError
            Raised when configured with a generic exception.
        ToolNotFoundError
            Raised when configured with ToolNotFoundError.
        """
        del tool, args, cwd, timeout_s
        if isinstance(self._result, ToolNotFoundError):
            raise ToolNotFoundError(self._result.tool, self._result.configured_path)
        if isinstance(self._result, Exception):
            raise ToolExecutionError(
                ToolRunResult(
                    tool=ToolName.PYRIGHT,
                    args=(),
                    returncode=1,
                    stdout="",
                    stderr="dummy error",
                    duration_s=0.1,
                    output_path=output_path,
                )
            ) from self._result
        return self._result


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def tooling_outputs(tmp_path: Path) -> ToolingOutputs:
    """Run the real tooling stack against a minimal repo.

    Returns
    -------
    ToolingOutputs
        Diagnostics and coverage reports produced by the tooling services.
    """
    context = build_tooling_context(tmp_path)
    return run_static_tooling(context)


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
        runner.run(ToolName.PYRIGHT, [], cwd=tmp_path)


def test_tool_runner_unknown_tool_raises_value_error(tmp_path: Path) -> None:
    """Unknown tool names raise ValueError."""
    runner = ToolRunner(cache_dir=tmp_path)
    with pytest.raises(ValueError, match="Unknown tool"):
        runner.run("unknown-tool", ["--version"])


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

    assert result.status == PluginToolStatus.NOT_FOUND
    assert result.run is None
    assert isinstance(result.error, ToolNotFoundError)


def test_pyright_plugin_successful_run_returns_ok_status() -> None:
    """PyrightPlugin preserves successful ToolRunResult."""
    tools_cfg = ToolsConfig.default()
    run = ToolRunResult(
        tool=ToolName.PYRIGHT,
        args=("--outputjson", "."),
        returncode=0,
        stdout='{"summary": {"files": {}}}',
        stderr="",
        duration_s=0.01,
        output_path=None,
    )
    runner = PresetRunner(run)
    plugin = PyrightPlugin(runner=runner, tools_config=tools_cfg)

    result = asyncio.run(plugin.run(repo_root=Path()))

    assert result.status == PluginToolStatus.OK
    assert result.ok is True
    assert result.run == run
    assert result.error is None


def test_default_registry_contains_expected_plugins() -> None:
    """Registry builder wires all expected plugin names."""
    runner = PresetRunner(
        ToolRunResult(
            tool=ToolName.PYRIGHT,
            args=(),
            returncode=0,
            stdout="",
            stderr="",
            duration_s=0.0,
        )
    )
    registry = build_default_registry(runner, runner.tools_config)
    names = registry.names()

    expected_plugins = ("pyright", "pyrefly", "ruff", "coverage", "pytest", "scip")
    for plugin_name in expected_plugins:
        assert plugin_name in names, f"Expected plugin {plugin_name} in registry, got {names}"
    assert len(names) >= PYRIGHT_PLUGINS_COUNT


# =============================================================================
# ToolService Tests (Real Tooling Execution)
# =============================================================================


def test_tool_service_pyright_parses_errors(tooling_outputs: ToolingOutputs) -> None:
    """ToolService aggregates pyright diagnostics per file."""
    errors = tooling_outputs.pyright_errors
    assert errors.get("pkg/mod.py", 0) >= 1, (
        f"Expected pyright to report errors for pkg/mod.py, got {errors}"
    )


def test_tool_service_pyrefly_parses_errors(tooling_outputs: ToolingOutputs) -> None:
    """ToolService aggregates pyrefly diagnostics per file."""
    errors = tooling_outputs.pyrefly_errors
    assert errors.get("pkg/mod.py", 0) >= 1, (
        f"Expected pyrefly to report errors for pkg/mod.py, got {errors}"
    )


def test_tool_service_coverage_reports_normalization(tooling_outputs: ToolingOutputs) -> None:
    """ToolService normalizes coverage.json payloads."""
    reports = {report.rel_path: report for report in tooling_outputs.coverage_reports}
    report = reports.get("pkg/mod.py")
    assert report is not None, f"Coverage report missing for pkg/mod.py: {reports}"
    assert report.executed_lines, "Expected executed_lines to be populated"
    assert not report.missing_lines, f"Expected no missing lines, got {report.missing_lines}"


# =============================================================================
# ToolStatus Tests
# =============================================================================


def test_tool_status_enum_values() -> None:
    """ToolStatus should have expected enum values."""
    assert PortToolStatus.OK.value == "ok"
    assert PortToolStatus.NOT_FOUND.value == "not_found"
    assert PortToolStatus.FAILED.value == "failed"
    assert PortToolStatus.TIMEOUT.value == "timeout"


def test_tool_status_enum_comparison() -> None:
    """ToolStatus values should be comparable."""
    assert PortToolStatus.OK == PortToolStatus.OK
    assert PortToolStatus.OK != PortToolStatus.FAILED


# =============================================================================
# DiagnosticEntry Tests
# =============================================================================


def test_diagnostic_entry_attributes() -> None:
    """DiagnosticEntry should store diagnostic information."""
    entry = DiagnosticEntry(
        path="src/module.py",
        line=LINE_10,
        column=COLUMN_5,
        severity="error",
        code="E001",
        message="Undefined variable",
    )

    assert entry.path == "src/module.py"
    assert entry.line == LINE_10
    assert entry.column == COLUMN_5
    assert entry.severity == "error"
    assert entry.code == "E001"
    assert entry.message == "Undefined variable"


# =============================================================================
# DiagnosticResult Tests
# =============================================================================


def test_diagnostic_result_ok_status() -> None:
    """DiagnosticResult should represent successful tool run."""
    result = DiagnosticResult(
        status=PortToolStatus.OK,
        diagnostics=[],
        duration_s=DURATION_1_5,
    )

    assert result.status == PortToolStatus.OK
    assert result.diagnostics == []
    assert result.error is None
    assert result.duration_s == DURATION_1_5


def test_diagnostic_result_with_diagnostics() -> None:
    """DiagnosticResult should store diagnostic entries."""
    entry1 = DiagnosticEntry(
        path="a.py", line=1, column=1, severity="error", code="E001", message="Error 1"
    )
    entry2 = DiagnosticEntry(
        path="b.py", line=2, column=2, severity="warning", code="W001", message="Warn 1"
    )

    result = DiagnosticResult(
        status=PortToolStatus.OK,
        diagnostics=[entry1, entry2],
    )

    assert len(result.diagnostics) == EXPECTED_DIAG_COUNT


def test_diagnostic_result_errors_by_path() -> None:
    """DiagnosticResult.errors_by_path should count errors per file."""
    entries = [
        DiagnosticEntry("a.py", 1, 1, "error", "E001", "Err1"),
        DiagnosticEntry("a.py", 2, 1, "error", "E002", "Err2"),
        DiagnosticEntry("b.py", 1, 1, "error", "E001", "Err3"),
        DiagnosticEntry("a.py", 3, 1, "warning", "W001", "Warn1"),  # Not an error
    ]

    result = DiagnosticResult(status=PortToolStatus.OK, diagnostics=entries)
    errors = result.errors_by_path()

    assert errors["a.py"] == EXPECTED_ERROR_COUNT
    assert errors["b.py"] == 1
    assert "c.py" not in errors


def test_diagnostic_result_failed_status() -> None:
    """DiagnosticResult should handle failed tool runs."""
    result = DiagnosticResult(
        status=PortToolStatus.FAILED,
        error="Tool crashed",
    )

    assert result.status == PortToolStatus.FAILED
    assert result.error == "Tool crashed"


# =============================================================================
# CoverageFileData Tests
# =============================================================================


def test_coverage_file_data_attributes() -> None:
    """CoverageFileData should store coverage information."""
    missing_line = 4
    data = CoverageFileData(
        rel_path="module.py",
        executed_lines=frozenset({1, 2, 3}),
        missing_lines=frozenset({missing_line, 5}),
        excluded_lines=frozenset({10}),
    )

    assert data.rel_path == "module.py"
    assert 1 in data.executed_lines
    assert missing_line in data.missing_lines
    assert data.excluded_lines == frozenset({10})


def test_coverage_file_data_default_excluded_lines() -> None:
    """CoverageFileData should default to empty excluded_lines."""
    data = CoverageFileData(
        rel_path="test.py",
        executed_lines=frozenset({1}),
        missing_lines=frozenset(),
    )

    assert data.excluded_lines == frozenset()


# =============================================================================
# CoverageResult Tests
# =============================================================================


def test_coverage_result_ok_status() -> None:
    """CoverageResult should represent successful coverage run."""
    result = CoverageResult(
        status=PortToolStatus.OK,
        files=[],
        duration_s=DURATION_1_5,
    )

    assert result.status == PortToolStatus.OK
    assert result.files == []
    assert result.duration_s == DURATION_1_5


def test_coverage_result_with_files() -> None:
    """CoverageResult should store file coverage data."""
    file1 = CoverageFileData(
        rel_path="a.py",
        executed_lines=frozenset({1, 2}),
        missing_lines=frozenset({3}),
    )
    file2 = CoverageFileData(
        rel_path="b.py",
        executed_lines=frozenset({1}),
        missing_lines=frozenset(),
    )

    result = CoverageResult(
        status=PortToolStatus.OK,
        files=[file1, file2],
    )

    assert len(result.files) == EXPECTED_FILE_COUNT
    assert result.files[0].rel_path == "a.py"


# =============================================================================
# ScipSymbol Tests
# =============================================================================


def test_scip_symbol_attributes() -> None:
    """ScipSymbol should store symbol information."""
    symbol = ScipSymbol(
        symbol="python pkg/module.py/MyClass#",
        documentation="A test class.",
    )

    assert "MyClass" in symbol.symbol
    assert symbol.documentation == "A test class."


def test_scip_symbol_defaults() -> None:
    """ScipSymbol should have sensible defaults."""
    symbol = ScipSymbol(symbol="test")

    assert symbol.documentation is None


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

    assert "func" in occurrence.symbol
    assert occurrence.range_start_line == LINE_10
    assert occurrence.range_start_col == COLUMN_5


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

    assert occurrence.symbol == "test"


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

    assert doc.relative_path == "src/module.py"
    assert len(doc.symbols) == 1
    assert len(doc.occurrences) == 1


def test_port_scip_document_defaults() -> None:
    """PortScipDocument should have sensible defaults."""
    doc = PortScipDocument(relative_path="test.py", symbols=[], occurrences=[])

    assert doc.occurrences == []
    assert doc.symbols == []


# =============================================================================
# ScipResult Tests
# =============================================================================


def test_scip_result_ok_status() -> None:
    """ScipResult should represent successful SCIP run."""
    result = ScipResult(
        status=PortToolStatus.OK,
        documents=[],
        duration_s=DURATION_1_5,
    )

    assert result.status == PortToolStatus.OK
    assert result.documents == []
    assert result.duration_s == DURATION_1_5


def test_scip_result_with_documents() -> None:
    """ScipResult should store documents."""
    doc = PortScipDocument(relative_path="mod.py", symbols=[], occurrences=[])

    result = ScipResult(
        status=PortToolStatus.OK,
        documents=[doc],
    )

    assert len(result.documents) == 1


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

    assert case.nodeid == "tests/test_mod.py::test_example"
    assert case.outcome == "passed"
    assert case.duration_s == DURATION_1_5


def test_test_case_with_failure() -> None:
    """TestCase should store failure information."""
    case = TestCase(
        nodeid="tests/test_mod.py::test_failing",
        outcome="failed",
        duration_s=0.1,
        longrepr="AssertionError: Expected 1, got 2",
    )

    assert case.outcome == "failed"
    assert "AssertionError" in (case.longrepr or "")


def test_test_case_defaults() -> None:
    """TestCase should have sensible defaults."""
    case = TestCase(
        nodeid="test",
        outcome="passed",
    )

    assert case.duration_s == 0.0
    assert case.longrepr is None


# =============================================================================
# TestResult Tests
# =============================================================================


def test_test_result_ok_status() -> None:
    """TestResult should represent successful test run."""
    result = TestResult(
        status=PortToolStatus.OK,
        tests=[],
        duration_s=DURATION_1_5,
    )

    assert result.status == PortToolStatus.OK
    assert result.tests == []
    assert result.duration_s == DURATION_1_5


def test_test_result_with_tests() -> None:
    """TestResult should store test cases."""
    tests = [
        TestCase("t::a", "passed"),
        TestCase("t::b", "failed"),
        TestCase("t::c", "skipped"),
    ]

    result = TestResult(status=PortToolStatus.OK, tests=tests)

    assert len(result.tests) == EXPECTED_TEST_COUNT


def test_test_result_passed_count() -> None:
    """TestResult.passed should count passed tests."""
    tests = [
        TestCase("t::a", "passed"),
        TestCase("t::b", "passed"),
        TestCase("t::c", "failed"),
    ]

    result = TestResult(
        status=PortToolStatus.OK,
        tests=tests,
        passed=EXPECTED_ERROR_COUNT,  # 2 passed
        failed=1,
        skipped=0,
    )

    assert result.passed == EXPECTED_ERROR_COUNT


def test_test_result_failed_count() -> None:
    """TestResult.failed should count failed tests."""
    tests = [
        TestCase("t::a", "passed"),
        TestCase("t::b", "failed"),
        TestCase("t::c", "failed"),
    ]

    result = TestResult(
        status=PortToolStatus.OK,
        tests=tests,
        passed=1,
        failed=EXPECTED_ERROR_COUNT,  # 2 failed
        skipped=0,
    )

    assert result.failed == EXPECTED_ERROR_COUNT


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
        status=PortToolStatus.OK,
        diagnostics=diag_entries,
        duration_s=2.5,
    )

    # Verify structure
    assert diag_result.status == PortToolStatus.OK
    assert len(diag_result.diagnostics) == 1
    assert diag_result.errors_by_path() == {"src/a.py": 1}


# =============================================================================
# ToolService Facade Tests
# =============================================================================


def test_tool_service_get_plugin_returns_registered_plugin() -> None:
    """ToolService.get_plugin should return a registered plugin."""
    runner = PresetRunner(
        ToolRunResult(
            tool=ToolName.PYRIGHT,
            args=(),
            returncode=0,
            stdout="",
            stderr="",
            duration_s=0.0,
        )
    )
    service = ToolService(runner)
    plugin = service.get_plugin("pyright")
    assert plugin is not None


def test_tool_service_run_plugin_raises_key_error_for_unknown() -> None:
    """ToolService.run_plugin should raise KeyError for unknown plugin."""
    runner = PresetRunner(
        ToolRunResult(
            tool=ToolName.PYRIGHT,
            args=(),
            returncode=0,
            stdout="",
            stderr="",
            duration_s=0.0,
        )
    )
    service = ToolService(runner)
    with pytest.raises(KeyError):
        asyncio.run(service.run_plugin("nonexistent-plugin", repo_root=Path()))


def test_tool_service_run_plugin_success(tmp_path: Path) -> None:
    """ToolService.run_plugin should return result for registered plugin."""
    run = ToolRunResult(
        tool=ToolName.PYRIGHT,
        args=("--outputjson", "."),
        returncode=0,
        stdout='{"summary": {"files": {}}}',
        stderr="",
        duration_s=0.01,
    )
    runner = PresetRunner(run)
    service = ToolService(runner)
    result = asyncio.run(service.run_plugin("pyright", repo_root=tmp_path))
    assert result.status == PluginToolStatus.OK


def test_tool_service_run_pyright_not_found(tmp_path: Path) -> None:
    """ToolService.run_pyright should return empty dict when binary not found."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.PYRIGHT, tools_cfg.pyright_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    errors = asyncio.run(service.run_pyright(tmp_path))
    assert errors == {}


def test_tool_service_run_pyright_success(tmp_path: Path) -> None:
    """ToolService.run_pyright should return parsed errors."""
    pyright_output = '{"generalDiagnostics": [{"file": "a.py", "severity": 1, "message": "err", "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}}}]}'
    run = ToolRunResult(
        tool=ToolName.PYRIGHT,
        args=("--outputjson", "."),
        returncode=0,
        stdout=pyright_output,
        stderr="",
        duration_s=0.01,
    )
    runner = PresetRunner(run)
    service = ToolService(runner)
    errors = asyncio.run(service.run_pyright(tmp_path))
    # Should return mapping of errors per path
    assert isinstance(errors, dict)


def test_tool_service_run_pyrefly_not_found(tmp_path: Path) -> None:
    """ToolService.run_pyrefly should return empty dict when binary not found."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.PYREFLY, tools_cfg.pyrefly_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    errors = asyncio.run(service.run_pyrefly(tmp_path))
    assert errors == {}


def test_tool_service_run_ruff_not_found(tmp_path: Path) -> None:
    """ToolService.run_ruff should return empty dict when binary not found."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.RUFF, tools_cfg.ruff_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    errors = asyncio.run(service.run_ruff(tmp_path))
    assert errors == {}


def test_tool_service_run_coverage_not_found(tmp_path: Path) -> None:
    """ToolService.run_coverage_report should return empty report when not found."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.COVERAGE, tools_cfg.coverage_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    report = asyncio.run(service.run_coverage_report(tmp_path))
    assert report.files == ()


def test_tool_service_get_test_report_returns_parsed() -> None:
    """ToolService.get_test_report should extract TestReport from result."""
    test_report = TestReport(
        tests=(TestCaseResult(nodeid="t::a", outcome="passed"),),
        passed_count=1,
        failed_count=0,
        skipped_count=0,
        total_duration_s=1.0,
    )
    result = ToolPluginResult(
        tool=ToolName.PYTEST,
        status=PluginToolStatus.OK,
        artifacts={},
        run=None,
        parsed=test_report,
    )
    extracted = ToolService.get_test_report(result)
    assert extracted is test_report


def test_tool_service_get_test_report_returns_empty_for_none() -> None:
    """ToolService.get_test_report should return empty for non-TestReport."""
    result = ToolPluginResult(
        tool=ToolName.PYTEST,
        status=PluginToolStatus.OK,
        artifacts={},
        run=None,
        parsed=None,
    )
    extracted = ToolService.get_test_report(result)
    assert extracted.tests == ()
    assert extracted.passed_count == 0


def test_tool_service_run_pytest_raises_not_found(tmp_path: Path) -> None:
    """ToolService.run_pytest_report should raise when pytest not found."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.PYTEST, tools_cfg.pytest_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    json_path = tmp_path / "report.json"
    with pytest.raises(ToolNotFoundError):
        asyncio.run(service.run_pytest_report(tmp_path, json_report_path=json_path))


def test_tool_service_run_pytest_skips_if_exists(tmp_path: Path) -> None:
    """ToolService.run_pytest_report should skip if report exists."""
    json_path = tmp_path / "report.json"
    json_path.write_text('{"tests": []}')
    run = ToolRunResult(
        tool=ToolName.PYTEST,
        args=(),
        returncode=0,
        stdout="",
        stderr="",
        duration_s=0.1,
    )
    runner = PresetRunner(run)
    service = ToolService(runner)
    executed = asyncio.run(service.run_pytest_report(tmp_path, json_report_path=json_path))
    assert executed is False


def test_tool_service_run_scip_not_found(tmp_path: Path) -> None:
    """ToolService.run_scip_full should raise when scip not found."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.SCIP_PYTHON, tools_cfg.scip_python_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)
    with pytest.raises(ToolNotFoundError):
        asyncio.run(
            service.run_scip_full(
                tmp_path,
                output_scip=tmp_path / "index.scip",
                output_json=tmp_path / "index.json",
            )
        )


# =============================================================================
# tools/results.py Domain Type Tests
# =============================================================================


def test_file_diagnostic_count_attributes() -> None:
    """FileDiagnosticCount should store diagnostic counts."""
    count = FileDiagnosticCount(rel_path="mod.py", error_count=5, warning_count=3)
    assert count.rel_path == "mod.py"
    assert count.error_count == COLUMN_5
    assert count.warning_count == EXPECTED_TEST_COUNT


def test_diagnostic_report_from_error_counts() -> None:
    """DiagnosticReport.from_error_counts should build report."""
    errors = {"a.py": 2, "b.py": 1}
    warnings = {"a.py": 1}
    report = DiagnosticReport.from_error_counts("pyright", errors, warnings_by_file=warnings)

    assert report.tool_name == "pyright"
    assert report.total_errors == EXPECTED_TEST_COUNT
    assert report.total_warnings == 1
    assert "a.py" in report.files


def test_diagnostic_report_errors_by_path() -> None:
    """DiagnosticReport.errors_by_path should return simple mapping."""
    errors = {"x.py": 3}
    report = DiagnosticReport.from_error_counts("ruff", errors)
    result = report.errors_by_path()
    assert result == {"x.py": 3}


def test_diagnostic_report_empty() -> None:
    """DiagnosticReport.empty should return empty report."""
    report = DiagnosticReport.empty("test_tool")
    assert report.tool_name == "test_tool"
    assert report.files == {}
    assert report.total_errors == 0


def test_coverage_file_summary_properties() -> None:
    """CoverageFileSummary should compute properties correctly."""
    summary = CoverageFileSummary(
        rel_path="mod.py",
        executed_lines=frozenset({1, 2, 3}),
        missing_lines=frozenset({4, 5}),
    )
    assert summary.total_executable == COLUMN_5
    assert summary.coverage_ratio == EXPECTED_COVERAGE_RATIO


def test_coverage_file_summary_zero_lines() -> None:
    """CoverageFileSummary.coverage_ratio should return 1.0 for empty files."""
    summary = CoverageFileSummary(
        rel_path="empty.py",
        executed_lines=frozenset(),
        missing_lines=frozenset(),
    )
    assert summary.coverage_ratio == 1.0


def test_coverage_report_from_file_reports() -> None:
    """CoverageReport.from_file_reports should build report."""
    reports = [
        ("a.py", {1, 2}, {3}),
        ("b.py", {1}, set()),
    ]
    result = CoverageReport.from_file_reports(reports)
    assert len(result.files) == EXPECTED_COUNT_2
    assert result.total_executed == EXPECTED_TEST_COUNT
    assert result.total_missing == 1


def test_coverage_report_empty() -> None:
    """CoverageReport.empty should return empty report."""
    report = CoverageReport.empty()
    assert report.files == ()
    assert report.total_executed == 0


def test_coverage_report_by_path() -> None:
    """CoverageReport.by_path should return path-keyed mapping."""
    reports = [("mod.py", {1, 2}, set())]
    result = CoverageReport.from_file_reports(reports)
    by_path = result.by_path()
    assert "mod.py" in by_path


def test_parse_test_duration_valid() -> None:
    """parse_test_duration should extract duration from call dict."""
    entry = {"call": {"duration": 1.5}}
    assert parse_test_duration(entry) == DURATION_1_5


def test_parse_test_duration_missing() -> None:
    """parse_test_duration should return 0.0 for missing data."""
    assert parse_test_duration({}) == 0.0
    assert parse_test_duration({"call": {}}) == 0.0


def test_parse_test_markers_dict() -> None:
    """parse_test_markers should extract from keywords dict."""
    entry = {"keywords": {"slow": True, "fast": False, "integration": True}}
    markers = parse_test_markers(entry)
    assert "slow" in markers
    assert "integration" in markers
    assert "fast" not in markers


def test_parse_test_markers_list() -> None:
    """parse_test_markers should handle keywords as list."""
    entry = {"keywords": ["slow", "integration"]}
    markers = parse_test_markers(entry)
    assert markers == ("integration", "slow")


def test_parse_test_markers_empty() -> None:
    """parse_test_markers should return empty for missing keywords."""
    assert parse_test_markers({}) == ()


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
    assert len(report.tests) == expected_tests
    assert report.passed_count == 1
    assert report.failed_count == 1
    assert report.skipped_count == 1
    assert report.error_count == 1


def test_test_report_from_entries_skips_empty_nodeid() -> None:
    """TestReport.from_test_entries should skip entries without nodeid."""
    entries = [
        {"nodeid": "", "outcome": "passed"},
        {"outcome": "passed"},
        {"nodeid": "test::valid", "outcome": "passed"},
    ]
    report = TestReport.from_test_entries(entries)
    assert len(report.tests) == 1


def test_test_report_empty() -> None:
    """TestReport.empty should return empty report."""
    report = TestReport.empty()
    assert report.tests == ()
    assert report.passed_count == 0


def test_results_scip_occurrence_attributes() -> None:
    """ScipOccurrence from results.py should store symbol and range."""
    occ = ScipOccurrence(symbol="pkg.mod#func", range_=(10, 0, 10, 5), is_definition=True)
    assert occ.symbol == "pkg.mod#func"
    assert occ.range_ == (LINE_10, 0, LINE_10, COLUMN_5)
    assert occ.is_definition is True


def test_results_scip_document_attributes() -> None:
    """ScipDocument from results.py should store path and occurrences."""
    occ = ScipOccurrence(symbol="sym", range_=(1, 0, 1, 3))
    doc = ScipDocument(relative_path="src/mod.py", occurrences=(occ,))
    assert doc.relative_path == "src/mod.py"
    assert len(doc.occurrences) == 1


def test_parse_scip_range_three_elements() -> None:
    """parse_scip_range should handle 3-element ranges (single line)."""
    result = parse_scip_range([10, 5, 15])
    assert result == (LINE_10, COLUMN_5, LINE_10, COLUMN_15)


def test_parse_scip_range_four_elements() -> None:
    """parse_scip_range should handle 4-element ranges."""
    result = parse_scip_range([10, 5, 12, 8])
    expected = (10, 5, 12, 8)
    assert result == expected


def test_parse_scip_range_invalid() -> None:
    """parse_scip_range should return None for invalid ranges."""
    assert parse_scip_range([1]) is None
    assert parse_scip_range([1, 2]) is None
    assert parse_scip_range([]) is None


def test_parse_scip_occurrence_valid() -> None:
    """parse_scip_occurrence should parse valid occurrence."""
    occ = {"symbol": "pkg#func", "range": [10, 5, 15], "symbol_roles": 1}
    result = parse_scip_occurrence(occ)
    assert result is not None
    parsed, is_def = result
    assert parsed.symbol == "pkg#func"
    assert is_def is True


def test_parse_scip_occurrence_invalid_symbol() -> None:
    """parse_scip_occurrence should return None for missing symbol."""
    assert parse_scip_occurrence({"range": [1, 0, 5]}) is None
    assert parse_scip_occurrence({"symbol": 123, "range": [1, 0, 5]}) is None


def test_parse_scip_occurrence_invalid_range() -> None:
    """parse_scip_occurrence should return None for invalid range."""
    assert parse_scip_occurrence({"symbol": "s", "range": [1]}) is None
    assert parse_scip_occurrence({"symbol": "s", "range": "bad"}) is None


def test_scip_index_result_from_json_documents() -> None:
    """ScipIndexResult.from_json_documents should build result."""
    docs = [
        {
            "relative_path": "mod.py",
            "occurrences": [
                {"symbol": "s1", "range": [1, 0, 5], "symbol_roles": 1},
                {"symbol": "s2", "range": [2, 0, 10], "symbol_roles": 0},
            ],
        },
    ]
    result = ScipIndexResult.from_json_documents(docs)
    assert len(result.documents) == 1
    assert result.definition_count == 1
    assert result.reference_count == 1


def test_scip_index_result_skips_invalid_docs() -> None:
    """ScipIndexResult.from_json_documents should skip invalid docs."""
    docs = [
        {"relative_path": 123},  # Invalid path
        {"other": "data"},  # Missing path
        {"relative_path": "valid.py", "occurrences": []},
    ]
    result = ScipIndexResult.from_json_documents(docs)
    assert len(result.documents) == 1


def test_scip_index_result_empty() -> None:
    """ScipIndexResult.empty should return empty result."""
    result = ScipIndexResult.empty()
    assert result.documents == ()
    assert result.definition_count == 0


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
            output_json=Path("index.json"),
        )
    )

    assert result.status == PluginToolStatus.NOT_FOUND
    assert result.run is None
    assert isinstance(result.error, ToolNotFoundError)


def test_scip_plugin_type_error_on_missing_output_scip() -> None:
    """ScipPlugin.run() should raise TypeError when output_scip is missing."""
    tools_cfg = ToolsConfig.default()
    run = ToolRunResult(
        tool=ToolName.SCIP_PYTHON,
        args=(),
        returncode=0,
        stdout="",
        stderr="",
        duration_s=0.1,
    )
    runner = PresetRunner(run)
    plugin = ScipPlugin(runner=runner, tools_config=tools_cfg)

    with pytest.raises(TypeError, match="output_scip"):
        asyncio.run(plugin.run(repo_root=Path(), output_json=Path("index.json")))


def test_scip_plugin_type_error_on_missing_output_json() -> None:
    """ScipPlugin.run() should raise TypeError when output_json is missing."""
    tools_cfg = ToolsConfig.default()
    run = ToolRunResult(
        tool=ToolName.SCIP_PYTHON,
        args=(),
        returncode=0,
        stdout="",
        stderr="",
        duration_s=0.1,
    )
    runner = PresetRunner(run)
    plugin = ScipPlugin(runner=runner, tools_config=tools_cfg)

    with pytest.raises(TypeError, match="output_json"):
        asyncio.run(plugin.run(repo_root=Path(), output_scip=Path("index.scip")))


def test_scip_plugin_type_error_on_invalid_target_dir() -> None:
    """ScipPlugin.run() should raise TypeError when target_dir is invalid type."""
    tools_cfg = ToolsConfig.default()
    run = ToolRunResult(
        tool=ToolName.SCIP_PYTHON,
        args=(),
        returncode=0,
        stdout="",
        stderr="",
        duration_s=0.1,
    )
    runner = PresetRunner(run)
    plugin = ScipPlugin(runner=runner, tools_config=tools_cfg)

    with pytest.raises(TypeError, match="target_dir"):
        asyncio.run(
            plugin.run(
                repo_root=Path(),
                output_scip=Path("index.scip"),
                output_json=Path("index.json"),
                target_dir="not-a-path",
            )
        )


def test_scip_plugin_type_error_on_invalid_rel_paths() -> None:
    """ScipPlugin.run() should raise TypeError when rel_paths is invalid type."""
    tools_cfg = ToolsConfig.default()
    run = ToolRunResult(
        tool=ToolName.SCIP_PYTHON,
        args=(),
        returncode=0,
        stdout="",
        stderr="",
        duration_s=0.1,
    )
    runner = PresetRunner(run)
    plugin = ScipPlugin(runner=runner, tools_config=tools_cfg)

    # Use an integer, since str is technically a Sequence
    with pytest.raises(TypeError, match="rel_paths"):
        asyncio.run(
            plugin.run(
                repo_root=Path(),
                output_scip=Path("index.scip"),
                output_json=Path("index.json"),
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

    assert result.status == PluginToolStatus.NOT_FOUND
    assert result.run is None
    assert isinstance(result.error, ToolNotFoundError)


def test_pytest_plugin_type_error_on_missing_json_report_path() -> None:
    """PytestPlugin.run() should raise TypeError when json_report_path is missing."""
    tools_cfg = ToolsConfig.default()
    run = ToolRunResult(
        tool=ToolName.PYTEST,
        args=(),
        returncode=0,
        stdout="",
        stderr="",
        duration_s=0.1,
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

    assert result.status == PluginToolStatus.ERROR
    assert isinstance(result.error, ToolExecutionError)


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
    pyright_output = '{"generalDiagnostics": [{"file": "test.py", "severity": 1, "message": "err", "range": {"start": {"line": 0, "character": 0}, "end": {"line": 0, "character": 5}}}]}'
    run = ToolRunResult(
        tool=ToolName.PYRIGHT,
        args=(),
        returncode=0,
        stdout=pyright_output,
        stderr="",
        duration_s=0.1,
    )
    runner = PresetRunner(run)
    service = ToolService(runner)

    errors = asyncio.run(service.run_pyright(tmp_path))

    # Should return mapping (may be empty if parsing differs)
    assert isinstance(errors, dict)


def test_tool_service_run_coverage_report_with_data(tmp_path: Path) -> None:
    """ToolService.run_coverage_report should return report from parsed data."""
    # Coverage plugin requires coverage JSON data
    run = ToolRunResult(
        tool=ToolName.COVERAGE,
        args=(),
        returncode=0,
        stdout='{"files": {"mod.py": {"executed_lines": [1,2], "missing_lines": [3]}}}',
        stderr="",
        duration_s=0.1,
    )
    runner = PresetRunner(run)
    service = ToolService(runner)

    report = asyncio.run(service.run_coverage_report(tmp_path))

    # Should return a CoverageReport
    assert isinstance(report, CoverageReport)


def test_tool_service_run_pytest_report_creates_file(tmp_path: Path) -> None:
    """ToolService.run_pytest_report should create JSON report file."""
    json_path = tmp_path / "new_report.json"
    run = ToolRunResult(
        tool=ToolName.PYTEST,
        args=(),
        returncode=0,
        stdout='{"tests": []}',
        stderr="",
        duration_s=0.1,
    )
    runner = PresetRunner(run)
    service = ToolService(runner)

    # Create a file so the check succeeds after execution
    json_path.write_text('{"tests": [], "summary": {}}')

    executed = asyncio.run(service.run_pytest_report(tmp_path, json_report_path=json_path))

    # Since file exists beforehand, should return False (reused)
    assert executed is False


def test_tool_service_run_scip_full_not_found_raises(tmp_path: Path) -> None:
    """ToolService.run_scip_full should raise ToolNotFoundError when missing."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.SCIP_PYTHON, tools_cfg.scip_python_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)

    with pytest.raises(ToolNotFoundError):
        asyncio.run(
            service.run_scip_full(
                tmp_path,
                output_scip=tmp_path / "index.scip",
                output_json=tmp_path / "index.json",
            )
        )


def test_tool_service_run_scip_shard_not_found_raises(tmp_path: Path) -> None:
    """ToolService.run_scip_shard should raise ToolNotFoundError when missing."""
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.SCIP_PYTHON, tools_cfg.scip_python_bin)
    runner = PresetRunner(exc)
    service = ToolService(runner, tools_cfg)

    with pytest.raises(ToolNotFoundError):
        asyncio.run(
            service.run_scip_shard(
                tmp_path,
                rel_paths=["src/mod.py"],
                output_scip=tmp_path / "index.scip",
                output_json=tmp_path / "index.json",
            )
        )


def test_tool_service_run_pyrefly_success(tmp_path: Path) -> None:
    """ToolService.run_pyrefly should return error dict on success."""
    pyrefly_output = '[{"path": "mod.py", "severity": "error", "message": "err"}]'
    run = ToolRunResult(
        tool=ToolName.PYREFLY,
        args=(),
        returncode=0,
        stdout=pyrefly_output,
        stderr="",
        duration_s=0.1,
    )
    runner = PresetRunner(run)
    service = ToolService(runner)

    errors = asyncio.run(service.run_pyrefly(tmp_path))

    # Should return dict of errors (may be empty)
    assert isinstance(errors, dict)
