"""Helpers that exercise the tooling stack for tests."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from coverage import Coverage

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.adapters.tool_runner import ToolRunnerAdapter
from codeintel.ingestion.engine.infrastructure import (
    ToolName,
    ToolRunner,
    ToolRunOptions,
)
from codeintel.ingestion.engine.service import ToolService
from tests._helpers.ingestion import write_coverage_file
from tests._helpers.scip_proto import ensure_proto_module
from tests._helpers.tool_payloads import coverage_json_payload
from tests._helpers.tooling_audit import require_tooling

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from codeintel.ingestion.engine.infrastructure import (
        ToolRunResult,
    )
    from codeintel.ingestion.engine.results import CoverageFileSummary


def _ensure_ok(result: ToolRunResult, *, action: str) -> None:
    if result.ok:
        return
    message = f"{action} failed: code={result.returncode} stderr={result.stderr}"
    raise RuntimeError(message)


def _write_tooling_repo(repo_root: Path) -> Path:
    repo_root.mkdir(parents=True, exist_ok=True)
    (repo_root / "__init__.py").write_text(
        '"""Package marker for tooling fixture repository."""',
        encoding="utf8",
    )
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    (pkg_dir / "__init__.py").write_text(
        '"""Test package used by tooling integration fixtures."""',
        encoding="utf8",
    )
    mod_path = pkg_dir / "mod.py"
    mod_path.write_text(
        "\n".join(
            [
                '"""Deliberately small module used to exercise tooling diagnostics."""',
                "",
                "def bad_type(x: int) -> int:",
                '    """Return a value with an incorrect type to trigger static errors."""',
                '    return x + "a"',
                "",
                "def add(x: int, y: int) -> int:",
                '    """Add two integers."""',
                "    return x + y",
            ]
        ),
        encoding="utf8",
    )
    driver_path = repo_root / "runner.py"
    driver_path.write_text(
        "\n".join(
            [
                '"""Lightweight driver to execute tooling fixture functions."""',
                "",
                "from contextlib import suppress",
                "",
                "from pkg import mod",
                "",
                "mod.add(1, 2)",
                "with suppress(Exception):",
                "    mod.bad_type(1)",
            ]
        ),
        encoding="utf8",
    )
    tests_dir = repo_root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_tooling_repo.py").write_text(
        "\n".join(
            [
                '"""Smoke tests for tooling fixtures."""',
                "",
                "from pkg import mod",
                "",
                "def test_add() -> None:",
                "    assert mod.add(1, 2) == 3",
            ]
        ),
        encoding="utf8",
    )
    return driver_path


def make_tools_config(**overrides: str | float | Path | None) -> ToolsConfig:
    """Return a fully populated ToolsConfig with optional overrides.

    Returns
    -------
    ToolsConfig
        Base configuration populated with defaults and optional overrides applied.
    """
    if overrides:
        return ToolsConfig.with_overrides(**overrides)
    return ToolsConfig.default()


@dataclass(frozen=True)
class ToolingContext:
    """Context for invoking the real ToolRunner and ToolService."""

    repo_root: Path
    tools_config: ToolsConfig
    runner: ToolRunner
    service: ToolService
    coverage_file: Path
    driver_path: Path


@dataclass(frozen=True)
class ToolingOutputs:
    """Collected outputs from running static tooling end-to-end."""

    pyright_errors: dict[str, int]
    pyrefly_errors: dict[str, int]
    ruff_errors: dict[str, int]
    coverage_reports: Sequence[CoverageFileSummary]
    context: ToolingContext


@dataclass(frozen=True)
class ToolingArtifacts:
    """Prebuilt artifact bundle for adapter/service tests."""

    coverage_file: Path
    pytest_report: Path
    scip_index: Path
    service: ToolService
    adapter: ToolRunnerAdapter
    context: ToolingContext


def build_tooling_context(base_dir: Path) -> ToolingContext:
    repo_root = base_dir / "repo"
    driver_path = _write_tooling_repo(repo_root)
    tools_cfg = make_tools_config(
        coverage_file=repo_root / ".coverage",
        pytest_report_path=repo_root / "build" / "test-results" / "pytest-report.json",
    )
    require_tooling(
        tools_cfg,
        required_tools=(
            ToolName.COVERAGE,
            ToolName.PYRIGHT,
            ToolName.PYREFLY,
            ToolName.RUFF,
        ),
        repo_root=repo_root,
    )
    runner = ToolRunner(cache_dir=base_dir / ".tool_cache", tools_config=tools_cfg)
    service = ToolService(runner, tools_cfg)
    return ToolingContext(
        repo_root=repo_root,
        tools_config=tools_cfg,
        runner=runner,
        service=service,
        coverage_file=tools_cfg.coverage_file or repo_root / ".coverage",
        driver_path=driver_path,
    )


def run_static_tooling(context: ToolingContext) -> ToolingOutputs:
    """Run static tooling and collect outputs.

    Parameters
    ----------
    context
        Tooling context with repo root, runner, and service.

    Returns
    -------
    ToolingOutputs
        Collected diagnostics and coverage data.
    """
    coverage_result = context.runner.run(
        ToolName.COVERAGE,
        [
            "run",
            "--data-file",
            str(context.coverage_file),
            str(context.driver_path),
        ],
        options=ToolRunOptions(cwd=context.repo_root),
    )
    _ensure_ok(coverage_result, action="coverage run")

    pyright_errors = asyncio.run(context.service.run_pyright(context.repo_root))
    pyrefly_errors = asyncio.run(context.service.run_pyrefly(context.repo_root))
    ruff_errors = asyncio.run(context.service.run_ruff(context.repo_root))
    coverage_report = asyncio.run(
        context.service.run_coverage_report(
            context.repo_root,
            coverage_file=context.coverage_file,
        )
    )
    return ToolingOutputs(
        pyright_errors=dict(pyright_errors),
        pyrefly_errors=dict(pyrefly_errors),
        ruff_errors=dict(ruff_errors),
        coverage_reports=coverage_report.files,
        context=context,
    )


def build_tooling_artifacts(
    tmp_path: Path,
    *,
    tooling_outputs: ToolingOutputs | None = None,
) -> ToolingArtifacts:
    """Create coverage, pytest, and SCIP artifacts with a ready ToolRunnerAdapter.

    Returns
    -------
    ToolingArtifacts
        Bundle containing artifact paths plus a configured service/adapter pair.
    """
    tooling_config = (
        tooling_outputs.context.tools_config if tooling_outputs else ToolsConfig.default()
    )
    tooling_repo_root = tooling_outputs.context.repo_root if tooling_outputs else tmp_path
    require_tooling(
        tooling_config,
        required_tools=(
            ToolName.COVERAGE,
            ToolName.PYRIGHT,
            ToolName.PYREFLY,
            ToolName.RUFF,
            ToolName.PYTEST,
            ToolName.SCIP_PYTHON,
        ),
        repo_root=tooling_repo_root,
    )
    outputs = tooling_outputs or run_static_tooling(build_tooling_context(tmp_path))
    build_dir = tmp_path / "artifacts"
    coverage_json = coverage_json_payload(
        files={
            summary.rel_path: {
                "executed_lines": sorted(summary.executed_lines),
                "missing_lines": sorted(summary.missing_lines),
            }
            for summary in outputs.coverage_reports
        }
    )
    coverage_file = write_coverage_file(build_dir, content=coverage_json)
    pytest_report = _run_pytest_report(outputs.context, build_dir)
    scip_index = _run_scip_index(outputs.context, build_dir)
    adapter = ToolRunnerAdapter(outputs.context.service)
    return ToolingArtifacts(
        coverage_file=coverage_file,
        pytest_report=pytest_report,
        scip_index=scip_index,
        service=outputs.context.service,
        adapter=adapter,
        context=outputs.context,
    )


def _run_pytest_report(context: ToolingContext, build_dir: Path) -> Path:
    report_path = context.tools_config.pytest_report_path or (
        build_dir / "test-results" / "pytest-report.json"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    asyncio.run(context.service.run_pytest_report(context.repo_root, json_report_path=report_path))
    return report_path


def _run_scip_index(context: ToolingContext, build_dir: Path) -> Path:
    scip_dir = build_dir / "scip"
    scip_dir.mkdir(parents=True, exist_ok=True)
    output_scip = scip_dir / "index.scip"
    proto_module_path = ensure_proto_module(build_dir)
    result = asyncio.run(
        context.service.run_scip_full(
            context.repo_root,
            output_scip=output_scip,
            proto_module_path=proto_module_path,
        )
    )
    scip_index = result.index_scip_path or output_scip
    if not scip_index.is_file():
        message = f"SCIP artifact missing at {scip_index}"
        raise RuntimeError(message)
    return scip_index


@pytest.fixture
def tooling_outputs(tmp_path: Path) -> ToolingOutputs:
    """Fixture that runs real tooling outputs against a minimal repo.

    Returns
    -------
    ToolingOutputs
        Aggregated diagnostics and coverage results.
    """
    context = build_tooling_context(tmp_path)
    return run_static_tooling(context)


@pytest.fixture(scope="session")
def tooling_outputs_session(tmp_path_factory: pytest.TempPathFactory) -> ToolingOutputs:
    """Session-scoped tooling outputs.

    Returns
    -------
    ToolingOutputs
        Aggregated diagnostics and coverage results.
    """
    base_dir = tmp_path_factory.mktemp("tooling-session")
    context = build_tooling_context(base_dir)
    return run_static_tooling(context)


@dataclass(frozen=True)
class GitRepoContext:
    """Git repository seeded with multiple commits for history tests."""

    repo_root: Path
    runner: ToolRunner
    commits: tuple[str, ...]
    file_path: Path


def init_git_repo_with_history(base_dir: Path) -> GitRepoContext:
    repo_root = base_dir / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    runner = ToolRunner(cache_dir=base_dir / ".tool_cache")

    _ensure_ok(
        runner.run(
            ToolName.GIT,
            ["init", "-b", "main"],
            options=ToolRunOptions(cwd=repo_root),
        ),
        action="git init",
    )
    _ensure_ok(
        runner.run(
            ToolName.GIT,
            ["config", "user.email", "codeintel@example.com"],
            options=ToolRunOptions(cwd=repo_root),
        ),
        action="git config user.email",
    )
    _ensure_ok(
        runner.run(
            ToolName.GIT,
            ["config", "user.name", "CodeIntel Tester"],
            options=ToolRunOptions(cwd=repo_root),
        ),
        action="git config user.name",
    )

    file_path = repo_root / "pkg" / "foo.py"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(
        "\n".join(
            [
                "def foo() -> int:",
                "    return 1",
            ]
        ),
        encoding="utf8",
    )
    _ensure_ok(
        runner.run(
            ToolName.GIT,
            ["add", "."],
            options=ToolRunOptions(cwd=repo_root),
        ),
        action="git add initial",
    )
    _ensure_ok(
        runner.run(
            ToolName.GIT,
            ["commit", "-m", "Initial commit"],
            options=ToolRunOptions(cwd=repo_root),
        ),
        action="git commit initial",
    )
    first_commit = runner.run(
        ToolName.GIT,
        ["rev-parse", "HEAD"],
        options=ToolRunOptions(cwd=repo_root),
    ).stdout.strip()

    file_path.write_text(
        "\n".join(
            [
                "def foo() -> int:",
                "    total = 1",
                "    return total + 2",
            ]
        ),
        encoding="utf8",
    )
    _ensure_ok(
        runner.run(
            ToolName.GIT,
            ["add", "."],
            options=ToolRunOptions(cwd=repo_root),
        ),
        action="git add update",
    )
    _ensure_ok(
        runner.run(
            ToolName.GIT,
            ["commit", "-m", "Update foo"],
            options=ToolRunOptions(cwd=repo_root),
        ),
        action="git commit update",
    )
    second_commit = runner.run(
        ToolName.GIT,
        ["rev-parse", "HEAD"],
        options=ToolRunOptions(cwd=repo_root),
    ).stdout.strip()

    return GitRepoContext(
        repo_root=repo_root,
        runner=runner,
        commits=(second_commit, first_commit),
        file_path=file_path,
    )


@dataclass(frozen=True)
class CoverageArtifact:
    """Coverage artifact capturing contexts for a single test id."""

    repo_root: Path
    coverage_file: Path


def _load_module(module_import: str, module_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_import, module_path)
    if spec is None or spec.loader is None:
        message = f"Unable to load module {module_import} from {module_path}"
        raise RuntimeError(message)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_import] = module
    spec.loader.exec_module(module)
    return module


def generate_coverage_for_function(
    *,
    repo_root: Path,
    module_import: str,
    function_name: str,
    test_id: str,
    coverage_file: Path | None = None,
) -> CoverageArtifact:
    """
    Execute a module function under a specific coverage context.

    Parameters
    ----------
    repo_root
        Root of the repo containing the module to execute.
    module_import
        Import path for the module (e.g., "pkg.mod").
    function_name
        Function to invoke to mark executable lines.
    test_id
        Coverage context label (typically pytest node id).
    coverage_file
        Optional override for the coverage database path.

    Returns
    -------
    CoverageArtifact
        Paths to the repo root and generated coverage file.

    Raises
    ------
    RuntimeError
        If the module or target function cannot be loaded.
    TypeError
        If the resolved attribute is not callable.
    """
    target_cov = coverage_file or (repo_root / ".coverage")
    repo_root.mkdir(parents=True, exist_ok=True)
    target_cov.parent.mkdir(parents=True, exist_ok=True)
    module_parts = module_import.split(".")
    pkg_path = repo_root / module_parts[0]
    pkg_path.mkdir(parents=True, exist_ok=True)
    (pkg_path / "__init__.py").write_text(
        '"""Test package used by tooling integration fixtures."""',
        encoding="utf8",
    )
    module_path = (repo_root / Path(*module_parts)).with_suffix(".py")
    if not module_path.exists():
        message = f"Module path not found for {module_import}: {module_path}"
        raise RuntimeError(message)

    coverage = Coverage(data_file=str(target_cov), config_file=False)
    coverage.start()
    coverage.switch_context(test_id)
    try:
        module = _load_module(module_import, module_path)
        attribute = getattr(module, function_name, None)
        if attribute is None:
            message = f"Function {function_name} not found in {module_import}"
            raise RuntimeError(message)
        if not callable(attribute):
            message = f"Attribute {function_name} on {module_import} is not callable"
            raise TypeError(message)
        attribute()
    finally:
        coverage.stop()
        coverage.save()
    return CoverageArtifact(repo_root=repo_root, coverage_file=target_cov)


__all__ = [
    "CoverageArtifact",
    "GitRepoContext",
    "ToolingContext",
    "ToolingOutputs",
    "build_tooling_context",
    "generate_coverage_for_function",
    "init_git_repo_with_history",
    "make_tools_config",
    "run_static_tooling",
]
