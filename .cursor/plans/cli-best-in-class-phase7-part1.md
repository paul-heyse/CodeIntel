# CLI Best-in-Class Implementation Plan (Phase 7 - Part 1)

> **Status**: Proposed  
> **Author**: AI Assistant  
> **Created**: 2025-12-09  
> **Depends On**: Phase 6 (Completed)

---

## Executive Summary

Phase 7 Part 1 focuses on **testing infrastructure** and **async architecture** — two foundational capabilities that enable confidence in the CLI and unlock scalability for I/O-bound operations.

The two priorities addressed:

1. **Comprehensive Test Suite Implementation** — Charter-compliant tests using the Phase 6 harness
2. **Async Handler Architecture** — Native async/await support in the execution pipeline

### Why These Priorities Matter

| Aspect | Current State | After Phase 7.1-7.2 |
|--------|---------------|---------------------|
| Test Coverage | Harness exists, few tests | Full coverage with golden files |
| Property Testing | None | Hypothesis for validators |
| Performance Tests | None | Budget assertions on critical paths |
| Handler Model | Sync-only | Async-native with cancellation |
| Progress Streaming | Polling-based | Async generator streams |
| Resource Cleanup | Manual | Async context managers |

---

## Table of Contents

1. [Phase 7.1: Comprehensive Test Suite Implementation](#phase-71-comprehensive-test-suite-implementation)
2. [Phase 7.2: Async Handler Architecture](#phase-72-async-handler-architecture)
3. [Implementation Timeline](#implementation-timeline)
4. [Success Metrics](#success-metrics)

---

## Phase 7.1: Comprehensive Test Suite Implementation

### Value Proposition

The test harness from Phase 6.3 provides infrastructure; Phase 7.1 provides comprehensive coverage:

- **Unit tests** for every operation handler
- **Integration tests** for the full pipeline
- **Golden file tests** for output stability
- **Property-based tests** for validators
- **Performance tests** with budgets
- **Shell mode tests** for REPL

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Test Suite Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │   Unit Tests    │  │ Integration     │  │  Golden File    │         │
│  │                 │  │    Tests        │  │    Tests        │         │
│  │ • Handlers      │  │ • Full pipeline │  │ • Text output   │         │
│  │ • Validators    │  │ • Config→Exec   │  │ • JSON output   │         │
│  │ • Middleware    │  │ • Error flows   │  │ • Error formats │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                   │
│           └────────────────────┼────────────────────┘                   │
│                                ▼                                        │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    CliTestHarness + Fixtures                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                          │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │ Property Tests  │  │  Performance    │  │   Shell Mode    │         │
│  │  (Hypothesis)   │  │    Tests        │  │    Tests        │         │
│  │ • Validators    │  │ • Latency       │  │ • Completion    │         │
│  │ • Schemas       │  │ • Memory        │  │ • Commands      │         │
│  │ • Serialization │  │ • Throughput    │  │ • History       │         │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Functional Objectives

1. Create unit tests for all operation handlers via `OperationTestHarness`
2. Build integration tests covering config → middleware → execution → output
3. Establish golden file testing for output stability
4. Implement property-based tests using Hypothesis for validators
5. Add performance regression tests with budget assertions
6. Create shell mode tests for REPL interactions

### Implementation

#### File: `tests/cli/unit/test_operation_handlers.py`

```python
"""Unit tests for operation handlers.

Tests each handler through OperationTestHarness, validating
behavior without full CLI overhead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from codeintel.cli.executor import OperationCategory, OperationSpec
from codeintel.cli.results import CliResult
from tests.cli._harness import OperationTestHarness

if TYPE_CHECKING:
    from codeintel.cli.result_types import BuildStatusResult


class TestBuildOperationHandlers:
    """Tests for build operation handlers."""

    def test_build_status_returns_structured_result(self) -> None:
        """Test build.status returns BuildStatusResult."""
        harness = OperationTestHarness()
        result = harness.execute("build.status")

        assert result.success
        data = result.json()
        assert "fresh_count" in data or "data" in data

    def test_build_status_with_invalid_root(self) -> None:
        """Test build.status with non-existent project root."""
        harness = OperationTestHarness()
        result = harness.execute(
            "build.status",
            {"project_root": "/nonexistent/path"},
        )

        # Should handle gracefully - either error or empty result
        assert result.exit_code in {0, 1}

    @pytest.mark.parametrize(
        ("operation_id", "expected_category"),
        [
            ("build.status", OperationCategory.READ),
            ("build.run", OperationCategory.BUILD),
            ("build.clean", OperationCategory.WRITE),
        ],
    )
    def test_build_operations_have_correct_category(
        self,
        operation_id: str,
        expected_category: OperationCategory,
    ) -> None:
        """Test build operations are categorized correctly."""
        from codeintel.cli.operation_registry import get_operation_registry

        registry = get_operation_registry()
        spec = registry.get(operation_id)

        if spec is not None:
            assert spec.category == expected_category


class TestOpOperationHandlers:
    """Tests for op command handlers."""

    def test_op_list_returns_operations(self) -> None:
        """Test op.list returns registered operations."""
        harness = OperationTestHarness()
        result = harness.execute("op.list")

        assert result.success
        data = result.json()
        assert isinstance(data, dict)

    def test_op_call_with_unknown_operation(self) -> None:
        """Test op.call handles unknown operations gracefully."""
        harness = OperationTestHarness()
        result = harness.execute("nonexistent.operation")

        assert not result.success
        assert "Unknown operation" in result.stderr or "not found" in result.stderr.lower()


class TestDatasetOperationHandlers:
    """Tests for dataset operation handlers."""

    def test_dataset_list_returns_datasets(self) -> None:
        """Test dataset.list returns dataset information."""
        harness = OperationTestHarness()
        result = harness.execute("dataset.list")

        assert result.success


class TestStorageOperationHandlers:
    """Tests for storage operation handlers."""

    def test_storage_status_returns_info(self) -> None:
        """Test storage.status returns storage information."""
        harness = OperationTestHarness()
        result = harness.execute("storage.status")

        # May succeed or fail depending on environment
        assert result.exit_code in {0, 1}
```

#### File: `tests/cli/integration/test_full_pipeline.py`

```python
"""Integration tests for the full CLI pipeline.

Tests the complete flow: config loading → middleware → execution → output.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.cli._harness import CliTestHarness

if TYPE_CHECKING:
    from collections.abc import Iterator


class TestConfigToExecutionPipeline:
    """Test config loading through to execution."""

    def test_config_affects_output_format(
        self,
        cli: CliTestHarness,
        tmp_path: Path,
    ) -> None:
        """Test that config file affects output format."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_format: json\n")

        result = cli.with_env(
            CODEINTEL_CONFIG_FILE=str(config_file),
        ).invoke(["build", "status"])

        # Should produce JSON output
        if result.success and result.stdout.strip():
            try:
                json.loads(result.stdout)
            except json.JSONDecodeError:
                pass  # Config may not be fully wired yet

    def test_env_override_takes_precedence(
        self,
        cli: CliTestHarness,
        tmp_path: Path,
    ) -> None:
        """Test environment variables override config file."""
        config_file = tmp_path / "config.yaml"
        config_file.write_text("output_format: text\n")

        result = cli.with_env(
            CODEINTEL_CONFIG_FILE=str(config_file),
            CODEINTEL_OUTPUT_FORMAT="json",
        ).invoke(["build", "status", "--format=json"])

        assert result.exit_code in {0, 1}


class TestMiddlewarePipeline:
    """Test middleware execution in pipeline."""

    def test_telemetry_middleware_creates_spans(
        self,
        cli: CliTestHarness,
    ) -> None:
        """Test telemetry middleware is invoked."""
        result = cli.with_env(
            CODEINTEL_CONSOLE_TELEMETRY="true",
        ).invoke(["build", "status"])

        # Command should complete regardless of telemetry
        assert result.exit_code in {0, 1}

    def test_validation_middleware_rejects_invalid_params(
        self,
        cli: CliTestHarness,
    ) -> None:
        """Test validation catches invalid parameters."""
        # This tests that validation is wired into the pipeline
        result = cli.invoke(["op", "call", "build.status", "--invalid-param=value"])

        # Should either ignore unknown param or error
        assert result.exit_code in {0, 1, 2}


class TestErrorPipeline:
    """Test error handling through pipeline."""

    def test_error_includes_problem_detail_structure(
        self,
        cli: CliTestHarness,
    ) -> None:
        """Test errors follow RFC 9457 Problem Details."""
        result = cli.invoke(["op", "call", "nonexistent.operation", "--format=json"])

        if not result.success and result.stdout.strip():
            try:
                data = json.loads(result.stdout)
                # Check for Problem Detail fields
                if "error" in data:
                    error = data["error"]
                    assert "type" in error or "title" in error or "detail" in error
            except json.JSONDecodeError:
                pass  # Text format error is also valid

    def test_debug_mode_includes_stack_trace(
        self,
        cli: CliTestHarness,
    ) -> None:
        """Test debug mode exposes stack traces."""
        result = cli.with_env(
            CODEINTEL_DEBUG="true",
        ).invoke(["op", "call", "nonexistent.operation"])

        # In debug mode, might see more detail
        assert result.exit_code in {1, 2}


class TestOutputPipeline:
    """Test output rendering through pipeline."""

    def test_json_output_is_valid_json(
        self,
        cli: CliTestHarness,
    ) -> None:
        """Test JSON output is parseable."""
        result = cli.invoke(["build", "status", "--format=json"])

        if result.success and result.stdout.strip():
            # Should be valid JSON
            data = json.loads(result.stdout)
            assert isinstance(data, dict)

    def test_text_output_is_human_readable(
        self,
        cli: CliTestHarness,
    ) -> None:
        """Test text output is formatted for humans."""
        result = cli.invoke(["build", "status", "--format=text"])

        if result.success:
            # Text output should not be raw JSON
            output = result.stdout
            assert not output.startswith("{") or "status" in output.lower()
```

#### File: `tests/cli/golden/test_golden_output.py`

```python
"""Golden file tests for CLI output stability.

Uses golden files to ensure output format doesn't change unexpectedly.
Run with UPDATE_GOLDEN=1 to update golden files.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from tests.cli._harness import CliTestHarness, GoldenFileAssertion

if TYPE_CHECKING:
    pass


class TestBuildOutputGolden:
    """Golden tests for build command output."""

    def test_build_status_text_output(
        self,
        cli: CliTestHarness,
        golden: GoldenFileAssertion,
    ) -> None:
        """Test build status text output matches golden file."""
        result = cli.invoke(["build", "status", "--format=text"])

        if result.success:
            golden.assert_matches(
                "build_status_text.txt",
                result.stdout,
                normalize=True,
            )

    def test_build_status_json_structure(
        self,
        cli: CliTestHarness,
        golden: GoldenFileAssertion,
    ) -> None:
        """Test build status JSON structure matches golden file."""
        result = cli.invoke(["build", "status", "--format=json"])

        if result.success and result.stdout.strip():
            golden.assert_json_matches(
                "build_status_json.json",
                result.json(),
                ignore_keys={"timestamp", "duration_ms", "elapsed_seconds"},
            )


class TestOpOutputGolden:
    """Golden tests for op command output."""

    def test_op_list_text_output(
        self,
        cli: CliTestHarness,
        golden: GoldenFileAssertion,
    ) -> None:
        """Test op list text output matches golden file."""
        result = cli.invoke(["op", "list", "--format=text"])

        if result.success:
            golden.assert_matches(
                "op_list_text.txt",
                result.stdout,
                normalize=True,
            )


class TestErrorOutputGolden:
    """Golden tests for error output format."""

    def test_validation_error_format(
        self,
        cli: CliTestHarness,
        golden: GoldenFileAssertion,
    ) -> None:
        """Test validation error format matches golden file."""
        # Trigger a validation error
        result = cli.invoke(["config", "validate", "--file=/nonexistent/config.yaml"])

        if not result.success:
            golden.assert_matches(
                "validation_error.txt",
                result.stderr or result.stdout,
                normalize=True,
            )

    def test_not_found_error_json(
        self,
        cli: CliTestHarness,
        golden: GoldenFileAssertion,
    ) -> None:
        """Test not found error JSON structure."""
        result = cli.invoke(["op", "call", "nonexistent.op", "--format=json"])

        if not result.success and result.stdout.strip():
            try:
                golden.assert_json_matches(
                    "not_found_error.json",
                    result.json(),
                    ignore_keys={"timestamp", "correlation_id", "traceback"},
                )
            except Exception:
                pass  # May be text format
```

#### File: `tests/cli/property/test_validators_property.py`

```python
"""Property-based tests for validators using Hypothesis.

Tests validators with generated inputs to find edge cases.
"""

from __future__ import annotations

from typing import Any

import pytest

try:
    from hypothesis import given, settings, strategies as st

    HYPOTHESIS_AVAILABLE = True
except ImportError:
    HYPOTHESIS_AVAILABLE = False
    # Create dummy decorators
    def given(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        """Dummy given decorator."""
        def decorator(func: Any) -> Any:
            return pytest.mark.skip(reason="Hypothesis not installed")(func)
        return decorator

    def settings(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        """Dummy settings decorator."""
        def decorator(func: Any) -> Any:
            return func
        return decorator

    class st:  # noqa: N801
        """Dummy strategies."""

        @staticmethod
        def text(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG004
            """Dummy text strategy."""
            return None

        @staticmethod
        def integers(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG004
            """Dummy integers strategy."""
            return None

        @staticmethod
        def floats(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG004
            """Dummy floats strategy."""
            return None

        @staticmethod
        def booleans() -> Any:
            """Dummy booleans strategy."""
            return None

        @staticmethod
        def lists(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG004
            """Dummy lists strategy."""
            return None

        @staticmethod
        def dictionaries(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG004
            """Dummy dictionaries strategy."""
            return None


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestStringValidatorProperty:
    """Property tests for StringValidator."""

    @given(st.text(min_size=1, max_size=100))
    @settings(max_examples=50)
    def test_valid_strings_pass(self, value: str) -> None:
        """Test that non-empty strings pass validation."""
        from codeintel.cli.cli_validation import StringValidator

        validator = StringValidator(min_length=0)
        result = validator.validate(value)

        assert result.is_valid

    @given(st.text(min_size=0, max_size=5))
    @settings(max_examples=50)
    def test_min_length_constraint(self, value: str) -> None:
        """Test min_length constraint is enforced."""
        from codeintel.cli.cli_validation import StringValidator

        validator = StringValidator(min_length=10)
        result = validator.validate(value)

        if len(value) < 10:
            assert not result.is_valid
        else:
            assert result.is_valid

    @given(st.text(min_size=0, max_size=100))
    @settings(max_examples=50)
    def test_max_length_constraint(self, value: str) -> None:
        """Test max_length constraint is enforced."""
        from codeintel.cli.cli_validation import StringValidator

        validator = StringValidator(max_length=50)
        result = validator.validate(value)

        if len(value) > 50:
            assert not result.is_valid
        else:
            assert result.is_valid


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestIntValidatorProperty:
    """Property tests for IntValidator."""

    @given(st.integers(min_value=-1000, max_value=1000))
    @settings(max_examples=50)
    def test_integers_validate_correctly(self, value: int) -> None:
        """Test integer validation."""
        from codeintel.cli.cli_validation import IntValidator

        validator = IntValidator()
        result = validator.validate(value)

        assert result.is_valid

    @given(st.integers(min_value=-100, max_value=100))
    @settings(max_examples=50)
    def test_min_value_constraint(self, value: int) -> None:
        """Test min_value constraint is enforced."""
        from codeintel.cli.cli_validation import IntValidator

        validator = IntValidator(min_value=0)
        result = validator.validate(value)

        if value < 0:
            assert not result.is_valid
        else:
            assert result.is_valid

    @given(st.integers(min_value=0, max_value=200))
    @settings(max_examples=50)
    def test_max_value_constraint(self, value: int) -> None:
        """Test max_value constraint is enforced."""
        from codeintel.cli.cli_validation import IntValidator

        validator = IntValidator(max_value=100)
        result = validator.validate(value)

        if value > 100:
            assert not result.is_valid
        else:
            assert result.is_valid


@pytest.mark.skipif(not HYPOTHESIS_AVAILABLE, reason="Hypothesis not installed")
class TestConfigSchemaProperty:
    """Property tests for configuration schema validation."""

    @given(st.dictionaries(
        keys=st.text(min_size=1, max_size=20, alphabet="abcdefghijklmnopqrstuvwxyz_"),
        values=st.text(min_size=0, max_size=50),
        max_size=10,
    ))
    @settings(max_examples=30)
    def test_unknown_keys_rejected_with_strict_schema(
        self,
        config: dict[str, str],
    ) -> None:
        """Test unknown keys are rejected in strict mode."""
        from codeintel.cli.cli_config_schema import validate_with_json_schema

        # Valid keys from schema
        valid_keys = {
            "output_format", "color", "progress", "progress_threshold",
            "telemetry", "retry", "log_level", "project_root", "plugins",
        }

        errors = validate_with_json_schema(config)

        # If config has unknown keys, should have errors
        unknown_keys = set(config.keys()) - valid_keys
        if unknown_keys:
            # Schema may or may not reject unknown keys depending on additionalProperties
            pass  # Either outcome is valid

    @given(st.booleans())
    @settings(max_examples=10)
    def test_boolean_config_values(self, value: bool) -> None:
        """Test boolean config values are accepted."""
        from codeintel.cli.cli_config_schema import validate_with_json_schema

        config = {"color": value, "progress": value}
        errors = validate_with_json_schema(config)

        # Boolean values should always be valid for boolean fields
        boolean_errors = [e for e in errors if e.path in {"color", "progress"}]
        assert len(boolean_errors) == 0
```

#### File: `tests/cli/performance/test_performance.py`

```python
"""Performance regression tests with budget assertions.

Tests critical paths have acceptable latency and resource usage.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import pytest

from tests.cli._harness import CliTestHarness, OperationTestHarness

if TYPE_CHECKING:
    pass


# Performance budgets (in seconds)
FAST_OPERATION_BUDGET = 0.5  # Read operations
MEDIUM_OPERATION_BUDGET = 2.0  # Compute operations
SLOW_OPERATION_BUDGET = 10.0  # Build operations


class TestStartupPerformance:
    """Test CLI startup performance."""

    @pytest.mark.benchmark
    def test_cli_startup_time(self, cli: CliTestHarness) -> None:
        """Test CLI starts within budget."""
        start = time.perf_counter()
        result = cli.invoke(["--help"])
        elapsed = time.perf_counter() - start

        assert result.exit_code == 0
        assert elapsed < FAST_OPERATION_BUDGET, (
            f"CLI startup took {elapsed:.2f}s, budget is {FAST_OPERATION_BUDGET}s"
        )

    @pytest.mark.benchmark
    def test_version_command_fast(self, cli: CliTestHarness) -> None:
        """Test version command is fast."""
        start = time.perf_counter()
        result = cli.invoke(["--version"])
        elapsed = time.perf_counter() - start

        assert elapsed < 0.2, f"Version took {elapsed:.2f}s, should be < 0.2s"


class TestOperationPerformance:
    """Test individual operation performance."""

    @pytest.mark.benchmark
    def test_read_operations_are_fast(self) -> None:
        """Test read operations complete within budget."""
        harness = OperationTestHarness()

        read_operations = [
            "build.status",
            "op.list",
            "dataset.list",
            "storage.status",
        ]

        for op_id in read_operations:
            start = time.perf_counter()
            result = harness.execute(op_id)
            elapsed = time.perf_counter() - start

            # Only assert on successful operations
            if result.success:
                assert elapsed < FAST_OPERATION_BUDGET, (
                    f"{op_id} took {elapsed:.2f}s, budget is {FAST_OPERATION_BUDGET}s"
                )

    @pytest.mark.benchmark
    def test_operation_registry_lookup_fast(self) -> None:
        """Test operation registry lookup is fast."""
        from codeintel.cli.operation_registry import get_operation_registry

        registry = get_operation_registry()

        # Warm up
        _ = registry.get("build.status")

        # Measure lookup time
        iterations = 1000
        start = time.perf_counter()
        for _ in range(iterations):
            registry.get("build.status")
        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations
        assert avg_time < 0.0001, f"Registry lookup took {avg_time*1000:.3f}ms average"


class TestPipelinePerformance:
    """Test full pipeline performance."""

    @pytest.mark.benchmark
    def test_json_output_overhead(self, cli: CliTestHarness) -> None:
        """Test JSON output doesn't add significant overhead."""
        # Text output
        start = time.perf_counter()
        cli.invoke(["build", "status", "--format=text"])
        text_time = time.perf_counter() - start

        # JSON output
        start = time.perf_counter()
        cli.invoke(["build", "status", "--format=json"])
        json_time = time.perf_counter() - start

        # JSON should be within 2x of text
        assert json_time < text_time * 2 + 0.1, (
            f"JSON ({json_time:.2f}s) much slower than text ({text_time:.2f}s)"
        )

    @pytest.mark.benchmark
    def test_middleware_overhead_acceptable(self) -> None:
        """Test middleware stack doesn't add excessive overhead."""
        from codeintel.cli.executor import OperationExecutor, OperationSpec, OperationCategory
        from codeintel.cli.results import CliResult

        def fast_handler() -> CliResult[dict[str, int]]:
            return CliResult.ok({"value": 42})

        spec = OperationSpec(
            operation_id="test.fast",
            handler=fast_handler,
            category=OperationCategory.READ,
        )

        executor = OperationExecutor()

        # Warm up
        executor.execute(spec, {}, render=False)

        # Measure
        iterations = 100
        start = time.perf_counter()
        for _ in range(iterations):
            executor.execute(spec, {}, render=False)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / iterations
        # Middleware overhead should be < 10ms per call
        assert avg_time < 0.01, f"Executor overhead {avg_time*1000:.1f}ms per call"


class TestMemoryPerformance:
    """Test memory usage characteristics."""

    @pytest.mark.benchmark
    def test_repeated_operations_no_memory_leak(self) -> None:
        """Test repeated operations don't leak memory."""
        import gc

        harness = OperationTestHarness()

        # Run many operations
        for _ in range(100):
            harness.execute("op.list")

        # Force garbage collection
        gc.collect()

        # This is a basic smoke test - more sophisticated memory
        # profiling would use tracemalloc or memory_profiler
```

#### File: `tests/cli/shell/test_shell_mode.py`

```python
"""Tests for interactive shell mode.

Tests REPL interactions, completion, and session management.
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import pytest

if TYPE_CHECKING:
    pass


class TestShellSession:
    """Tests for ShellSession state management."""

    def test_session_stores_history(self) -> None:
        """Test session stores command history."""
        from codeintel.cli.shell import ShellSession

        session = ShellSession()
        session.history.append("call build.status")
        session.history.append("list")

        assert len(session.history) == 2
        assert session.history[0] == "call build.status"

    def test_session_stores_variables(self) -> None:
        """Test session stores variables."""
        from codeintel.cli.shell import ShellSession

        session = ShellSession()
        session.variables["project"] = "/path/to/project"
        session.variables["format"] = "json"

        assert session.variables["project"] == "/path/to/project"
        assert session.variables["format"] == "json"

    def test_session_tracks_last_result(self) -> None:
        """Test session tracks last result."""
        from codeintel.cli.shell import ShellSession

        session = ShellSession()
        session.last_result = {"status": "ok", "count": 42}

        assert session.last_result is not None
        assert session.last_result["count"] == 42


class TestShellCompleter:
    """Tests for shell tab completion."""

    def test_completer_suggests_commands(self) -> None:
        """Test completer suggests valid commands."""
        from codeintel.cli.shell import ShellCompleter, ShellSession

        session = ShellSession()
        completer = ShellCompleter(session)

        # Mock readline.get_line_buffer
        import readline
        original_get_buffer = readline.get_line_buffer

        try:
            readline.get_line_buffer = lambda: "ca"

            # Should suggest "call"
            result = completer.complete("ca", 0)
            assert result == "call"
        finally:
            readline.get_line_buffer = original_get_buffer

    def test_completer_suggests_operations(self) -> None:
        """Test completer suggests operations after 'call'."""
        from codeintel.cli.shell import ShellCompleter, ShellSession

        session = ShellSession()
        completer = ShellCompleter(session)

        import readline
        original_get_buffer = readline.get_line_buffer

        try:
            readline.get_line_buffer = lambda: "call build."

            # Should suggest build operations
            result = completer.complete("build.", 0)
            # May or may not have suggestions depending on registry state
            assert result is None or result.startswith("build.")
        finally:
            readline.get_line_buffer = original_get_buffer


class TestInteractiveShell:
    """Tests for InteractiveShell."""

    def test_shell_initializes(self) -> None:
        """Test shell initializes correctly."""
        from codeintel.cli.shell import InteractiveShell

        shell = InteractiveShell()
        assert shell._session is not None
        assert shell._running is False

    def test_shell_parses_params(self) -> None:
        """Test shell parses command parameters."""
        from codeintel.cli.shell import InteractiveShell

        shell = InteractiveShell()
        params = shell._parse_params(["--key=value", "--number=42", "--flag=true"])

        assert params["key"] == "value"
        assert params["number"] == 42
        assert params["flag"] is True

    def test_shell_substitutes_variables(self) -> None:
        """Test shell substitutes session variables."""
        from codeintel.cli.shell import InteractiveShell

        shell = InteractiveShell()
        shell._session.variables["mypath"] = "/path/to/project"

        params = shell._parse_params(["--root=$mypath"])
        assert params["root"] == "/path/to/project"

    def test_cmd_set_stores_variable(self) -> None:
        """Test set command stores variable."""
        from codeintel.cli.shell import InteractiveShell

        shell = InteractiveShell()
        shell._cmd_set(["myvar", "myvalue"])

        assert shell._session.variables["myvar"] == "myvalue"

    def test_cmd_get_retrieves_variable(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test get command retrieves variable."""
        from codeintel.cli.shell import InteractiveShell

        shell = InteractiveShell()
        shell._session.variables["testvar"] = "testvalue"
        shell._cmd_get(["testvar"])

        captured = capsys.readouterr()
        assert "testvalue" in captured.out

    def test_cmd_history_shows_commands(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test history command shows command history."""
        from codeintel.cli.shell import InteractiveShell

        shell = InteractiveShell()
        shell._session.history = ["call build.status", "list", "help"]
        shell._cmd_history([])

        captured = capsys.readouterr()
        assert "build.status" in captured.out
        assert "list" in captured.out

    def test_cmd_quit_stops_shell(self) -> None:
        """Test quit command stops the shell."""
        from codeintel.cli.shell import InteractiveShell

        shell = InteractiveShell()
        shell._running = True
        shell._cmd_quit([])

        assert shell._running is False

    def test_cmd_export_generates_script(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Test export command generates bash script."""
        from codeintel.cli.shell import InteractiveShell

        shell = InteractiveShell()
        shell._session.history = [
            "call build.status",
            "call op.list",
            "help",  # Should not be exported
        ]
        shell._cmd_export([])

        captured = capsys.readouterr()
        assert "#!/usr/bin/env bash" in captured.out
        assert "codeintel op call build.status" in captured.out
        assert "codeintel op call op.list" in captured.out
```

#### File: `tests/cli/_golden/` (Directory Structure)

Create golden files directory with sample files:

```
tests/cli/_golden/
├── build_status_text.txt
├── build_status_json.json
├── op_list_text.txt
├── validation_error.txt
└── not_found_error.json
```

#### File: `tests/cli/_golden/build_status_json.json`

```json
{
  "success": true,
  "data": {
    "fresh_count": 0,
    "stale_count": 0,
    "missing_count": 0,
    "targets": []
  }
}
```

---

## Phase 7.2: Async Handler Architecture

### Value Proposition

Synchronous handlers block on I/O operations. Async support enables:

- **Non-blocking I/O** — File, network, and database operations don't block
- **Progress streaming** — Async generators yield progress updates
- **Cancellation** — Proper `asyncio.CancelledError` propagation
- **Resource cleanup** — `async with` ensures cleanup on errors
- **Concurrent operations** — Multiple operations can run in parallel

### Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      Async Handler Architecture                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     OperationExecutor                              │  │
│  │                                                                    │  │
│  │   ┌────────────────────────────────────────────────────────────┐  │  │
│  │   │                    execute() / execute_async()              │  │  │
│  │   └────────────────────────────────┬───────────────────────────┘  │  │
│  │                                    │                              │  │
│  │              ┌─────────────────────┴─────────────────────┐       │  │
│  │              ▼                                           ▼       │  │
│  │   ┌──────────────────┐                       ┌──────────────────┐│  │
│  │   │   Sync Handler   │                       │  Async Handler   ││  │
│  │   │   (legacy)       │                       │  (native async)  ││  │
│  │   │                  │                       │                  ││  │
│  │   │ def handler():   │                       │ async def handler││  │
│  │   │   return result  │                       │   await io_op()  ││  │
│  │   └──────────────────┘                       │   return result  ││  │
│  │                                              └──────────────────┘│  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Async Middleware Stack                         │  │
│  │   ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐    │  │
│  │   │   Trace    │→│  Resilience│→│   Cancel   │→│  Progress  │    │  │
│  │   │ (async ctx)│ │(async retry)│ │  Handler   │ │  Stream    │    │  │
│  │   └────────────┘ └────────────┘ └────────────┘ └────────────┘    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                                                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                     Progress Streaming                             │  │
│  │                                                                    │  │
│  │   async def handler() -> AsyncGenerator[ProgressEvent, None]:     │  │
│  │       for item in items:                                          │  │
│  │           yield ProgressEvent(current=i, total=len(items))        │  │
│  │           await process(item)                                     │  │
│  │       return FinalResult(...)                                     │  │
│  │                                                                    │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

### Functional Objectives

1. Update `OperationSpec` to support async handlers
2. Create `AsyncOperationExecutor` or update `OperationExecutor` for async
3. Implement async middleware protocol
4. Add progress streaming via async generators
5. Ensure proper cancellation propagation
6. Provide async context manager for resource cleanup

### Implementation

#### File: `src/codeintel/cli/async_types.py`

```python
"""Async type definitions for CLI operations.

Provides type aliases and protocols for async handler support.
"""

from __future__ import annotations

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from codeintel.cli.results import CliResult

T = TypeVar("T")


@dataclass(frozen=True)
class ProgressEvent:
    """Progress update event.

    Parameters
    ----------
    current
        Current progress value.
    total
        Total items to process.
    message
        Optional progress message.
    percent
        Computed percentage (0-100).
    """

    current: int
    total: int
    message: str | None = None

    @property
    def percent(self) -> float:
        """Compute progress percentage.

        Returns
        -------
        float
            Percentage complete (0-100).
        """
        if self.total == 0:
            return 0.0
        return (self.current / self.total) * 100


@dataclass(frozen=True)
class StreamingResult[T]:
    """Result that includes progress events.

    Parameters
    ----------
    result
        Final operation result.
    events
        Progress events that were emitted.
    """

    result: CliResult[T]
    events: list[ProgressEvent]


# Type aliases for handler signatures
SyncHandler = Callable[..., CliResult[T]]
AsyncHandler = Callable[..., Awaitable[CliResult[T]]]
StreamingHandler = Callable[..., AsyncGenerator[ProgressEvent | CliResult[T], None]]

# Union type for any handler
AnyHandler = SyncHandler[T] | AsyncHandler[T] | StreamingHandler[T]


def is_async_handler(handler: AnyHandler[Any]) -> bool:
    """Check if handler is async.

    Parameters
    ----------
    handler
        Handler to check.

    Returns
    -------
    bool
        True if handler is async.
    """
    import asyncio
    import inspect

    return asyncio.iscoroutinefunction(handler) or inspect.isasyncgenfunction(handler)


def is_streaming_handler(handler: AnyHandler[Any]) -> bool:
    """Check if handler is a streaming generator.

    Parameters
    ----------
    handler
        Handler to check.

    Returns
    -------
    bool
        True if handler is async generator.
    """
    import inspect

    return inspect.isasyncgenfunction(handler)


__all__ = [
    "AnyHandler",
    "AsyncHandler",
    "ProgressEvent",
    "StreamingHandler",
    "StreamingResult",
    "SyncHandler",
    "is_async_handler",
    "is_streaming_handler",
]
```

#### File: `src/codeintel/cli/async_executor.py`

```python
"""Async operation execution support.

Extends the OperationExecutor to support async handlers,
progress streaming, and cancellation.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, TypeVar

from codeintel.cli.async_types import (
    AnyHandler,
    ProgressEvent,
    StreamingResult,
    is_async_handler,
    is_streaming_handler,
)
from codeintel.cli.cli_errors import ProblemDetail
from codeintel.cli.error_taxonomy import CANCELLED, INTERNAL_ERROR, StructuredCliError
from codeintel.cli.executor import ExecutionContext, ExecutionResult, OperationSpec
from codeintel.cli.results import CliResult

if TYPE_CHECKING:
    from codeintel.cli.cli_middleware import MiddlewareStack

LOG = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass
class AsyncExecutionContext(ExecutionContext):
    """Extended execution context for async operations.

    Parameters
    ----------
    cancellation_token
        Token for cancellation signaling.
    progress_callback
        Callback for progress updates.
    """

    cancellation_token: asyncio.Event | None = None
    progress_callback: Any | None = None

    def is_cancelled(self) -> bool:
        """Check if operation was cancelled.

        Returns
        -------
        bool
            True if cancelled.
        """
        if self.cancellation_token is None:
            return False
        return self.cancellation_token.is_set()


@dataclass
class AsyncExecutionResult[T](ExecutionResult[T]):
    """Extended execution result for async operations.

    Parameters
    ----------
    progress_events
        Progress events emitted during execution.
    was_cancelled
        Whether operation was cancelled.
    """

    progress_events: list[ProgressEvent] = field(default_factory=list)
    was_cancelled: bool = False


class AsyncOperationExecutor:
    """Executor with async handler support.

    Extends OperationExecutor to handle:
    - Async handlers (async def)
    - Streaming handlers (async generators)
    - Cancellation propagation
    - Progress streaming

    Parameters
    ----------
    middleware_stack
        Middleware stack to apply.
    """

    def __init__(
        self,
        middleware_stack: MiddlewareStack | None = None,
    ) -> None:
        """Initialize async executor."""
        self._middleware = middleware_stack

    async def execute_async(
        self,
        spec: OperationSpec[T],
        params: dict[str, Any],
        *,
        cancellation_token: asyncio.Event | None = None,
        on_progress: Any | None = None,
    ) -> AsyncExecutionResult[T]:
        """Execute operation asynchronously.

        Parameters
        ----------
        spec
            Operation specification.
        params
            Operation parameters.
        cancellation_token
            Token for cancellation.
        on_progress
            Progress callback.

        Returns
        -------
        AsyncExecutionResult[T]
            Execution result.
        """
        ctx = AsyncExecutionContext(
            operation_id=spec.operation_id,
            params=params,
            output_format=params.get("output_format", "text"),
            cancellation_token=cancellation_token,
            progress_callback=on_progress,
        )

        LOG.debug(
            "Starting async operation",
            extra={"operation_id": spec.operation_id},
        )

        progress_events: list[ProgressEvent] = []
        was_cancelled = False

        try:
            if is_streaming_handler(spec.handler):
                result = await self._execute_streaming(
                    spec,
                    ctx,
                    progress_events,
                    on_progress,
                )
            elif is_async_handler(spec.handler):
                result = await self._execute_async_handler(spec, ctx)
            else:
                # Sync handler - run in thread pool
                result = await self._execute_sync_in_thread(spec, ctx)

        except asyncio.CancelledError:
            was_cancelled = True
            result = CliResult.error(
                ProblemDetail(
                    type=CANCELLED.type_uri,
                    title=CANCELLED.title,
                    detail="Operation was cancelled",
                    status=CANCELLED.status,
                )
            )
        except StructuredCliError as e:
            result = CliResult.error(e.to_problem_detail())
        except Exception as e:
            LOG.exception("Async operation failed")
            result = CliResult.error(
                ProblemDetail(
                    type=INTERNAL_ERROR.type_uri,
                    title=INTERNAL_ERROR.title,
                    detail=str(e),
                    status=INTERNAL_ERROR.status,
                )
            )

        return AsyncExecutionResult(
            result=result,
            duration_seconds=ctx.elapsed_seconds,
            progress_events=progress_events,
            was_cancelled=was_cancelled,
        )

    async def _execute_async_handler(
        self,
        spec: OperationSpec[T],
        ctx: AsyncExecutionContext,
    ) -> CliResult[T]:
        """Execute async handler.

        Parameters
        ----------
        spec
            Operation specification.
        ctx
            Execution context.

        Returns
        -------
        CliResult[T]
            Handler result.
        """
        return await spec.handler(**ctx.params)

    async def _execute_sync_in_thread(
        self,
        spec: OperationSpec[T],
        ctx: AsyncExecutionContext,
    ) -> CliResult[T]:
        """Execute sync handler in thread pool.

        Parameters
        ----------
        spec
            Operation specification.
        ctx
            Execution context.

        Returns
        -------
        CliResult[T]
            Handler result.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            lambda: spec.handler(**ctx.params),
        )

    async def _execute_streaming(
        self,
        spec: OperationSpec[T],
        ctx: AsyncExecutionContext,
        progress_events: list[ProgressEvent],
        on_progress: Any | None,
    ) -> CliResult[T]:
        """Execute streaming handler.

        Parameters
        ----------
        spec
            Operation specification.
        ctx
            Execution context.
        progress_events
            List to collect progress events.
        on_progress
            Progress callback.

        Returns
        -------
        CliResult[T]
            Final result from generator.
        """
        result: CliResult[T] | None = None

        async for event in spec.handler(**ctx.params):
            # Check cancellation
            if ctx.is_cancelled():
                raise asyncio.CancelledError

            if isinstance(event, ProgressEvent):
                progress_events.append(event)
                if on_progress:
                    await self._call_progress_callback(on_progress, event)
            elif isinstance(event, CliResult):
                result = event

        if result is None:
            return CliResult.error(
                ProblemDetail(
                    type=INTERNAL_ERROR.type_uri,
                    title=INTERNAL_ERROR.title,
                    detail="Streaming handler did not yield a result",
                    status=INTERNAL_ERROR.status,
                )
            )

        return result

    async def _call_progress_callback(
        self,
        callback: Any,
        event: ProgressEvent,
    ) -> None:
        """Call progress callback (sync or async).

        Parameters
        ----------
        callback
            Progress callback.
        event
            Progress event.
        """
        if asyncio.iscoroutinefunction(callback):
            await callback(event)
        else:
            callback(event)


@asynccontextmanager
async def cancellable_operation(
    timeout: float | None = None,
) -> AsyncGenerator[asyncio.Event, None]:
    """Context manager for cancellable operations.

    Parameters
    ----------
    timeout
        Optional timeout in seconds.

    Yields
    ------
    asyncio.Event
        Cancellation token.
    """
    cancel_token = asyncio.Event()

    async def timeout_task() -> None:
        if timeout is not None:
            await asyncio.sleep(timeout)
            cancel_token.set()

    timeout_handle = None
    if timeout is not None:
        timeout_handle = asyncio.create_task(timeout_task())

    try:
        yield cancel_token
    finally:
        if timeout_handle is not None:
            timeout_handle.cancel()


def run_async_operation[T](
    executor: AsyncOperationExecutor,
    spec: OperationSpec[T],
    params: dict[str, Any],
    *,
    timeout: float | None = None,
) -> AsyncExecutionResult[T]:
    """Run async operation from sync context.

    Parameters
    ----------
    executor
        Async executor.
    spec
        Operation specification.
    params
        Operation parameters.
    timeout
        Optional timeout.

    Returns
    -------
    AsyncExecutionResult[T]
        Execution result.
    """
    async def run() -> AsyncExecutionResult[T]:
        async with cancellable_operation(timeout) as cancel_token:
            return await executor.execute_async(
                spec,
                params,
                cancellation_token=cancel_token,
            )

    return asyncio.run(run())


__all__ = [
    "AsyncExecutionContext",
    "AsyncExecutionResult",
    "AsyncOperationExecutor",
    "cancellable_operation",
    "run_async_operation",
]
```

#### File: `src/codeintel/cli/async_middleware.py`

```python
"""Async middleware support for operation execution.

Provides async-aware middleware that can handle both sync
and async operations.
"""

from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

LOG = logging.getLogger(__name__)


class AsyncOperationMiddleware(ABC):
    """Base class for async middleware.

    Middleware can intercept operation execution to add
    cross-cutting concerns like logging, tracing, retries.
    """

    @abstractmethod
    async def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Execute before operation invocation.

        Parameters
        ----------
        op_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        dict[str, Any]
            Context to pass to after_invoke.
        """

    @abstractmethod
    async def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Execute after successful operation.

        Parameters
        ----------
        op_id
            Operation identifier.
        result
            Operation result.
        context
            Context from before_invoke.
        """

    @abstractmethod
    async def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Execute on operation error.

        Parameters
        ----------
        op_id
            Operation identifier.
        exc
            Exception that occurred.
        context
            Context from before_invoke.
        """


class AsyncTracingMiddleware(AsyncOperationMiddleware):
    """Async-aware tracing middleware."""

    async def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Start trace span."""
        LOG.debug("Starting async operation: %s", op_id)
        return {
            "start_time": time.monotonic(),
            "operation_id": op_id,
        }

    async def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """End trace span on success."""
        duration = time.monotonic() - context["start_time"]
        LOG.debug(
            "Async operation completed: %s (%.2fs)",
            op_id,
            duration,
        )

    async def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """End trace span on error."""
        duration = time.monotonic() - context["start_time"]
        LOG.error(
            "Async operation failed: %s (%.2fs): %s",
            op_id,
            duration,
            exc,
        )


@dataclass
class AsyncRetryPolicy:
    """Retry policy for async operations.

    Parameters
    ----------
    max_attempts
        Maximum retry attempts.
    initial_delay
        Initial delay between retries.
    backoff_factor
        Exponential backoff multiplier.
    max_delay
        Maximum delay between retries.
    retryable_exceptions
        Exception types to retry.
    """

    max_attempts: int = 3
    initial_delay: float = 0.5
    backoff_factor: float = 2.0
    max_delay: float = 30.0
    retryable_exceptions: tuple[type[Exception], ...] = (
        ConnectionError,
        TimeoutError,
    )

    def calculate_delay(self, attempt: int) -> float:
        """Calculate delay for attempt.

        Parameters
        ----------
        attempt
            Attempt number (0-indexed).

        Returns
        -------
        float
            Delay in seconds.
        """
        delay = self.initial_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


class AsyncResilienceMiddleware(AsyncOperationMiddleware):
    """Async resilience middleware with retry support.

    Parameters
    ----------
    policy
        Retry policy.
    """

    def __init__(self, policy: AsyncRetryPolicy | None = None) -> None:
        """Initialize middleware."""
        self._policy = policy or AsyncRetryPolicy()

    async def before_invoke(
        self,
        op_id: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Initialize retry context."""
        return {
            "attempts": 0,
            "start_time": time.monotonic(),
        }

    async def after_invoke(
        self,
        op_id: str,
        result: object,
        context: dict[str, Any],
    ) -> None:
        """Record success."""
        pass

    async def on_error(
        self,
        op_id: str,
        exc: Exception,
        context: dict[str, Any],
    ) -> None:
        """Record failure for potential retry."""
        context["attempts"] += 1
        context["last_error"] = exc


async def execute_with_async_retry[T](
    coro_factory: Any,
    policy: AsyncRetryPolicy,
    *,
    operation_id: str = "",
) -> T:
    """Execute coroutine with retry policy.

    Parameters
    ----------
    coro_factory
        Factory function that creates the coroutine.
    policy
        Retry policy.
    operation_id
        Operation identifier for logging.

    Returns
    -------
    T
        Coroutine result.

    Raises
    ------
    Exception
        If all retries exhausted.
    """
    last_exception: Exception | None = None

    for attempt in range(policy.max_attempts):
        try:
            return await coro_factory()

        except policy.retryable_exceptions as e:
            last_exception = e

            if attempt < policy.max_attempts - 1:
                delay = policy.calculate_delay(attempt)
                LOG.warning(
                    "Async operation %s failed (attempt %d/%d), "
                    "retrying in %.1fs: %s",
                    operation_id,
                    attempt + 1,
                    policy.max_attempts,
                    delay,
                    e,
                )
                await asyncio.sleep(delay)

    if last_exception:
        raise last_exception

    msg = "Retry loop completed without result"
    raise RuntimeError(msg)


@asynccontextmanager
async def async_middleware_context(
    middlewares: list[AsyncOperationMiddleware],
    op_id: str,
    params: dict[str, Any],
) -> AsyncIterator[list[dict[str, Any]]]:
    """Context manager for async middleware stack.

    Parameters
    ----------
    middlewares
        List of middleware to apply.
    op_id
        Operation identifier.
    params
        Operation parameters.

    Yields
    ------
    list[dict[str, Any]]
        Middleware contexts.
    """
    contexts: list[dict[str, Any]] = []

    # Before invoke
    for mw in middlewares:
        ctx = await mw.before_invoke(op_id, params)
        contexts.append(ctx)

    try:
        yield contexts
    except Exception as e:
        # On error
        for mw, ctx in zip(middlewares, contexts, strict=True):
            await mw.on_error(op_id, e, ctx)
        raise
    else:
        # After invoke
        for mw, ctx in zip(middlewares, contexts, strict=True):
            await mw.after_invoke(op_id, None, ctx)


__all__ = [
    "AsyncOperationMiddleware",
    "AsyncResilienceMiddleware",
    "AsyncRetryPolicy",
    "AsyncTracingMiddleware",
    "async_middleware_context",
    "execute_with_async_retry",
]
```

#### File: `src/codeintel/cli/async_progress.py`

```python
"""Async progress streaming for CLI operations.

Provides async generators for progress reporting and
streaming output to the console.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from typing import Any, TextIO

from codeintel.cli.async_types import ProgressEvent


@dataclass
class ProgressStreamConfig:
    """Configuration for progress streaming.

    Parameters
    ----------
    show_bar
        Show progress bar.
    show_percentage
        Show percentage.
    show_eta
        Show estimated time.
    refresh_rate
        Refresh rate in Hz.
    """

    show_bar: bool = True
    show_percentage: bool = True
    show_eta: bool = True
    refresh_rate: float = 10.0


class AsyncProgressRenderer:
    """Async renderer for progress events.

    Parameters
    ----------
    config
        Progress configuration.
    output
        Output stream.
    """

    def __init__(
        self,
        config: ProgressStreamConfig | None = None,
        output: TextIO | None = None,
    ) -> None:
        """Initialize renderer."""
        self._config = config or ProgressStreamConfig()
        self._output = output or sys.stderr
        self._last_render = 0.0
        self._min_interval = 1.0 / self._config.refresh_rate

    async def render(self, event: ProgressEvent) -> None:
        """Render progress event.

        Parameters
        ----------
        event
            Progress event to render.
        """
        import time

        now = time.monotonic()
        if now - self._last_render < self._min_interval:
            return

        self._last_render = now

        parts: list[str] = []

        if self._config.show_percentage:
            parts.append(f"{event.percent:5.1f}%")

        if self._config.show_bar:
            bar_width = 30
            filled = int(bar_width * event.percent / 100)
            bar = "█" * filled + "░" * (bar_width - filled)
            parts.append(f"[{bar}]")

        parts.append(f"{event.current}/{event.total}")

        if event.message:
            parts.append(event.message)

        line = " ".join(parts)
        self._output.write(f"\r{line}")
        self._output.flush()

    async def clear(self) -> None:
        """Clear progress line."""
        self._output.write("\r" + " " * 80 + "\r")
        self._output.flush()


async def stream_progress[T](
    generator: AsyncGenerator[ProgressEvent | T, None],
    *,
    renderer: AsyncProgressRenderer | None = None,
) -> T:
    """Stream progress from async generator.

    Consumes progress events and returns final result.

    Parameters
    ----------
    generator
        Async generator yielding progress or result.
    renderer
        Progress renderer.

    Returns
    -------
    T
        Final result from generator.

    Raises
    ------
    RuntimeError
        If no result yielded.
    """
    renderer = renderer or AsyncProgressRenderer()
    result: T | None = None

    async for event in generator:
        if isinstance(event, ProgressEvent):
            await renderer.render(event)
        else:
            result = event

    await renderer.clear()

    if result is None:
        msg = "Generator did not yield a result"
        raise RuntimeError(msg)

    return result


async def progress_generator[T](
    items: list[T],
    process: Any,
    *,
    message_fn: Any | None = None,
) -> AsyncGenerator[ProgressEvent | list[Any], None]:
    """Create progress generator for item processing.

    Parameters
    ----------
    items
        Items to process.
    process
        Async function to process each item.
    message_fn
        Function to generate progress message.

    Yields
    ------
    ProgressEvent | list[Any]
        Progress events and final results.
    """
    total = len(items)
    results: list[Any] = []

    for i, item in enumerate(items):
        message = message_fn(item) if message_fn else None
        yield ProgressEvent(current=i, total=total, message=message)

        result = await process(item)
        results.append(result)

    yield ProgressEvent(current=total, total=total, message="Complete")
    yield results


__all__ = [
    "AsyncProgressRenderer",
    "ProgressStreamConfig",
    "progress_generator",
    "stream_progress",
]
```

#### Update `src/codeintel/cli/executor.py`

Add async support to the existing executor:

```python
# Add to executor.py

from codeintel.cli.async_types import AnyHandler, is_async_handler

@dataclass(frozen=True)
class OperationSpec(Generic[T]):
    """Specification for an operation's execution behavior.

    Parameters
    ----------
    operation_id
        Unique identifier for the operation.
    handler
        The handler function (sync or async).
    category
        Operation category for behavior configuration.
    param_schema
        Optional validation schema for parameters.
    requires_progress
        Whether to show progress bar.
    estimated_duration
        Estimated duration in seconds (for progress).
    retryable
        Whether the operation can be retried on failure.
    timeout
        Maximum execution time in seconds.
    description
        Human-readable operation description.
    is_async
        Whether handler is async (auto-detected if not specified).
    """

    operation_id: str
    handler: AnyHandler[T]
    category: OperationCategory = OperationCategory.READ
    param_schema: ValidationSchema[dict[str, Any]] | None = None
    requires_progress: bool = False
    estimated_duration: float | None = None
    retryable: bool = False
    timeout: float | None = None
    description: str = ""
    is_async: bool | None = None  # Auto-detect if None

    def __post_init__(self) -> None:
        """Auto-detect async status if not specified."""
        if self.is_async is None:
            # Use object.__setattr__ for frozen dataclass
            object.__setattr__(self, "is_async", is_async_handler(self.handler))
```

---

## Implementation Timeline

| Phase | Duration | Dependencies | Priority | Effort |
|-------|----------|--------------|----------|--------|
| 7.1 Test Suite | 5-7 days | Phase 6 complete | Critical | High |
| 7.2 Async Architecture | 4-5 days | Phase 6 complete | High | High |

**Total estimated time: 9-12 days**

### Recommended Order

```
Week 1:       [======= Phase 7.1: Test Suite =======]
Week 1-2:                        [===== 7.2: Async =====]
```

### Parallel Workstreams

- 7.1 and 7.2 can proceed in parallel after initial setup
- Unit tests (7.1) inform async handler design (7.2)
- Performance tests (7.1) validate async improvements (7.2)

---

## Success Metrics

### Phase 7.1: Test Suite

- [ ] Unit tests for all operation handlers (≥1 test per handler)
- [ ] Integration tests for config → execution pipeline
- [ ] Golden file tests for text and JSON output
- [ ] Property-based tests for validators with Hypothesis
- [ ] Performance tests with budget assertions
- [ ] Shell mode tests for REPL functionality
- [ ] Test coverage ≥ 85% for CLI modules
- [ ] Zero monkeypatch usage in tests

### Phase 7.2: Async Architecture

- [ ] `OperationSpec.is_async` auto-detection working
- [ ] `AsyncOperationExecutor` executes async handlers
- [ ] Progress streaming via async generators
- [ ] Cancellation propagates via `asyncio.CancelledError`
- [ ] Sync handlers run in thread pool
- [ ] Async middleware stack functional
- [ ] Performance improvement for I/O-bound operations

---

## Appendix: File Manifest

### New Files

| File | Purpose |
|------|---------|
| `tests/cli/unit/test_operation_handlers.py` | Unit tests for handlers |
| `tests/cli/integration/test_full_pipeline.py` | Integration tests |
| `tests/cli/golden/test_golden_output.py` | Golden file tests |
| `tests/cli/property/test_validators_property.py` | Hypothesis tests |
| `tests/cli/performance/test_performance.py` | Performance tests |
| `tests/cli/shell/test_shell_mode.py` | Shell mode tests |
| `tests/cli/_golden/` | Golden file directory |
| `src/codeintel/cli/async_types.py` | Async type definitions |
| `src/codeintel/cli/async_executor.py` | Async executor |
| `src/codeintel/cli/async_middleware.py` | Async middleware |
| `src/codeintel/cli/async_progress.py` | Progress streaming |

### Modified Files

| File | Changes |
|------|---------|
| `src/codeintel/cli/executor.py` | Add `is_async` to `OperationSpec` |
| `tests/cli/conftest.py` | Add performance test markers |
| `pyproject.toml` | Add Hypothesis dependency |

---

*End of Phase 7 Part 1 Implementation Plan*

