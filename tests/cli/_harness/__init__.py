"""CLI test harness for charter-compliant testing.

Provide tools for testing CLI commands through real entry points
without monkeypatching, using dependency injection and test doubles.

This module follows the Testing Charter from AGENTS.md, which requires:
- No monkeypatching of production code
- Testing through real entry points
- Using the same tech stack as production
- Proper dependency injection
"""

from __future__ import annotations

import io
import json
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from codeintel.cli.commands import app as cli_app
from codeintel.cli.config import load_config
from codeintel.cli.context import CommandContextBuilder
from codeintel.cli.core import CliResult
from codeintel.cli.introspection import get_registry
from codeintel.cli.rendering.types import OutputFormat
from tests._helpers.gateway import GatewayFactory


@dataclass
class CliInvocationResult:
    """Result of a CLI invocation.

    Parameters
    ----------
    exit_code
        Process exit code.
    stdout
        Captured standard output.
    stderr
        Captured standard error.
    exception
        Exception if raised.
    """

    exit_code: int
    stdout: str
    stderr: str
    exception: Exception | None = None

    @property
    def success(self) -> bool:
        """Check if invocation succeeded.

        Returns
        -------
        bool
            True if exit_code is 0.
        """
        return self.exit_code == 0

    @property
    def output(self) -> str:
        """Get combined stdout and stderr.

        Returns
        -------
        str
            Combined output.
        """
        return self.stdout + self.stderr

    @property
    def data(self) -> dict[str, object] | None:
        """Get parsed JSON data from stdout.

        Returns
        -------
        dict[str, object] | None
            Parsed JSON data or None if not valid JSON.
        """
        try:
            return json.loads(self.stdout)
        except (json.JSONDecodeError, ValueError):
            return None

    @property
    def error(self) -> str | None:
        """Get error message.

        Returns
        -------
        str | None
            Error message from stderr or exception.
        """
        if self.exception:
            return str(self.exception)
        if self.stderr:
            return self.stderr
        return None

    def json(self) -> dict[str, object]:
        """Parse stdout as JSON.

        Returns
        -------
        dict[str, object]
            Parsed JSON output.
        """
        return json.loads(self.stdout)

    def lines(self) -> list[str]:
        """Get stdout as lines.

        Returns
        -------
        list[str]
            Output lines.
        """
        return self.stdout.strip().split("\n")


@dataclass
class CliTestHarness:
    """Harness for testing CLI commands.

    Provide a clean way to invoke CLI commands and capture output
    without subprocess overhead.

    Parameters
    ----------
    env_overrides
        Environment variable overrides.
    config_overrides
        Configuration overrides.
    working_dir
        Working directory for invocation.
    """

    env_overrides: dict[str, str] = field(default_factory=dict)
    config_overrides: dict[str, object] = field(default_factory=dict)
    working_dir: Path | None = None

    def with_env(self, **env: str) -> CliTestHarness:
        """Create harness with environment overrides.

        Parameters
        ----------
        **env
            Environment variables.

        Returns
        -------
        CliTestHarness
            New harness with overrides.
        """
        return CliTestHarness(
            env_overrides={**self.env_overrides, **env},
            config_overrides=self.config_overrides,
            working_dir=self.working_dir,
        )

    def with_config(self, **config: object) -> CliTestHarness:
        """Create harness with config overrides.

        Parameters
        ----------
        **config
            Configuration values.

        Returns
        -------
        CliTestHarness
            New harness with overrides.
        """
        return CliTestHarness(
            env_overrides=self.env_overrides,
            config_overrides={**self.config_overrides, **config},
            working_dir=self.working_dir,
        )

    def with_cwd(self, path: Path) -> CliTestHarness:
        """Create harness with working directory.

        Parameters
        ----------
        path
            Working directory.

        Returns
        -------
        CliTestHarness
            New harness with cwd.
        """
        return CliTestHarness(
            env_overrides=self.env_overrides,
            config_overrides=self.config_overrides,
            working_dir=path,
        )

    @contextmanager
    def _capture_context(
        self,
        args: list[str],
    ) -> Iterator[tuple[io.StringIO, io.StringIO]]:
        """Set up capture context for CLI invocation.

        Parameters
        ----------
        args
            Command line arguments.

        Yields
        ------
        tuple[io.StringIO, io.StringIO]
            Stdout and stderr capture objects.
        """
        _ = args  # Used only for documentation
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        old_argv = sys.argv
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        old_cwd = Path.cwd()
        old_env = dict(os.environ)

        try:
            # Set up environment
            sys.argv = ["codeintel", *args]
            sys.stdout = stdout_capture
            sys.stderr = stderr_capture

            for key, value in self.env_overrides.items():
                os.environ[key] = value

            if self.working_dir:
                os.chdir(self.working_dir)

            yield stdout_capture, stderr_capture

        finally:
            sys.argv = old_argv
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            os.chdir(old_cwd)
            os.environ.clear()
            os.environ.update(old_env)

    def invoke(self, args: list[str]) -> CliInvocationResult:
        """Invoke CLI with arguments.

        Parameters
        ----------
        args
            Command line arguments.

        Returns
        -------
        CliInvocationResult
            Invocation result.
        """
        with self._capture_context(args) as (stdout_capture, stderr_capture):
            exit_code = 0
            exception = None
            try:
                cli_app()
            except SystemExit as e:
                exit_code = e.code if isinstance(e.code, int) else 1
            except (RuntimeError, ValueError, OSError) as e:
                exception = e
                exit_code = 1

            return CliInvocationResult(
                exit_code=exit_code,
                stdout=stdout_capture.getvalue(),
                stderr=stderr_capture.getvalue(),
                exception=exception,
            )

    def invoke_json(self, args: list[str]) -> dict[str, object]:
        """Invoke CLI and parse JSON output.

        Parameters
        ----------
        args
            Command line arguments (--format=json added if not present).

        Returns
        -------
        dict[str, object]
            Parsed JSON output.

        Raises
        ------
        RuntimeError
            If invocation failed.
        """
        # Add JSON format flag if not present
        if "--format=json" not in args and "--output-format" not in " ".join(args):
            args = [*args, "--format=json"]

        result = self.invoke(args)
        if not result.success:
            msg = f"CLI failed: {result.stderr}"
            raise RuntimeError(msg)
        return result.json()


@dataclass
class GoldenFileAssertion:
    """Helper for golden file testing.

    Golden files contain expected output for comparison.
    This helper supports updating golden files when tests run
    with UPDATE_GOLDEN=1.

    Parameters
    ----------
    golden_dir
        Directory containing golden files.
    update_mode
        Whether to update golden files instead of comparing.
    """

    golden_dir: Path
    update_mode: bool = False

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize whitespace in text.

        Parameters
        ----------
        text
            Text to normalize.

        Returns
        -------
        str
            Normalized text.
        """
        return " ".join(text.split())

    @staticmethod
    def filter_json(
        obj: object,
        ignore_keys: set[str],
    ) -> dict[str, object]:
        """Filter JSON object by removing specified keys.

        Parameters
        ----------
        obj
            JSON object (must be a dict).
        ignore_keys
            Keys to remove.

        Returns
        -------
        dict[str, object]
            Filtered object.
        """
        if not isinstance(obj, dict):
            return {}

        def remove_keys(item: object) -> object:
            if isinstance(item, dict):
                return {k: remove_keys(v) for k, v in item.items() if k not in ignore_keys}
            if isinstance(item, list):
                return [remove_keys(i) for i in item]
            return item

        result = remove_keys(obj)
        if isinstance(result, dict):
            return result
        return {}

    def assert_matches(
        self,
        name: str,
        actual: str,
        *,
        normalize: bool = True,
    ) -> None:
        """Assert output matches golden file.

        Parameters
        ----------
        name
            Golden file name.
        actual
            Actual output.
        normalize
            Normalize whitespace.

        Raises
        ------
        AssertionError
            If output doesn't match.
        """
        golden_path = self.golden_dir / name

        if normalize:
            actual = actual.strip() + "\n"

        if self.update_mode:
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            golden_path.write_text(actual)
            return

        if not golden_path.exists():
            msg = (
                f"Golden file not found: {golden_path}\n"
                f"Run with UPDATE_GOLDEN=1 to create it.\n"
                f"Actual output:\n{actual}"
            )
            raise AssertionError(msg)

        expected = golden_path.read_text()
        if normalize:
            expected = expected.strip() + "\n"

        if actual != expected:
            msg = (
                f"Output doesn't match golden file: {golden_path}\n"
                f"Expected:\n{expected}\n"
                f"Actual:\n{actual}"
            )
            raise AssertionError(msg)

    def assert_json_matches(
        self,
        name: str,
        actual: dict[str, object],
        *,
        ignore_keys: set[str] | None = None,
    ) -> None:
        """Assert JSON output matches golden file.

        Parameters
        ----------
        name
            Golden file name.
        actual
            Actual JSON data.
        ignore_keys
            Keys to ignore in comparison.

        Notes
        -----
        This method delegates to assert_matches which raises AssertionError
        if the output doesn't match the golden file.
        """
        if ignore_keys:
            keys_to_ignore = ignore_keys

            def remove_keys(obj: object) -> object:
                if isinstance(obj, dict):
                    return {k: remove_keys(v) for k, v in obj.items() if k not in keys_to_ignore}
                if isinstance(obj, list):
                    return [remove_keys(item) for item in obj]
                return obj

            actual = cast("dict[str, object]", remove_keys(actual))

        actual_str = json.dumps(actual, indent=2, sort_keys=True)
        self.assert_matches(name, actual_str)


@dataclass
class OperationTestHarness:
    """Harness for testing individual operations.

    Test operations through the executor without going through
    the full CLI parsing.

    Parameters
    ----------
    render
        Whether to render output (False for programmatic use).
    """

    render: bool = False

    @staticmethod
    def execute(
        operation_id: str,
        params: dict[str, object] | None = None,
    ) -> CliInvocationResult:
        """Execute an operation directly.

        Parameters
        ----------
        operation_id
            Operation identifier.
        params
            Operation parameters.

        Returns
        -------
        CliInvocationResult
            Result of execution.
        """
        params = params or {}
        registry = get_registry()
        spec = registry.get(operation_id)

        if spec is None:
            return CliInvocationResult(
                exit_code=1,
                stdout="",
                stderr=f"Unknown operation: {operation_id}",
            )

        params = dict(params or {})

        with TemporaryDirectory() as tmp_dir:
            gateway = None
            try:
                builder = (
                    CommandContextBuilder()
                    .with_params(params)
                    .with_output_format(OutputFormat.JSON)
                    .with_verbosity(0)
                    .with_operation_id(spec.operation_id)
                )
                if spec.require_runtime:
                    builder = builder.with_runtime(project_root=Path(tmp_dir))
                if spec.require_gateway:
                    gateway = GatewayFactory().open()
                    builder = builder.with_injected_gateway(gateway)
                needs_serving = bool(
                    getattr(spec, "require_serving", False)
                    or spec.serving_op_id is not None
                    or spec.backend_method is not None
                )
                if needs_serving:
                    builder = builder.with_serving()

                with builder.build() as ctx:
                    try:
                        result = spec.handler(ctx)
                    except Exception as exc:  # noqa: BLE001
                        return CliInvocationResult(
                            exit_code=1,
                            stdout="",
                            stderr=str(exc),
                        )
            finally:
                if gateway is not None:
                    gateway.close()

        if result.success:
            data = result.data
            if data is not None and hasattr(data, "to_dict"):
                stdout = json.dumps(data.to_dict(), indent=2)
            else:
                stdout = str(data) if data is not None else ""
            return CliInvocationResult(
                exit_code=0,
                stdout=stdout,
                stderr="",
            )

        error_msg = result.error.detail if result.error else "Unknown error"
        return CliInvocationResult(
            exit_code=1,
            stdout="",
            stderr=str(error_msg),
        )


__all__ = [
    "CliInvocationResult",
    "CliResult",
    "CliTestHarness",
    "GoldenFileAssertion",
    "OperationTestHarness",
    "load_config",
]
