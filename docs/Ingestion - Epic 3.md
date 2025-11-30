Nice, this is a really fun one. Let’s turn your tool layer into an actual “plugin runtime” without blowing up existing call-sites.

Below is a **step-by-step, patch-level plan** with concrete code snippets.

High-level shape:

* New **plugin package**: `codeintel.ingestion.tools`.
* Plugins expose a uniform interface: status, artifacts, metrics.
* `ToolService` becomes a **thin façade** over a **plugin registry**.
* Existing ingestion modules (`typing_ingest`, `coverage_ingest`, `tests_ingest`, `scip_ingest`) keep using `ToolService` APIs; no big signature changes.

I’ll show **full detail** for:

* The base plugin types & registry.
* A full **Pyright plugin**.
* A full **Coverage plugin**.
* How to wire them into `ToolService.run_pyright` / `run_coverage_json`.
* Patterns for Pyrefly, Ruff, Pytest, SCIP.
* Tests.

---

## 1. New plugin package layout

Under `src/codeintel/ingestion/`, add a `tools` package:

```text
codeintel/ingestion/
  tools/
    __init__.py
    plugins.py          # base types + registry
    pyright.py          # PyrightPlugin
    pyrefly.py          # PyreflyPlugin
    ruff.py             # RuffPlugin
    coverage.py         # CoveragePlugin
    pytest.py           # PytestPlugin
    scip.py             # ScipPlugin
```

We’ll keep **all shared interfaces** in `plugins.py` and put the logic of each tool into its own module.

---

## 2. Base plugin types & registry (`ingestion/tools/plugins.py`)

### 2.1. Base types

```python
# src/codeintel/ingestion/tools/plugins.py

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import ToolName, ToolRunResult, ToolRunner

log = logging.getLogger(__name__)


class ToolStatus(StrEnum):
    """Normalized status for plugin invocations."""

    OK = "ok"
    NOT_FOUND = "not_found"
    TIMEOUT = "timeout"
    ERROR = "error"


@dataclass(frozen=True)
class ToolPluginResult:
    """
    High-level result from a tool plugin execution.

    Attributes
    ----------
    tool:
        Logical tool identifier (from ToolName).
    status:
        Normalized status code describing the outcome.
    artifacts:
        Logical name -> on-disk artifact path (e.g. 'json_report' -> Path).
    run:
        Underlying ToolRunResult when a subprocess ran, otherwise None.
    error:
        Exception captured by the plugin, if any.
    """

    tool: ToolName
    status: ToolStatus
    artifacts: Mapping[str, Path]
    run: ToolRunResult | None
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        """Return True when the plugin completed successfully."""
        return self.status == ToolStatus.OK


@dataclass(frozen=True)
class ToolPluginMetadata:
    """
    Declarative metadata for a plugin.

    Attributes
    ----------
    name:
        Registry name (e.g., 'pyright').
    produces_artifacts:
        Logical artifact names exposed by this plugin.
    consumes_configs:
        ToolConfig fields this plugin depends on (e.g., 'pyright_bin').
    datasets:
        Datasets (table keys) that conceptually rely on this tool.
    """

    name: str
    produces_artifacts: tuple[str, ...]
    consumes_configs: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()


@runtime_checkable
class ToolPlugin(Protocol):
    """
    Protocol implemented by all tool plugins.

    Plugins are designed to be stateless wrappers around ToolRunner + ToolsConfig.
    """

    metadata: ToolPluginMetadata
    runner: ToolRunner
    tools_config: ToolsConfig

    async def run(self, *, repo_root: Path, **kwargs: Any) -> ToolPluginResult:
        """
        Execute the tool for the given repository root.

        Additional keyword arguments are tool-specific (e.g., coverage_file,
        json_output_path, rel_paths for sharded SCIP, etc.).
        """
        ...
```

### 2.2. Plugin registry

```python
@dataclass
class ToolPluginRegistry:
    """
    Registry of tool plugins keyed by logical name.

    Parameters
    ----------
    runner:
        Shared ToolRunner used by plugins.
    tools_config:
        Effective ToolsConfig configuration.
    plugins:
        Initial mapping of name -> plugin instance (optional).
    """

    runner: ToolRunner
    tools_config: ToolsConfig
    _plugins: MutableMapping[str, ToolPlugin] = field(default_factory=dict)

    def register(self, plugin: ToolPlugin) -> None:
        """Register or overwrite a plugin."""
        name = plugin.metadata.name
        self._plugins[name] = plugin
        log.debug("Registered tool plugin %s", name)

    def get(self, name: str) -> ToolPlugin:
        """Return a plugin by name or raise KeyError."""
        try:
            return self._plugins[name]
        except KeyError as exc:  # pragma: no cover - trivial guard
            message = f"Unknown tool plugin: {name!r}"
            raise KeyError(message) from exc

    def names(self) -> tuple[str, ...]:
        """Return all registered plugin names."""
        return tuple(self._plugins.keys())

    def items(self) -> Mapping[str, ToolPlugin]:
        """Return an immutable view of registered plugins."""
        return dict(self._plugins)


def build_default_registry(runner: ToolRunner, tools_config: ToolsConfig) -> ToolPluginRegistry:
    """
    Construct a registry with all built-in tool plugins.

    This import is intentionally local to avoid import cycles between
    tool_service and plugin implementations.
    """
    from codeintel.ingestion.tools.pyright import PyrightPlugin
    from codeintel.ingestion.tools.pyrefly import PyreflyPlugin
    from codeintel.ingestion.tools.ruff import RuffPlugin
    from codeintel.ingestion.tools.coverage import CoveragePlugin
    from codeintel.ingestion.tools.pytest import PytestPlugin
    from codeintel.ingestion.tools.scip import ScipPlugin

    registry = ToolPluginRegistry(runner=runner, tools_config=tools_config)

    registry.register(PyrightPlugin(runner=runner, tools_config=tools_config))
    registry.register(PyreflyPlugin(runner=runner, tools_config=tools_config))
    registry.register(RuffPlugin(runner=runner, tools_config=tools_config))
    registry.register(CoveragePlugin(runner=runner, tools_config=tools_config))
    registry.register(PytestPlugin(runner=runner, tools_config=tools_config))
    registry.register(ScipPlugin(runner=runner, tools_config=tools_config))

    return registry
```

### 2.3. Package `__init__.py`

```python
# src/codeintel/ingestion/tools/__init__.py

from .plugins import (
    ToolStatus,
    ToolPluginResult,
    ToolPluginMetadata,
    ToolPlugin,
    ToolPluginRegistry,
    build_default_registry,
)

__all__ = [
    "ToolStatus",
    "ToolPluginResult",
    "ToolPluginMetadata",
    "ToolPlugin",
    "ToolPluginRegistry",
    "build_default_registry",
]
```

---

## 3. Pyright plugin (`ingestion/tools/pyright.py`)

This plugin centralizes **how** we call pyright and how we normalize errors, but still lets `ToolService.run_pyright` keep its existing return type (`Mapping[str, int]`).

```python
# src/codeintel/ingestion/tools/pyright.py

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunResult,
    ToolRunner,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)

log = logging.getLogger(__name__)


@dataclass
class PyrightPlugin(ToolPlugin):
    """
    Plugin responsible for running pyright and normalizing failures.

    This plugin does **not** parse the diagnostics into a mapping; that
    remains the responsibility of ToolService.run_pyright, so tests and
    callers keep the same semantics.
    """

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = ToolPluginMetadata(
        name="pyright",
        produces_artifacts=(),
        consumes_configs=("pyright_bin",),
        datasets=("analytics.typedness", "analytics.static_diagnostics"),
    )

    async def run(self, *, repo_root: Path, **_: Any) -> ToolPluginResult:
        """
        Invoke pyright with --outputjson and normalize outcomes.

        Returns a ToolPluginResult that *never* raises; ToolService can
        decide whether to downgrade or re-raise based on status.
        """
        try:
            result = await self.runner.run_async(
                ToolName.PYRIGHT,
                ["--outputjson", str(repo_root)],
                cwd=repo_root,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError as exc:
            log.warning("pyright binary not found; treating all files as 0 errors")
            return ToolPluginResult(
                tool=ToolName.PYRIGHT,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
            )
        except ToolExecutionError as exc:
            # Preserve the underlying ToolRunResult for diagnostics
            return ToolPluginResult(
                tool=ToolName.PYRIGHT,
                status=ToolStatus.ERROR,
                artifacts={},
                run=exc.result,
                error=exc,
            )

        status = ToolStatus.OK if result.ok else ToolStatus.ERROR
        return ToolPluginResult(
            tool=result.tool,
            status=status,
            artifacts={},  # stdout holds the JSON payload
            run=result,
            error=None if status is ToolStatus.OK else ToolExecutionError(result),
        )

    @staticmethod
    def parse_diagnostics(result: ToolRunResult) -> dict[str, int]:
        """
        Parse pyright JSON from stdout into path -> error_count mapping.

        This is factored out here so ToolService can call it directly.
        """
        if not result.stdout.strip():
            return {}

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            message = f"pyright returned non-JSON output: {exc}"
            raise ToolExecutionError(result) from exc

        # This mirrors your existing logic: adjust as needed to match tests.
        summary = payload.get("summary", {})
        if not isinstance(summary, dict):
            log.warning("Unexpected pyright JSON structure; missing 'summary'")
            return {}

        by_file = summary.get("files", {})
        if not isinstance(by_file, dict):
            return {}

        errors: dict[str, int] = {}
        for path, info in by_file.items():
            if not isinstance(info, dict):
                continue
            count = int(info.get("errorCount", 0))
            errors[str(path)] = count
        return errors
```

---

## 4. Coverage plugin (`ingestion/tools/coverage.py`)

This plugin will be used by `ToolService.run_coverage_json`. It encapsulates CLI invocation and error normalization; parsing of JSON into `CoverageFileReport` remains in `ToolService`.

```python
# src/codeintel/ingestion/tools/coverage.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunResult,
    ToolRunner,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)

log = logging.getLogger(__name__)


@dataclass
class CoveragePlugin(ToolPlugin):
    """
    Plugin for running `coverage json` to produce a JSON coverage report.

    The plugin ensures the output directory exists and normalizes errors;
    it does *not* parse the JSON, leaving that to ToolService.
    """

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = ToolPluginMetadata(
        name="coverage",
        produces_artifacts=("coverage_json",),
        consumes_configs=("coverage_bin",),
        datasets=("analytics.coverage_lines",),
    )

    async def run(
        self,
        *,
        repo_root: Path,
        coverage_file: Path,
        output_path: Path,
        **_: Any,
    ) -> ToolPluginResult:
        """
        Run coverage CLI to produce a JSON report.

        Parameters
        ----------
        repo_root:
            Repository root passed to the CLI via `cwd`.
        coverage_file:
            Path to `.coverage` data file.
        output_path:
            Target JSON file path.
        """
        await to_thread.run_sync(output_path.parent.mkdir, parents=True, exist_ok=True)

        args = [
            "json",
            "-i",
            f"--data-file={coverage_file}",
            f"--output={output_path}",
        ]
        try:
            result = await self.runner.run_async(
                ToolName.COVERAGE,
                args,
                cwd=repo_root,
                output_path=output_path,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError as exc:
            log.warning("coverage binary not found; skipping coverage ingestion")
            return ToolPluginResult(
                tool=ToolName.COVERAGE,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.COVERAGE,
                status=ToolStatus.ERROR,
                artifacts={"coverage_json": output_path},
                run=exc.result,
                error=exc,
            )

        # Even on non-zero exit, coverage may still produce JSON.
        status = ToolStatus.OK if result.ok else ToolStatus.ERROR
        artifacts = {"coverage_json": output_path}

        return ToolPluginResult(
            tool=result.tool,
            status=status,
            artifacts=artifacts,
            run=result,
            error=None if status is ToolStatus.OK else ToolExecutionError(result),
        )
```

---

## 5. Pytest plugin (`ingestion/tools/pytest.py`)

This plugin wraps `pytest` with the JSON-report plugin; ToolService will keep the same behaviour (generate or reuse report).

```python
# src/codeintel/ingestion/tools/pytest.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)

log = logging.getLogger(__name__)


@dataclass
class PytestPlugin(ToolPlugin):
    """
    Plugin for running pytest with the JSON-report plugin enabled.

    This plugin generates a JSON report at the requested path.
    """

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = ToolPluginMetadata(
        name="pytest",
        produces_artifacts=("pytest_json_report",),
        consumes_configs=("pytest_bin",),
        datasets=("analytics.test_catalog",),
    )

    async def run(
        self,
        *,
        repo_root: Path,
        json_report_path: Path,
        **_: Any,
    ) -> ToolPluginResult:
        await to_thread.run_sync(json_report_path.parent.mkdir, parents=True, exist_ok=True)

        args = [
            "-m",
            "pytest",
            "-q",
            "--disable-warnings",
            "--maxfail=1",
            "--json-report",
            f"--json-report-file={json_report_path}",
        ]

        try:
            result = await self.runner.run_async(
                ToolName.PYTEST,
                args,
                cwd=repo_root,
                output_path=json_report_path,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError as exc:
            log.warning("pytest binary not found; skipping test ingestion")
            return ToolPluginResult(
                tool=ToolName.PYTEST,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.PYTEST,
                status=ToolStatus.ERROR,
                artifacts={"pytest_json_report": json_report_path},
                run=exc.result,
                error=exc,
            )

        status = ToolStatus.OK if result.ok else ToolStatus.ERROR
        artifacts = {"pytest_json_report": json_report_path}

        return ToolPluginResult(
            tool=result.tool,
            status=status,
            artifacts=artifacts,
            run=result,
            error=None if status is ToolStatus.OK else ToolExecutionError(result),
        )
```

---

## 6. SCIP plugin (`ingestion/tools/scip.py`) – pattern

SCIP is a bit more involved, but conceptually the same. You’ll wrap:

* `scip-python` invocation (full or sharded) and
* `scip` conversion to JSON

and expose artifacts (`index_scip`, `index_json`, or per-shard JSON).

Sketch:

```python
# src/codeintel/ingestion/tools/scip.py

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)

log = logging.getLogger(__name__)


@dataclass
class ScipPlugin(ToolPlugin):
    """
    Plugin for SCIP indexing via scip-python + scip CLI.

    Exposes two primary operations:
      - full index for a repo (index_scip + index_json)
      - sharded index for selected paths (shard_scip + shard_json)
    """

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = ToolPluginMetadata(
        name="scip",
        produces_artifacts=("index_scip", "index_json", "shard_json"),
        consumes_configs=("scip_python_bin", "scip_bin"),
        datasets=("core.scip_symbols", "core.goid_crosswalk"),
    )

    async def run(
        self,
        *,
        repo_root: Path,
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
        rel_paths: Sequence[str] | None = None,
        **_: Any,
    ) -> ToolPluginResult:
        """
        Run scip-python and scip to produce SCIP + JSON index.

        When `rel_paths` is provided, a shard is produced; otherwise a full
        index is created under `target_dir` or repo_root.
        """
        # This should internally mirror your existing _run_scip_python and
        # scip JSON export logic; omitted here in detail for brevity.
        # The key is: assemble ToolRunResult(s), catch ToolNotFoundError /
        # ToolExecutionError, and return a ToolPluginResult with artifacts:
        #   {"index_scip": output_scip, "index_json": output_json}
        ...
```

You’ll lift the logic now in `ToolService._run_scip_python`, `run_scip_full`, `run_scip_shard` into this plugin and then have those service methods simply call the plugin (see next section).

---

## 7. Wire plugins into `ToolService` (`ingestion/tool_service.py`)

Now we make `ToolService` a thin façade over the plugin registry.

### 7.1. Imports & `__init__`

At the top of `tool_service.py`, add:

```python
from codeintel.ingestion.tools import (
    ToolPlugin,
    ToolPluginResult,
    ToolPluginRegistry,
    ToolStatus,
    build_default_registry,
)
```

Then adapt the constructor:

```python
class ToolService:
    """Orchestrate external tooling and parse outputs for ingestion modules."""

    def __init__(self, runner: ToolRunner, tools_config: ToolsConfig | None = None) -> None:
        self.runner = runner
        self.tools_config = tools_config or runner.tools_config
        self._plugins: ToolPluginRegistry = build_default_registry(self.runner, self.tools_config)
```

Add helpers:

```python
    def get_plugin(self, name: str) -> ToolPlugin:
        """Return a registered plugin by name."""
        return self._plugins.get(name)

    async def run_plugin(self, name: str, **kwargs: Any) -> ToolPluginResult:
        """
        Execute a tool plugin by name.

        Tool-specific kwargs are passed through to the plugin's run() method.
        """
        plugin = self.get_plugin(name)
        return await plugin.run(**kwargs)
```

### 7.2. Refactor `run_pyright` to use the plugin

**Before** (simplified):

```python
async def run_pyright(self, repo_root: Path) -> Mapping[str, int]:
    try:
        result = await self.runner.run_async(
            ToolName.PYRIGHT,
            ["--outputjson", str(repo_root)],
            cwd=repo_root,
            timeout_s=self.tools_config.default_timeout_s,
        )
    except ToolNotFoundError:
        log.warning("pyright binary not found; treating all files as 0 errors")
        return {}

    if result.returncode not in {0, 1}:
        raise ToolExecutionError(result)

    # parse result.stdout JSON to path -> error count
    ...
```

**After** – delegate to plugin:

```python
    async def run_pyright(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run pyright and return error counts keyed by repo-relative path.

        This implementation delegates CLI invocation to the PyrightPlugin
        and preserves existing behaviour for NOT_FOUND and ERROR cases.
        """
        plugin_result = await self.run_plugin("pyright", repo_root=repo_root)

        if plugin_result.status is ToolStatus.NOT_FOUND:
            # Preserve old downgrade behaviour.
            log.warning("pyright binary not found; treating all files as 0 errors")
            return {}

        if plugin_result.status is not ToolStatus.OK:
            # Preserve old semantics: raise ToolExecutionError on unexpected exit.
            if isinstance(plugin_result.error, ToolExecutionError):
                raise plugin_result.error
            if plugin_result.run is not None:
                raise ToolExecutionError(plugin_result.run)
            # Fallback: synthesize a minimal ToolRunResult if needed.
            raise RuntimeError("pyright plugin failed without ToolRunResult")

        assert plugin_result.run is not None
        return PyrightPlugin.parse_diagnostics(plugin_result.run)
```

### 7.3. Refactor `run_pyrefly` / `run_ruff`

Implement the same pattern:

* Call `run_plugin("pyrefly", repo_root=repo_root)` / `run_plugin("ruff", repo_root=repo_root)`.
* Map `ToolStatus.NOT_FOUND` → empty map and log warning.
* Map `ToolStatus.ERROR` → either degrade or raise `ToolExecutionError` as your existing code does.
* Parse JSON from `plugin_result.run.stdout` exactly as before.

You’ll mirror your current logic in these methods, just replacing direct `runner.run_async(...)` with plugin calls.

### 7.4. Refactor `run_coverage_json` to use `CoveragePlugin`

**Before** (simplified):

```python
async def run_coverage_json(
    self,
    repo_root: Path,
    *,
    coverage_file: Path,
    output_path: Path,
) -> list[CoverageFileReport]:
    json_path = output_path
    try:
        result = await self.runner.run_async(
            ToolName.COVERAGE,
            args,
            cwd=repo_root,
            output_path=json_path,
            timeout_s=self.tools_config.default_timeout_s,
        )
    except (ToolExecutionError, ToolNotFoundError) as exc:
        log.warning("coverage CLI failed; falling back to API parsing: %s", exc)
        return []
    # parse json_path into CoverageFileReport instances
    ...
```

**After**:

```python
    async def run_coverage_json(
        self,
        repo_root: Path,
        *,
        coverage_file: Path,
        output_path: Path,
    ) -> list[CoverageFileReport]:
        """
        Run coverage CLI via the CoveragePlugin and parse its JSON output.

        Returns an empty list when the tool is missing or when parsing fails.
        """
        plugin_result = await self.run_plugin(
            "coverage",
            repo_root=repo_root,
            coverage_file=coverage_file,
            output_path=output_path,
        )

        if plugin_result.status is ToolStatus.NOT_FOUND:
            log.warning("coverage binary not found; skipping coverage ingestion")
            return []

        if plugin_result.status is not ToolStatus.OK:
            # Preserve old “fall back to API parsing” behaviour by logging and
            # returning an empty list; coverage_ingest will decide whether to
            # use API mode instead.
            log.warning(
                "coverage CLI failed or returned non-zero exit; status=%s error=%r",
                plugin_result.status,
                plugin_result.error,
            )
            return []

        json_path = plugin_result.artifacts.get("coverage_json", output_path)
        payload = await to_thread.run_sync(_load_json_file, json_path)
        if payload is None:
            log.warning("coverage json report missing or empty at %s", json_path)
            return []

        return self._parse_coverage_json(payload)
```

Here `_load_json_file` + `_parse_coverage_json` are small helpers that you can factor from your existing implementation; they don’t need to move into the plugin.

### 7.5. Refactor `run_pytest_report` to use `PytestPlugin`

**Before**: you call `runner.run_async` with pytest CLI and return `True/False`.

**After**:

```python
    async def run_pytest_report(
        self,
        repo_root: Path,
        *,
        json_report_path: Path,
    ) -> bool:
        """
        Generate a pytest JSON report when missing.

        Returns True when pytest was executed to produce a new report,
        False when an existing report was reused.
        """
        if await to_thread.run_sync(_path_is_file, json_report_path):
            return False

        plugin_result = await self.run_plugin(
            "pytest",
            repo_root=repo_root,
            json_report_path=json_report_path,
        )

        if plugin_result.status is ToolStatus.NOT_FOUND:
            raise ToolNotFoundError(ToolName.PYTEST, self.tools_config.pytest_bin)

        if plugin_result.status is not ToolStatus.OK:
            if isinstance(plugin_result.error, ToolExecutionError):
                raise plugin_result.error
            if plugin_result.run is not None:
                raise ToolExecutionError(plugin_result.run)
            raise RuntimeError("pytest plugin failed without ToolRunResult")

        exists = await to_thread.run_sync(_path_is_file, json_report_path)
        if not exists:
            raise ToolExecutionError(
                plugin_result.run
                or ToolRunResult(  # type: ignore[call-arg]
                    tool=ToolName.PYTEST,
                    args=(),
                    returncode=1,
                    stdout="",
                    stderr="pytest completed but report missing",
                    duration_s=0.0,
                    output_path=json_report_path,
                )
            )
        return True
```

### 7.6. Refactor SCIP methods to use `ScipPlugin`

* `run_scip_full` → call `run_plugin("scip", repo_root=repo_root, output_scip=output_scip, output_json=output_json, target_dir=target_dir)`, check `status`, raise `ToolExecutionError` if failure.
* `run_scip_shard` → call `run_plugin("scip", repo_root=repo_root, output_scip=shard_scip, output_json=shard_json, rel_paths=rel_paths)`.

In both cases, return or use the artifacts (`index_scip`, `index_json`, `shard_json`) from `plugin_result.artifacts`.

---

## 8. Ingestion modules: no signature changes

The nice thing: **all ingestion modules keep calling `ToolService` in the same way**:

* `typing_ingest` still does:

  ```python
  error_maps = asyncio.run(_collect_error_maps(repo_root, active_service))
  ```

  where `_collect_error_maps` calls `service.run_pyrefly`, `service.run_pyright`, `service.run_ruff`.

* `coverage_ingest` still does:

  ```python
  reports = asyncio.run(
      service.run_coverage_json(
          repo_root,
          coverage_file=coverage_file,
          output_path=json_path,
      )
  )
  ```

* `tests_ingest` still uses `service.run_pytest_report`.

* `scip_ingest` still uses `service.run_scip_full` / `run_scip_shard`.

You don’t have to touch those modules for Epic 3; all the behaviour change is centralized in `ToolService` + plugins.

---

## 9. Tests: `tests/ingestion/test_tool_plugins.py`

Finally, add focused tests for the plugin runtime. Example:

```python
# tests/tests/ingestion/test_tool_plugins.py

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import ToolName, ToolRunResult, ToolRunner, ToolExecutionError, ToolNotFoundError
from codeintel.ingestion.tools import ToolStatus, build_default_registry
from codeintel.ingestion.tools.pyright import PyrightPlugin


class DummyRunner(ToolRunner):
    """Test double that bypasses real subprocess execution."""

    def __init__(self, result: ToolRunResult | Exception) -> None:
        self._result = result
        super().__init__(tools_config=ToolsConfig.default(), cache_dir=Path("build/.tool_cache"))

    async def run_async(  # type: ignore[override]
        self,
        tool: ToolName | str,
        args: list[str] | tuple[str, ...],
        *,
        cwd: Path | None = None,
        output_path: Path | None = None,
        timeout_s: float | None = None,
    ) -> ToolRunResult:
        if isinstance(self._result, Exception):
            if isinstance(self._result, ToolNotFoundError):
                raise self._result
            raise ToolExecutionError(
                ToolRunResult(
                    tool=ToolName.PYRIGHT,
                    args=tuple(args),
                    returncode=1,
                    stdout="",
                    stderr="dummy error",
                    duration_s=0.1,
                    output_path=output_path,
                )
            )
        return self._result


@pytest.mark.asyncio
async def test_pyright_plugin_not_found_downgrades() -> None:
    tools_cfg = ToolsConfig.default()
    exc = ToolNotFoundError(ToolName.PYRIGHT, tools_cfg.pyright_bin)
    runner = DummyRunner(exc)
    plugin = PyrightPlugin(runner=runner, tools_config=tools_cfg)

    result = await plugin.run(repo_root=Path("."))
    assert result.status is ToolStatus.NOT_FOUND
    assert result.run is None
    assert isinstance(result.error, ToolNotFoundError)


@pytest.mark.asyncio
async def test_pyright_plugin_ok_status() -> None:
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
    runner = DummyRunner(run)
    plugin = PyrightPlugin(runner=runner, tools_config=tools_cfg)

    result = await plugin.run(repo_root=Path("."))
    assert result.status is ToolStatus.OK
    assert result.ok
    assert result.run is run
```

You can also add a small test that:

* Builds `build_default_registry`.
* Asserts it contains `"pyright"`, `"pyrefly"`, `"ruff"`, `"coverage"`, `"pytest"`, `"scip"`.
* Confirms that `ToolService.__init__` wires `self._plugins` correctly by running a simple happy-path through one plugin with a `DummyRunner`.

---

If you’d like, next we can zoom in on **just one service method**, e.g. `run_pyright` or `run_scip_full`, and I can give you a literal **before/after diff** against your current `tool_service.py` so you can patch it with zero ambiguity, like we did for Epic 1 & 2.

# zoom in on run_scip_full #

Here’s the zoom-in on **`ToolService.run_scip_full`** (and its sibling `run_scip_shard`) with a literal before/after, wired into the `ScipPlugin` via `run_plugin("scip", ...)`.

I’ll assume you’ve already:

* Introduced the **plugin registry** (`ToolPluginRegistry`, `build_default_registry`) and
* Added `ToolService.run_plugin(...)` as in the previous Epic 3 plan.

If not, this will be the only extra thing you need to add.

---

## 1. Current code (BEFORE)

From your updated `ingestion/tool_service.py`, the SCIP wiring currently looks like this:

```python
    async def run_scip_full(
        self,
        repo_root: Path,
        *,
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> None:
        """Run scip-python for a full index and export to JSON."""
        await self._run_scip_python(
            repo_root,
            output_scip=output_scip,
            target_dir=target_dir,
            target_only=None,
        )
        await self._run_scip_print(output_scip, output_json)

    async def run_scip_shard(
        self,
        repo_root: Path,
        *,
        rel_paths: Sequence[str],
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> None:
        """Run scip-python for a subset of files and export to JSON."""
        await self._run_scip_python(
            repo_root,
            output_scip=output_scip,
            target_dir=target_dir,
            target_only=rel_paths,
        )
        await self._run_scip_print(output_scip, output_json)

    async def _run_scip_python(
        self,
        repo_root: Path,
        *,
        output_scip: Path,
        target_dir: Path | None,
        target_only: Sequence[str] | None,
    ) -> None:
        target_base = await to_thread.run_sync(_resolve_target_base, repo_root, target_dir)
        await to_thread.run_sync(_mkdir_parents, output_scip.parent)
        args: list[str] = ["index", str(target_base), "--output", str(output_scip)]
        for rel_path in target_only or ():
            args.extend(["--target-only", rel_path])
        result = await self.runner.run_async(
            ToolName.SCIP_PYTHON,
            args,
            cwd=repo_root,
            output_path=output_scip,
            timeout_s=self.tools_config.default_timeout_s,
        )
        if not result.ok:
            raise ToolExecutionError(result)

    async def _run_scip_print(self, scip_path: Path, output_json: Path) -> None:
        args = ["print", "--json", str(scip_path)]
        await to_thread.run_sync(_mkdir_parents, output_json.parent)
        result = await self.runner.run_async(
            ToolName.SCIP,
            args,
            cwd=scip_path.parent,
            output_path=output_json,
            timeout_s=self.tools_config.default_timeout_s,
        )
        if not result.ok:
            raise ToolExecutionError(result)
        await to_thread.run_sync(_write_text, output_json, result.stdout or "")
```

This is the direct ToolRunner-based path.

---

## 2. New ScipPlugin (context for the AFTER)

To make `run_scip_full` plugin-driven, we need a `ScipPlugin` implementation.

Create **`src/codeintel/ingestion/tools/scip.py`** (if you haven’t already) like this:

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunResult,
    ToolRunner,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)

log = logging.getLogger(__name__)


def _mkdir_parents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf8")


def _resolve_target_base(repo_root: Path, target_dir: Path | None) -> Path:
    if target_dir is not None:
        return target_dir
    src_dir = repo_root / "src"
    return src_dir if src_dir.is_dir() else repo_root


@dataclass
class ScipPlugin(ToolPlugin):
    """
    Plugin for SCIP indexing via scip-python + scip CLI.

    Produces both a .scip index and a JSON export.
    """

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = ToolPluginMetadata(
        name="scip",
        produces_artifacts=("index_scip", "index_json"),
        consumes_configs=("scip_python_bin", "scip_bin"),
        datasets=("core.scip_symbols", "core.goid_crosswalk"),
    )

    async def run(
        self,
        *,
        repo_root: Path,
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
        rel_paths: Sequence[str] | None = None,
        **_: Any,
    ) -> ToolPluginResult:
        """
        Run scip-python index and scip print to produce SCIP + JSON.

        When rel_paths is provided, only those paths are targeted; otherwise
        the full repo (or target_dir/src) is indexed.
        """
        # 1) scip-python index
        try:
            target_base = await to_thread.run_sync(_resolve_target_base, repo_root, target_dir)
            await to_thread.run_sync(_mkdir_parents, output_scip.parent)

            args: list[str] = ["index", str(target_base), "--output", str(output_scip)]
            for rel_path in rel_paths or ():
                args.extend(["--target-only", rel_path])

            scip_result = await self.runner.run_async(
                ToolName.SCIP_PYTHON,
                args,
                cwd=repo_root,
                output_path=output_scip,
                timeout_s=self.tools_config.default_timeout_s,
            )
            if not scip_result.ok:
                raise ToolExecutionError(scip_result)
        except ToolNotFoundError as exc:
            log.warning("scip-python binary not found; SCIP index cannot be built")
            return ToolPluginResult(
                tool=ToolName.SCIP_PYTHON,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.SCIP_PYTHON,
                status=ToolStatus.ERROR,
                artifacts={"index_scip": output_scip},
                run=exc.result,
                error=exc,
            )

        # 2) scip print --json
        try:
            await to_thread.run_sync(_mkdir_parents, output_json.parent)
            print_result = await self.runner.run_async(
                ToolName.SCIP,
                ["print", "--json", str(output_scip)],
                cwd=output_scip.parent,
                output_path=output_json,
                timeout_s=self.tools_config.default_timeout_s,
            )
            if not print_result.ok:
                raise ToolExecutionError(print_result)

            await to_thread.run_sync(_write_text, output_json, print_result.stdout or "")
        except ToolNotFoundError as exc:
            log.warning("scip binary not found; JSON export cannot be built")
            return ToolPluginResult(
                tool=ToolName.SCIP,
                status=ToolStatus.NOT_FOUND,
                artifacts={"index_scip": output_scip},
                run=None,
                error=exc,
            )
        except ToolExecutionError as exc:
            return ToolPluginResult(
                tool=ToolName.SCIP,
                status=ToolStatus.ERROR,
                artifacts={"index_scip": output_scip, "index_json": output_json},
                run=exc.result,
                error=exc,
            )

        # Success: both index and JSON exist
        artifacts = {
            "index_scip": output_scip,
            "index_json": output_json,
        }
        return ToolPluginResult(
            tool=ToolName.SCIP,
            status=ToolStatus.OK,
            artifacts=artifacts,
            run=print_result,
            error=None,
        )
```

And make sure your plugin registry registers it:

```python
# ingestion/tools/plugins.py (inside build_default_registry)
from codeintel.ingestion.tools.scip import ScipPlugin

...

registry.register(ScipPlugin(runner=runner, tools_config=tools_config))
```

---

## 3. Imports needed in `tool_service.py` (AFTER)

At the top of `ingestion/tool_service.py`, add imports for the plugin status/result:

```python
from codeintel.ingestion.tools import (
    ToolPluginResult,
    ToolStatus,
    build_default_registry,
)
```

And in `ToolService.__init__`, ensure you build the registry (something like):

```python
class ToolService:
    def __init__(self, runner: ToolRunner, tools_config: ToolsConfig | None = None) -> None:
        self.runner = runner
        self.tools_config = tools_config or runner.tools_config
        self._plugins = build_default_registry(self.runner, self.tools_config)
```

Plus a helper:

```python
    async def run_plugin(self, name: str, **kwargs: Any) -> ToolPluginResult:
        plugin = self._plugins.get(name)
        return await plugin.run(**kwargs)
```

(That’s all context for the new `run_scip_full` / `run_scip_shard`.)

---

## 4. New code (AFTER) for `run_scip_full` and `run_scip_shard`

Now, replace the **existing** `run_scip_full` and `run_scip_shard` definitions in `ToolService` with these plugin-backed versions.

### 4.1. Replace `run_scip_full` with:

```python
    async def run_scip_full(
        self,
        repo_root: Path,
        *,
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> None:
        """
        Run scip-python for a full index and export to JSON using the ScipPlugin.

        This preserves the previous behaviour: a non-zero exit from scip-python
        or scip print results in a ToolExecutionError.
        """
        plugin_result = await self.run_plugin(
            "scip",
            repo_root=repo_root,
            output_scip=output_scip,
            output_json=output_json,
            target_dir=target_dir,
            rel_paths=None,
        )

        if plugin_result.status is ToolStatus.NOT_FOUND:
            # Treat missing scip/scip-python as a hard failure, as before.
            if isinstance(plugin_result.error, ToolNotFoundError):
                raise plugin_result.error
            raise RuntimeError("SCIP tools not found and no detailed error provided")

        if plugin_result.status is not ToolStatus.OK:
            # Preserve old semantics: propagate ToolExecutionError when available.
            if isinstance(plugin_result.error, ToolExecutionError):
                raise plugin_result.error
            if plugin_result.run is not None:
                raise ToolExecutionError(plugin_result.run)
            raise RuntimeError("SCIP plugin failed without ToolRunResult")
```

### 4.2. Replace `run_scip_shard` with:

```python
    async def run_scip_shard(
        self,
        repo_root: Path,
        *,
        rel_paths: Sequence[str],
        output_scip: Path,
        output_json: Path,
        target_dir: Path | None = None,
    ) -> None:
        """
        Run scip-python for a subset of files and export to JSON using the ScipPlugin.

        The same error semantics as run_scip_full are preserved.
        """
        plugin_result = await self.run_plugin(
            "scip",
            repo_root=repo_root,
            output_scip=output_scip,
            output_json=output_json,
            target_dir=target_dir,
            rel_paths=list(rel_paths),
        )

        if plugin_result.status is ToolStatus.NOT_FOUND:
            if isinstance(plugin_result.error, ToolNotFoundError):
                raise plugin_result.error
            raise RuntimeError("SCIP tools not found and no detailed error provided")

        if plugin_result.status is not ToolStatus.OK:
            if isinstance(plugin_result.error, ToolExecutionError):
                raise plugin_result.error
            if plugin_result.run is not None:
                raise ToolExecutionError(plugin_result.run)
            raise RuntimeError("SCIP plugin failed without ToolRunResult")
```

### 4.3. What about `_run_scip_python` and `_run_scip_print`?

After this change:

* `run_scip_full` / `run_scip_shard` no longer call `_run_scip_python` or `_run_scip_print`.
* Those helper methods (and their helpers `_resolve_target_base`, etc.) have effectively been **inlined** into `ScipPlugin`.

You have two options:

1. **Keep them for now**, in case tests still reference them or you want them as reference.
2. **Delete them** once you’ve migrated everything and test suite is green.

If you delete them, also delete `_resolve_target_base` from `tool_service.py` (you already re-implemented it inside `ScipPlugin`).

---

## 5. Behaviour check

With this patch:

* Any caller of `ToolService.run_scip_full` / `run_scip_shard` (e.g. `scip_ingest`) still sees exactly the same surface API.
* Failures from either `scip-python` or `scip` still result in `ToolExecutionError` (or `ToolNotFoundError`) just as before.
* Success still yields both `.scip` and `.json` artifacts, now recorded in the plugin’s `ToolPluginResult.artifacts` map.
* The SCIP logic is now part of the **plugin runtime**, consistent with Pyright, Pyrefly, Ruff, Coverage, and Pytest.

If you’d like, I can next do the same tight before/after treatment for **`run_pyright`** so the style is absolutely uniform across tools.


# zoom in on run_pyright #

Here we go — let’s make `run_pyright` look like `run_scip_full` in the new plugin world.

I’ll:

1. Show your **current** `run_pyright` (from the uploaded `tool_service.py`).
2. Show the **new** plugin-backed version.
3. Show the minimal helper you need in `ToolService` if you haven’t already added `run_plugin`.

I won’t touch `_parse_pyright_errors` — it stays exactly as-is.

---

## 1. Current `run_pyright` (BEFORE)

From your current `ingestion/tool_service.py`:

```python
async def run_pyright(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run pyright and return error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the pyright invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to error counts.

        Raises
        ------
        ToolExecutionError
            When pyright exits with an unexpected status code or returns
            output that cannot be parsed.
        """
        try:
            result = await self.runner.run_async(
                ToolName.PYRIGHT,
                ["--outputjson", str(repo_root)],
                cwd=repo_root,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError:
            log.warning("pyright binary not found; treating all files as 0 errors")
            return {}

        if result.returncode not in {0, 1}:
            raise ToolExecutionError(result)
        return _parse_pyright_errors(result.stdout, repo_root)
```

That’s the direct `ToolRunner` version.

---

## 2. Plugin-backed `run_pyright` (AFTER)

Now replace that entire function with this version that uses the `PyrightPlugin` via `run_plugin("pyright", ...)` and `ToolStatus`:

```python
async def run_pyright(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run pyright and return error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the pyright invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to error counts.

        Raises
        ------
        ToolExecutionError
            When pyright exits with an unexpected status code or returns
            output that cannot be parsed.
        """
        # Delegate process execution + normalization to the PyrightPlugin
        plugin_result = await self.run_plugin("pyright", repo_root=repo_root)

        # Preserve old downgrade behaviour when the binary is missing.
        if plugin_result.status is ToolStatus.NOT_FOUND:
            log.warning("pyright binary not found; treating all files as 0 errors")
            return {}

        # For any non-OK status, propagate the underlying ToolExecutionError
        # when present, otherwise synthesize one from the ToolRunResult.
        if plugin_result.status is not ToolStatus.OK:
            err = plugin_result.error
            if isinstance(err, ToolExecutionError):
                raise err
            if plugin_result.run is not None:
                raise ToolExecutionError(plugin_result.run)
            raise RuntimeError("pyright plugin failed without ToolRunResult")

        # Happy path: we have a valid ToolRunResult with JSON on stdout.
        assert plugin_result.run is not None
        return _parse_pyright_errors(plugin_result.run.stdout, repo_root)
```

Key points:

* **Missing binary** (`ToolStatus.NOT_FOUND`) still downgrades to “treat everything as 0 errors” with a warning.
* Any other non-OK status gets converted into a `ToolExecutionError` just like before.
* On success we still use `_parse_pyright_errors(stdout, repo_root)`, so test expectations / formats stay the same.

---

## 3. Minimal plumbing this depends on

If you’ve already wired in the plugin runtime (per the Epic 3 plan + `run_scip_full` patch), you should already have these bits.

If not, here’s what `run_pyright` depends on in `ToolService`:

### 3.1. Imports at the top of `tool_service.py`

Add (or confirm you have):

```python
from codeintel.ingestion.tools import (
    ToolPluginResult,
    ToolStatus,
    build_default_registry,
)
```

And keep your existing import:

```python
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunner,
)
```

### 3.2. Initialize the plugin registry in `ToolService.__init__`

In `ToolService.__init__`:

```python
class ToolService:
    """Orchestrate external tooling and parse outputs for ingestion modules."""

    def __init__(self, runner: ToolRunner, tools_config: ToolsConfig | None = None) -> None:
        self.runner = runner
        self.tools_config = tools_config or runner.tools_config
        # NEW: plugin registry
        self._plugins = build_default_registry(self.runner, self.tools_config)
```

### 3.3. A small helper to run plugins

Add this method to `ToolService` (e.g. just above `run_pyright`):

```python
    async def run_plugin(self, name: str, **kwargs: Any) -> ToolPluginResult:
        """
        Execute a tool plugin by name and return its normalized result.

        Parameters
        ----------
        name
            Plugin registry name (e.g., 'pyright', 'coverage', 'pytest').
        **kwargs
            Plugin-specific arguments (e.g. repo_root, coverage_file, output_path).

        Returns
        -------
        ToolPluginResult
            Normalized status, artifacts, and underlying ToolRunResult.
        """
        plugin = self._plugins.get(name)
        return await plugin.run(**kwargs)
```

With that in place:

* `run_scip_full`, `run_scip_shard`, and `run_pyright` can all look the same stylistically.
* You can later convert `run_pyrefly`, `run_ruff`, `run_coverage_json`, and `run_pytest_report` to the same pattern without touching their external signatures.

---

If you want, next we can do **`run_pyrefly`** or **`run_ruff`** in the same before/after style so the whole static-analysis stack is perfectly uniform.


# zoom in on run_pyrefly #

Awesome, let’s make `run_pyrefly` match the plugin style you now have for `run_scip_full` and `run_pyright`.

I’ll show:

1. The **current** `run_pyrefly` from your `tool_service.py`.
2. The **new** plugin-backed version that preserves behaviour (no exceptions, degrade to `{}`).
3. The expectations it has of `PyreflyPlugin` (so you can keep things consistent).

---

## 1. Current `run_pyrefly` (BEFORE)

From your current `ingestion/tool_service.py`:

```python
    async def run_pyrefly(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run pyrefly and return error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the pyrefly invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to error counts.
        """
        output_path = self.runner.cache_dir / "pyrefly.json"
        await to_thread.run_sync(_mkdir_parents, output_path.parent)
        args = [
            "check",
            str(repo_root),
            "--output-format",
            "json",
            "--output",
            str(output_path),
            "--summary",
            "none",
            "--count-errors=0",
        ]
        try:
            result = await self.runner.run_async(
                ToolName.PYREFLY,
                args,
                cwd=repo_root,
                output_path=output_path,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError:
            log.warning("pyrefly binary not found; treating all files as 0 errors")
            await to_thread.run_sync(_unlink_missing, output_path)
            return {}

        output_exists = await to_thread.run_sync(_path_is_file, output_path)
        if result.returncode != 0 and not output_exists:
            log.warning(
                "pyrefly exited with code %s and produced no output; stdout=%s stderr=%s",
                result.returncode,
                result.stdout.strip(),
                result.stderr.strip(),
            )
            await to_thread.run_sync(_unlink_missing, output_path)
            return {}

        payload = await to_thread.run_sync(ToolRunner.load_json, output_path) or {}
        await to_thread.run_sync(_unlink_missing, output_path)
        return _parse_pyrefly_errors(payload, repo_root)
```

Semantics to preserve:

* If **binary missing** → log warning, delete any stale JSON, return `{}`.
* If pyrefly exits non-zero **and** **no JSON file** → log warning, delete file, return `{}`.
* If JSON exists (even on non-zero exit) → load it and parse; no exceptions raised.

---

## 2. Plugin-backed `run_pyrefly` (AFTER)

Now we’ll rewrite `run_pyrefly` so it:

* Delegates **process execution + error normalization** to `PyreflyPlugin` via `run_plugin("pyrefly", ...)`.
* Still handles:

  * `ToolStatus.NOT_FOUND` → warn + `{}`.
  * Other non-OK statuses → warn + `{}`.
  * OK status → load JSON file from `plugin_result.artifacts` and feed `_parse_pyrefly_errors`.

Replace the entire function above with this:

```python
    async def run_pyrefly(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run pyrefly and return error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the pyrefly invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to error counts.
        """
        output_path = self.runner.cache_dir / "pyrefly.json"

        # Delegate CLI invocation to the PyreflyPlugin. The plugin is responsible
        # for creating the output directory and deciding whether a non-zero exit
        # is still usable (i.e. JSON was produced).
        plugin_result = await self.run_plugin(
            "pyrefly",
            repo_root=repo_root,
            output_path=output_path,
        )

        # Binary missing → degrade to 0 errors, as before.
        if plugin_result.status is ToolStatus.NOT_FOUND:
            log.warning("pyrefly binary not found; treating all files as 0 errors")
            await to_thread.run_sync(_unlink_missing, output_path)
            return {}

        # Any other non-OK status → log and degrade to 0 errors (no exceptions).
        if plugin_result.status is not ToolStatus.OK:
            log.warning(
                "pyrefly invocation failed or produced unusable output; status=%s error=%r",
                plugin_result.status,
                plugin_result.error,
            )
            await to_thread.run_sync(_unlink_missing, output_path)
            return {}

        # Happy path: PyreflyPlugin ensured a JSON file exists.
        json_path = plugin_result.artifacts.get("pyrefly_json", output_path)
        payload = await to_thread.run_sync(ToolRunner.load_json, json_path) or {}
        await to_thread.run_sync(_unlink_missing, json_path)
        return _parse_pyrefly_errors(payload, repo_root)
```

This keeps the external contract identical (returns `Mapping[str, int]`, never raises for pyrefly) while centralizing process management in the plugin.

---

## 3. Expected behaviour of `PyreflyPlugin` (for consistency)

`run_pyrefly` above assumes you have a `PyreflyPlugin` roughly like:

```python
# src/codeintel/ingestion/tools/pyrefly.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from anyio import to_thread

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunResult,
    ToolRunner,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)

log = logging.getLogger(__name__)


def _mkdir_parents(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


@dataclass
class PyreflyPlugin(ToolPlugin):
    """
    Plugin responsible for running pyrefly and deciding whether its output
    is usable based on exit code and JSON file presence.
    """

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = ToolPluginMetadata(
        name="pyrefly",
        produces_artifacts=("pyrefly_json",),
        consumes_configs=("pyrefly_bin",),
        datasets=("analytics.static_diagnostics", "analytics.typedness"),
    )

    async def run(
        self,
        *,
        repo_root: Path,
        output_path: Path,
        **_: Any,
    ) -> ToolPluginResult:
        await to_thread.run_sync(_mkdir_parents, output_path.parent)

        args = [
            "check",
            str(repo_root),
            "--output-format",
            "json",
            "--output",
            str(output_path),
            "--summary",
            "none",
            "--count-errors=0",
        ]

        try:
            result = await self.runner.run_async(
                ToolName.PYREFLY,
                args,
                cwd=repo_root,
                output_path=output_path,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError as exc:
            # Binary missing → no JSON, NOT_FOUND status.
            return ToolPluginResult(
                tool=ToolName.PYREFLY,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
            )

        output_exists = await to_thread.run_sync(output_path.is_file)
        if not output_exists and result.returncode != 0:
            # Match old semantics: non-zero exit + no JSON = degrade.
            log.warning(
                "pyrefly exited with code %s and produced no output; stdout=%s stderr=%s",
                result.returncode,
                result.stdout.strip(),
                result.stderr.strip(),
            )
            return ToolPluginResult(
                tool=result.tool,
                status=ToolStatus.ERROR,
                artifacts={},
                run=result,
                error=ToolExecutionError(result),
            )

        # If we have JSON, we consider this OK even if returncode != 0,
        # mirroring the previous behaviour.
        artifacts = {"pyrefly_json": output_path}
        status = ToolStatus.OK
        return ToolPluginResult(
            tool=result.tool,
            status=status,
            artifacts=artifacts,
            run=result,
            error=None,
        )
```

And you register it in `build_default_registry`:

```python
from codeintel.ingestion.tools.pyrefly import PyreflyPlugin

...

registry.register(PyreflyPlugin(runner=runner, tools_config=tools_config))
```

With that, `run_pyrefly` + `PyreflyPlugin` together exactly reproduce your old semantics, but now the whole static analysis stack (Pyright, Pyrefly, Ruff, etc.) shares the same plugin runtime.

If you’d like, we can do the same for `run_ruff` next so the trio (pyright/pyrefly/ruff) is fully uniform.

# run_ruff zoom in #

Nice, let’s finish the static-analysis trio.

We’ll do the same treatment for **`run_ruff`** that you now have for `run_scip_full` and `run_pyright`:

* Show the **current** implementation.
* Provide a **plugin-backed** version that preserves behaviour.
* Show the expected **`RuffPlugin`** implementation so everything lines up.

I’ll assume you already have:

* `ToolStatus`, `ToolPluginResult`, `build_default_registry` and
* `ToolService.run_plugin(...)`.

If not, you can drop those in from the earlier Epic 3 plan.

---

## 1. Current `run_ruff` (BEFORE)

From your `ingestion/tool_service.py` (reconstructed without the ellipsis):

```python
async def run_ruff(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run ruff and return lint error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the ruff invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to lint error counts.

        Raises
        ------
        ToolExecutionError
            Raised when ruff exits with an unexpected status.
        """
        try:
            result = await self.runner.run_async(
                ToolName.RUFF,
                ["check", str(repo_root), "--output-format", "json"],
                cwd=repo_root,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError:
            log.warning("ruff binary not found; treating all files as 0 errors")
            return {}

        if result.returncode not in {0, 1}:
            raise ToolExecutionError(result)
        return _parse_ruff_errors(result.stdout, repo_root)
```

Semantics to preserve:

* If **binary missing** → log warning and return `{}` (no exception).
* If **returncode not in {0, 1}** → raise `ToolExecutionError`.
* If **returncode in {0, 1}** → parse stdout JSON via `_parse_ruff_errors`.

---

## 2. Plugin-backed `run_ruff` (AFTER)

Replace the entire `run_ruff` function above with this plugin-driven version:

```python
async def run_ruff(self, repo_root: Path) -> Mapping[str, int]:
        """
        Run ruff and return lint error counts keyed by repo-relative path.

        Parameters
        ----------
        repo_root
            Repository root supplied to the ruff invocation.

        Returns
        -------
        Mapping[str, int]
            Mapping from relative file paths to lint error counts.

        Raises
        ------
        ToolExecutionError
            Raised when ruff exits with an unexpected status.
        """
        # Delegate CLI invocation and exit-code handling to the RuffPlugin.
        plugin_result = await self.run_plugin(
            "ruff",
            repo_root=repo_root,
        )

        # Binary missing → degrade to 0 errors, as before.
        if plugin_result.status is ToolStatus.NOT_FOUND:
            log.warning("ruff binary not found; treating all files as 0 errors")
            return {}

        # Any non-OK status → propagate ToolExecutionError (or synthesize one).
        if plugin_result.status is not ToolStatus.OK:
            err = plugin_result.error
            if isinstance(err, ToolExecutionError):
                raise err
            if plugin_result.run is not None:
                raise ToolExecutionError(plugin_result.run)
            raise RuntimeError("ruff plugin failed without ToolRunResult")

        # Happy path: we have a valid ToolRunResult with JSON diagnostics on stdout.
        assert plugin_result.run is not None
        return _parse_ruff_errors(plugin_result.run.stdout, repo_root)
```

This preserves your external contract and semantics:

* **NOT_FOUND** → `{}` with warning.
* Other failures → `ToolExecutionError`.
* Success → parsing via `_parse_ruff_errors`.

---

## 3. Expected `RuffPlugin` implementation

`run_ruff` above expects a plugin registered under the name `"ruff"` that:

* Calls `ruff check <repo_root> --output-format json`.
* Maps:

  * `ToolNotFoundError` → `ToolStatus.NOT_FOUND`.
  * `returncode not in {0, 1}` → `ToolStatus.ERROR` + `ToolExecutionError`.
  * `returncode in {0, 1}` → `ToolStatus.OK`.

Here’s a concrete `RuffPlugin` you can drop in:

**File:** `src/codeintel/ingestion/tools/ruff.py`

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from codeintel.config.models import ToolsConfig
from codeintel.ingestion.tool_runner import (
    ToolExecutionError,
    ToolName,
    ToolNotFoundError,
    ToolRunResult,
    ToolRunner,
)
from codeintel.ingestion.tools.plugins import (
    ToolPlugin,
    ToolPluginMetadata,
    ToolPluginResult,
    ToolStatus,
)

log = logging.getLogger(__name__)


@dataclass
class RuffPlugin(ToolPlugin):
    """
    Plugin responsible for running ruff and normalizing failures.

    This plugin does not parse diagnostics itself; ToolService.run_ruff
    still calls _parse_ruff_errors(result.stdout, repo_root) so tests and
    callers keep the same behaviour.
    """

    runner: ToolRunner
    tools_config: ToolsConfig
    metadata: ToolPluginMetadata = ToolPluginMetadata(
        name="ruff",
        produces_artifacts=(),
        consumes_configs=("ruff_bin",),
        datasets=("analytics.static_diagnostics",),
    )

    async def run(self, *, repo_root: Path, **_: Any) -> ToolPluginResult:
        """
        Invoke ruff with JSON output and normalize the outcome.

        Returns a ToolPluginResult that does not raise; ToolService decides
        whether to downgrade (NOT_FOUND) or raise ToolExecutionError.
        """
        try:
            result = await self.runner.run_async(
                ToolName.RUFF,
                ["check", str(repo_root), "--output-format", "json"],
                cwd=repo_root,
                timeout_s=self.tools_config.default_timeout_s,
            )
        except ToolNotFoundError as exc:
            log.warning("ruff binary not found; treating all files as 0 errors")
            return ToolPluginResult(
                tool=ToolName.RUFF,
                status=ToolStatus.NOT_FOUND,
                artifacts={},
                run=None,
                error=exc,
            )

        # Match old semantics: only return codes 0 or 1 are considered usable.
        if result.returncode not in {0, 1}:
            err = ToolExecutionError(result)
            return ToolPluginResult(
                tool=result.tool,
                status=ToolStatus.ERROR,
                artifacts={},
                run=result,
                error=err,
            )

        # OK: even if there are lint errors, ruff's JSON is usable.
        return ToolPluginResult(
            tool=result.tool,
            status=ToolStatus.OK,
            artifacts={},
            run=result,
            error=None,
        )
```

And make sure you register it in the default registry:

**File:** `src/codeintel/ingestion/tools/plugins.py` (inside `build_default_registry`):

```python
from codeintel.ingestion.tools.ruff import RuffPlugin

...

registry.register(RuffPlugin(runner=runner, tools_config=tools_config))
```

---

With this:

* `run_pyright`, `run_pyrefly`, and `run_ruff` all share the same **plugin-driven pattern**.
* Your ingestion code (`typing_ingest`) still just calls `_collect_error_maps(service)` and gets the same `Mapping[str, int]` shapes back.
* All process management, exit-code normalization, and “binary missing vs error vs ok” decisions are centralized in the plugins, which makes it much easier to evolve (e.g., add metrics, caching, etc.) without touching ingestion logic again.

