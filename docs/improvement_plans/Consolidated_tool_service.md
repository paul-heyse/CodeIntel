Nice, I’m really glad this is all getting wired in cleanly. Let’s turn the “ToolRunner-based tool service” into a concrete, implementable plan.

I’ll structure this as:

1. Goals & invariants
2. Inventory & search tasks
3. Tighten / extend `ToolRunner`
4. Define / reuse `ToolsConfig`
5. Implement `ToolService` (core design + methods)
6. Wire `ToolService` into orchestration / contexts
7. Refactor `typing_ingest`
8. Refactor `coverage_ingest`
9. Refactor `tests_ingest`
10. Refactor `scip_ingest`
11. Testing and guardrails

---

## 1. Goals & invariants

After this refactor:

* **All external binaries** (pyright, pyrefly, ruff, coverage, pytest, scip-python, etc.) are executed **only** through:

  ```python
  ToolService -> ToolRunner -> subprocess
  ```

* There are **no `_run_command` one-offs** in ingestion modules.

* All external-tool semantics (argv, working directory, env, output format, error handling) live in **one module**.

* Ingestion/analytics code asks questions like “give me a map of pyright errors per file” or “run coverage and return line-coverage rows,” and **never** does CLI orchestration directly.

---

## 2. Inventory & search tasks (for the agent)

These are “prep” tasks an agent should do in the repo:

1. Find the existing runner:

   * `src/codeintel/ingestion/tool_runner.py`

   Confirm what you have:

   * `ToolRunner` dataclass
   * `ToolRunResult` / similar
   * Whether it’s async (`async def run(...)`) or sync
   * If it already accepts a `ToolsConfig` or tool name enum.

2. Find all custom subprocess logic in ingestion/orchestration:

   * `src/codeintel/ingestion/typing_ingest.py`
   * `src/codeintel/ingestion/scip_ingest.py`
   * `src/codeintel/ingestion/coverage_ingest.py`
   * `src/codeintel/ingestion/tests_ingest.py`
   * Any others using `asyncio.create_subprocess_exec`, `subprocess.run`, or manual `Popen`.

3. Find the “tool configuration” abstraction:

   * `ToolsConfig` is likely defined in `orchestration.prefect_flow` or `orchestration.context` or similar.

Document:

* What tools are tracked (pyright path, pyrefly path, ruff path, coverage, pytest, scip-python).
* How those are currently passed into ingestion modules (via contexts, env, or direct lookup).

We want to **reuse** that instead of inventing a new config.

---

## 3. Tighten / extend `ToolRunner`

Your current `ToolRunner` already does most of the good stuff:

* Validates tool name vs config
* Resolves tool path or uses `shutil.which`
* Uses `asyncio` to spawn subprocess
* Captures stdout / stderr / returncode

We want to make sure it gives `ToolService` exactly what it needs.

### 3.1 Define a tool name enum (if you don’t have one)

In `tool_runner.py`:

```python
from enum import Enum

class ToolName(str, Enum):
    PYRIGHT = "pyright"
    PYREFLY = "pyrefly"
    RUFF = "ruff"
    COVERAGE = "coverage"
    PYTEST = "pytest"
    SCIP = "scip-python"
    # add any others that matter
```

### 3.2 Make `ToolRunner` resolve tool paths via `ToolsConfig`

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .tools_config import ToolsConfig  # wherever this lives
from .tool_names import ToolName


@dataclass
class ToolRunResult:
    tool: ToolName
    args: tuple[str, ...]
    returncode: int
    stdout: str
    stderr: str


@dataclass
class ToolRunner:
    tools_config: ToolsConfig
    base_env: Mapping[str, str] | None = None

    async def run(
        self,
        tool: ToolName,
        args: Sequence[str],
        *,
        cwd: Path | None = None,
        timeout: float | None = None,
    ) -> ToolRunResult:
        # 1) Resolve path
        tool_path = self.tools_config.resolve_path(tool)
        # 2) Build env (merge base_env + tool-specific env)
        env = self.tools_config.build_env(tool, base_env=self.base_env)
        # 3) Spawn subprocess (existing logic using asyncio)
        ...
        return ToolRunResult(
            tool=tool,
            args=tuple(args),
            returncode=proc.returncode,
            stdout=stdout,
            stderr=stderr,
        )
```

If you already have `ToolRunResult` etc., just adjust names accordingly.

### 3.3 Standardize error semantics

Add two exceptions:

```python
class ToolNotFoundError(RuntimeError):
    def __init__(self, tool: ToolName, message: str) -> None:
        super().__init__(message)
        self.tool = tool


class ToolExecutionError(RuntimeError):
    def __init__(self, result: ToolRunResult) -> None:
        msg = (
            f"Tool {result.tool.value} failed with exit code {result.returncode}\n"
            f"Args: {result.args}\n"
            f"stderr:\n{result.stderr}"
        )
        super().__init__(msg)
        self.result = result
```

`ToolRunner` should:

* Raise `ToolNotFoundError` if `resolve_path` fails.
* Return `ToolRunResult` with non-zero code; let `ToolService` decide whether to interpret that as error or “tool produced diagnostics”.

---

## 4. Define / reuse `ToolsConfig`

You already have something like `ToolsConfig` in orchestration. We want it to be the **single source of truth** for:

* Which tools are enabled / paths.
* Any extra env vars.
* Timeouts.

Example shape (adapt to your real one, don’t blindly overwrite):

```python
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .tool_names import ToolName


@dataclass(frozen=True)
class ToolsConfig:
    pyright_path: Path | None = None
    pyrefly_path: Path | None = None
    ruff_path: Path | None = None
    coverage_path: Path | None = None
    pytest_path: Path | None = None
    scip_path: Path | None = None
    default_timeout_s: float = 300.0

    def resolve_path(self, tool: ToolName) -> str:
        # If explicit path is configured, use that; otherwise fall back to tool.value.
        if tool is ToolName.PYRIGHT and self.pyright_path is not None:
            return str(self.pyright_path)
        ...
        return tool.value  # rely on PATH

    def build_env(
        self,
        tool: ToolName,
        *,
        base_env: Mapping[str, str] | None,
    ) -> Mapping[str, str]:
        # Typically: merge base_env with PATH modifications, etc.
        # For now, you can just pass base_env through unchanged.
        return dict(base_env or {})
```

The important thing is just that `ToolService` can get:

* The executable path
* A timeout (either per tool or default)

If you have a more detailed `ToolsConfig` already, just make sure it exposes equivalent `resolve_path` / `build_env` helpers.

---

## 5. Implement `ToolService`

Create a new module, e.g.:

* `src/codeintel/ingestion/tool_service.py`

### 5.1 Core class

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from codeintel.ingestion.tool_runner import ToolRunner, ToolRunResult, ToolName, ToolExecutionError
from codeintel.ingestion.tools_config import ToolsConfig  # or wherever


@dataclass
class ToolService:
    """
    High-level facade for external tools used in ingestion.

    All CLI details, env quirks, and output parsing live here.
    """
    runner: ToolRunner
    tools_config: ToolsConfig

    # --- Typing tools ---

    async def run_pyright(self, repo_root: Path) -> dict[str, int]:
        ...

    async def run_pyrefly(self, repo_root: Path) -> dict[str, int]:
        ...

    async def run_ruff(self, repo_root: Path) -> dict[str, int]:
        ...

    # --- Coverage ---

    async def run_coverage_json(self, repo_root: Path) -> list[CoverageLineRow]:
        ...

    # --- Pytest ---

    async def run_pytest_report(
        self,
        repo_root: Path,
        *,
        json_report_path: Path,
    ) -> bool:
        """Run pytest to generate JSON report if it does not exist. Return True if run succeeded."""

    # --- SCIP ---

    async def run_scip_full(
        self,
        repo_root: Path,
        *,
        output_json: Path,
    ) -> None:
        ...

    async def run_scip_shard(
        self,
        repo_root: Path,
        *,
        rel_paths: list[str],
        output_json: Path,
    ) -> None:
        ...
```

We’ll define the types like `CoverageLineRow` below; they should go into a shared place (probably `coverage_ingest`).

### 5.2 Typing tools methods

**Goal:** give `typing_ingest` already-normalized “file → error count” maps.

Assume your CLIs look like (roughly):

* `pyright --outputjson` or similar
* `pyrefly --output-json`
* `ruff check --format=json`

Implementation sketch:

```python
import json
from typing import Any


def _parse_error_map_from_json_output(stdout: str) -> dict[str, int]:
    data = json.loads(stdout)
    # You can tailor this to the exact tool’s JSON schema.
    # But we want a uniform: { rel_path: error_count }
    error_map: dict[str, int] = {}
    ...
    return error_map


class ToolService:
    ...

    async def run_pyright(self, repo_root: Path) -> dict[str, int]:
        result = await self.runner.run(
            ToolName.PYRIGHT,
            args=["--outputjson"],
            cwd=repo_root,
        )

        # pyright uses non-zero exit if errors found; treat as "success with diagnostics"
        if result.returncode not in (0, 1):
            raise ToolExecutionError(result)

        return _parse_error_map_from_json_output(result.stdout)

    async def run_pyrefly(self, repo_root: Path) -> dict[str, int]:
        result = await self.runner.run(
            ToolName.PYREFLY,
            args=["--format=json"],
            cwd=repo_root,
        )
        if result.returncode != 0:
            raise ToolExecutionError(result)
        return _parse_error_map_from_json_output(result.stdout)

    async def run_ruff(self, repo_root: Path) -> dict[str, int]:
        result = await self.runner.run(
            ToolName.RUFF,
            args=["check", "--format=json"],
            cwd=repo_root,
        )
        # ruff returns 1 when there are lint violations
        if result.returncode not in (0, 1):
            raise ToolExecutionError(result)
        return _parse_error_map_from_json_output(result.stdout)
```

You can adjust parsing to match each tool’s exact JSON schema. The important bit: **call sites never see JSON**; they get maps.

### 5.3 Coverage

Define a small row type for coverage ingestion:

```python
# coverage_ingest.py or a shared types module
from typing import TypedDict

class CoverageLineRow(TypedDict):
    rel_path: str
    lineno: int
    covered: bool
```

In `ToolService`:

```python
from .coverage_parsing import parse_coverage_json  # new helper

class ToolService:
    ...

    async def run_coverage_json(self, repo_root: Path) -> list[CoverageLineRow]:
        result = await self.runner.run(
            ToolName.COVERAGE,
            args=["json", "-o", "coverage.json"],
            cwd=repo_root,
        )
        if result.returncode != 0:
            # Optionally: fallback to coverage.py API here
            raise ToolExecutionError(result)

        json_path = repo_root / "coverage.json"
        return parse_coverage_json(json_path)
```

`parse_coverage_json` can live in `coverage_ingest.py` or a sibling and handle the specifics of the coverage JSON schema, returning a list of `CoverageLineRow`.

### 5.4 Pytest JSON report

For tests ingestion you already do something like: “if the JSON report is missing, run pytest with the JSON-report plugin”.

Move that into `ToolService`:

```python
class ToolService:
    ...

    async def run_pytest_report(
        self,
        repo_root: Path,
        *,
        json_report_path: Path,
    ) -> bool:
        if json_report_path.exists():
            return False  # nothing to do

        result = await self.runner.run(
            ToolName.PYTEST,
            args=[
                "-q",
                f"--json-report-file={json_report_path}",
                "--maxfail=1",
            ],
            cwd=repo_root,
        )

        if result.returncode != 0:
            # Up to you: raise, or log and return False.
            raise ToolExecutionError(result)

        return True
```

### 5.5 SCIP

The SCIP indexing CLI is something like:

* Full index: `scip-python index . --project-name=... --output-format=json`
* Per-module: `scip-python index . --target-only=src/my_module.py --output-format=json`

We want `ToolService` to own that:

```python
class ToolService:
    ...

    async def run_scip_full(
        self,
        repo_root: Path,
        *,
        output_json: Path,
    ) -> None:
        args = [
            "index",
            ".",
            f"--project-name={repo_root.name}",
            "--output-format=json",
            f"--output={output_json}",
        ]
        result = await self.runner.run(
            ToolName.SCIP,
            args=args,
            cwd=repo_root,
        )
        if result.returncode != 0:
            raise ToolExecutionError(result)

    async def run_scip_shard(
        self,
        repo_root: Path,
        *,
        rel_paths: list[str],
        output_json: Path,
    ) -> None:
        # one shard may cover multiple target-only files
        args = [
            "index",
            ".",
            f"--project-name={repo_root.name}",
            "--output-format=json",
            f"--output={output_json}",
        ]
        for rel in rel_paths:
            args.append(f"--target-only={rel}")
        result = await self.runner.run(
            ToolName.SCIP,
            args=args,
            cwd=repo_root,
        )
        if result.returncode != 0:
            raise ToolExecutionError(result)
```

Later, you can add concurrency/fan-out logic in `scip_ingest` around this, but **all CLI details stay here**.

---

## 6. Wire `ToolService` into orchestration / contexts

### 6.1 Pipeline / ingestion context

Wherever you currently have a `ToolsConfig` and/or `ToolRunner` in context (likely in `PipelineContext`), add `tool_service: ToolService`.

Example:

```python
# orchestration/context.py or steps.py

from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.ingestion.tools_config import ToolsConfig

@dataclass
class PipelineContext:
    gateway: StorageGateway
    snapshot: SnapshotConfig
    tools_config: ToolsConfig
    tool_runner: ToolRunner
    tool_service: ToolService
    # ...
```

In your pipeline initialization (e.g. Prefect flow):

```python
tools_config = build_tools_config_from_env(...)
tool_runner = ToolRunner(tools_config, base_env=os.environ)
tool_service = ToolService(runner=tool_runner, tools_config=tools_config)

ctx = PipelineContext(
    gateway=gateway,
    snapshot=snapshot,
    tools_config=tools_config,
    tool_runner=tool_runner,
    tool_service=tool_service,
    ...
)
```

Steps now access tools through `ctx.tool_service`.

---

## 7. Refactor `typing_ingest` to use `ToolService`

Right now `typing_ingest`:

* Has its own async `_run_command` / `_run_pyright` / `_run_pyrefly` / `_run_ruff`.
* Parses JSON itself.
* Merges results with AST-level typing signals.

We want to strip out the subprocess bits.

Conceptual before → after:

**Before** (rough):

```python
async def ingest_typing_signals(...):
    pyright_errors = await _run_pyright(...)
    pyrefly_errors = await _run_pyrefly(...)
    ruff_errors = await _run_ruff(...)
    # combine & write to DB
```

**After:**

```python
from codeintel.ingestion.tool_service import ToolService

async def ingest_typing_signals(
    gateway: StorageGateway,
    tool_service: ToolService,
    ctx: IngestionContext,
) -> None:
    repo_root = ctx.repo_root

    pyright_map = await tool_service.run_pyright(repo_root)
    pyrefly_map = await tool_service.run_pyrefly(repo_root)
    ruff_map = await tool_service.run_ruff(repo_root)

    # everything below this line should be pure data merging + DuckDB writes
    merged = merge_typing_maps(pyright_map, pyrefly_map, ruff_map, ctx.modules)
    write_typing_signals(gateway.con, merged)
```

**Implementation tasks:**

1. Delete or deprecate `_run_command`, `_run_pyright`, `_run_pyrefly`, `_run_ruff` from `typing_ingest.py`.
2. Replace all call sites with `tool_service.run_*`.
3. Make `TypingStep` in `orchestration.steps` pass `ctx.tool_service` into `ingest_typing_signals`.

You’ll keep all the AST and DB logic intact; only the subprocess layer moves.

---

## 8. Refactor `coverage_ingest` to use `ToolService`

Currently `coverage_ingest` probably:

* Checks for `coverage.json`.
* If missing, uses `ToolRunner` or manually calls `coverage`.
* Parses JSON, builds `coverage_lines.*` rows, writes to DB.

**After refactor:**

```python
from codeintel.ingestion.tool_service import ToolService

def ingest_coverage(
    gateway: StorageGateway,
    tool_service: ToolService,
    ctx: IngestionContext,
) -> None:
    repo_root = ctx.repo_root

    # 1. Run coverage CLI and parse JSON
    coverage_rows = asyncio.run(tool_service.run_coverage_json(repo_root))

    # 2. Insert into DuckDB
    write_coverage_rows(gateway.con, coverage_rows)
```

Or, if `ingest_coverage` is already async, you can keep it async and await the call.

Tasks:

1. Move any parsing logic into a separate helper (`parse_coverage_json`) used by `ToolService`.
2. Remove direct calls to `ToolRunner` or `subprocess` from `coverage_ingest.py`.
3. Update orchestration step to supply `ctx.tool_service`.

---

## 9. Refactor `tests_ingest` to use `ToolService`

Currently, `tests_ingest`:

* Finds / expects a pytest JSON report.
* If missing, calls pytest with the JSON-report plugin.
* Parses the JSON into `test_profile.*` rows.

**After:**

```python
from codeintel.ingestion.tool_service import ToolService

def ingest_tests(
    gateway: StorageGateway,
    tool_service: ToolService,
    ctx: IngestionContext,
) -> None:
    repo_root = ctx.repo_root
    json_report_path = ctx.paths.pytest_report_path

    # 1. Ensure JSON report exists
    asyncio.run(tool_service.run_pytest_report(repo_root, json_report_path=json_report_path))

    # 2. Parse JSON
    report = load_pytest_report(json_report_path)
    rows = convert_pytest_report_to_rows(report, ctx.modules)

    # 3. Write to DB
    write_test_rows(gateway.con, rows)
```

Tasks:

1. Move the “run pytest with JSON plugin” logic into `ToolService.run_pytest_report`.
2. Keep JSON parsing in `tests_ingest`, or move to a small `pytest_parsing` helper used there.
3. Update orchestration step to pass `ctx.tool_service`.

---

## 10. Refactor `scip_ingest` to use `ToolService`

Currently `scip_ingest`:

* Builds `scip-python` command lines.
* Has its own async `_run_command`, `_run_scip_full`, `_run_scip_shard`.
* Handles exit codes, possibly per-module shards.

After the refactor:

* All command-line specifics live in `ToolService.run_scip_full` / `run_scip_shard`.
* `scip_ingest` focuses on:

  * deciding incremental vs full (now via `ChangeTracker`)
  * splitting changed modules into shards
  * reading JSON output
  * writing data into `core.scip_*` tables.

**After:**

```python
from codeintel.ingestion.change_tracker import ChangeTracker
from codeintel.ingestion.tool_service import ToolService

async def ingest_scip(
    gateway: StorageGateway,
    tool_service: ToolService,
    tracker: ChangeTracker,
    ctx: IngestionContext,
) -> None:
    repo_root = ctx.repo_root
    tmp_dir = ctx.paths.scip_tmp_dir

    # Decide incremental vs full using ChangeTracker (as in your new infra)
    # Suppose you’ve computed: shard_paths: list[list[str]]
    # 1) For each shard, call tool_service.run_scip_shard
    for i, shard in enumerate(shard_paths):
        shard_output = tmp_dir / f"scip_shard_{i}.json"
        await tool_service.run_scip_shard(
            repo_root,
            rel_paths=shard,
            output_json=shard_output,
        )
        # read JSON, convert to rows, and insert into DuckDB
        rows = parse_scip_json(shard_output)
        write_scip_rows(gateway.con, rows)

    # Or for full rebuild:
    # output_json = tmp_dir / "scip_full.json"
    # await tool_service.run_scip_full(repo_root, output_json=output_json)
    # ...
```

Tasks:

1. Delete `_run_command`, `_run_scip_full`, `_run_scip_shard` from `scip_ingest.py`.
2. Use `ChangeTracker` + `IncrementalIngestOps` (from your new change-tracker framework) to decide shards and deletions.
3. Let `ToolService` own the pure CLI/execution part.

---

## 11. Testing and guardrails

### 11.1 Unit tests for `ToolService`

Create tests like:

* `tests/ingestion/test_tool_service_typing.py`
* `tests/ingestion/test_tool_service_scip.py`

Use monkeypatching/mocking of `ToolRunner.run` to return canned `ToolRunResult` objects, then verify:

* `run_pyright` parses stdout JSON into the expected `{path: error_count}` map.
* `run_ruff` handles returncode 1 as “OK with diagnostics”.
* `run_coverage_json` calls `ToolRunner.run` with the right args and then calls parser.
* `run_scip_shard` constructs args matching your desired CLI.

### 11.2 Integration tests for ingestion modules

For each of:

* `typing_ingest`
* `coverage_ingest`
* `tests_ingest`
* `scip_ingest`

Add tests that:

* Use a fake `ToolService` (or a real one with a fake `ToolRunner`) that returns known outputs.
* Assert that ingestion writes the correct rows into a temporary DuckDB.

This proves that the ingestion logic is now independent of CLI details.

### 11.3 Architecture test to forbid stray `_run_command` calls

Add a simple test that scans your src tree for:

* `subprocess.run(`, `create_subprocess_exec(`, `_run_command(`
  outside of `tool_runner.py` and `tool_service.py`.

Example:

```python
def test_no_direct_subprocess_calls_outside_tooling() -> None:
    for path, text in _iter_python_source_files(Path("src/codeintel")):
        if "tool_runner.py" in str(path) or "tool_service.py" in str(path):
            continue
        if "subprocess.run(" in text or "create_subprocess_exec(" in text or "_run_command(" in text:
            pytest.fail(f"Direct subprocess usage not allowed in {path}")
```

This keeps the “all tools go through ToolService” invariant true over time.

---

If you’d like, we can next zoom in on one specific module (for example `typing_ingest.py`) and produce a more concrete “before/after” patch sketch that an AI agent can apply directly, with function signature changes and internal call replacements spelled out.
