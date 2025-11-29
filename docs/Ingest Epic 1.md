Here’s a concrete, patch-oriented plan to implement **Epic 1: ingestion step registry**, tailored to your current CodeIntel layout.

I’ll walk through:

1. New `codeintel.ingestion.steps` module (step interface, concrete steps, registry).
2. Wiring `runner.py` to route all `run_*` functions through the registry.
3. Exporting the new APIs from `ingestion/__init__.py`.
4. Tests: new `tests/ingestion/test_step_registry.py`.

I’ll keep everything compatible with existing callers (pipeline, serving, CLI).

---

## 1. New module: `codeintel/ingestion/steps.py`

**Goal:** define a *dataset-centric* ingestion step interface, concrete step implementations, and a registry that knows dependencies and can produce a topological order.

Create a new file:

`src/codeintel/ingestion/steps.py`:

```python
"""Ingestion step interface and registry.

Each step:

- Has a stable ``name`` used across the system.
- Declares which DuckDB tables it populates (``produces_tables``).
- Declares which other ingestion steps it depends on (``requires``).
- Implements a ``run(ctx: IngestionContext)`` method.

This allows agents, CLI, and pipeline orchestration to:

- Enumerate ingestion datasets.
- Inspect dependencies and affected tables.
- Run subsets of ingestion in a safe, dependency-aware order.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.config import DocstringStepConfig, ScipIngestStepConfig
from codeintel.config.builder import (
    ConfigIngestStepConfig,
    CoverageIngestStepConfig,
    RepoScanStepConfig,
    TestsIngestStepConfig,
    TypingIngestStepConfig,
)
from codeintel.config.models import ToolsConfig
from codeintel.config.primitives import BuildPaths
from codeintel.ingestion import (
    config_ingest,
    coverage_ingest,
    cst_extract,
    docstrings_ingest,
    py_ast_extract,
    repo_scan,
    scip_ingest,
    tests_ingest,
    typing_ingest,
)
from codeintel.ingestion import change_tracker as change_tracker_module
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    # Imported only for type checking to avoid runtime import cycles.
    from codeintel.ingestion.runner import IngestionContext

log = __import__("logging").getLogger(__name__)


@runtime_checkable
class IngestStep(Protocol):
    """Protocol implemented by ingestion steps."""

    name: str
    description: str
    produces_tables: Sequence[str]
    requires: Sequence[str]

    def run(self, ctx: IngestionContext) -> object | None:  # pragma: no cover - protocol
        """Execute the ingestion step against the provided context."""
        raise NotImplementedError


@dataclass(frozen=True)
class IngestStepMetadata:
    """
    Machine-readable metadata for an ingestion step.

    Parameters
    ----------
    name
        Unique step identifier (e.g. ``"repo_scan"``).
    description
        Human-readable description of what the step does.
    produces_tables
        DuckDB tables populated by this step.
    requires
        Names of steps this step depends on.
    """

    name: str
    description: str
    produces_tables: tuple[str, ...]
    requires: tuple[str, ...]


# ---------------------------------------------------------------------------
# Concrete step implementations
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RepoScanStep:
    """Scan repository tree into core tables and change-tracker state."""

    name: str = "repo_scan"
    description: str = "Scan repository modules and build change-tracker state."
    produces_tables: tuple[str, ...] = (
        "core.file_state",
        "core.modules",
        "core.repo_map",
        "analytics.tags_index",
    )
    requires: tuple[str, ...] = ()

    def run(self, ctx: IngestionContext) -> change_tracker_module.ChangeTracker:
        cfg = RepoScanStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            tool_runner=ctx.tool_runner,
        )
        tracker = repo_scan.ingest_repo(
            ctx.gateway,
            cfg=cfg,
            code_profile=ctx.code_profile,
        )
        # Subsequent steps can incrementally ingest from this change tracker.
        ctx.change_tracker = tracker
        return tracker


@dataclass(frozen=True)
class ScipIngestStep:
    """Run scip-python and register SCIP artifacts/view."""

    name: str = "scip_ingest"
    description: str = "Run scip-python and persist symbols and GOID crosswalk."
    produces_tables: tuple[str, ...] = (
        "index.scip",
        "core.scip_symbols",
        "core.goid_crosswalk",
    )
    requires: tuple[str, ...] = ("repo_scan",)

    def run(self, ctx: IngestionContext) -> scip_ingest.ScipIngestResult:
        binaries = ctx.active_tools  # ToolsConfig
        # Map into ScipIngestStepConfig's ToolBinaries
        cfg = ScipIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            binaries=scip_ingest.ToolBinaries(
                scip_python_bin=binaries.scip_python_bin,
                scip_bin=binaries.scip_bin,
                pyright_bin=binaries.pyright_bin,
                pyrefly_bin=binaries.pyrefly_bin,
                ruff_bin=binaries.ruff_bin,
                coverage_bin=binaries.coverage_bin,
                pytest_bin=binaries.pytest_bin,
                git_bin=binaries.git_bin,
                default_timeout_s=binaries.default_timeout_s,
            ),
            scip_runner=ctx.scip_runner,
            artifact_writer=ctx.artifact_writer,
        )
        tracker = ctx.change_tracker
        if tracker is None:
            # Mirror runner._require_change_tracker semantics.
            message = "change_tracker is not set; run repo_scan before incremental ingest"
            raise RuntimeError(message)

        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)
        return scip_ingest.ingest_scip(
            ctx.gateway,
            cfg=cfg,
            tracker=tracker,
            tool_service=service,
        )


@dataclass(frozen=True)
class CstExtractStep:
    """Parse CST and persist rows."""

    name: str = "cst_extract"
    description: str = "Parse CST via LibCST and write rows into core.cst_nodes."
    produces_tables: tuple[str, ...] = ("core.cst_nodes",)
    requires: tuple[str, ...] = ("repo_scan",)

    def run(self, ctx: IngestionContext) -> None:
        tracker = ctx.change_tracker
        if tracker is None:
            message = "change_tracker is not set; run repo_scan before incremental ingest"
            raise RuntimeError(message)
        cst_extract.ingest_cst(
            tracker,
            executor_kind=os.getenv("CODEINTEL_CST_EXECUTOR", "process"),  # type: ignore[name-defined]
        )


@dataclass(frozen=True)
class AstExtractStep:
    """Parse stdlib AST and persist rows/metrics."""

    name: str = "ast_extract"
    description: str = "Parse Python AST and persist rows + metrics into core.ast_* tables."
    produces_tables: tuple[str, ...] = ("core.ast_nodes", "core.ast_metrics")
    requires: tuple[str, ...] = ("repo_scan",)

    def run(self, ctx: IngestionContext) -> None:
        tracker = ctx.change_tracker
        if tracker is None:
            message = "change_tracker is not set; run repo_scan before incremental ingest"
            raise RuntimeError(message)
        py_ast_extract.ingest_python_ast(tracker)


@dataclass(frozen=True)
class TypingIngestStep:
    """Compute typedness and static diagnostics."""

    name: str = "typing_ingest"
    description: str = "Populate analytics.typedness and analytics.static_diagnostics."
    produces_tables: tuple[str, ...] = ("analytics.typedness", "analytics.static_diagnostics")
    requires: tuple[str, ...] = ()

    def run(self, ctx: IngestionContext) -> None:
        cfg = TypingIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            tool_runner=ctx.tool_runner,
        )
        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)
        typing_ingest.ingest_typing_signals(
            gateway=ctx.gateway,
            cfg=cfg,
            code_profile=ctx.code_profile,
            tools=ctx.active_tools,
            tool_service=service,
        )


@dataclass(frozen=True)
class CoverageIngestStep:
    """Load coverage.py data into analytics.coverage_lines."""

    name: str = "coverage_ingest"
    description: str = "Load coverage.py data into analytics.coverage_lines."
    produces_tables: tuple[str, ...] = ("analytics.coverage_lines",)
    requires: tuple[str, ...] = ()

    def run(self, ctx: IngestionContext) -> None:
        cfg = CoverageIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            coverage_file=ctx.active_tools.coverage_file,  # type: ignore[arg-type]
            tool_runner=ctx.tool_runner,
        )
        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)
        coverage_ingest.ingest_coverage_lines(
            gateway=ctx.gateway,
            cfg=cfg,
            tools=ctx.active_tools,
            tool_service=service,
            json_output_path=ctx.paths.coverage_json,
        )


@dataclass(frozen=True)
class TestsIngestStep:
    """Load pytest results into analytics.test_catalog."""

    name: str = "tests_ingest"
    description: str = "Ingest pytest JSON report into analytics.test_catalog."
    produces_tables: tuple[str, ...] = ("analytics.test_catalog",)
    requires: tuple[str, ...] = ()

    def run(self, ctx: IngestionContext) -> None:
        cfg = TestsIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            pytest_report_path=ctx.paths.pytest_report,
        )
        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)
        tests_ingest.ingest_tests(
            gateway=ctx.gateway,
            cfg=cfg,
            report_path=ctx.paths.pytest_report,
            tools=ctx.active_tools,
            tool_service=service,
        )


@dataclass(frozen=True)
class DocstringsIngestStep:
    """Extract and persist docstrings."""

    name: str = "docstrings_ingest"
    description: str = "Extract docstrings and persist structured rows into core.docstrings."
    produces_tables: tuple[str, ...] = ("core.docstrings",)
    requires: tuple[str, ...] = ("repo_scan",)

    def run(self, ctx: IngestionContext) -> None:
        cfg = DocstringStepConfig(snapshot=ctx.snapshot)
        docstrings_ingest.ingest_docstrings(
            ctx.gateway,
            cfg,
            code_profile=ctx.code_profile,
        )


@dataclass(frozen=True)
class ConfigIngestStep:
    """Flatten config files into analytics.config_values."""

    name: str = "config_ingest"
    description: str = "Flatten config files into analytics.config_values."
    produces_tables: tuple[str, ...] = ("analytics.config_values",)
    requires: tuple[str, ...] = ()

    def run(self, ctx: IngestionContext) -> None:
        cfg = ConfigIngestStepConfig(snapshot=ctx.snapshot)
        config_ingest.ingest_config_values(
            ctx.gateway,
            cfg=cfg,
            config_profile=ctx.config_profile,
        )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestStepRegistry:
    """
    Registry of ingestion steps with dependency-aware ordering helpers.

    Parameters
    ----------
    _steps
        Mapping of step names to step instances.
    _sequence
        Ordered sequence of step names (defines default ordering).
    """

    _steps: Mapping[str, IngestStep]
    _sequence: tuple[str, ...] = field(default_factory=tuple)

    def __iter__(self) -> Iterator[IngestStep]:
        """Iterate over steps in default sequence order."""
        for name in self._sequence:
            yield self._steps[name]

    def __contains__(self, name: str) -> bool:
        """Return True if a step with this name is registered."""
        return name in self._steps

    def __len__(self) -> int:
        """Return the number of registered steps."""
        return len(self._steps)

    def step_names(self) -> tuple[str, ...]:
        """Return the tuple of step names in default sequence order."""
        return self._sequence

    def get(self, name: str) -> IngestStep:
        """
        Look up a step by name.

        Raises
        ------
        KeyError
            If the step name is not registered.
        """
        try:
            return self._steps[name]
        except KeyError as exc:  # pragma: no cover - trivial
            message = f"Unknown ingestion step: {name}"
            raise KeyError(message) from exc

    def metadata_for(self, name: str) -> IngestStepMetadata:
        """Return metadata for a single step."""
        step = self.get(name)
        return IngestStepMetadata(
            name=step.name,
            description=step.description,
            produces_tables=tuple(step.produces_tables),
            requires=tuple(step.requires),
        )

    def all_metadata(self) -> list[IngestStepMetadata]:
        """Return metadata for all steps in default order."""
        return [self.metadata_for(name) for name in self._sequence]

    # ---- dependency helpers -------------------------------------------------

    def dependency_graph(self) -> dict[str, tuple[str, ...]]:
        """Return mapping of step name -> direct dependencies."""
        return {name: tuple(self._steps[name].requires) for name in self._steps}

    def _expand_recursive(self, name: str, expanded: set[str]) -> None:
        """Recursively expand dependencies for a name."""
        if name in expanded:
            return
        step = self.get(name)
        for dep in step.requires:
            self._expand_recursive(dep, expanded)
        expanded.add(name)

    def expand_with_deps(self, names: Sequence[str]) -> set[str]:
        """
        Expand a set of step names to include all transitive dependencies.
        """
        expanded: set[str] = set()
        for name in names:
            self._expand_recursive(name, expanded)
        return expanded

    def topological_order(self, names: Sequence[str]) -> list[str]:
        """
        Return a topological ordering of the requested steps.

        Raises
        ------
        RuntimeError
            If a dependency cycle is detected.
        KeyError
            If any step name is not registered.
        """
        # Validate all names exist
        for name in names:
            if name not in self._steps:
                message = f"Unknown ingestion step: {name}"
                raise KeyError(message)

        deps = {name: set(self._steps[name].requires) & set(names) for name in names}
        remaining = set(names)
        ordered: list[str] = []
        no_deps = [name for name in names if not deps[name]]

        while no_deps:
            name = no_deps.pop()
            ordered.append(name)
            remaining.discard(name)
            for other in list(remaining):
                deps[other].discard(name)
                if not deps[other]:
                    no_deps.append(other)

        if remaining:
            message = f"Circular dependencies detected in ingestion steps: {sorted(remaining)}"
            raise RuntimeError(message)
        return ordered


def _build_default_registry() -> IngestStepRegistry:
    """Construct the default registry with all built-in ingestion steps."""
    steps: dict[str, IngestStep] = {
        "repo_scan": RepoScanStep(),
        "scip_ingest": ScipIngestStep(),
        "cst_extract": CstExtractStep(),
        "ast_extract": AstExtractStep(),
        "typing_ingest": TypingIngestStep(),
        "coverage_ingest": CoverageIngestStep(),
        "tests_ingest": TestsIngestStep(),
        "docstrings_ingest": DocstringsIngestStep(),
        "config_ingest": ConfigIngestStep(),
    }
    # Default sequence is simply the insertion order.
    sequence = tuple(steps.keys())
    return IngestStepRegistry(_steps=steps, _sequence=sequence)


DEFAULT_REGISTRY: IngestStepRegistry = _build_default_registry()


def get_ingest_step(name: str) -> IngestStep:
    """Convenience accessor for the default registry."""
    return DEFAULT_REGISTRY.get(name)


__all__ = [
    "IngestStep",
    "IngestStepMetadata",
    "IngestStepRegistry",
    "DEFAULT_REGISTRY",
    "get_ingest_step",
]
```

Notes / tweaks you might want to make when actually implementing:

* I referenced `os.getenv` and `scip_ingest.ToolBinaries` inline; if you prefer, you can keep the ToolBinaries import in `runner.py` and pass it through via `ctx` instead.
* `produces_tables` is honest about **DuckDB** writes; `index.scip` is included because it’s a logical dataset even if not in DuckDB.

---

## 2. Wire `ingestion/runner.py` through the registry

Now use that registry as the canonical mechanism for running steps, while keeping your existing `run_*` API.

### 2.1. Imports

At the top of `ingestion/runner.py`, extend imports:

```python
from collections.abc import Callable, Sequence  # add Sequence

...

from codeintel.ingestion.steps import DEFAULT_REGISTRY, IngestStep, get_ingest_step
```

(Leave existing imports of `repo_scan`, `scip_ingest`, etc. for now; you may later slim them down once you’re confident everything routes via steps.)

### 2.2. Add a private helper `_run_ingest_step`

Just below `_log_step_done` (or near the top of the run_* functions), add:

```python
def _run_ingest_step(ctx: IngestionContext, name: str) -> object | None:
    """
    Run a single ingestion step by name with logging.

    Parameters
    ----------
    ctx
        Shared ingestion context.
    name
        Name of the ingestion step to execute.

    Returns
    -------
    object | None
        Any value returned by the underlying step (usually None, except
        for 'repo_scan' and 'scip_ingest').
    """
    start = _log_step_start(name, ctx)
    step = DEFAULT_REGISTRY.get(name)
    result = step.run(ctx)
    _log_step_done(name, start, ctx)
    return result
```

### 2.3. Implement `run_ingest_steps` and `list_ingest_steps`

Still in `runner.py`:

```python
def list_ingest_steps() -> list[dict[str, object]]:
    """
    Return machine-readable metadata for all ingestion steps.

    Returns
    -------
    list[dict[str, object]]
        Dictionaries with name, description, produces_tables, and requires.
    """
    return [
        {
            "name": meta.name,
            "description": meta.description,
            "produces_tables": meta.produces_tables,
            "requires": meta.requires,
        }
        for meta in DEFAULT_REGISTRY.all_metadata()
    ]


def run_ingest_steps(
    ctx: IngestionContext,
    selected_steps: Sequence[str] | None = None,
) -> None:
    """
    Run ingestion steps in dependency order.

    Parameters
    ----------
    ctx
        Shared ingestion context.
    selected_steps
        Optional subset of step names to run. If None, all steps
        are executed in the default registry order, respecting
        declared dependencies.
    """
    if selected_steps is None:
        names = list(DEFAULT_REGISTRY.step_names())
    else:
        expanded = DEFAULT_REGISTRY.expand_with_deps(selected_steps)
        names = DEFAULT_REGISTRY.topological_order(sorted(expanded))
    for name in names:
        _run_ingest_step(ctx, name)
```

### 2.4. Rewrite existing `run_*` entrypoints to delegate

Now replace the current bodies of the existing `run_*` functions with thin wrappers that call `_run_ingest_step`.

#### `run_repo_scan`

```python
def run_repo_scan(ctx: IngestionContext) -> change_tracker_module.ChangeTracker:
    """Ingest repository structure and modules using the provided storage gateway."""
    _run_ingest_step(ctx, "repo_scan")
    tracker = ctx.change_tracker
    if tracker is None:
        message = "repo_scan step did not populate change_tracker"
        raise RuntimeError(message)
    return tracker
```

#### `run_scip_ingest`

```python
def run_scip_ingest(ctx: IngestionContext) -> scip_ingest.ScipIngestResult:
    """
    Execute scip-python indexing and register outputs.

    Returns
    -------
    scip_ingest.ScipIngestResult
        Status and artifact paths for the SCIP run.
    """
    result = _run_ingest_step(ctx, "scip_ingest")
    assert isinstance(result, scip_ingest.ScipIngestResult)
    return result
```

#### Pure side-effect steps

For the remaining steps (`run_cst_extract`, `run_ast_extract`, `run_typing_ingest`, `run_coverage_ingest`, `run_tests_ingest`, `run_docstrings_ingest`, `run_config_ingest`), each becomes:

```python
def run_cst_extract(ctx: IngestionContext) -> None:
    """Extract LibCST nodes for the repository using the gateway connection."""
    _run_ingest_step(ctx, "cst_extract")


def run_ast_extract(ctx: IngestionContext) -> None:
    """Extract stdlib AST nodes and metrics using the gateway connection."""
    _run_ingest_step(ctx, "ast_extract")


def run_typing_ingest(ctx: IngestionContext) -> None:
    """Collect static typing diagnostics and typedness via the gateway connection."""
    _run_ingest_step(ctx, "typing_ingest")


def run_coverage_ingest(ctx: IngestionContext) -> None:
    """Load coverage lines via the gateway connection."""
    _run_ingest_step(ctx, "coverage_ingest")


def run_tests_ingest(ctx: IngestionContext) -> None:
    """Ingest pytest catalog rows via the gateway connection."""
    _run_ingest_step(ctx, "tests_ingest")


def run_docstrings_ingest(ctx: IngestionContext) -> None:
    """Extract docstrings and persist structured rows via the gateway connection."""
    _run_ingest_step(ctx, "docstrings_ingest")


def run_config_ingest(ctx: IngestionContext) -> None:
    """Flatten configuration files into analytics.config_values via the gateway connection."""
    _run_ingest_step(ctx, "config_ingest")
```

You can delete the old bodies (cfg construction, ToolRunner instantiation, etc.) because those are now in the step implementations in `steps.py`.

---

## 3. Export registry APIs from `ingestion/__init__.py`

Update `codeintel/ingestion/__init__.py` to expose the new functions so agents / MCP can easily discover them.

### 3.1. Extend imports

Change the runner import to include the new functions:

```python
from codeintel.ingestion.runner import (
    IngestionContext,
    list_ingest_steps,
    run_ingest_steps,
    run_ast_extract,
    run_config_ingest,
    run_coverage_ingest,
    run_cst_extract,
    run_docstrings_ingest,
    run_repo_scan,
    run_scip_ingest,
    run_tests_ingest,
    run_typing_ingest,
)
```

### 3.2. Extend `__all__`

At the bottom, extend the exported names:

```python
__all__ = [
    "IngestionContext",
    "ensure_repo_root",
    "normalize_rel_path",
    "relpath_to_module",
    "repo_relpath",
    "list_ingest_steps",
    "run_ingest_steps",
    "run_ast_extract",
    "run_config_ingest",
    "run_coverage_ingest",
    "run_cst_extract",
    "run_docstrings_ingest",
    "run_repo_scan",
    "run_scip_ingest",
    "run_tests_ingest",
    "run_typing_ingest",
]
```

No existing callers break: `from codeintel.ingestion import run_repo_scan` still works; now they just happen to route via the registry.

---

## 4. Tests: `tests/ingestion/test_step_registry.py`

Add a new test file to validate the registry is wired correctly, without spinning up DuckDB or touching the filesystem.

`tests/ingestion/test_step_registry.py`:

```python
"""Tests for the ingestion step registry wiring and metadata."""

from __future__ import annotations

from codeintel.ingestion.steps import DEFAULT_REGISTRY


def test_registry_includes_all_expected_steps() -> None:
    names = set(DEFAULT_REGISTRY.step_names())
    # Keep in sync with concrete steps in codeintel.ingestion.steps
    expected = {
        "repo_scan",
        "scip_ingest",
        "cst_extract",
        "ast_extract",
        "typing_ingest",
        "coverage_ingest",
        "tests_ingest",
        "docstrings_ingest",
        "config_ingest",
    }
    missing = expected - names
    extra = names - expected
    assert not missing, f"Missing ingestion steps in registry: {sorted(missing)}"
    assert not extra, f"Unexpected ingestion steps in registry: {sorted(extra)}"


def test_metadata_exposes_tables_and_deps() -> None:
    meta_by_name = {m.name: m for m in DEFAULT_REGISTRY.all_metadata()}

    repo_scan = meta_by_name["repo_scan"]
    assert "core.modules" in repo_scan.produces_tables
    assert repo_scan.requires == ()

    scip = meta_by_name["scip_ingest"]
    assert "core.scip_symbols" in scip.produces_tables
    assert "core.goid_crosswalk" in scip.produces_tables
    assert ("repo_scan",) == scip.requires

    docstrings = meta_by_name["docstrings_ingest"]
    assert "core.docstrings" in docstrings.produces_tables
    assert "repo_scan" in docstrings.requires


def test_topological_order_respects_dependencies() -> None:
    names = ["repo_scan", "scip_ingest", "ast_extract", "cst_extract", "docstrings_ingest"]
    order = DEFAULT_REGISTRY.topological_order(names)
    pos = {name: order.index(name) for name in names}

    assert pos["repo_scan"] < pos["scip_ingest"]
    assert pos["repo_scan"] < pos["ast_extract"]
    assert pos["repo_scan"] < pos["cst_extract"]
    assert pos["repo_scan"] < pos["docstrings_ingest"]
```

This gives you:

* A guardrail that registry names can’t silently drift.
* A structural test that dependency relations are correct.
* No DB / IO dependence; tests should be very fast.

Optionally, you can later add a tiny smoke test that creates an `IngestionContext` for a temp repo and runs `run_ingest_steps(ctx, ["repo_scan", "ast_extract"])`, but that crosses over into the integration testing you already do at the pipeline layer.

---

## 5. Rollout / migration notes

* **Backwards compatibility:** all existing entrypoints (`run_repo_scan`, etc.) keep their signatures and logging shape. Pipeline orchestration in `pipeline/orchestration/steps_ingestion.py` continues to call them unchanged.
* **Refactor locality:** the only behavioural changes are:

  * Logging for steps run via `run_ingest_steps` now goes through `_run_ingest_step` (same pattern as old `run_*`).
  * Ingestion logic is now centralized in `steps.py`, which should make Epic 2 (unifying incremental ingestion) significantly easier.
* **Agent friendliness:** agents can now call:

  * `codeintel.ingestion.list_ingest_steps()` to discover datasets.
  * `codeintel.ingestion.run_ingest_steps(ctx, [...])` to run subsets safely.

If you’d like, I can next take just one step (e.g. `scip_ingest` + `ast_extract`) and sketch the *exact* before/after code for those functions so you can patch with zero ambiguity.

# zoom in on one step to be converted #

Perfect, let’s zoom in on **just `scip_ingest` and `ast_extract`** and make this very patchable.

I’ll assume:

* You’re implementing the **step registry** in a new `codeintel/ingestion/steps.py`.
* You’re okay with `runner.py` delegating to the registry via a helper `_run_ingest_step`.

Below I’ll show:

1. The **current** `run_scip_ingest` + **new** version.
2. The **new** `ScipIngestStep` in `steps.py`.
3. The **current** `run_ast_extract` + **new** version.
4. The **new** `AstExtractStep` in `steps.py`.
5. The relevant slice of `DEFAULT_REGISTRY` that wires these two in.

You can literally search for these functions and replace them verbatim.

---

## 1. `run_scip_ingest` in `ingestion/runner.py` — before vs after

### 1.1 Current `run_scip_ingest` (before)

Search in `src/codeintel/ingestion/runner.py` and you should find this:

```python
def run_scip_ingest(ctx: IngestionContext) -> scip_ingest.ScipIngestResult:
    """
    Execute scip-python indexing and register outputs.

    Returns
    -------
    scip_ingest.ScipIngestResult
        Status and artifact paths for the SCIP run.
    """
    start = _log_step_start("scip_ingest", ctx)
    binaries = ToolBinaries(
        scip_python_bin=ctx.active_tools.scip_python_bin,
        scip_bin=ctx.active_tools.scip_bin,
        pyright_bin=ctx.active_tools.pyright_bin,
        pyrefly_bin=ctx.active_tools.pyrefly_bin,
        ruff_bin=ctx.active_tools.ruff_bin,
        coverage_bin=ctx.active_tools.coverage_bin,
        pytest_bin=ctx.active_tools.pytest_bin,
        git_bin=ctx.active_tools.git_bin,
        default_timeout_s=ctx.active_tools.default_timeout_s,
    )
    cfg = ScipIngestStepConfig(
        snapshot=ctx.snapshot,
        paths=ctx.paths,
        binaries=binaries,
        scip_runner=ctx.scip_runner,
        artifact_writer=ctx.artifact_writer,
    )
    tracker = ctx.change_tracker
    if tracker is None:
        tracker = _require_change_tracker(ctx)
    runner = ctx.tool_runner or ToolRunner(
        cache_dir=ctx.paths.tool_cache,
        tools_config=ctx.active_tools,
    )
    service = ctx.tool_service or ToolService(runner, ctx.active_tools)
    result = scip_ingest.ingest_scip(
        ctx.gateway,
        cfg=cfg,
        tracker=tracker,
        tool_service=service,
    )
    _log_step_done("scip_ingest", start, ctx)
    return result
```

### 1.2 New `run_scip_ingest` (after)

Replace that entire function with this **thin delegating wrapper**:

```python
def run_scip_ingest(ctx: IngestionContext) -> scip_ingest.ScipIngestResult:
    """
    Execute scip-python indexing and register outputs.

    This wrapper delegates to the ingestion step registry so that the
    SCIP ingestion logic can live in a pluggable IngestStep implementation.
    """
    result = _run_ingest_step(ctx, "scip_ingest")
    assert isinstance(result, scip_ingest.ScipIngestResult)
    return result
```

This assumes you’ve added `_run_ingest_step` and wired in the registry (I’ll summarize the helper at the end in case you haven’t yet).

---

## 2. New `ScipIngestStep` in `ingestion/steps.py`

Now we move the **real logic** for SCIP ingestion into the step implementation.

In your new `src/codeintel/ingestion/steps.py` (or whatever filename you chose), make sure you have the right imports at the top:

```python
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from codeintel.config import ScipIngestStepConfig, ToolBinaries
from codeintel.ingestion import py_ast_extract, scip_ingest
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService

if TYPE_CHECKING:
    from codeintel.ingestion.runner import IngestionContext
```

Then define the `IngestStep` protocol (if you haven’t already):

```python
@runtime_checkable
class IngestStep(Protocol):
    name: str
    description: str
    produces_tables: Sequence[str]
    requires: Sequence[str]

    def run(self, ctx: IngestionContext) -> object | None: ...
```

And then add the concrete **SCIP step**:

```python
@dataclass(frozen=True)
class ScipIngestStep:
    """Run scip-python and register SCIP artifacts/view."""

    name: str = "scip_ingest"
    description: str = "Run scip-python and persist symbols and GOID crosswalk."
    produces_tables: tuple[str, ...] = (
        "index.scip",
        "core.scip_symbols",
        "core.goid_crosswalk",
    )
    requires: tuple[str, ...] = ("repo_scan",)

    def run(self, ctx: IngestionContext) -> scip_ingest.ScipIngestResult:
        # Mirror the existing ToolBinaries + ScipIngestStepConfig wiring
        binaries = ToolBinaries(
            scip_python_bin=ctx.active_tools.scip_python_bin,
            scip_bin=ctx.active_tools.scip_bin,
            pyright_bin=ctx.active_tools.pyright_bin,
            pyrefly_bin=ctx.active_tools.pyrefly_bin,
            ruff_bin=ctx.active_tools.ruff_bin,
            coverage_bin=ctx.active_tools.coverage_bin,
            pytest_bin=ctx.active_tools.pytest_bin,
            git_bin=ctx.active_tools.git_bin,
            default_timeout_s=ctx.active_tools.default_timeout_s,
        )
        cfg = ScipIngestStepConfig(
            snapshot=ctx.snapshot,
            paths=ctx.paths,
            binaries=binaries,
            scip_runner=ctx.scip_runner,
            artifact_writer=ctx.artifact_writer,
        )

        tracker = ctx.change_tracker
        if tracker is None:
            # Keep the same semantics and error message as _require_change_tracker.
            message = "change_tracker is not set; run repo_scan before incremental ingest"
            raise RuntimeError(message)

        runner = ctx.tool_runner or ToolRunner(
            cache_dir=ctx.paths.tool_cache,
            tools_config=ctx.active_tools,
        )
        service = ctx.tool_service or ToolService(runner, ctx.active_tools)

        return scip_ingest.ingest_scip(
            ctx.gateway,
            cfg=cfg,
            tracker=tracker,
            tool_service=service,
        )
```

That is **exactly** the logic that used to live in `run_scip_ingest`, just moved into the step, with the same ToolBinaries wiring and error semantics.

---

## 3. `run_ast_extract` in `ingestion/runner.py` — before vs after

### 3.1 Current `run_ast_extract` (before)

In `runner.py` today you should have:

```python
def run_ast_extract(ctx: IngestionContext) -> None:
    """Extract stdlib AST nodes and metrics using the gateway connection."""
    start = _log_step_start("ast_extract", ctx)
    tracker = _require_change_tracker(ctx)
    py_ast_extract.ingest_python_ast(tracker)
    _log_step_done("ast_extract", start, ctx)
```

### 3.2 New `run_ast_extract` (after)

Replace the whole function with:

```python
def run_ast_extract(ctx: IngestionContext) -> None:
    """Extract stdlib AST nodes and metrics using the gateway connection."""
    _run_ingest_step(ctx, "ast_extract")
```

Logging and timing are now handled in `_run_ingest_step` instead of being duplicated here.

---

## 4. New `AstExtractStep` in `ingestion/steps.py`

In the same `steps.py`, add the **AST** step implementation:

```python
@dataclass(frozen=True)
class AstExtractStep:
    """Parse stdlib AST and persist rows/metrics."""

    name: str = "ast_extract"
    description: str = "Parse Python AST and persist rows + metrics into core.ast_* tables."
    produces_tables: tuple[str, ...] = ("core.ast_nodes", "core.ast_metrics")
    requires: tuple[str, ...] = ("repo_scan",)

    def run(self, ctx: IngestionContext) -> None:
        tracker = ctx.change_tracker
        if tracker is None:
            message = "change_tracker is not set; run repo_scan before incremental ingest"
            raise RuntimeError(message)
        py_ast_extract.ingest_python_ast(tracker)
```

Again, this is exactly what `run_ast_extract` did before, just lifted into a step.

---

## 5. Wiring into `DEFAULT_REGISTRY`

In `steps.py`, you’ll also want your default registry to include these two steps.

Somewhere near the bottom, you should have something like:

```python
@dataclass(frozen=True)
class IngestStepRegistry:
    _steps: Mapping[str, IngestStep]
    _sequence: tuple[str, ...] = ()

    # ... methods (get, all_metadata, topological_order, etc.) ...
```

And then the default registry builder:

```python
def _build_default_registry() -> IngestStepRegistry:
    """Construct the default registry with all built-in ingestion steps."""
    steps: dict[str, IngestStep] = {
        "repo_scan": RepoScanStep(),
        "scip_ingest": ScipIngestStep(),
        "cst_extract": CstExtractStep(),
        "ast_extract": AstExtractStep(),
        # ... other steps like typing_ingest, coverage_ingest, tests_ingest, docstrings_ingest, config_ingest ...
    }
    sequence = tuple(steps.keys())
    return IngestStepRegistry(_steps=steps, _sequence=sequence)


DEFAULT_REGISTRY: IngestStepRegistry = _build_default_registry()
```

As long as `"scip_ingest"` and `"ast_extract"` appear in the `_steps` mapping, your `run_*` wrappers will resolve to the right logic.

---

## 6. `_run_ingest_step` helper in `runner.py` (if you haven’t added it yet)

Finally, here’s the small helper in `runner.py` that makes the **before/after** deltas work cleanly:

At the top of `runner.py` add:

```python
from collections.abc import Callable, Sequence  # Sequence might be new
...
from codeintel.ingestion.steps import DEFAULT_REGISTRY
```

Then somewhere above the `run_*` functions (e.g. right after `_log_step_done`), define:

```python
def _run_ingest_step(ctx: IngestionContext, name: str) -> object | None:
    """
    Run a single ingestion step by name with logging.

    Parameters
    ----------
    ctx
        Shared ingestion context.
    name
        Name of the ingestion step to execute.

    Returns
    -------
    object | None
        Any value returned by the underlying step (usually None, except
        for steps like 'repo_scan' or 'scip_ingest').
    """
    start = _log_step_start(name, ctx)
    step = DEFAULT_REGISTRY.get(name)
    result = step.run(ctx)
    _log_step_done(name, start, ctx)
    return result
```

With that in place:

* `run_scip_ingest` and `run_ast_extract` become thin, registry-backed wrappers.
* All **SCIP** + **AST** logic lives in the new, declarative `IngestStep` implementations.
* You now have a concrete pattern you can copy for the rest of the steps (CST, typing, coverage, tests, docstrings, config).

If you’d like, next we can do the same **before/after** treatment for one of the more complex “multi-tool” steps (e.g. `typing_ingest`), which will exercise the tool-service side of the registry more heavily.
