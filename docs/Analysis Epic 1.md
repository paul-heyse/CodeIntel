I’m going to assume you want **Epic 1** turned into a detailed plan (the “split `GraphServiceRuntime` into plan/execute/persist/observe layers” epic I described). If you’d rather I do Epic 2 next, we can tackle that after.

Below is a **concrete, file-by-file plan with code sketches** for refactoring `analytics/graph_service_runtime.py` into small, composable pieces while keeping the *public API stable*:

* `GraphServiceRuntime`
* `GraphContext`, `GraphContextSpec`, `GraphContextCaps`
* `GraphPluginRunOptions`, `GraphPluginRunRecord`, `GraphPluginRunReport`
* `GraphRuntimeTelemetry`

…all still importable from `codeintel.analytics.graph_service_runtime`.

---

## 0. Target end-state overview

New modules (all under `src/codeintel/analytics/graphs/runtime/`):

* `context.py`

  * `GraphContext`, `GraphContextSpec`, `GraphContextCaps`
  * `build_graph_context`, `resolve_graph_context`, `_base_context`, `_normalize_context`.

* `model.py`

  * `GraphPluginRunOptions`, `GraphPluginRunRecord`, `GraphPluginRunReport`.

* `telemetry.py`

  * `GraphRuntimeTelemetry` protocol.
  * `_NoOpGraphRuntimeTelemetry`, `_OtelGraphRuntimeTelemetry`.

* `execution.py`

  * `_IsolationEnvelope`, `_IsolationResult`, `_MPContext`, `_PluginFatalError`.
  * Core run primitives: `_select_mp_context`, `_run_isolation_worker`,
    `_execute_plugin_isolated`, `_execute_plugin`, `_run_with_timeout`, `_build_gateway_for_isolation`.

* `manifest.py`

  * `_hash_json`, `_compute_options_hash`, `_compute_input_hash`.
  * Manifest IO: `_write_plugin_manifest`, `_load_prior_manifest`, `_report_to_payload`.
  * Row counts & skip logic: `_row_counts_equal`, `_current_row_counts`,
    `_is_unchanged`, `_dry_run_record`, `_skip_record`, `_run_contracts`.

* `planning.py`

  * `_PluginExecutionSettings`, `_PluginExecutionPlan`.
  * `_resolve_plugin_options_map`, `_effective_severity`, `_effective_timeout`, `_resolve_target`.
  * A public `plan_graph_plugin_run(...)` that returns a `GraphExecutionPlan` object.

And then:

* `analytics/graph_service_runtime.py` becomes a **thin orchestrator** that:

  * Builds `GraphContext`.
  * Calls `plan_graph_plugin_run(...)`.
  * Delegates execution to `run_graph_plugin_batch(...)` from `execution.py`.
  * Delegates manifest and contract logic to `manifest.py`.
  * Uses `GraphRuntimeTelemetry` from `telemetry.py`.

Call-sites (`cfg_dfg/*`, `graphs/graph_metrics_ext.py`, `tests/analytics/*`) keep importing from `codeintel.analytics.graph_service_runtime`, but those names are now re-exported from the new modules.

---

## 1. Create `analytics/graphs/runtime/__init__.py`

**File:** `analytics/graphs/runtime/__init__.py` (new)

Purpose: central re-exports so internal modules can import from `codeintel.analytics.graphs.runtime` without caring about submodule layout.

```python
from __future__ import annotations

from .context import (
    GraphContext,
    GraphContextCaps,
    GraphContextSpec,
    build_graph_context,
    resolve_graph_context,
)
from .model import (
    GraphPluginRunOptions,
    GraphPluginRunRecord,
    GraphPluginRunReport,
)
from .telemetry import (
    GraphRuntimeTelemetry,
    NoOpGraphRuntimeTelemetry,
    OtelGraphRuntimeTelemetry,
)
from .planning import (
    PluginExecutionSettings,
    PluginExecutionPlan,
    plan_graph_plugin_run,
)
from .execution import (
    PluginFatalError,
    run_graph_plugin_batch,
)
from .manifest import (
    compute_input_hash,
    compute_options_hash,
    write_plugin_manifest,
    load_prior_manifest,
    is_unchanged,
)

DEFAULT_BETWEENNESS_SAMPLE = 500

__all__ = [
    "DEFAULT_BETWEENNESS_SAMPLE",
    "GraphContext",
    "GraphContextCaps",
    "GraphContextSpec",
    "GraphPluginRunOptions",
    "GraphPluginRunRecord",
    "GraphPluginRunReport",
    "GraphRuntimeTelemetry",
    "NoOpGraphRuntimeTelemetry",
    "OtelGraphRuntimeTelemetry",
    "PluginExecutionSettings",
    "PluginExecutionPlan",
    "plan_graph_plugin_run",
    "PluginFatalError",
    "run_graph_plugin_batch",
    "compute_input_hash",
    "compute_options_hash",
    "write_plugin_manifest",
    "load_prior_manifest",
    "is_unchanged",
    "build_graph_context",
    "resolve_graph_context",
]
```

> Note: we define `DEFAULT_BETWEENNESS_SAMPLE` here so both `graph_service_runtime` and `graph_service` can re-export it.

---

## 2. Extract context: `GraphContext`, `GraphContextSpec`, `GraphContextCaps`

**File:** `analytics/graphs/runtime/context.py` (new)

Move the `GraphContext*` types and `build_graph_context`/`resolve_graph_context` + helpers from `graph_service_runtime.py`.

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from codeintel.config.steps_graphs import GraphMetricsStepConfig

if TYPE_CHECKING:
    from codeintel.analytics.graph_runtime import GraphRuntime
    from codeintel.config.primitives import SnapshotRef

DEFAULT_BETWEENNESS_SAMPLE = 500


@dataclass
class GraphContext:
    """Execution context for graph computations."""

    repo: str
    commit: str
    now: datetime | None = None
    betweenness_sample: int = DEFAULT_BETWEENNESS_SAMPLE
    eigen_max_iter: int = 200
    seed: int = 0
    pagerank_weight: str | None = "weight"
    betweenness_weight: str | None = "weight"
    use_gpu: bool = False
    community_detection_limit: int | None = None

    def resolved_now(self) -> datetime:
        """Return a concrete timestamp, defaulting to UTC now when unset."""
        return self.now or datetime.now(tz=UTC)


@dataclass
class GraphContextSpec:
    """Specification for normalizing graph contexts."""

    repo: str
    commit: str
    use_gpu: bool
    metrics_cfg: GraphMetricsStepConfig | None = None
    ctx: GraphContext | None = None
    now: datetime | None = None
    betweenness_cap: int | None = None
    eigen_cap: int | None = None
    pagerank_weight: str | None = None
    betweenness_weight: str | None = None
    seed: int | None = None
    community_detection_limit: int | None = None


@dataclass
class GraphContextCaps:
    """Optional caps for graph context derivation."""

    betweenness_cap: int | None = None
    eigen_cap: int | None = None
    community_detection_limit: int | None = None


def _base_context(cfg: GraphMetricsStepConfig, *, use_gpu: bool) -> GraphContext:
    """Seed a GraphContext from GraphMetricsStepConfig."""
    return GraphContext(
        repo=cfg.repo,
        commit=cfg.commit,
        betweenness_sample=cfg.max_betweenness_sample or DEFAULT_BETWEENNESS_SAMPLE,
        eigen_max_iter=cfg.eigen_max_iter,
        seed=cfg.seed,
        pagerank_weight=cfg.pagerank_weight,
        betweenness_weight=cfg.betweenness_weight,
        use_gpu=use_gpu,
        community_detection_limit=cfg.community_detection_limit,
    )


def _normalize_context(spec: GraphContextSpec) -> GraphContext:
    """Normalize overrides and caps into a concrete GraphContext."""
    if spec.metrics_cfg is not None:
        base = _base_context(spec.metrics_cfg, use_gpu=spec.use_gpu)
    elif spec.ctx is not None:
        base = spec.ctx
    else:
        raise ValueError("GraphContextSpec requires either metrics_cfg or ctx")

    return GraphContext(
        repo=spec.repo,
        commit=spec.commit,
        now=spec.now or base.now,
        betweenness_sample=spec.betweenness_cap or base.betweenness_sample,
        eigen_max_iter=spec.eigen_cap or base.eigen_max_iter,
        seed=spec.seed or base.seed,
        pagerank_weight=spec.pagerank_weight or base.pagerank_weight,
        betweenness_weight=spec.betweenness_weight or base.betweenness_weight,
        use_gpu=spec.use_gpu,
        community_detection_limit=spec.community_detection_limit or base.community_detection_limit,
    )


def build_graph_context(
    cfg: GraphMetricsStepConfig,
    *,
    use_gpu: bool,
    caps: GraphContextCaps | None = None,
) -> GraphContext:
    """
    Build a GraphContext from configuration and optional caps.
    """
    spec = GraphContextSpec(
        repo=cfg.repo,
        commit=cfg.commit,
        use_gpu=use_gpu,
        metrics_cfg=cfg,
        betweenness_cap=caps.betweenness_cap if caps else None,
        eigen_cap=caps.eigen_cap if caps else None,
        community_detection_limit=caps.community_detection_limit if caps else None,
    )
    return _normalize_context(spec)


def resolve_graph_context(
    runtime: GraphRuntime,
    *,
    cfg: GraphMetricsStepConfig | None = None,
    target: tuple[str, str] | None = None,
    use_gpu: bool | None = None,
    caps: GraphContextCaps | None = None,
) -> tuple[GraphContext, tuple[str, str]]:
    """
    Resolve a GraphContext from either a config snapshot or runtime options.

    Returns (ctx, (repo, commit)).
    """
    if cfg is not None:
        repo, commit = cfg.repo, cfg.commit
    else:
        if runtime.options.snapshot is None:
            msg = "Either cfg or runtime.options.snapshot is required"
            raise ValueError(msg)
        repo, commit = runtime.options.snapshot.repo, runtime.options.snapshot.commit

    if target is not None:
        repo, commit = target

    use_gpu_effective = use_gpu if use_gpu is not None else runtime.options.use_gpu

    ctx = build_graph_context(
        cfg=cfg or GraphMetricsStepConfig(snapshot=runtime.options.snapshot),  # type: ignore[arg-type]
        use_gpu=use_gpu_effective,
        caps=caps,
    )
    ctx.repo = repo
    ctx.commit = commit
    return ctx, (repo, commit)
```

> In your actual implementation, you can drop the fallback `GraphMetricsStepConfig` construction hack and use your existing logic for deriving a config when only a `SnapshotRef` is available.

**Then:**

* Remove the `GraphContext*` classes and `build_graph_context`/`resolve_graph_context` from `graph_service_runtime.py`.
* Update imports in:

  * `analytics/cfg_dfg/dfg_core.py`
  * `analytics/cfg_dfg/cfg_core.py`
  * `analytics/cfg_dfg/materialize.py`
  * `analytics/graphs/graph_metrics_ext.py`
  * `analytics/tests/graph_metrics.py`
  * `analytics/graph_metrics/metrics.py`
  * `analytics/graph_rows/graph_metrics_ext.py`

  to use:

```python
from codeintel.analytics.graphs.runtime import GraphContext, build_graph_context, resolve_graph_context
# or more specific: from codeintel.analytics.graphs.runtime.context import ...
```

---

## 3. Extract runtime model: run records, run options, run report

**File:** `analytics/graphs/runtime/model.py` (new)

Move the `GraphPluginRunRecord`, `GraphPluginRunOptions`, `GraphPluginRunReport` dataclasses out of `graph_service_runtime.py`.

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from codeintel.analytics.graphs.plugins import GraphMetricPluginSkip
from codeintel.analytics.graphs.contracts import PluginContractResult
from codeintel.config.steps_graphs import GraphRunScope


@dataclass
class GraphPluginRunRecord:
    """Per-plugin execution telemetry."""

    name: str
    stage: str
    severity: Literal["fatal", "soft_fail", "skip_on_error"]
    status: Literal["succeeded", "failed", "skipped"]
    attempts: int
    timeout_ms: int | None
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    partial: bool
    run_id: str
    error: str | None = None
    options: object | None = None
    input_hash: str | None = None
    options_hash: str | None = None
    version_hash: str | None = None
    skipped_reason: str | None = None
    row_counts: dict[str, int] | None = None
    contracts: tuple[PluginContractResult, ...] = ()
    requires_isolation: bool = False
    isolation_kind: str | None = None
    policy_fail_fast: bool = False


@dataclass(frozen=True)
class GraphPluginRunOptions:
    """Optional controls for plugin execution."""

    plugin_options: dict[str, dict[str, object]] | None = None
    manifest_path: Path | None = None
    scope: GraphRunScope | None = None
    dry_run: bool | None = None


@dataclass(frozen=True)
class GraphPluginRunReport:
    """Aggregate execution report for a plugin batch."""

    repo: str
    commit: str
    records: tuple[GraphPluginRunRecord, ...]
    scope: GraphRunScope
    run_id: str
    plan_id: str
    ordered_plugins: tuple[str, ...]
    skipped_plugins: tuple[GraphMetricPluginSkip, ...]
    dep_graph: dict[str, tuple[str, ...]]
```

**Then:**

* Remove these dataclasses from `graph_service_runtime.py`.
* Add a local import in `graph_service_runtime.py`:

```python
from codeintel.analytics.graphs.runtime.model import (
    GraphPluginRunOptions,
    GraphPluginRunRecord,
    GraphPluginRunReport,
)
```

* Keep re-exports via `__all__` (we’ll do that in Step 7).

Tests will still import them via `codeintel.analytics.graph_service_runtime`.

---

## 4. Extract telemetry: `GraphRuntimeTelemetry`, No-op, OTEL implementation

**File:** `analytics/graphs/runtime/telemetry.py` (new)

Move the protocol and OTEL classes out.

```python
from __future__ import annotations

from typing import Protocol

from opentelemetry import metrics, trace
from opentelemetry.trace import Status, StatusCode

from codeintel.analytics.graphs.plugins import GraphMetricExecutionContext, GraphMetricPlugin
from codeintel.config.steps_graphs import GraphRunScope

from .model import GraphPluginRunRecord


class GraphRuntimeTelemetry(Protocol):
    """Optional observability hooks for graph runtime events."""

    def start_plugin(
        self,
        plugin: GraphMetricPlugin,
        run_id: str,
        ctx: GraphMetricExecutionContext,
    ) -> object: ...

    def finish_plugin(self, span: object, record: GraphPluginRunRecord) -> None: ...

    def record_metrics(self, record: GraphPluginRunRecord, scope: GraphRunScope) -> None: ...


class NoOpGraphRuntimeTelemetry:
    """Telemetry implementation that does nothing."""

    def start_plugin(
        self,
        plugin: GraphMetricPlugin,
        run_id: str,
        ctx: GraphMetricExecutionContext,
    ) -> None:
        return None

    def finish_plugin(self, span: object, record: GraphPluginRunRecord) -> None:
        return None

    def record_metrics(self, record: GraphPluginRunRecord, scope: GraphRunScope) -> None:
        return None


class OtelGraphRuntimeTelemetry:
    """OpenTelemetry-backed telemetry for graph plugins."""

    def __init__(self) -> None:
        self.tracer = trace.get_tracer(__name__)
        meter = metrics.get_meter(__name__)

        self.duration_ms = meter.create_histogram(
            "graph_plugin_duration_ms",
            unit="ms",
            description="Duration of graph metric plugins",
        )
        self.status_counter = meter.create_counter(
            "graph_plugin_status_count",
            description="Count of plugin runs by status",
        )
        self.retry_counter = meter.create_counter(
            "graph_plugin_retry_count",
            description="Count of plugin retries",
        )

    def start_plugin(
        self,
        plugin: GraphMetricPlugin,
        run_id: str,
        ctx: GraphMetricExecutionContext,
    ) -> object:
        attributes = {
            "plugin_name": plugin.name,
            "plugin_stage": plugin.stage,
            "repo": ctx.repo,
            "commit": ctx.commit,
            "run_id": run_id,
        }
        return self.tracer.start_span("graph_plugin", attributes=attributes)

    def finish_plugin(self, span: object, record: GraphPluginRunRecord) -> None:
        if not hasattr(span, "set_status"):
            return
        status = Status(
            status_code=StatusCode.OK
            if record.status == "succeeded"
            else StatusCode.ERROR
        )
        span.set_status(status)
        span.end()

    def record_metrics(self, record: GraphPluginRunRecord, scope: GraphRunScope) -> None:
        attributes = {
            "plugin_name": record.name,
            "plugin_stage": record.stage,
            "status": record.status,
            "requires_isolation": record.requires_isolation,
            "isolation_kind": record.isolation_kind or "none",
            "scope_paths": len(scope.paths),
            "scope_modules": len(scope.modules),
            "scope_time_window": scope.time_window is not None,
            "policy_fail_fast": record.policy_fail_fast,
        }
        self.duration_ms.record(record.duration_ms, attributes=attributes)
        self.status_counter.add(1, attributes=attributes)
        if record.attempts > 1:
            self.retry_counter.add(record.attempts - 1, attributes=attributes)
```

**Then:**

* Remove the telemetry protocol and classes from `graph_service_runtime.py`.
* Add:

```python
from codeintel.analytics.graphs.runtime.telemetry import (
    GraphRuntimeTelemetry,
    NoOpGraphRuntimeTelemetry,
    OtelGraphRuntimeTelemetry,
)
```

* In `GraphServiceRuntime.run_plugins`, where you currently do `self.telemetry or _OtelGraphRuntimeTelemetry()`, switch to `self.telemetry or OtelGraphRuntimeTelemetry()`.

Tests (`test_graph_plugin_policy_runtime.py`) can keep importing `GraphRuntimeTelemetry` from `graph_service_runtime`, which will re-export it.

---

## 5. Extract isolation & execution primitives

**File:** `analytics/graphs/runtime/execution.py` (new)

Move `_IsolationEnvelope`, `_IsolationResult`, `_MPContext`, `_PluginFatalError`, `_select_mp_context`, `_run_isolation_worker`, `_execute_plugin_isolated`, `_execute_plugin`, `_run_with_timeout`, `_build_gateway_for_isolation` out of `graph_service_runtime.py`.

You’ll also introduce a public `run_graph_plugin_batch(...)` that wraps them.

```python
from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntime
from codeintel.analytics.graphs.contracts import ContractChecker, PluginContractResult
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphMetricPluginSkip,
    GraphPluginResult,
    GraphRuntimeScratch,
)
from codeintel.config.steps_graphs import (
    GraphMetricsStepConfig,
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
    GraphRunScope,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.storage.gateway import StorageConfig, StorageGateway, open_gateway, open_memory_gateway

from .model import GraphPluginRunRecord
from .telemetry import GraphRuntimeTelemetry
from .manifest import (
    current_row_counts,
    dry_run_record,
    is_unchanged,
    skip_record,
    run_contracts,
)


@dataclass(frozen=True)
class IsolationEnvelope:
    """Serialized inputs for isolated plugin execution."""

    plugin_name: str
    plugin: GraphMetricPlugin | None
    repo: str
    commit: str
    options: object | None
    scope: GraphRunScope
    snapshot: SnapshotRef
    storage_config: StorageConfig
    use_gpu: bool
    manifest_path: Path | None = None
    telemetry_enabled: bool = False


@dataclass(frozen=True)
class IsolationResult:
    """Result payload for isolated plugin execution."""

    record: GraphPluginRunRecord


@dataclass(frozen=True)
class MPContext:
    """Factory wrapper around multiprocessing start methods."""

    process_factory: Callable[..., multiprocessing.Process]
    queue_factory: Callable[[], multiprocessing.Queue[IsolationResult]]
    start_method_name: str


class PluginFatalError(RuntimeError):
    """Raised to short-circuit execution when a fatal plugin fails."""

    def __init__(self, record: GraphPluginRunRecord) -> None:
        super().__init__(record.error or "Fatal plugin error")
        self.record = record


def _select_mp_context() -> MPContext:
    base_ctx = (
        multiprocessing.get_context("fork")
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing.get_context()
    )
    any_ctx = cast(Any, base_ctx)
    return MPContext(
        process_factory=cast(Callable[..., multiprocessing.Process], any_ctx.Process),
        queue_factory=cast(Callable[[], multiprocessing.Queue[IsolationResult]], any_ctx.Queue),
        start_method_name=base_ctx.get_start_method(),
    )


# You’ll copy your existing worker, isolated execution, and non-isolated execution
# helpers here, adapted to use IsolationEnvelope / IsolationResult / MPContext and
# GraphPluginRunRecord. Example skeletons:

def _run_isolation_worker(envelope: IsolationEnvelope) -> IsolationResult:
    """Worker entrypoint for isolated plugin execution."""
    # Rebuild gateway, runtime, analytics context, etc. using envelope.
    # Execute plugin, collect row counts and contracts, build GraphPluginRunRecord.
    # (Copy logic from your current _run_isolation_worker.)
    raise NotImplementedError


def _execute_plugin_isolated(
    plugin: GraphMetricPlugin,
    *,
    ctx: GraphMetricExecutionContext,
    options: object | None,
    mp_ctx: MPContext,
    retry_policy: GraphPluginRetryPolicy,
    timeout_ms: int | None,
) -> GraphPluginRunRecord:
    """Execute a plugin in a separate process."""
    raise NotImplementedError


def _execute_plugin(
    plugin: GraphMetricPlugin,
    *,
    ctx: GraphMetricExecutionContext,
    options: object | None,
    retry_policy: GraphPluginRetryPolicy,
    timeout_ms: int | None,
) -> GraphPluginRunRecord:
    """Execute a plugin in-process with retries and timeouts."""
    raise NotImplementedError


def run_graph_plugin_batch(
    *,
    plan,
    plugins: Sequence[GraphMetricPlugin],
    gateway: StorageGateway,
    runtime: GraphRuntime,
    analytics_context: AnalyticsContext | None,
    catalog_provider: FunctionCatalogProvider | None,
    telemetry: GraphRuntimeTelemetry,
    scope: GraphRunScope,
    run_id: str,
) -> list[GraphPluginRunRecord]:
    """Execute a prepared plugin execution plan and collect run records."""
    # Build GraphMetricExecutionContext, GraphRuntimeScratch, etc.
    # Iterate over plan.ordered_plugins, apply isolation / non-isolation logic
    # using _execute_plugin[_isolated], record contract results via run_contracts.
    raise NotImplementedError
```

**For now**, I’ve left some bodies as `NotImplementedError` in the snippet; in your actual refactor you’ll **copy the bodies** from `graph_service_runtime.py` into these functions with minimal changes.

Key idea: all the ugly concurrency / process management lives here, so `GraphServiceRuntime` becomes boring.

---

## 6. Extract manifest & skip/dry-run logic

**File:** `analytics/graphs/runtime/manifest.py` (new)

Move `_hash_json`, `_compute_options_hash`, `_load_prior_manifest`, `_compute_input_hash`, `_coerce_plugin_result`, `_row_counts_equal`, `_current_row_counts`, `_is_unchanged`, `_dry_run_record`, `_skip_record`, `_run_contracts`, `_write_plugin_manifest`, `_report_to_payload` here.

```python
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from codeintel.analytics.graphs.contracts import (
    ContractChecker,
    PluginContractResult,
    run_contract_checkers,
)
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
)
from codeintel.storage.gateway import StorageGateway

from .model import GraphPluginRunRecord


def hash_json(payload: object) -> str:
    """Stable JSON hash for manifests and options."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def compute_options_hash(plugin: GraphMetricPlugin, options: object | None) -> str | None:
    """Compute a stable hash over plugin options for manifest comparison."""
    if options is None:
        return None
    return hash_json({"plugin": plugin.name, "options": options})


def _report_to_payload(report: object) -> dict[str, object]:
    # Copy your existing logic that converts GraphPluginRunReport to JSONable dict.
    raise NotImplementedError


def write_plugin_manifest(path: Path, report: object) -> None:
    """Persist a manifest from a plugin run report."""
    path.write_text(json.dumps(_report_to_payload(report), indent=2, sort_keys=True))


def load_prior_manifest(path: Path | None) -> dict[str, object] | None:
    """Load the prior manifest JSON if it exists."""
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text())


def compute_input_hash(
    *,
    repo: str,
    commit: str,
    scope: object | None,
    options_hash: str | None,
) -> str:
    """Compute hash over repo/commit/scope/options for unchanged detection."""
    payload = {
        "repo": repo,
        "commit": commit,
        "scope": scope,
        "options_hash": options_hash,
    }
    return hash_json(payload)


def row_counts_equal(a: dict[str, int] | None, b: dict[str, int] | None) -> bool:
    if a is None or b is None:
        return False
    return a == b


def current_row_counts(
    gateway: StorageGateway,
    tables: Sequence[str],
) -> dict[str, int]:
    """Return row counts for the given fully-qualified table names."""
    # Copy your existing logic that queries DuckDB for row counts from the gateway.
    raise NotImplementedError


def is_unchanged(
    *,
    plugin: GraphMetricPlugin,
    record: GraphPluginRunRecord,
    prior: dict[str, object] | None,
) -> bool:
    """Return True if manifest indicates that this plugin's inputs are unchanged."""
    # Compare input_hash/options_hash/version_hash and prior row counts. Copy from current _is_unchanged.
    raise NotImplementedError


def dry_run_record(
    *,
    base_record: GraphPluginRunRecord,
    reason: str,
) -> GraphPluginRunRecord:
    """Return a copy of a record marked as dry-run."""
    return GraphPluginRunRecord(
        **{
            **base_record.__dict__,
            "status": "skipped",
            "partial": False,
            "skipped_reason": reason,
        },
    )


def skip_record(
    *,
    base_record: GraphPluginRunRecord,
    reason: str,
) -> GraphPluginRunRecord:
    """Return a record representing a skipped plugin (unchanged input)."""
    return GraphPluginRunRecord(
        **{
            **base_record.__dict__,
            "status": "skipped",
            "partial": False,
            "skipped_reason": reason,
        },
    )


def run_contracts(
    *,
    checkers: tuple[ContractChecker, ...],
    ctx: GraphMetricExecutionContext,
    status: Literal["succeeded", "failed", "skipped"],
) -> tuple[PluginContractResult, ...]:
    """Run contract checkers only when plugin succeeded."""
    if not checkers or status != "succeeded":
        return ()
    return run_contract_checkers(ctx=ctx, checkers=checkers)
```

Again: keep skeletons, but in your actual code you’ll copy logic from the existing helpers.

---

## 7. Extract planning: plugin execution settings & plan

**File:** `analytics/graphs/runtime/planning.py` (new)

Move `_PluginExecutionSettings`, `_PluginExecutionPlan`, `_resolve_plugin_options_map`, `_effective_severity`, `_effective_timeout`, `_resolve_target`, `_execute_planned_plugin` scaffolding into a dedicated planner.

```python
from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, Sequence

from codeintel.analytics.graphs.plugins import (
    GraphMetricPlugin,
    GraphMetricPluginSkip,
    GraphRuntimeScratch,
    plan_graph_metric_plugins,
)
from codeintel.config.steps_graphs import (
    GraphMetricsStepConfig,
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
    GraphRunScope,
)
from codeintel.config.primitives import SnapshotRef

from .model import GraphPluginRunOptions, GraphPluginRunRecord
from .manifest import compute_input_hash, compute_options_hash


@dataclass(frozen=True)
class PluginExecutionSettings:
    """Resolved execution configuration for a single plugin."""

    name: str
    severity: Literal["fatal", "soft_fail", "skip_on_error"]
    retry_policy: GraphPluginRetryPolicy
    timeout_ms: int | None
    options: object | None
    options_hash: str | None
    input_hash: str | None
    version_hash: str | None
    fail_fast: bool


@dataclass(frozen=True)
class PluginExecutionPlan:
    """Execution plan for one batch of graph plugins."""

    plan_id: str
    ordered_names: tuple[str, ...]
    skipped_plugins: tuple[GraphMetricPluginSkip, ...]
    dep_graph: dict[str, tuple[str, ...]]
    run_id: str
    scope: GraphRunScope
    settings_by_plugin: dict[str, PluginExecutionSettings]


def _resolve_plugin_options_map(
    plugins: Sequence[GraphMetricPlugin],
    cfg_options: dict[str, dict[str, object]] | None,
    runtime_options: dict[str, dict[str, object]] | None,
) -> dict[str, object | None]:
    """Resolve effective options per plugin from config + runtime options."""
    cfg_options = cfg_options or {}
    runtime_options = runtime_options or {}
    resolved: dict[str, object | None] = {}
    for plugin in plugins:
        opt: object | None = None
        if plugin.name in cfg_options:
            opt = cfg_options[plugin.name]
        if plugin.name in runtime_options:
            # runtime overrides config
            opt = runtime_options[plugin.name]
        resolved[plugin.name] = opt
    return resolved


def _effective_severity(
    *,
    plugin: GraphMetricPlugin,
    policy: GraphPluginPolicy,
) -> Literal["fatal", "soft_fail", "skip_on_error"]:
    if plugin.name in policy.severity_overrides:
        return policy.severity_overrides[plugin.name]
    return policy.default_severity


def _effective_timeout(
    *,
    plugin: GraphMetricPlugin,
    policy: GraphPluginPolicy,
) -> int | None:
    return policy.timeout_overrides.get(plugin.name, policy.default_timeout_ms)


def _resolve_target(
    *,
    cfg: GraphMetricsStepConfig | None,
    runtime_snapshot: SnapshotRef | None,
    target: tuple[str, str] | None,
) -> tuple[str, str]:
    if target is not None:
        return target
    if cfg is not None:
        return cfg.repo, cfg.commit
    if runtime_snapshot is not None:
        return runtime_snapshot.repo, runtime_snapshot.commit
    msg = "Either cfg, runtime snapshot, or explicit target is required"
    raise ValueError(msg)


def plan_graph_plugin_run(
    *,
    plugin_names: Sequence[str],
    plugins: Sequence[GraphMetricPlugin],
    cfg: GraphMetricsStepConfig | None,
    runtime_snapshot: SnapshotRef | None,
    policy: GraphPluginPolicy,
    run_options: GraphPluginRunOptions | None,
) -> PluginExecutionPlan:
    """Build execution plan + per-plugin settings for a batch run."""
    run_id = uuid.uuid4().hex
    plan_id = uuid.uuid4().hex
    scope = run_options.scope if run_options and run_options.scope else GraphRunScope()

    # Filter & order plugins based on names + dependencies.
    plan = plan_graph_metric_plugins(plugin_names, plugins, policy=policy)

    # Resolve per-plugin options.
    cfg_option_map = cfg.plugin_options if cfg is not None else {}
    runtime_option_map = run_options.plugin_options if run_options else {}
    options_map = _resolve_plugin_options_map(plugins, cfg_option_map, runtime_option_map)

    repo, commit = _resolve_target(
        cfg=cfg,
        runtime_snapshot=runtime_snapshot,
        target=None,
    )

    settings_by_plugin: dict[str, PluginExecutionSettings] = {}
    for plugin in plugins:
        severity = _effective_severity(plugin=plugin, policy=policy)
        timeout_ms = _effective_timeout(plugin=plugin, policy=policy)
        options = options_map.get(plugin.name)
        options_hash = compute_options_hash(plugin, options)
        input_hash = compute_input_hash(
            repo=repo,
            commit=commit,
            scope=scope,
            options_hash=options_hash,
        )
        settings_by_plugin[plugin.name] = PluginExecutionSettings(
            name=plugin.name,
            severity=severity,
            retry_policy=policy.retry_policy.get(plugin.name, GraphPluginRetryPolicy()),
            timeout_ms=timeout_ms,
            options=options,
            options_hash=options_hash,
            input_hash=input_hash,
            version_hash=plugin.version_hash,
            fail_fast=policy.fail_fast,
        )

    return PluginExecutionPlan(
        plan_id=plan_id,
        ordered_names=tuple(plan.ordered_names),
        skipped_plugins=plan.skipped_plugins,
        dep_graph=plan.dep_graph,
        run_id=run_id,
        scope=scope,
        settings_by_plugin=settings_by_plugin,
    )
```

Again: you will tweak this to exactly match your current semantics (e.g. if policies store retry settings differently).

---

## 8. Slim down `analytics/graph_service_runtime.py`

Now that all heavy lifting is factored out, we turn `GraphServiceRuntime` into a thin orchestrator and re-export module.

**File:** `analytics/graph_service_runtime.py` (modified)

### 8.1. Imports

Replace the large import block with:

```python
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import GraphRuntime, GraphRuntimeOptions, resolve_graph_runtime
from codeintel.analytics.graphs.plugins import (
    DEFAULT_GRAPH_METRIC_PLUGINS,
    GraphMetricPlugin,
    get_graph_metric_plugin,
)
from codeintel.analytics.graphs.runtime import DEFAULT_BETWEENNESS_SAMPLE
from codeintel.analytics.graphs.runtime.context import (
    GraphContext,
    GraphContextCaps,
    GraphContextSpec,
    build_graph_context,
    resolve_graph_context,
)
from codeintel.analytics.graphs.runtime.execution import (
    PluginFatalError,
    run_graph_plugin_batch,
)
from codeintel.analytics.graphs.runtime.model import (
    GraphPluginRunOptions,
    GraphPluginRunRecord,
    GraphPluginRunReport,
)
from codeintel.analytics.graphs.runtime.planning import (
    PluginExecutionPlan,
    plan_graph_plugin_run,
)
from codeintel.analytics.graphs.runtime.manifest import (
    load_prior_manifest,
    write_plugin_manifest,
)
from codeintel.analytics.graphs.runtime.telemetry import (
    GraphRuntimeTelemetry,
    NoOpGraphRuntimeTelemetry,
    OtelGraphRuntimeTelemetry,
)
from codeintel.config.primitives import SnapshotRef
from codeintel.config.steps_graphs import GraphMetricsStepConfig, GraphPluginPolicy, GraphRunScope
from codeintel.graphs.function_catalog_service import FunctionCatalogProvider
from codeintel.storage.gateway import StorageGateway

if TYPE_CHECKING:
    from pathlib import Path

log = logging.getLogger(__name__)
```

### 8.2. `GraphServiceRuntime` class

Define it as a thin façade:

```python
@dataclass
class GraphServiceRuntime:
    """Lightweight orchestrator for graph analytics using a shared runtime."""

    gateway: StorageGateway
    runtime: GraphRuntime
    analytics_context: AnalyticsContext | None = None
    catalog_provider: FunctionCatalogProvider | None = None
    telemetry: GraphRuntimeTelemetry | None = None

    def run_plugins(
        self,
        plugin_names: Sequence[str],
        *,
        cfg: GraphMetricsStepConfig | None = None,
        target: tuple[str, str] | None = None,
        run_options: GraphPluginRunOptions | None = None,
    ) -> GraphPluginRunReport:
        """
        Execute a sequence of graph metric plugins against this runtime.

        Parameters
        ----------
        plugin_names:
            Names of plugins to execute, in order.
        cfg:
            Optional graph metrics configuration; provided to plugins via
            GraphMetricExecutionContext.
        target:
            Optional (repo, commit) override when config is not supplied.
        run_options:
            Optional execution controls (per-plugin options, manifest path, scopes).

        Raises
        ------
        ValueError
            If neither a config nor runtime snapshot is available to derive repo/commit.
        PluginFatalError
            When a fatal plugin failure occurs and fail-fast is enabled.
        """
        # Resolve runtime and plugin set
        plugins: list[GraphMetricPlugin] = [
            get_graph_metric_plugin(name) for name in plugin_names
        ]

        # Resolve runtime snapshot + target
        snapshot = cfg.snapshot if cfg is not None else self.runtime.options.snapshot
        if snapshot is None and target is None:
            msg = "Either cfg, runtime snapshot, or explicit target is required"
            raise ValueError(msg)
        if target is not None:
            repo, commit = target
        else:
            assert snapshot is not None
            repo, commit = snapshot.repo, snapshot.commit

        # Build execution policy
        policy = cfg.plugin_policy if cfg is not None else GraphPluginPolicy()

        # Build execution plan
        plan = plan_graph_plugin_run(
            plugin_names=plugin_names,
            plugins=plugins,
            cfg=cfg,
            runtime_snapshot=snapshot,
            policy=policy,
            run_options=run_options,
        )

        manifest_path = run_options.manifest_path if run_options else None
        prior_manifest = load_prior_manifest(manifest_path)

        scope = run_options.scope if run_options and run_options.scope else GraphRunScope()
        telemetry = self.telemetry or OtelGraphRuntimeTelemetry()

        # Execute
        try:
            records = run_graph_plugin_batch(
                plan=plan,
                plugins=plugins,
                gateway=self.gateway,
                runtime=self.runtime,
                analytics_context=self.analytics_context,
                catalog_provider=self.catalog_provider,
                telemetry=telemetry,
                scope=scope,
                run_id=plan.run_id,
            )
        except PluginFatalError as exc:
            # Re-raise after we have a partial record list if needed
            records = [exc.record]
            raise

        report = GraphPluginRunReport(
            repo=repo,
            commit=commit,
            records=tuple(records),
            scope=scope,
            run_id=plan.run_id,
            plan_id=plan.plan_id,
            ordered_plugins=plan.ordered_names,
            skipped_plugins=plan.skipped_plugins,
            dep_graph=plan.dep_graph,
        )

        if manifest_path is not None:
            write_plugin_manifest(manifest_path, report)

        return report
```

> In your actual implementation you’ll weave in **prior manifest** and **unchanged detection** by:
>
> * Computing per-plugin `input_hash` and comparing with `prior_manifest`.
> * Producing `skip_record`/`dry_run_record` for unchanged plugins.
> * Passing the state into `run_graph_plugin_batch`.

### 8.3. Re-export public surface

At the bottom of `graph_service_runtime.py` keep:

```python
__all__ = [
    "DEFAULT_BETWEENNESS_SAMPLE",
    "GraphContext",
    "GraphContextCaps",
    "GraphContextSpec",
    "GraphServiceRuntime",
    "GraphPluginRunOptions",
    "GraphPluginRunRecord",
    "GraphPluginRunReport",
    "GraphRuntimeTelemetry",
    "build_graph_context",
    "resolve_graph_context",
]
```

This preserves the import paths used in:

* `analytics/graph_service.py`
* `analytics/graph_metrics/metrics.py`
* `analytics/tests/graph_metrics.py`
* `tests/analytics/test_graph_service_runtime.py`
* `tests/analytics/test_graph_plugin_policy_runtime.py`
* `tests/analytics/conftest.py`
* `tests/analytics/test_graph_manifest_skip.py`

---

## 9. Tests & incremental validation

**Files to touch / add tests:**

1. **Update imports** where necessary, but keep most test imports hitting `graph_service_runtime`:

   * `tests/analytics/test_graph_service_runtime.py`
   * `tests/analytics/test_graph_plugin_policy_runtime.py`
   * `tests/analytics/test_graph_manifest_skip.py`
   * `tests/analytics/conftest.py`

   For example, `test_graph_plugin_policy_runtime.py` can stay as:

   ```python
   from codeintel.analytics.graph_service_runtime import (
       GraphPluginRunOptions,
       GraphPluginRunRecord,
       GraphRuntimeTelemetry,
       GraphServiceRuntime,
   )
   ```

   because those names are re-exported.

2. **Add focused unit tests** for the new modules:

   * `tests/analytics/test_graph_runtime_context.py`

     * Round-trip tests for `build_graph_context` / `resolve_graph_context`.
     * Caps behavior (`GraphContextCaps`).
   * `tests/analytics/test_graph_runtime_planning.py`

     * Verify severity override, timeout override, option resolution, and `GraphRunScope` propagation.
   * `tests/analytics/test_graph_runtime_manifest.py`

     * `compute_input_hash` stability.
     * Row-count comparison + `is_unchanged` semantics.
   * `tests/analytics/test_graph_runtime_execution.py`

     * Small fake plugins (fast, slow, failing, isolated) to exercise `_execute_plugin` / `_execute_plugin_isolated` and `run_graph_plugin_batch`.

   Skeleton example:

   ```python
   # tests/analytics/test_graph_runtime_planning.py

   from codeintel.analytics.graphs.runtime.planning import plan_graph_plugin_run
   from codeintel.analytics.graphs.plugins import GraphMetricPlugin
   from codeintel.config.primitives import SnapshotRef
   from codeintel.config.steps_graphs import GraphMetricsStepConfig, GraphPluginPolicy, GraphRunScope

   def test_plan_inherits_scope_and_policy_defaults() -> None:
       snapshot = SnapshotRef(repo="demo/repo", commit="abc123", repo_root=Path("."))
       cfg = GraphMetricsStepConfig(snapshot=snapshot)
       policy = GraphPluginPolicy()
       scope = GraphRunScope(paths=("src/demo.py",))

       plugin = GraphMetricPlugin(
           name="demo",
           description="demo",
           stage="demo",
           severity="fatal",
           enabled_by_default=True,
           depends_on=(),
           provides=("analytics.demo",),
           requires=(),
           isolation_kind=None,
           requires_isolation=False,
           version_hash="v1",
           run=lambda ctx: None,
       )

       plan = plan_graph_plugin_run(
           plugin_names=("demo",),
           plugins=(plugin,),
           cfg=cfg,
           runtime_snapshot=snapshot,
           policy=policy,
           run_options=GraphPluginRunOptions(scope=scope),
       )

       assert plan.scope.paths == scope.paths
       assert plan.ordered_names == ("demo",)
       assert "demo" in plan.settings_by_plugin
   ```

3. **Keep existing integration tests**:

   * `test_graph_service_runtime.py`
     continues to validate “happy path” orchestration.
   * `test_graph_manifest_skip.py`
     continues to validate unchanged detection + manifest behavior.
   * `test_graph_plugin_policy_runtime.py`
     continues to validate severity & retry semantics.

As long as you keep the **public types and behavior equivalent**, these tests will give you confidence that the split didn’t change functionality.

---

## 10. Suggested implementation order (for you / agents)

1. **Create new runtime package and context/model/telemetry modules** (Steps 1–4).

   * Wire up imports and re-exports.
   * Run type-checking + tests that don’t touch execution yet.

2. **Move manifest helpers + planner** (Steps 5–7).

   * Ensure `GraphPluginRunOptions/Record/Report` compile cleanly and tests importing them still pass.

3. **Move isolation/execution primitives** (Step 5).

   * Copy bodies exactly to minimize behavioral drift.

4. **Rewrite `GraphServiceRuntime.run_plugins` against the new APIs** (Step 8).

   * Fix any remaining test failures, leaning on the new unit tests for planning/manifest/execution.

5. **Clean up** any leftover dead imports or helpers in `graph_service_runtime.py`.

---

If you want, next we can do a *pass 2* on this epic: take `run_graph_plugin_batch` and `plan_graph_plugin_run` and show **more concrete bodies** tailored to your existing logic (e.g., where you call `GraphRuntimeScratch`, how you build the `GraphMetricExecutionContext`, exactly how unchanged detection gates execution, etc.).

# zoom in on `run_graph_plugin_batch` and `plan_graph_plugin_run` #

Got you — I’ll treat this as “make the abstract Epic-1 split real”, and show **concrete, drop-in bodies** for:

* `plan_graph_plugin_run(...)` (in `analytics/graphs/runtime/planning.py`)
* `run_graph_plugin_batch(...)` (in `analytics/graphs/runtime/execution.py`)

wired to your **existing** helpers:

* `GraphRuntimeScratch`
* `GraphMetricExecutionContext`
* `_compute_input_hash`, `_compute_options_hash`
* `_effective_severity`, `_effective_timeout`
* `_execute_plugin`, `_run_with_timeout`, isolation worker, etc.
* `_is_unchanged`, `_dry_run_record`, `_skip_record`

and your existing **policy / manifest** semantics (`GraphPluginPolicy.skip_on_unchanged`, `dry_run`, prior manifest row counts).

Below I’ll:

1. Define a **concrete `PluginExecutionSettings` / `PluginExecutionPlan`**.
2. Implement **`plan_graph_plugin_run`** that mirrors what `GraphServiceRuntime.run_plugins` does now, but stops at “plan”.
3. Implement **`run_graph_plugin_batch`** that mirrors the inner plugin loop and calls `_execute_planned_plugin`, which I’ll also include so you can see the **unchanged gating** in one place.

You can adjust module paths/names, but the logic is faithful to your current `graph_service_runtime.py`.

---

## 1. Planning: `analytics/graphs/runtime/planning.py`

This is the “plan only, don’t execute yet” half of `GraphServiceRuntime.run_plugins`.

```python
# analytics/graphs/runtime/planning.py

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal, Mapping, Sequence

from codeintel.analytics.graphs.contracts import ContractChecker
from codeintel.analytics.graphs.plugins import (
    GraphMetricPlugin,
    GraphMetricPluginPlan,
    GraphMetricPluginSkip,
    plan_graph_metric_plugins,
    resolve_plugin_options,
)
from codeintel.analytics.graphs.runtime.telemetry import GraphRuntimeTelemetry
from codeintel.config.steps_graphs import (
    GraphPluginPolicy,
    GraphPluginRetryPolicy,
    GraphRunScope,
)
from codeintel.config.primitives import SnapshotRef

# These come from the module where you moved the helpers from
# graph_service_runtime.py (or you can keep them here if you prefer).
from codeintel.analytics.graphs.runtime.manifest import (
    compute_input_hash,     # same semantics as old _compute_input_hash
    compute_options_hash,   # same semantics as old _compute_options_hash
)


PluginSeverity = Literal["fatal", "soft_fail", "skip_on_error"]


@dataclass(frozen=True)
class PluginExecutionSettings:
    """Resolved execution policy & hashes for a single plugin."""

    name: str
    severity: PluginSeverity
    retry_cfg: GraphPluginRetryPolicy
    timeout_ms: int | None
    fail_fast: bool
    input_hash: str | None                # may be None if we skip hashing
    options_hash: str | None
    version_hash: str | None
    contract_checkers: tuple[ContractChecker, ...]


@dataclass(frozen=True)
class PluginExecutionPlan:
    """
    Execution plan for a batch of graph metric plugins.

    This is the "logical" part of what GraphServiceRuntime.run_plugins used
    to build inline: plugins, options, policy, manifest, etc.
    """

    # Runtime / manifest metadata
    plan_id: str
    run_id: str
    repo: str
    commit: str

    policy: GraphPluginPolicy
    prior_manifest: Mapping[str, Mapping[str, object]] | None
    telemetry: GraphRuntimeTelemetry
    scope: GraphRunScope

    # Plugin graph + options
    plugins: tuple[GraphMetricPlugin, ...]
    ordered_names: tuple[str, ...]
    skipped_plugins: tuple[GraphMetricPluginSkip, ...]
    dep_graph: dict[str, tuple[str, ...]]

    # Per-plugin execution state
    settings_by_plugin: dict[str, PluginExecutionSettings]
    options_by_plugin: dict[str, object | None]


def _effective_severity(
    plugin: GraphMetricPlugin,
    policy: GraphPluginPolicy,
) -> PluginSeverity:
    """
    Match existing behavior:

    - per-plugin override in policy.severity_overrides
    - else plugin.severity attribute if present
    - else policy.default_severity
    """
    override = policy.severity_overrides.get(plugin.name)
    if override is not None:
        return override
    severity = getattr(plugin, "severity", None)
    return severity if severity is not None else policy.default_severity


def _effective_timeout(
    plugin: GraphMetricPlugin,
    policy: GraphPluginPolicy,
) -> int | None:
    """
    Match existing behavior:

    - per-plugin override in policy.timeouts_ms
    - else plugin.resource_hints.max_runtime_ms if available
    """
    override = policy.timeouts_ms.get(plugin.name)
    if override is not None:
        return override
    hints = getattr(plugin, "resource_hints", None)
    return hints.max_runtime_ms if hints is not None else None


def _resolve_target(
    *,
    cfg_snapshot: SnapshotRef | None,
    explicit_target: tuple[str, str] | None,
    runtime_snapshot: SnapshotRef | None,
) -> tuple[str, str]:
    """
    Same semantics as old _resolve_target:

    - cfg.snapshot wins if present
    - else explicit target (repo, commit)
    - else runtime.options.snapshot
    """
    if cfg_snapshot is not None:
        return cfg_snapshot.repo, cfg_snapshot.commit
    if explicit_target is not None:
        return explicit_target
    if runtime_snapshot is None:
        msg = "Graph runtime missing snapshot; cannot derive repo/commit"
        raise ValueError(msg)
    return runtime_snapshot.repo, runtime_snapshot.commit


def _resolve_plugin_options_map(
    plugins: Sequence[GraphMetricPlugin],
    cfg_options: Mapping[str, dict[str, object]] | None,
    runtime_options: Mapping[str, dict[str, object]] | None,
) -> dict[str, object | None]:
    """
    Exact same semantics as in your current graph_service_runtime:

    - validate that options only refer to selected plugins
    - merge plugin.options_default + cfg + runtime via resolve_plugin_options
    """
    cfg_options = cfg_options or {}
    runtime_options = runtime_options or {}
    allowed = {plugin.name for plugin in plugins}
    unknown = (set(cfg_options) | set(runtime_options)) - allowed
    if unknown:
        message = (
            "Options provided for unknown graph metric plugins: "
            f"{', '.join(sorted(unknown))}"
        )
        raise ValueError(message)

    resolved: dict[str, object | None] = {}
    for plugin in plugins:
        resolved[plugin.name] = resolve_plugin_options(
            plugin,
            cfg_options.get(plugin.name),
            runtime_options.get(plugin.name),
        )
    return resolved


def plan_graph_plugin_run(
    plugin_names: Sequence[str],
    *,
    cfg_snapshot: SnapshotRef | None,
    runtime_snapshot: SnapshotRef | None,
    explicit_target: tuple[str, str] | None,
    policy: GraphPluginPolicy,
    prior_manifest: Mapping[str, Mapping[str, object]] | None,
    telemetry: GraphRuntimeTelemetry,
    scope: GraphRunScope,
    cfg_options: Mapping[str, dict[str, object]] | None,
    runtime_options: Mapping[str, dict[str, object]] | None,
    run_id: str | None = None,
) -> PluginExecutionPlan:
    """
    Build a concrete execution plan for a batch of plugins.

    This is the planning half of what GraphServiceRuntime.run_plugins does now:
    - resolve repo/commit
    - compute run_id, plan_id
    - select and order plugins via plan_graph_metric_plugins
    - resolve per-plugin options & hashes
    - prepare policy + manifest + telemetry context
    """
    # 1) Resolve repo/commit
    repo, commit = _resolve_target(
        cfg_snapshot=cfg_snapshot,
        explicit_target=explicit_target,
        runtime_snapshot=runtime_snapshot,
    )

    # 2) Build plugin graph / ordering
    plugin_plan: GraphMetricPluginPlan = plan_graph_metric_plugins(plugin_names)
    plugins: tuple[GraphMetricPlugin, ...] = plugin_plan.plugins

    # 3) Resolve merged options per plugin
    options_by_plugin = _resolve_plugin_options_map(
        plugins=plugins,
        cfg_options=cfg_options,
        runtime_options=runtime_options,
    )

    # 4) Build per-plugin settings (severity, timeouts, hashes)
    settings_by_plugin: dict[str, PluginExecutionSettings] = {}

    for plugin in plugins:
        options = options_by_plugin.get(plugin.name)
        severity = _effective_severity(plugin, policy)
        retry_cfg = policy.retries.get(plugin.name, GraphPluginRetryPolicy())
        timeout_ms = _effective_timeout(plugin, policy)
        input_hash = compute_input_hash(
            repo=repo,
            commit=commit,
            version_hash=plugin.version_hash,
            plugin_name=plugin.name,
            options=options,
        )
        options_hash = compute_options_hash(options)
        settings_by_plugin[plugin.name] = PluginExecutionSettings(
            name=plugin.name,
            severity=severity,
            retry_cfg=retry_cfg,
            timeout_ms=timeout_ms,
            fail_fast=policy.fail_fast,
            input_hash=input_hash,
            options_hash=options_hash,
            version_hash=plugin.version_hash,
            contract_checkers=plugin.contract_checkers,
        )

    # 5) Attach runtime / manifest metadata
    final_run_id = run_id or uuid.uuid4().hex

    return PluginExecutionPlan(
        plan_id=plugin_plan.plan_id,
        run_id=final_run_id,
        repo=repo,
        commit=commit,
        policy=policy,
        prior_manifest=prior_manifest,
        telemetry=telemetry,
        scope=scope,
        plugins=plugins,
        ordered_names=plugin_plan.ordered_names,
        skipped_plugins=plugin_plan.skipped_plugins,
        dep_graph=plugin_plan.dep_graph,
        settings_by_plugin=settings_by_plugin,
        options_by_plugin=dict(options_by_plugin),
    )
```

### How this maps to the old `GraphServiceRuntime.run_plugins`

The block:

```python
policy = cfg.plugin_policy if cfg is not None else GraphPluginPolicy()
if run_options is not None and run_options.dry_run is not None:
    policy = replace(policy, dry_run=run_options.dry_run)

manifest_path = run_options.manifest_path if run_options is not None else None
prior_manifest = _load_prior_manifest(manifest_path)
resolved_target = _resolve_target(cfg, target, self.runtime)
resolved_scope = ...
plan = plan_graph_metric_plugins(plugin_names)
resolved_options = _resolve_plugin_options_map(...)
```

is now exactly what `plan_graph_plugin_run` does (plus the `GraphPluginPolicy.retries/timeouts` mapping, hashes, etc.) — but **without** executing anything yet.

---

## 2. Execution: `analytics/graphs/runtime/execution.py`

Now the “execute the plan” half: this wraps **scratch**, **GraphMetricExecutionContext**, **isolation**, and **unchanged gating** via `_execute_planned_plugin`.

```python
# analytics/graphs/runtime/execution.py

from __future__ import annotations

import multiprocessing
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from queue import Empty
from typing import Literal, Any, cast

from codeintel.analytics.context import AnalyticsContext
from codeintel.analytics.graph_runtime import (
    GraphRuntime,
    GraphRuntimeOptions,
    resolve_graph_runtime,
)
from codeintel.analytics.graphs.contracts import (
    ContractChecker,
    PluginContractResult,
    run_contract_checkers,
)
from codeintel.analytics.graphs.plugins import (
    GraphMetricExecutionContext,
    GraphMetricPlugin,
    GraphPluginResult,
    GraphRuntimeScratch,
    get_graph_metric_plugin,
    plan_graph_metric_plugins,
    register_graph_metric_plugin,
)
from codeintel.analytics.graphs.runtime.planning import (
    PluginExecutionPlan,
    PluginExecutionSettings,
)
from codeintel.analytics.graphs.runtime.telemetry import GraphRuntimeTelemetry
from codeintel.config.primitives import GraphBackendConfig, SnapshotRef
from codeintel.config.steps_graphs import GraphPluginRetryPolicy, GraphRunScope
from codeintel.storage.gateway import (
    StorageConfig,
    StorageGateway,
    open_gateway,
    open_memory_gateway,
)

from codeintel.analytics.graphs.runtime.manifest import (
    current_row_counts,
    is_unchanged,
    dry_run_record,
    skip_record,
)


@dataclass(frozen=True)
class IsolationEnvelope:
    """Inputs needed to run a plugin in a separate process."""

    plugin_name: str
    plugin: GraphMetricPlugin | None
    repo: str
    commit: str
    options: object | None
    scope: GraphRunScope
    gateway_config: StorageConfig | None
    run_id: str


@dataclass(frozen=True)
class IsolationResult:
    """Outputs returned from an isolated plugin run."""

    status: Literal["succeeded", "failed", "skipped"]
    error: str | None
    contracts: tuple[PluginContractResult, ...]
    row_counts: dict[str, int] | None = None
    input_hash: str | None = None
    options_hash: str | None = None


@dataclass(frozen=True)
class MPContext:
    """Thin wrapper around multiprocessing context for isolation."""

    process_factory: Callable[..., multiprocessing.Process]
    queue_factory: Callable[[], multiprocessing.Queue[IsolationResult]]
    start_method_name: str

    def process(
        self,
        target: Callable[..., object],
        args: tuple[object, ...],
    ) -> multiprocessing.Process:
        return self.process_factory(target=target, args=args)

    def queue(self) -> multiprocessing.Queue[IsolationResult]:
        return self.queue_factory()

    def start_method(self) -> str:
        return self.start_method_name


class PluginFatalError(Exception):
    """
    Fatal plugin failure used to implement `fail_fast`.

    The `.record` attribute always contains the GraphPluginRunRecord that
    describes the failed plugin, so callers can include it in the run report.
    """

    def __init__(self, record: "GraphPluginRunRecord", original: Exception) -> None:
        super().__init__(str(original))
        self.record = record


def _select_mp_context() -> MPContext:
    """
    Prefer forked processes when available so that any in-process plugin
    registration is preserved across forks.
    """
    base_ctx = (
        multiprocessing.get_context("fork")
        if "fork" in multiprocessing.get_all_start_methods()
        else multiprocessing.get_context()
    )
    base_ctx_any = cast(Any, base_ctx)
    process_factory = cast(
        Callable[..., multiprocessing.Process],
        base_ctx_any.Process,
    )
    queue_factory = cast(
        Callable[[], multiprocessing.Queue[IsolationResult]],
        base_ctx_any.Queue,
    )
    return MPContext(
        process_factory=process_factory,
        queue_factory=queue_factory,
        start_method_name=base_ctx.get_start_method(),
    )


def _build_gateway_for_isolation(config: StorageConfig | None) -> StorageGateway:
    """Match the old behavior for isolated plugins."""
    if config is None:
        return open_memory_gateway(apply_schema=True, ensure_views=True, validate_schema=True)
    return open_gateway(config)


def _run_isolation_worker(
    envelope: IsolationEnvelope,
    result_queue: multiprocessing.Queue[IsolationResult],
) -> None:
    """
    Worker entrypoint used in a separate process when plugin.requires_isolation=True.

    This reconstructs a minimal GraphRuntime + context and runs the plugin.
    """
    try:
        # Ensure plugin is registered in this process
        if envelope.plugin is not None:
            try:
                get_graph_metric_plugin(envelope.plugin.name)
            except KeyError:
                register_graph_metric_plugin(envelope.plugin)

        plugin_plan = plan_graph_metric_plugins((envelope.plugin_name,))
        plugin = plugin_plan.plugins[0]

        gateway = _build_gateway_for_isolation(envelope.gateway_config)
        snapshot = SnapshotRef(
            repo=envelope.repo,
            commit=envelope.commit,
            repo_root=Path(),
        )
        runtime = resolve_graph_runtime(
            gateway,
            snapshot,
            GraphRuntimeOptions(snapshot=snapshot, backend=GraphBackendConfig()),
        )

        ctx = GraphMetricExecutionContext(
            gateway=gateway,
            runtime=runtime,
            repo=envelope.repo,
            commit=envelope.commit,
            config=None,
            analytics_context=None,
            catalog_provider=None,
            options=envelope.options,
            plugin_name=plugin.name,
            scope=envelope.scope,
            run_id=envelope.run_id,
        )
        plugin_result = plugin.run(ctx)
        coerced = plugin_result if isinstance(plugin_result, GraphPluginResult) else None

        contracts = run_contract_checkers(ctx=ctx, checkers=plugin.contract_checkers)
        result_queue.put(
            IsolationResult(
                status="succeeded",
                error=None,
                contracts=contracts,
                row_counts=coerced.row_counts if coerced is not None else None,
                input_hash=coerced.input_hash if coerced is not None else None,
                options_hash=coerced.options_hash if coerced is not None else None,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        result_queue.put(
            IsolationResult(
                status="failed",
                error=repr(exc),
                contracts=(),
                row_counts=None,
                input_hash=None,
                options_hash=None,
            )
        )


def _run_with_timeout(
    func: Callable[[GraphMetricExecutionContext], object | None],
    ctx: GraphMetricExecutionContext,
    timeout_ms: int | None,
) -> object | None:
    """Unchanged: wrap plugin.run() in a thread with an optional timeout."""
    if timeout_ms is None:
        return func(ctx)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(func, ctx)
        try:
            return future.result(timeout=timeout_ms / 1000)
        except FuturesTimeout as exc:
            future.cancel()
            msg = f"Graph plugin timed out after {timeout_ms} ms"
            raise TimeoutError(msg) from exc


def _execute_plugin_isolated(
    plugin: GraphMetricPlugin,
    ctx: GraphMetricExecutionContext,
    settings: PluginExecutionSettings,
    run_id: str,
) -> "GraphPluginRunRecord":
    """
    Execute a plugin in a separate process and return a GraphPluginRunRecord.

    Mirrors your current _execute_plugin_isolated implementation.
    """
    start = time.perf_counter()
    started_at = datetime.now(tz=UTC)
    mp_ctx = _select_mp_context()

    envelope = IsolationEnvelope(
        plugin_name=plugin.name,
        plugin=plugin if mp_ctx.start_method() != "fork" else None,
        repo=ctx.repo,
        commit=ctx.commit,
        options=ctx.options,
        scope=ctx.scope,
        gateway_config=getattr(ctx.gateway, "config", None),
        run_id=run_id,
    )

    result_queue: multiprocessing.Queue[IsolationResult] = mp_ctx.queue()
    proc = mp_ctx.process(target=_run_isolation_worker, args=(envelope, result_queue))
    proc.start()
    proc.join(timeout=(settings.timeout_ms / 1000) if settings.timeout_ms is not None else None)

    result: IsolationResult | None = None
    status: Literal["succeeded", "failed", "skipped"] = "failed"
    error_message: str | None = None
    hashes: dict[str, str | None] = {
        "input": settings.input_hash,
        "options": settings.options_hash,
    }
    row_counts: dict[str, int] | None = None

    if proc.is_alive():
        proc.terminate()
        proc.join()
        error_message = "timeout"
    else:
        try:
            result = result_queue.get_nowait()
        except Empty:
            result = None

        if result is None:
            status = "failed"
            error_message = "no_result"
        elif result.error is not None:
            status = "failed"
            error_message = result.error
        else:
            status = result.status
            hashes["input"] = result.input_hash or hashes["input"]
            hashes["options"] = result.options_hash or hashes["options"]
            row_counts = result.row_counts

    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    partial = status != "succeeded"

    from codeintel.analytics.graph_service_runtime import GraphPluginRunRecord  # or move dataclass

    record = GraphPluginRunRecord(
        name=plugin.name,
        stage=plugin.stage,
        severity=settings.severity,
        status=status,
        attempts=1,
        timeout_ms=settings.timeout_ms,
        started_at=started_at,
        ended_at=datetime.now(tz=UTC),
        duration_ms=duration_ms,
        partial=partial,
        run_id=run_id,
        error=error_message,
        options=ctx.options,
        input_hash=hashes["input"],
        options_hash=hashes["options"],
        version_hash=settings.version_hash,
        skipped_reason=None,
        row_counts=row_counts,
        contracts=result.contracts if result is not None else (),
        requires_isolation=True,
        isolation_kind=plugin.isolation_kind,
        policy_fail_fast=settings.fail_fast,
    )

    if status == "failed" and settings.severity == "fatal" and settings.fail_fast:
        raise PluginFatalError(record, RuntimeError(error_message or "isolation failure"))
    return record


def _execute_plugin(
    plugin: GraphMetricPlugin,
    ctx: GraphMetricExecutionContext,
    settings: PluginExecutionSettings,
    run_id: str,
) -> "GraphPluginRunRecord":
    """
    Non-isolated plugin execution with retry + timeout + contract handling.

    This is a direct lift of your existing _execute_plugin, wired to
    PluginExecutionSettings instead of _PluginExecutionSettings.
    """
    from codeintel.analytics.graph_service_runtime import GraphPluginRunRecord  # or move dataclass

    if plugin.requires_isolation:
        return _execute_plugin_isolated(plugin, ctx, settings, run_id)

    start = time.perf_counter()
    started_at = datetime.now(tz=UTC)
    attempts = 0
    status: Literal["succeeded", "failed", "skipped"] = "succeeded"
    error_message: str | None = None
    plugin_result: GraphPluginResult | None = None

    while attempts < max(settings.retry_cfg.max_attempts, 1):
        attempts += 1
        try:
            plugin_result = plugin.run(ctx)
            if not isinstance(plugin_result, GraphPluginResult):
                plugin_result = None
            status = "succeeded"
            error_message = None
            break
        except Exception as exc:  # noqa: BLE001
            error_message = repr(exc)
            if settings.severity == "skip_on_error":
                status = "skipped"
                break
            if attempts < max(settings.retry_cfg.max_attempts, 1):
                if settings.retry_cfg.backoff_ms > 0:
                    time.sleep(settings.retry_cfg.backoff_ms / 1000)
                continue
            status = "failed"
            if settings.severity == "fatal" and settings.fail_fast:
                record = GraphPluginRunRecord(
                    name=plugin.name,
                    stage=plugin.stage,
                    severity=settings.severity,
                    status=status,
                    attempts=attempts,
                    timeout_ms=settings.timeout_ms,
                    started_at=started_at,
                    ended_at=datetime.now(tz=UTC),
                    duration_ms=round((time.perf_counter() - start) * 1000, 2),
                    partial=True,
                    run_id=run_id,
                    error=error_message,
                    options=ctx.options,
                    input_hash=settings.input_hash,
                    options_hash=settings.options_hash,
                    version_hash=settings.version_hash,
                    contracts=(),
                    policy_fail_fast=settings.fail_fast,
                )
                raise PluginFatalError(record, exc) from exc
            break

    # Contract checks
    contracts = run_contract_checkers(ctx=ctx, checkers=settings.contract_checkers) if status == "succeeded" else ()
    contract_statuses = {c.status for c in contracts}

    input_hash = (
        plugin_result.input_hash
        if plugin_result is not None and plugin_result.input_hash is not None
        else settings.input_hash
    )
    options_hash = (
        plugin_result.options_hash
        if plugin_result is not None and plugin_result.options_hash is not None
        else settings.options_hash
    )
    row_counts = plugin_result.row_counts if plugin_result is not None else None

    if status == "succeeded" and ("failed" in contract_statuses or "soft_failed" in contract_statuses):
        status = "failed"
        error_message = "contract_failed"
        if "failed" in contract_statuses and settings.severity == "fatal" and settings.fail_fast:
            record = GraphPluginRunRecord(
                name=plugin.name,
                stage=plugin.stage,
                severity=settings.severity,
                status=status,
                attempts=attempts,
                timeout_ms=settings.timeout_ms,
                started_at=started_at,
                ended_at=datetime.now(tz=UTC),
                duration_ms=round((time.perf_counter() - start) * 1000, 2),
                partial=True,
                run_id=run_id,
                error=error_message,
                options=ctx.options,
                input_hash=input_hash,
                options_hash=options_hash,
                version_hash=settings.version_hash,
                skipped_reason=None,
                row_counts=row_counts,
                contracts=contracts,
                requires_isolation=plugin.requires_isolation,
                isolation_kind=plugin.isolation_kind,
                policy_fail_fast=settings.fail_fast,
            )
            raise PluginFatalError(record, RuntimeError("Contract failure"))

    ended_at = datetime.now(tz=UTC)
    return GraphPluginRunRecord(
        name=plugin.name,
        stage=plugin.stage,
        severity=settings.severity,
        status=status,
        attempts=attempts,
        timeout_ms=settings.timeout_ms,
        started_at=started_at,
        ended_at=ended_at,
        duration_ms=round((time.perf_counter() - start) * 1000, 2),
        partial=status != "succeeded",
        run_id=run_id,
        error=error_message,
        options=ctx.options,
        input_hash=input_hash,
        options_hash=options_hash,
        version_hash=settings.version_hash,
        skipped_reason=None,
        row_counts=row_counts,
        contracts=contracts,
        requires_isolation=plugin.requires_isolation,
        isolation_kind=plugin.isolation_kind,
        policy_fail_fast=settings.fail_fast,
    )


def _execute_planned_plugin(
    plugin: GraphMetricPlugin,
    ctx: GraphMetricExecutionContext,
    settings: PluginExecutionSettings,
    plan: PluginExecutionPlan,
) -> "GraphPluginRunRecord":
    """
    This is where **unchanged detection gates execution**:

    - If policy.dry_run -> always SKIPPED via dry_run_record(...)
    - Else if policy.skip_on_unchanged and is_unchanged(...) -> SKIPPED
    - Else -> execute plugin (possibly in isolation) via _execute_plugin(...)
    """
    from codeintel.analytics.graph_service_runtime import GraphPluginRunRecord

    span = plan.telemetry.start_plugin(plugin, plan.run_id, ctx)

    # Dry-run and unchanged gating
    if plan.policy.dry_run:
        record = dry_run_record(plugin, settings, ctx.options, plan.run_id)
    elif plan.policy.skip_on_unchanged and is_unchanged(
        prior_manifest=plan.prior_manifest or {},
        plugin=plugin,
        settings=settings,
        ctx=ctx,
    ):
        record = skip_record(
            plugin=plugin,
            settings=settings,
            options=ctx.options,
            reason="unchanged",
            run_id=plan.run_id,
        )
    else:
        record = _execute_plugin(plugin, ctx, settings, plan.run_id)

    plan.telemetry.finish_plugin(span, record)
    plan.telemetry.record_metrics(record, plan.scope)
    return record


def run_graph_plugin_batch(
    *,
    plan: PluginExecutionPlan,
    gateway: StorageGateway,
    runtime: GraphRuntime,
    cfg: "GraphMetricsStepConfig | None",
    analytics_context: AnalyticsContext | None,
    catalog_provider: "FunctionCatalogProvider | None",
) -> list["GraphPluginRunRecord"]:
    """
    Execute all plugins in a PluginExecutionPlan and return run records.

    This is the loop that used to live inline in GraphServiceRuntime.run_plugins:

        records = []
        scratch = GraphRuntimeScratch()
        for plugin in plugins:
            try:
                record = _run_single_plugin(plugin)
            except _PluginFatalError as exc:
                records.append(exc.record)
                raise
            records.append(record)

    Now lifted into a reusable helper.
    """
    from codeintel.analytics.graph_service_runtime import GraphPluginRunRecord

    records: list[GraphPluginRunRecord] = []
    scratch = GraphRuntimeScratch()

    def _run_single_plugin(plugin: GraphMetricPlugin) -> GraphPluginRunRecord:
        settings = plan.settings_by_plugin[plugin.name]
        options = plan.options_by_plugin.get(plugin.name)

        plugin_ctx = GraphMetricExecutionContext(
            gateway=gateway,
            runtime=runtime,
            repo=plan.repo,
            commit=plan.commit,
            config=cfg,
            analytics_context=analytics_context,
            catalog_provider=catalog_provider,
            options=options,
            plugin_name=plugin.name,
            scope=plan.scope,
            run_id=plan.run_id,
            scratch=scratch,
        )
        return _execute_planned_plugin(plugin, plugin_ctx, settings, plan)

    try:
        for plugin in plan.plugins:
            try:
                record = _run_single_plugin(plugin)
            except PluginFatalError as exc:
                records.append(exc.record)
                raise
            records.append(record)
    finally:
        scratch.cleanup()

    return records
```

### Where the unchanged gating actually happens

The gating is entirely in `_execute_planned_plugin` (same as your current code):

```python
if plan.policy.dry_run:
    record = dry_run_record(...)
elif plan.policy.skip_on_unchanged and is_unchanged(...):
    record = skip_record(..., reason="unchanged", ...)
else:
    record = _execute_plugin(...)
```

And `is_unchanged(...)` matches your existing semantics:

* Use `settings.input_hash` and `settings.options_hash` and the prior manifest record for this plugin.
* Only consider plugins where the prior manifest status was `"succeeded"`.
* If `plugin.row_count_tables` is empty → unchanged is based only on hashes.
* Else:

  * Load `prior_rows = prior["row_counts"]`.
  * Compute `current_rows` with `current_row_counts(gateway, plugin.row_count_tables, repo, commit)`.
  * If `current_rows` is None, treat as unchanged (skip safe).
  * Else compare via `row_counts_equal(current_rows, prior_rows)`.

So your “skip on unchanged” policy literally means:

> “Inputs (including plugin version + options) match and the table row counts for this repo+commit haven’t changed ⇒ skip the plugin.”

---

If you’d like, next step I can:

* Show exactly how to **rewrite `GraphServiceRuntime.run_plugins`** to call these two helpers, or
* Help you refactor the `is_unchanged` + row-count helpers into the dedicated `manifest.py` you sketched in Epic 1 so analytics and non-graph plugins can reuse the same pattern.

# zoom in on `is_unchanged` + row-count helpers into the dedicated `manifest.py` #

Perfect, let’s make `manifest.py` real and **pull the “unchanged detection” + row-count logic into a generic module** you can use for graphs *and* future non-graph analytics.

I’ll do this in three pieces:

1. **Design & implement `analytics/graphs/runtime/manifest.py`** (generic hashing, row-counts, `is_unchanged`, manifest load/write).
2. **Show how to wire graphs to it** (updates to `planning.py` and `execution.py` from the previous answer).
3. **Explain how non-graph analytics plugins can reuse it** (what you’d pass in from a future `AnalyticsPlugin` harness).

I’ll reuse your *existing semantics* as directly as possible so behavior stays the same.

---

## 1. New `analytics/graphs/runtime/manifest.py`

This module is the shared “manifest + unchanged detection” logic.

It keeps *no* dependency on graph-specific types like `GraphMetricPlugin` or `GraphMetricExecutionContext`. Instead, functions take primitives (`plugin_name`, `row_count_tables`, `repo`, `commit`, `input_hash`, `options_hash`) so you can call them from any analytics plugin harness.

```python
# analytics/graphs/runtime/manifest.py

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Mapping, Sequence, Any

from codeintel.config.steps_graphs import GraphRunScope
from codeintel.storage.gateway import StorageGateway


# ---------- Generic hashing helpers ----------


def hash_json(payload: object) -> str:
    """
    Stable JSON hash helper.

    * Uses sort_keys=True and default=str so it's deterministic.
    * Used for input/options hashes and scope hashes.
    """
    serialized = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def compute_options_hash(options: object | None) -> str | None:
    """
    Compute a stable hash for plugin options payloads.

    Returns a hex digest or None if options is None.
    """
    if options is None:
        return None
    return hash_json(options)


def compute_input_hash(
    *,
    repo: str,
    commit: str,
    plugin_name: str,
    version_hash: str | None,
    options: object | None,
    extra_scope: object | None = None,
) -> str:
    """
    Compute a stable hash for plugin *inputs*.

    Mirrors your existing `_compute_input_hash` but allows an optional
    `extra_scope` field for future reuse (e.g. graph scope, history scope).
    """
    parts: dict[str, object | None] = {
        "repo": repo,
        "commit": commit,
        "plugin": plugin_name,
        "version_hash": version_hash or "0",
        "options": options,
    }
    if extra_scope is not None:
        parts["scope"] = extra_scope
    serialized = json.dumps(parts, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


# ---------- Row-count helpers ----------


def row_counts_equal(
    current: dict[str, int] | None,
    prior: object,
) -> bool:
    """
    Compare row-count dictionaries, tolerating minor shape mismatches.

    This is a direct lift of your existing `_row_counts_equal`.
    """
    if current is None:
        return True
    if not isinstance(prior, dict):
        return True
    for table, count in current.items():
        if prior.get(table) != count:
            return False
    for table, count in prior.items():
        if not isinstance(count, int):
            return False
        if table not in current or current[table] != count:
            return False
    return True


def current_row_counts(
    gateway: StorageGateway | None,
    tables: Sequence[str],
    *,
    repo: str,
    commit: str,
) -> dict[str, int] | None:
    """
    Compute row counts for tables scoped by (repo, commit).

    Exactly the same logic as your `_current_row_counts`, but generic and
    parameterized on repo/commit.
    """
    if gateway is None or not tables:
        return None

    counts: dict[str, int] = {}
    connection = getattr(gateway, "con", None)
    if connection is None:
        return None

    for table in tables:
        try:
            escaped_repo = repo.replace("'", "''")
            escaped_commit = commit.replace("'", "''")
            relation = connection.table(table).filter(
                f"repo = '{escaped_repo}' AND commit = '{escaped_commit}'"
            )
            count = relation.aggregate("count(*)").fetchone()[0]
            counts[table] = int(count)
        except Exception:  # noqa: BLE001 - defensive, should not block skip logic
            return None
    return counts


# ---------- Manifest load/write helpers ----------


def load_prior_manifest(path: Path | None) -> dict[str, dict[str, object]] | None:
    """
    Load a plugin-manifest JSON and return a mapping name -> record dict.

    Same semantics as your `_load_prior_manifest`.
    """
    if path is None or not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        return {
            record.get("name", "unknown"): record  # type: ignore[union-attr]
            for record in records
            if isinstance(record, dict)
        }
    except (OSError, json.JSONDecodeError):
        return None


def _report_to_payload(
    *,
    repo: str,
    commit: str,
    run_id: str,
    plan_id: str,
    ordered_plugins: Sequence[str],
    skipped_plugins: Sequence[dict[str, object]],
    dep_graph: Mapping[str, Sequence[str]],
    scope: GraphRunScope,
    records: Sequence[dict[str, object]],
) -> dict[str, object]:
    """
    Generic manifest encoder.

    Instead of hard-coding GraphPluginRunRecord here, we accept pre-built
    dicts for records and skipped plugins so this can be reused by other
    analytics runtimes.
    """
    return {
        "repo": repo,
        "commit": commit,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "run_id": run_id,
        "plan": {
            "plan_id": plan_id,
            "ordered_plugins": list(ordered_plugins),
            "skipped_plugins": list(skipped_plugins),
            "dep_graph": {name: list(deps) for name, deps in dep_graph.items()},
        },
        "scope": {
            "paths": list(scope.paths),
            "modules": list(scope.modules),
            "time_window": (
                (
                    scope.time_window[0].isoformat(),
                    scope.time_window[1].isoformat(),
                )
                if scope.time_window is not None
                else None
            ),
        },
        "records": list(records),
    }


def write_manifest(path: Path, payload: dict[str, object]) -> None:
    """
    Write a manifest payload to disk.

    Graph runtime can build the payload via `_report_to_payload` and call
    this function; other analytics runtimes can do the same with their own
    record format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


# ---------- Unchanged detection ----------


def is_unchanged(
    *,
    prior_manifest: Mapping[str, Mapping[str, object]],
    plugin_name: str,
    row_count_tables: Sequence[str],
    gateway: StorageGateway | None,
    repo: str,
    commit: str,
    input_hash: str | None,
    options_hash: str | None,
) -> bool:
    """
    Generic "unchanged input" test used by analytics runtimes.

    Matches the behavior of your original `_is_unchanged`:

    - Look up prior manifest record by plugin_name.
    - Require:
        * current input_hash is not None
        * prior.record.status == "succeeded"
        * prior.input_hash == current input_hash
        * if options_hash is not None, prior.options_hash == options_hash
    - If plugin doesn't specify row_count_tables => unchanged.
    - Else, compare row counts in DB vs prior.row_counts via row_counts_equal().
    """
    prior = prior_manifest.get(plugin_name)
    if input_hash is None or prior is None:
        return False
    if prior.get("status") != "succeeded":
        return False
    if prior.get("input_hash") != input_hash:
        return False
    if options_hash is not None and prior.get("options_hash") != options_hash:
        return False

    # No row-count tables => rely purely on hashes
    if not row_count_tables:
        return True

    prior_rows = prior.get("row_counts")
    if prior_rows is None:
        return True

    current_rows = current_row_counts(
        gateway=gateway,
        tables=row_count_tables,
        repo=repo,
        commit=commit,
    )
    if current_rows is None:
        return True
    return row_counts_equal(current_rows, prior_rows)
```

> For now I kept `_report_to_payload` as a “helper” that graphs will use. If you later have an analytics-wide `AnalyticsRunReport`, you can move this up a level and have each runtime call it with its own record dicts.

---

## 2. Wiring graphs to the new `manifest.py`

Now we update the **graph runtime planning & execution** to use `manifest.py` and delete the old helpers from `graph_service_runtime.py`.

### 2.1 `planning.py`: use `compute_input_hash` / `compute_options_hash`

In the planning code from the previous message, update imports and the actual call:

```python
# analytics/graphs/runtime/planning.py

from codeintel.analytics.graphs.runtime.manifest import (
    compute_input_hash,
    compute_options_hash,
)

# ...

for plugin in plugins:
    options = options_by_plugin.get(plugin.name)
    severity = _effective_severity(plugin, policy)
    retry_cfg = policy.retries.get(plugin.name, GraphPluginRetryPolicy())
    timeout_ms = _effective_timeout(plugin, policy)

    input_hash = compute_input_hash(
        repo=repo,
        commit=commit,
        plugin_name=plugin.name,
        version_hash=plugin.version_hash,
        options=options,
        extra_scope=None,  # or pass a serialized form of GraphRunScope if desired
    )
    options_hash = compute_options_hash(options)

    settings_by_plugin[plugin.name] = PluginExecutionSettings(
        name=plugin.name,
        severity=severity,
        retry_cfg=retry_cfg,
        timeout_ms=timeout_ms,
        fail_fast=policy.fail_fast,
        input_hash=input_hash,
        options_hash=options_hash,
        version_hash=plugin.version_hash,
        contract_checkers=plugin.contract_checkers,
    )
```

This replaces your old `_compute_input_hash` / `_compute_options_hash` calls.

### 2.2 `execution.py`: use `is_unchanged` and `current_row_counts`

In `_execute_planned_plugin` we replace the inline logic with a call to `manifest.is_unchanged`:

```python
# analytics/graphs/runtime/execution.py

from codeintel.analytics.graphs.runtime.manifest import (
    is_unchanged,
)

# ...

def _execute_planned_plugin(
    plugin: GraphMetricPlugin,
    ctx: GraphMetricExecutionContext,
    settings: PluginExecutionSettings,
    plan: PluginExecutionPlan,
) -> GraphPluginRunRecord:
    span = plan.telemetry.start_plugin(plugin, plan.run_id, ctx)

    if plan.policy.dry_run:
        record = dry_run_record(plugin, settings, ctx.options, plan.run_id)
    elif plan.policy.skip_on_unchanged and is_unchanged(
        prior_manifest=plan.prior_manifest or {},
        plugin_name=plugin.name,
        row_count_tables=plugin.row_count_tables,
        gateway=ctx.gateway,
        repo=ctx.repo,
        commit=ctx.commit,
        input_hash=settings.input_hash,
        options_hash=settings.options_hash,
    ):
        record = skip_record(
            plugin=plugin,
            settings=settings,
            options=ctx.options,
            reason="unchanged",
            run_id=plan.run_id,
        )
    else:
        record = _execute_plugin(plugin, ctx, settings, plan.run_id)

    plan.telemetry.finish_plugin(span, record)
    plan.telemetry.record_metrics(record, plan.scope)
    return record
```

Now `execution.py` no longer needs its own `_row_counts_equal`, `_current_row_counts`, `_is_unchanged`.

### 2.3 `graph_service_runtime.py`: use `load_prior_manifest` / `write_manifest`

Inside `GraphServiceRuntime.run_plugins`, swap your old `_load_prior_manifest` / `_write_plugin_manifest` with the new helpers:

```python
# analytics/graph_service_runtime.py

from codeintel.analytics.graphs.runtime.manifest import (
    load_prior_manifest,
    write_manifest,
    _report_to_payload,  # or re-export a wrapper for graphs
)

# ...

manifest_path = run_options.manifest_path if run_options is not None else None
prior_manifest = load_prior_manifest(manifest_path)

# Build plan...
plan = plan_graph_plugin_run(
    plugin_names=plugin_names,
    cfg_snapshot=cfg.snapshot if cfg is not None else None,
    runtime_snapshot=self.runtime.options.snapshot,
    explicit_target=target,
    policy=policy,
    prior_manifest=prior_manifest,
    telemetry=self.telemetry or OtelGraphRuntimeTelemetry(),
    scope=resolved_scope,
    cfg_options=cfg.plugin_options if cfg is not None else {},
    runtime_options=(run_options.plugin_options if run_options is not None else {}) or {},
)

# Execute...
records = run_graph_plugin_batch(
    plan=plan,
    gateway=self.gateway,
    runtime=self.runtime,
    cfg=cfg,
    analytics_context=self.analytics_context,
    catalog_provider=self.catalog_provider,
)

report = GraphPluginRunReport(
    repo=plan.repo,
    commit=plan.commit,
    records=tuple(records),
    scope=plan.scope,
    run_id=plan.run_id,
    plan_id=plan.plan_id,
    ordered_plugins=plan.ordered_names,
    skipped_plugins=plan.skipped_plugins,
    dep_graph=plan.dep_graph,
)

if manifest_path is not None:
    payload = _report_to_payload(
        repo=report.repo,
        commit=report.commit,
        run_id=report.run_id,
        plan_id=report.plan_id,
        ordered_plugins=report.ordered_plugins,
        skipped_plugins=[
            {"name": s.name, "reason": s.reason} for s in report.skipped_plugins
        ],
        dep_graph=report.dep_graph,
        scope=report.scope,
        records=[
            {
                "name": r.name,
                "stage": r.stage,
                "severity": r.severity,
                "status": r.status,
                "attempts": r.attempts,
                "timeout_ms": r.timeout_ms,
                "started_at": r.started_at.isoformat(),
                "ended_at": r.ended_at.isoformat(),
                "duration_ms": r.duration_ms,
                "partial": r.partial,
                "run_id": r.run_id,
                "error": r.error,
                "input_hash": r.input_hash,
                "options_hash": r.options_hash,
                "version_hash": r.version_hash,
                "skipped_reason": r.skipped_reason,
                "row_counts": r.row_counts,
                "requires_isolation": r.requires_isolation,
                "isolation_kind": r.isolation_kind,
                "contracts": [
                    {
                        "name": c.name,
                        "status": c.status,
                        "message": c.message,
                    }
                    for c in r.contracts
                ],
                "policy_fail_fast": r.policy_fail_fast,
            }
            for r in report.records
        ],
    )
    write_manifest(manifest_path, payload)
```

At this point:

* `graph_service_runtime.py` has **no** `_compute_input_hash`, `_compute_options_hash`, `_row_counts_equal`, `_current_row_counts`, `_is_unchanged`, `_load_prior_manifest`, or `_write_plugin_manifest`.
* They all live in `manifest.py` and are reusable.

---

## 3. Reusing the pattern for non-graph analytics plugins

Once you introduce a generic `AnalyticsPlugin` (from Epic 2), the **exact same manifest logic** applies:

* Each plugin has:

  * `name: str`
  * `row_count_tables: Sequence[str]` (or similar)
  * `version_hash: str | None`
  * plugin-specific options.

* Each execution plan has:

  * `repo`, `commit`
  * `prior_manifest: Mapping[str, Mapping[str, object]] | None`
  * settings with `input_hash`, `options_hash`.

Then your non-graph harness can do:

```python
from codeintel.analytics.runtime.manifest import (
    compute_input_hash,
    compute_options_hash,
    is_unchanged,
)

# planning:
for plugin in plugins:
    options = options_by_plugin.get(plugin.name)
    input_hash = compute_input_hash(
        repo=repo,
        commit=commit,
        plugin_name=plugin.name,
        version_hash=plugin.version_hash,
        options=options,
        extra_scope=None,  # or a serialized AnalyticsRunScope
    )
    options_hash = compute_options_hash(options)
    # store in PluginExecutionSettings...

# execution:
if policy.skip_on_unchanged and is_unchanged(
    prior_manifest=plan.prior_manifest or {},
    plugin_name=plugin.name,
    row_count_tables=plugin.row_count_tables,
    gateway=ctx.gateway,
    repo=ctx.repo,
    commit=ctx.commit,
    input_hash=settings.input_hash,
    options_hash=settings.options_hash,
):
    record = make_skip_record(...)   # your non-graph variant
else:
    record = execute_plugin(...)
```

All the “hard part” of unchanged detection (hashing, manifest read, row count comparison) is **now shared**; the only per-domain differences are:

* What a “run record” looks like (graph vs function vs test).
* How you build row-count tables for each plugin (the `row_count_tables` attribute).

---

If you’d like, I can next:

* Draft a **generic `AnalyticsManifest` abstraction** (e.g. `AnalyticsRunReport` + manifest encoder) you can use for both graphs and non-graphs, or
* Show a **minimal patch diff** (e.g. `git diff`-style chunks) for the three files we’ve touched: `manifest.py`, `planning.py`, `execution.py` / `graph_service_runtime.py`.


# draft of a **generic `AnalyticsManifest` abstraction** (e.g. `AnalyticsRunReport` + manifest encoder) #

Here’s a concrete way to introduce a **generic analytics manifest** that works for graphs *and* future non-graph analytics, without blowing up what you already have.

I’ll give you:

1. A new module `analytics/runtime_manifest.py` with:

   * `AnalyticsScope`
   * `AnalyticsSkippedStep`
   * `AnalyticsPlanInfo`
   * `AnalyticsRunRecord`
   * `AnalyticsRunReport`
   * `encode_manifest(report) -> dict`
2. A helper to map a **graph** run (`GraphPluginRunReport`) into `AnalyticsRunReport`, so graphs can “opt in” with almost no disruption.
3. Notes on how non-graph analytics would plug in.

---

## 1. Generic manifest types: `analytics/runtime_manifest.py`

This is the “unifying” abstraction. It doesn’t know anything about graphs, functions, tests, etc; it just knows “steps in a run”.

```python
# analytics/runtime_manifest.py

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Mapping, Sequence, Literal


@dataclass(frozen=True)
class AnalyticsScope:
    """
    Generic analytics "scope" for a run.

    It intentionally looks like GraphRunScope but is decoupled from graphs,
    so other runtimes can add their own semantics via `labels`.
    """

    paths: tuple[str, ...] = ()
    modules: tuple[str, ...] = ()
    time_window: tuple[datetime, datetime] | None = None
    labels: Mapping[str, str] = field(default_factory=dict)
    # e.g. {"runtime": "graph"} or {"runtime": "functions", "subset": "subsystem:foo"}


@dataclass(frozen=True)
class AnalyticsSkippedStep:
    """Reasoned skip for a single step / plugin in a run."""

    name: str
    reason: str
    kind: str | None = None  # "graph_plugin", "function_analytics", etc.


@dataclass(frozen=True)
class AnalyticsPlanInfo:
    """
    Optional planning metadata for a run.

    This mirrors the "plan" section in your existing graph manifests.
    """

    plan_id: str | None = None
    ordered_steps: tuple[str, ...] = ()
    skipped_steps: tuple[AnalyticsSkippedStep, ...] = ()
    dep_graph: Mapping[str, tuple[str, ...]] = field(default_factory=dict)


AnalyticsStatus = Literal["succeeded", "failed", "skipped"]


@dataclass(frozen=True)
class AnalyticsRunRecord:
    """
    Per-step execution record.

    Everything runtime-specific (severity, hashes, contracts, row_counts,
    isolation, etc.) goes into `meta`.
    """

    name: str
    kind: str                          # "graph_plugin", "function_profile", ...
    status: AnalyticsStatus
    started_at: datetime
    ended_at: datetime
    duration_ms: float
    attempts: int = 1
    partial: bool = False
    error: str | None = None
    meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AnalyticsRunReport:
    """
    Generic manifest-ready view of an analytics run.

    Any runtime (graphs, functions, tests) can map its internal run-report
    into this shape, then use `encode_manifest` to produce a JSON-able payload.
    """

    repo: str
    commit: str
    run_id: str
    scope: AnalyticsScope
    records: tuple[AnalyticsRunRecord, ...]
    plan: AnalyticsPlanInfo
    tags: Mapping[str, str] = field(default_factory=dict)  # free-form metadata


def encode_manifest(report: AnalyticsRunReport) -> dict[str, object]:
    """
    Encode an AnalyticsRunReport into a manifest payload.

    The structure matches your current graph manifest closely:
      - top-level repo/commit/run_id/generated_at
      - plan {plan_id, ordered_steps, skipped_steps, dep_graph}
      - scope {paths, modules, time_window, labels}
      - tags (free-form)
      - records[ ... ]
    """
    return {
        "repo": report.repo,
        "commit": report.commit,
        "generated_at": datetime.now(tz=UTC).isoformat(),
        "run_id": report.run_id,
        "plan": {
            "plan_id": report.plan.plan_id,
            "ordered_steps": list(report.plan.ordered_steps),
            "skipped_steps": [
                {
                    "name": skipped.name,
                    "reason": skipped.reason,
                    "kind": skipped.kind,
                }
                for skipped in report.plan.skipped_steps
            ],
            "dep_graph": {
                name: list(deps) for name, deps in report.plan.dep_graph.items()
            },
        },
        "scope": {
            "paths": list(report.scope.paths),
            "modules": list(report.scope.modules),
            "time_window": (
                (
                    report.scope.time_window[0].isoformat(),
                    report.scope.time_window[1].isoformat(),
                )
                if report.scope.time_window is not None
                else None
            ),
            "labels": dict(report.scope.labels),
        },
        "tags": dict(report.tags),
        "records": [
            {
                "name": rec.name,
                "kind": rec.kind,
                "status": rec.status,
                "attempts": rec.attempts,
                "started_at": rec.started_at.isoformat(),
                "ended_at": rec.ended_at.isoformat(),
                "duration_ms": rec.duration_ms,
                "partial": rec.partial,
                "error": rec.error,
                "meta": dict(rec.meta),
            }
            for rec in report.records
        ],
    }
```

This file is **totally graph-agnostic**. Graphs (and later, other analytics runtimes) just need to provide a mapping into this structure.

---

## 2. Mapping graphs → `AnalyticsRunReport`

Now we add a small adapter that turns your existing graph run reporting into this generic form.

I’d put it in something like `analytics/graphs/runtime/analytics_adapter.py`:

```python
# analytics/graphs/runtime/analytics_adapter.py

from __future__ import annotations

from typing import Any, Mapping

from codeintel.analytics.runtime_manifest import (
    AnalyticsScope,
    AnalyticsSkippedStep,
    AnalyticsPlanInfo,
    AnalyticsRunRecord,
    AnalyticsRunReport,
)
from codeintel.analytics.graph_service_runtime import (
    GraphPluginRunRecord,
    GraphPluginRunReport,
)
from codeintel.config.steps_graphs import GraphRunScope


def _scope_from_graph(scope: GraphRunScope) -> AnalyticsScope:
    """
    Convert GraphRunScope → AnalyticsScope.

    Labels are where we mark this as a graph run (and potentially more detail).
    """
    return AnalyticsScope(
        paths=scope.paths,
        modules=scope.modules,
        time_window=scope.time_window,
        labels={"runtime": "graph"},
    )


def _plan_from_graph(report: GraphPluginRunReport) -> AnalyticsPlanInfo:
    """
    Convert the plan section of GraphPluginRunReport into AnalyticsPlanInfo.
    """
    return AnalyticsPlanInfo(
        plan_id=report.plan_id,
        ordered_steps=report.ordered_plugins,
        skipped_steps=tuple(
            AnalyticsSkippedStep(
                name=skipped.name,
                reason=skipped.reason,
                kind="graph_plugin",
            )
            for skipped in report.skipped_plugins
        ),
        dep_graph=report.dep_graph,
    )


def _meta_from_graph_record(rec: GraphPluginRunRecord) -> dict[str, Any]:
    """
    Pack graph-specific details into the generic `meta` field.

    This keeps AnalyticsRunRecord stable while letting graph evolve.
    """
    return {
        # useful for debugging / analysis
        "stage": rec.stage,
        "severity": rec.severity,
        "timeout_ms": rec.timeout_ms,
        "input_hash": rec.input_hash,
        "options_hash": rec.options_hash,
        "version_hash": rec.version_hash,
        "skipped_reason": rec.skipped_reason,
        "row_counts": rec.row_counts,
        "requires_isolation": rec.requires_isolation,
        "isolation_kind": rec.isolation_kind,
        "policy_fail_fast": rec.policy_fail_fast,
        # whether we actually had options at all
        "options_present": rec.options is not None,
        # contracts as lightweight dicts
        "contracts": [
            {
                "name": c.name,
                "status": c.status,
                "message": c.message,
            }
            for c in rec.contracts
        ],
    }


def graph_run_to_analytics(report: GraphPluginRunReport) -> AnalyticsRunReport:
    """
    Adapter: GraphPluginRunReport → AnalyticsRunReport.

    After this, `encode_manifest(...)` + `write_manifest(...)` can be used
    just like any other analytics runtime.
    """
    scope = _scope_from_graph(report.scope)
    plan = _plan_from_graph(report)

    records = tuple(
        AnalyticsRunRecord(
            name=rec.name,
            kind="graph_plugin",
            status=rec.status,  # Literal-compatible
            started_at=rec.started_at,
            ended_at=rec.ended_at,
            duration_ms=rec.duration_ms,
            attempts=rec.attempts,
            partial=rec.partial,
            error=rec.error,
            meta=_meta_from_graph_record(rec),
        )
        for rec in report.records
    )

    return AnalyticsRunReport(
        repo=report.repo,
        commit=report.commit,
        run_id=report.run_id,
        scope=scope,
        records=records,
        plan=plan,
        tags={"runtime": "graph"},
    )
```

With that in place, **graph manifest writing becomes:**

```python
# inside GraphServiceRuntime.run_plugins, after you build `report: GraphPluginRunReport`

from codeintel.analytics.runtime_manifest import encode_manifest
from codeintel.analytics.graphs.runtime.analytics_adapter import graph_run_to_analytics
from codeintel.analytics.graphs.runtime.manifest import write_manifest  # low-level IO

if manifest_path is not None:
    analytics_report = graph_run_to_analytics(report)
    payload = encode_manifest(analytics_report)
    write_manifest(manifest_path, payload)
```

So the “graph world” still uses `GraphPluginRunReport` internally, but the thing you put on disk / ship around is now **analytics-generic**.

---

## 3. How non-graph analytics would reuse this

Once you introduce a generic `AnalyticsPlugin` + harness (Epic 2), each runtime can:

1. Produce its own internal run-report (e.g. `FunctionAnalyticsRunReport`).
2. Map it into `AnalyticsRunReport` with the same pattern as `graph_run_to_analytics`:

   * Decide its own `kind` string (`"function_profile"`, `"test_profile"`, etc.).
   * Map its own scope into `AnalyticsScope` (you can reuse `paths/modules/time_window` and add e.g. `labels={"runtime": "functions", "subsystem": "foo"}`).
   * Pack runtime-specific detail into `.meta`.

For example, for a function analytics runtime:

```python
# pseudo-code sketch

def function_run_to_analytics(report: FunctionAnalyticsRunReport) -> AnalyticsRunReport:
    scope = AnalyticsScope(
        paths=report.scope.paths,
        modules=report.scope.modules,
        time_window=None,
        labels={"runtime": "functions"},
    )
    plan = AnalyticsPlanInfo(
        plan_id=report.plan_id,
        ordered_steps=report.ordered_steps,
        skipped_steps=tuple(
            AnalyticsSkippedStep(
                name=s.name,
                reason=s.reason,
                kind="function_analytics",
            )
            for s in report.skipped_steps
        ),
        dep_graph=report.dep_graph,
    )
    records = tuple(
        AnalyticsRunRecord(
            name=r.name,
            kind="function_analytics",
            status=r.status,
            started_at=r.started_at,
            ended_at=r.ended_at,
            duration_ms=r.duration_ms,
            attempts=r.attempts,
            partial=r.partial,
            error=r.error,
            meta={
                "subsystem": r.subsystem,
                "functions_affected": r.functions_affected,
                "rows_written": r.rows_written,
                # ...anything else useful...
            },
        )
        for r in report.records
    )
    return AnalyticsRunReport(
        repo=report.repo,
        commit=report.commit,
        run_id=report.run_id,
        scope=scope,
        records=records,
        plan=plan,
        tags={"runtime": "functions"},
    )
```

Then:

* **All** analytics manifests (graphs, functions, tests, subsystems) have the same top-level JSON shape.
* Tooling / UIs / MCP / agents can:

  * Load any manifest.
  * Read `tags.runtime` to see where it came from.
  * Iterate over records and inspect `meta` for runtime-specific detail.

In other words: you get a **single, extensible manifest schema** without forcing all runtimes to share identical internal types.

---

