# Hamilton Integration Phase 1 Implementation Report

> **Status**: Phase 1 Complete (Refined)  
> **Date**: December 2024

---

## Executive Summary

Phase 1 transforms Hamilton from a Phase 0 proof-of-concept into a **production-grade, end-to-end orchestration engine** for the entire CodeIntel build graph. This phase delivers:

- **Full DAG coverage** via dynamic node generation for all targets (not just the Phase 0 chain)
- **CLI parity** with the legacy executor (`--force` flag, accurate executed/skipped/failed lists)
- **Production correctness** with upstream failure gating (downstream targets don't run if upstream fails)
- **Run tracking** integration with `build.runs` for build history and status
- **Universal dataset lineage** with auto-populated `DatasetRef` and generated dataset nodes
- **DAG observability** with export/visualization tools for debugging and PR review

### Code Quality Refinements

The implementation has been hardened with:
- **Dataclass-based context objects** (`_RunContext`, `_SuccessRecordParams`) to reduce argument counts and local variables
- **Explicit type guards** for DataFrame validation with clear error paths
- **Full NumPy-style docstrings** on all helpers with `Returns` sections
- **Optimized edge construction** using `list.extend()` with generator comprehensions
- **Sorted `__all__` exports** for consistency across modules

---

## Phase 1 Implementation Summary

| PR | Feature | Files Modified |
|----|---------|----------------|
| PR-01 | Hamilton Node Mode + Target-Node Mapping | `driver_factory.py`, `executor.py` |
| PR-02 | Execute Dependency Closure + Complete Results | `executor.py`, `cli/handlers/build.py` |
| PR-03 | Upstream Failure Gating | `targets_phase0.py`, `node_factory.py` |
| PR-04 | CLI Parity - Force Flag | `env.py`, `cli/handlers/build.py`, `targets_phase0.py` |
| PR-05 | Run Tracking Parity | `executor.py` |
| PR-06 | Dataset Lineage Scale-Out | `targets_phase0.py`, `node_factory.py`, `manifest_hook.py` |
| PR-07 | Observability + DAG Visualization | `observability.py` (new), `cli/commands/build.py`, `cli/handlers/build.py` |

---

## PR-01: Hamilton Node Mode + Target-Node Mapping

### Problem

Phase 0 hardcoded the target-to-node mapping in `targets_phase0.TARGET_TO_NODE`, limiting Hamilton to only the explicit Phase 0 chain. The executor couldn't address targets beyond this subset.

### Solution

Introduced `HamiltonNodeMode` and expanded `HamiltonRuntime` to carry bidirectional mappings that work for both Phase 0 and generated node modes.

### Implementation

#### 1. HamiltonNodeMode Type (`driver_factory.py`)

```python
from typing import Literal

HamiltonNodeMode = Literal["phase0", "generated"]
```

#### 2. Expanded HamiltonRuntime (`driver_factory.py`)

```python
@dataclass(frozen=True)
class HamiltonRuntime:
    """Bundled Hamilton Driver and TargetGraph for build execution.
    
    Attributes
    ----------
    dr : driver.Driver
        Configured Hamilton Driver.
    graph : TargetGraph
        Target dependency graph.
    mode : HamiltonNodeMode
        Current node mode ("phase0" or "generated").
    target_to_node : dict[str, str]
        Mapping from target name to Hamilton node name.
    node_to_target : dict[str, str]
        Reverse mapping from node name to target name.
    """
    dr: driver.Driver
    graph: TargetGraph
    mode: HamiltonNodeMode
    target_to_node: dict[str, str]
    node_to_target: dict[str, str]
```

#### 3. Mode-Aware Mapping Builder (`driver_factory.py`)

```python
def _build_target_to_node_map(
    graph: TargetGraph,
    *,
    mode: HamiltonNodeMode,
) -> dict[str, str]:
    """Build target-to-node mapping based on mode.
    
    For phase0 mode, uses the static TARGET_TO_NODE from targets_phase0.
    For generated mode, uses the dynamically built module's mapping
    or computes names using the stable naming convention.
    """
    if mode == "phase0":
        return dict(targets_phase0.TARGET_TO_NODE)

    # generated mode
    mod = get_generated_module()
    mapping = getattr(mod, "TARGET_TO_NODE", None)
    if isinstance(mapping, dict) and mapping:
        return dict(mapping)

    # fallback: compute stable names
    return {t.name: target_node(t.name) for t in graph.all_targets}
```

#### 4. Updated build_driver (`driver_factory.py`)

```python
def build_driver(
    *,
    config: dict[str, Any] | None = None,
    profile: str | None = None,
    mode: HamiltonNodeMode = "generated",
) -> HamiltonRuntime:
    """Build Hamilton Driver for build execution.
    
    Parameters
    ----------
    config
        Configuration dict passed to Hamilton Driver.
    profile
        Build profile for TargetGraph selection.
    mode
        Node mode: "phase0" for explicit nodes, "generated" for dynamic.
    """
    graph = get_target_graph(profile=profile)

    if mode == "generated":
        nodes_module = get_generated_module()
    else:
        nodes_module = targets_phase0

    dr = driver.Driver(config or {}, nodes_module)

    t2n = _build_target_to_node_map(graph, mode=mode)
    n2t = {v: k for k, v in t2n.items()}
    return HamiltonRuntime(
        dr=dr, graph=graph, mode=mode, target_to_node=t2n, node_to_target=n2t
    )
```

#### 5. Runtime-Aware target_to_node_name (`driver_factory.py`)

```python
def target_to_node_name(
    target_name: str,
    *,
    runtime: HamiltonRuntime | None = None,
    mode: HamiltonNodeMode = "generated",
) -> str | None:
    """Convert target name to Hamilton node name.
    
    When runtime is provided, uses the runtime's cached mapping.
    Otherwise falls back to computing the name directly.
    """
    if runtime is not None:
        return runtime.target_to_node.get(target_name)

    # fallback if used outside runtime
    return target_node(target_name) if mode == "generated" else None
```

---

## PR-02: Execute Dependency Closure + Complete Results

### Problem

When requesting `targets=["risk_factors"]`, the executor only tracked the requested target, not the full dependency closure that Hamilton actually executed.

### Solution

Expanded `HamiltonBuildResult` with closure tracking and implemented proper categorization of computed/skipped/failed targets across the entire closure. Introduced `_RunContext` dataclass to bundle execution state and reduce local variable counts.

### Implementation

#### 1. Run Context for State Management (`executor.py`)

```python
@dataclass(frozen=True)
class _RunContext:
    """Execution context shared across run steps."""

    env: BuildEnv
    targets: tuple[str, ...]
    runtime: HamiltonRuntime
    run_id: str
    start_time: float
    started_at: datetime

    @property
    def duration_ms(self) -> float:
        """Return elapsed milliseconds for the run."""
        return (time.perf_counter() - self.start_time) * 1000
```

#### 2. Expanded HamiltonBuildResult (`executor.py`)

```python
@dataclass(frozen=True)
class HamiltonBuildResult:
    """Result of a Hamilton-based build execution."""

    requested: tuple[str, ...]
    closure: tuple[str, ...] = ()
    computed_targets: tuple[str, ...] = ()
    skipped_targets: tuple[str, ...] = ()
    failed_targets: tuple[str, ...] = ()
    outputs: dict[str, Any] = field(default_factory=dict)
    success: bool = True
    duration_ms: float = 0.0
    error: str | None = None
    run_id: str = ""
    runtime: HamiltonRuntime | None = None

    def get_record(self, target_name: str) -> TargetRunRecord | None:
        """Get execution record for a target using runtime mapping."""
        node_name = target_to_node_name(target_name, runtime=self.runtime)
        if node_name is None:
            return None
        value = self.outputs.get(node_name)
        return value if isinstance(value, TargetRunRecord) else None
```

> **Note**: Default values use `()` instead of `field(default_factory=tuple)` for immutable tuple defaults, and `run_id` defaults to empty string for simpler initialization.

#### 3. Closure Computation and Execution (`executor.py`)

```python
def run(self, *, env: BuildEnv, targets: list[str]) -> HamiltonBuildResult:
    # Bundle all execution state into a context object
    context = _RunContext(
        env=env,
        targets=tuple(targets),
        runtime=self._build_runtime(),
        run_id=_generate_run_id(),
        start_time=time.perf_counter(),
        started_at=datetime.now(tz=UTC),
    )
    
    # Compute full dependency closure
    closure = self._compute_closure(context.runtime, targets, context.run_id)
    if closure is None:
        return self._make_error_result(context, "Failed to compute closure")
    
    # Map closure to Hamilton node names
    final_vars, missing = _map_closure_to_nodes(closure, context.runtime)
    if missing:
        return self._make_missing_result(context, closure, missing)
    
    # Track run and execute DAG
    _start_build_run(context.env, context.run_id, targets, context.started_at)
    outputs, error = self._execute_dag(
        context.runtime, final_vars, context.env, context.run_id
    )
    
    # Categorize results using context duration
    computed, skipped, failed = _categorize_outputs(closure, outputs, context.runtime)
    duration_ms = context.duration_ms  # Property computes elapsed time
```

> **Note**: Static helper methods now accept `_RunContext` instead of multiple individual parameters, reducing argument counts and improving readability.

#### 4. Output Categorization (`executor.py`)

```python
def _categorize_outputs(
    closure: tuple[str, ...],
    outputs: dict[str, Any],
    runtime: HamiltonRuntime,
) -> tuple[list[str], list[str], list[str]]:
    """Categorize outputs into computed/skipped/failed lists.

    Returns
    -------
    tuple[list[str], list[str], list[str]]
        Computed, skipped, and failed targets in that order.
    """
    computed: list[str] = []
    skipped: list[str] = []
    failed: list[str] = []

    for target in closure:
        node_name = target_to_node_name(target, runtime=runtime)
        if node_name is None:
            failed.append(target)
            continue

        record = outputs.get(node_name)
        if not isinstance(record, TargetRunRecord):
            failed.append(target)
        elif record.status == "succeeded":
            computed.append(target)
        elif record.status == "skipped":
            skipped.append(target)
        else:
            failed.append(target)

    return computed, skipped, failed
```

> **Note**: All helper functions now include full NumPy-style docstrings with `Returns` sections.

#### 5. Updated CLI Adapter (`cli/handlers/build.py`)

```python
class _HamiltonResultAdapter:
    """Adapter to make HamiltonBuildResult compatible with BuildResult interface."""

    def __init__(self, hamilton_result: HamiltonBuildResult) -> None:
        self._result = hamilton_result

    @property
    def completed_targets(self) -> tuple[str, ...]:
        """Return targets that completed successfully."""
        return self._result.computed_targets

    @property
    def skipped_targets(self) -> tuple[str, ...]:
        """Return targets that were skipped."""
        return self._result.skipped_targets

    @property
    def failed_targets(self) -> tuple[str, ...]:
        """Return targets that failed."""
        return self._result.failed_targets

    @property
    def duration_ms(self) -> float:
        """Return total duration in milliseconds."""
        return self._result.duration_ms
```

---

## PR-03: Upstream Failure Gating

### Problem

If an upstream target failed, downstream targets would still attempt execution, potentially producing incorrect results from stale data.

### Solution

Added upstream failure checking in `_run_target()` that gates execution—if any upstream target failed, the downstream target is marked as skipped with a clear error message.

### Implementation

#### 1. Upstream Parameter in _run_target (`targets_phase0.py`)

```python
def _run_target(
    *,
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    upstream: tuple[TargetRunRecord, ...] = (),
) -> TargetRunRecord:
    """Execute target plugin with upstream failure gating."""
    
    # Check for upstream failures first
    failed_upstream = _check_upstream_failures(upstream)
    if failed_upstream:
        log.info("build.hamilton.upstream_failed target=%s", target_name)
        return TargetRunRecord(
            target=target_name,
            plugin_name=meta.name,
            status="skipped",
            error=f"upstream_failed:{','.join(failed_upstream)}",
        )
    
    # ... continue with skip check and execution ...
```

#### 2. Upstream Failure Check Helper (`targets_phase0.py`)

```python
def _check_upstream_failures(
    upstream: tuple[TargetRunRecord, ...],
) -> list[str]:
    """Check for failed upstream targets.
    
    Returns list of failed target names, empty if all succeeded.
    """
    return [r.target for r in upstream if r.status == "failed"]
```

#### 3. Wiring Upstream in Phase 0 Nodes (`targets_phase0.py`)

```python
@tag(domain="graphs", target="call_graph")
def t__call_graph(
    env: BuildEnv,
    graph: TargetGraph,
    t__goids: TargetRunRecord,
    t__scip: TargetRunRecord,
) -> TargetRunRecord:
    """Execute call_graph target with upstream gating."""
    return _run_target(
        env=env,
        graph=graph,
        target_name="call_graph",
        upstream=(t__goids, t__scip),
    )
```

#### 4. Wiring Upstream in Generated Nodes (`node_factory.py`)

```python
def _create_node_function(
    target: OutputTarget,
    dep_node_names: list[str],
    domain: str,
) -> Callable[..., TargetRunRecord]:
    """Create a Hamilton node function for a target."""
    target_name = target.name

    def node_fn(
        env: BuildEnv,
        graph: TargetGraph,
        **dependencies: TargetRunRecord,
    ) -> TargetRunRecord:
        # Extract upstream records from dependencies
        upstream = tuple(
            r for r in dependencies.values() 
            if isinstance(r, TargetRunRecord)
        )
        return _run_target(
            env=env, graph=graph, target_name=target_name, upstream=upstream
        )

    # ... signature and metadata setup ...
    return node_fn
```

---

## PR-04: CLI Parity - Force Flag

### Problem

The `--force` flag existed in the legacy executor but Hamilton had no way to bypass skip logic for specific targets.

### Solution

Added `force_targets` field to `BuildEnv` and implemented bypass logic in `_run_target()`.

### Implementation

#### 1. Extended BuildEnv (`env.py`)

```python
@dataclass(frozen=True)
class BuildEnv:
    """Bundled execution dependencies for Hamilton node execution."""
    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    providers: Providers
    config: BuildConfig
    profile: str | None = None
    force_targets: frozenset[str] = field(default_factory=frozenset)

    def is_forced(self, target_name: str) -> bool:
        """Check if a target is explicitly forced to recompute."""
        return target_name in self.force_targets
```

#### 2. CLI Handler Integration (`cli/handlers/build.py`)

```python
def _execute_build_hamilton(
    runtime: ResolvedRuntime,
    gateway: StorageGateway,
    goals: list[str],
    run_mode: RunMode,
    force: list[str] | None,
) -> BuildResultLike | None:
    # ...
    
    env = BuildEnv(
        gateway=gateway,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        providers=providers,
        config=config,
        profile="default",
        force_targets=frozenset(force or ()),  # Pass forced targets
    )

    executor = HamiltonBuildExecutor(profile="default", mode="generated")
    return _HamiltonResultAdapter(executor.run(env=env, targets=goals))
```

#### 3. Skip Bypass in _run_target (`targets_phase0.py`)

```python
def _should_skip_target(
    env: BuildEnv,
    target_name: str,
    input_hash: str | None,
) -> bool:
    """Check if target should be skipped (respects force flag)."""
    # Forced targets are never skipped
    if env.is_forced(target_name):
        log.info("build.hamilton.forced target=%s", target_name)
        return False
    
    # Check manifest for skip eligibility
    return should_skip(
        gateway=env.gateway,
        target=target_name,
        repo=env.repo,
        commit=env.commit,
        input_hash=input_hash,
    )
```

---

## PR-05: Run Tracking Parity

### Problem

Hamilton builds didn't show up in build history/status the way legacy builds did.

### Solution

Integrated run tracking by calling `gateway.build.start_run()` and `gateway.build.complete_run()` with proper timing and status.

### Implementation

#### 1. Run ID Generation (`executor.py`)

```python
def _generate_run_id() -> str:
    """Generate a unique run ID for build tracking."""
    timestamp = datetime.now(tz=UTC).strftime("%Y%m%d%H%M%S")
    suffix = uuid.uuid4().hex[:8]
    return f"hamilton-{timestamp}-{suffix}"
```

#### 2. Start Run Helper (`executor.py`)

```python
def _start_build_run(
    env: BuildEnv,
    run_id: str,
    targets: list[str],
    start_datetime: datetime,
) -> None:
    """Record the start of a build run."""
    try:
        record = BuildRunRecord(
            run_id=run_id,
            repo=env.repo,
            commit=env.commit,
            requested_targets=tuple(targets),
            computed_targets=(),
            skipped_targets=(),
            started_at=start_datetime,
            status="running",
        )
        env.gateway.build.start_run(record)
    except Exception:  # noqa: BLE001 - Best effort tracking
        log.warning("build.hamilton.executor.start_run_failed run_id=%s", run_id)
```

> **Note**: Uses `BuildRunRecord` dataclass for structured run state. Exception handling uses `noqa: BLE001` annotation since run tracking is best-effort and shouldn't fail the build.

#### 3. Complete Run Helper (`executor.py`)

```python
@dataclass(frozen=True)
class _RunCompletionParams:
    """Parameters for completing a build run."""
    env: BuildEnv
    run_id: str
    success: bool
    computed: tuple[str, ...]
    skipped: tuple[str, ...]
    error_summary: str | None


def _complete_build_run(params: _RunCompletionParams) -> None:
    """Record build run completion in gateway."""
    try:
        params.env.gateway.build.complete_run(
            run_id=params.run_id,
            status="succeeded" if params.success else "failed",
            computed_targets=params.computed,
            skipped_targets=params.skipped,
            error_summary=params.error_summary,
        )
    except Exception:
        log.exception(
            "build.hamilton.complete_run_failed run_id=%s", params.run_id
        )
```

#### 4. Integration in run() (`executor.py`)

```python
def run(self, *, env: BuildEnv, targets: list[str]) -> HamiltonBuildResult:
    run_id = _generate_run_id()
    start_time = time.perf_counter()
    start_datetime = datetime.now(tz=UTC)

    # ... build runtime and compute closure ...

    _start_build_run(env, run_id, targets, start_datetime)
    
    try:
        outputs, error = self._execute_dag(runtime, final_vars, env, run_id)
        computed, skipped, failed = _categorize_outputs(...)
    finally:
        success = not failed and error is None
        _complete_build_run(_RunCompletionParams(
            env=env,
            run_id=run_id,
            success=success,
            computed=tuple(computed),
            skipped=tuple(skipped),
            error_summary=error or (f"{len(failed)} targets failed" if failed else None),
        ))

    return HamiltonBuildResult(
        ...,
        run_id=run_id,
        duration_ms=(time.perf_counter() - start_time) * 1000,
    )
```

---

## PR-06: Dataset Lineage Scale-Out

### Problem

Dataset lineage was limited to hardcoded Phase 0 examples. The system needed to auto-populate `TargetRunRecord.datasets` for all targets and generate dataset nodes dynamically.

### Solution

Extended `_run_target()` to populate datasets from target contracts on success, and added dataset node generation to `build_target_module()`.

### Implementation

#### 1. TargetRunRecord.datasets Field (`manifest_hook.py`)

```python
@dataclass(frozen=True)
class TargetRunRecord:
    """Record of Hamilton node execution."""
    target: str
    plugin_name: str
    status: str
    input_hash: str | None = None
    options_hash: str | None = None
    duration_ms: float = 0.0
    row_counts: Mapping[str, int] | None = None
    error: str | None = None
    datasets: tuple[DatasetRef, ...] = field(default_factory=tuple)

    def get_dataset(self, table_key: str) -> DatasetRef | None:
        """Get specific dataset ref by table key."""
        for ds in self.datasets:
            if ds.table_key == table_key:
                return ds
        return None
```

#### 2. Success Record Builder (`targets_phase0.py`)

```python
@dataclass(frozen=True)
class _SuccessRecordParams:
    """Parameters for building a success record."""

    env: BuildEnv
    target: OutputTarget
    target_name: str
    meta_name: str
    input_hash: str | None
    options_hash: str | None
    duration_ms: float
    row_counts: dict[str, int]


def _build_success_record(params: _SuccessRecordParams) -> TargetRunRecord:
    """Build record and save manifest for successful execution."""
    if params.input_hash is not None:
        request = ManifestSaveRequest(...)
        save_manifest(gateway=params.env.gateway, request=request)

    table_keys = params.target.contract.table_keys or params.target.table_keys
    refs = refs_from_target_result(
        target_name=params.target_name,
        table_keys=table_keys,
        row_counts=params.row_counts,
    )
    datasets = refs_to_tuple(refs)

    return TargetRunRecord(
        target=params.target_name,
        plugin_name=params.meta_name,
        status="succeeded",
        input_hash=params.input_hash,
        options_hash=params.options_hash,
        duration_ms=params.duration_ms,
        row_counts=params.row_counts,
        datasets=datasets,  # Auto-populated from contract!
    )
```

> **Note**: The `_SuccessRecordParams` dataclass bundles all parameters needed to build a success record, reducing argument counts in the helper function.

#### 3. Dataset Node Generator (`node_factory.py`)

```python
def _create_dataset_node_function(
    *,
    table_key: str,
    target_name: str,
) -> Callable[..., DatasetRef]:
    """Create a Hamilton node function for a dataset.
    
    The generated node extracts a specific DatasetRef from
    the parent target's TargetRunRecord.
    """
    d_name = dataset_node(table_key)
    t_name = target_node(target_name)

    def fn(**kwargs: object) -> DatasetRef:
        rec = kwargs[t_name]
        assert isinstance(rec, TargetRunRecord)
        ds = rec.get_dataset(table_key)
        if ds is None:
            raise ValueError(f"Missing DatasetRef for {table_key} from {target_name}")
        return ds

    fn.__name__ = d_name
    fn.__module__ = "codeintel.build.hamilton.nodes.generated"
    fn.__signature__ = inspect.Signature(
        parameters=[
            inspect.Parameter(
                t_name, 
                inspect.Parameter.POSITIONAL_OR_KEYWORD, 
                annotation=TargetRunRecord
            ),
        ],
        return_annotation=DatasetRef,
    )
    return tag(domain=table_key.split(".", 1)[0], table=table_key)(fn)
```

#### 4. Dataset Node Registration in build_target_module (`node_factory.py`)

```python
def build_target_module(...) -> ModuleType:
    # ... generate target nodes ...
    
    # Generate dataset nodes for all contract tables
    dataset_to_node: dict[str, str] = {}
    for target in graph.all_targets:
        if target.name not in include or target.name in exclude:
            continue
        for table_key in target.contract.table_keys:
            dataset_fn = _create_dataset_node_function(
                table_key=table_key,
                target_name=target.name,
            )
            d_node_name = dataset_node(table_key)
            setattr(module, d_node_name, dataset_fn)
            dataset_to_node[table_key] = d_node_name

    module.DATASET_TO_NODE = dataset_to_node
    return module
```

---

## PR-07: Observability + DAG Visualization

### Problem

Developers had no way to inspect what Hamilton would execute before running a build, making debugging and PR review difficult.

### Solution

Created `observability.py` with DAG export functions and added a `codeintel build graph` CLI command.

### Implementation

#### 1. Observability Module (`observability.py`)

```python
"""Hamilton DAG observability utilities."""

def list_execution_order(
    runtime: HamiltonRuntime,
    targets: list[str],
) -> list[str]:
    """Return Hamilton node names in topological execution order."""
    closure = runtime.graph.topological_order(targets)
    return [runtime.target_to_node[t] for t in closure if t in runtime.target_to_node]


def list_execution_targets(
    runtime: HamiltonRuntime,
    targets: list[str],
) -> list[str]:
    """Return target names in topological execution order."""
    return list(runtime.graph.topological_order(targets))


def get_dag_info(
    runtime: HamiltonRuntime,
    targets: list[str],
) -> dict[str, Any]:
    """Get detailed DAG information for targets.
    
    Returns dict with:
    - requested: Original target list
    - closure: Full dependency closure
    - mode: Hamilton node mode
    - nodes: List of node info dicts (name, module, plugin, tables, deps)
    - edges: List of edge dicts (from, to)
    - node_count, edge_count: Counts for quick reference
    """
    closure = runtime.graph.topological_order(targets)

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, str]] = []

    for target_name in closure:
        target = runtime.graph.get(target_name)
        node_info = {
            "name": target_name,
            "node_name": runtime.target_to_node.get(target_name),
            "module": target.module,
            "plugin": target.plugin,
            "tables": list(target.table_keys),
            "dependencies": list(target.dependencies),
        }
        nodes.append(node_info)

        # Optimized edge construction using list.extend with generator
        edges.extend(
            {"from": dep, "to": target_name}
            for dep in target.dependencies
            if dep in runtime.target_to_node
        )

    return {
        "requested": targets,
        "closure": list(closure),
        "mode": runtime.mode,
        "nodes": nodes,
        "edges": edges,
        "node_count": len(nodes),
        "edge_count": len(edges),
    }


def export_dag_json(
    runtime: HamiltonRuntime,
    targets: list[str],
    *,
    indent: int | None = 2,
) -> str:
    """Export DAG information as JSON string."""
    return json.dumps(get_dag_info(runtime, targets), indent=indent)


def export_execution_json(
    runtime: HamiltonRuntime,
    *,
    targets: list[str],
    env: BuildEnv,
) -> str:
    """Export execution plan as JSON with input context."""
    dag_info = get_dag_info(runtime, targets)
    execution_info = {
        **dag_info,
        "execution_order": list_execution_order(runtime, targets),
        "inputs": {
            "env": {
                "repo": env.repo,
                "commit": env.commit,
                "profile": env.profile,
                "force_targets": list(env.force_targets),
            },
        },
    }
    return json.dumps(execution_info, indent=2)
```

#### 2. CLI Command (`cli/commands/build.py`)

```python
@cli_command("build.graph", handler=build_graph_handler, config=_BUILD_CONFIG)
@build_app.command(name="graph")
@dataclass
class BuildGraphCommand:
    """Export Hamilton DAG for specified targets."""

    targets: Annotated[
        list[str] | None,
        Parameter(
            name=None,
            help="Target names to include in the DAG.",
        ),
    ] = None
    module: Annotated[
        str | None,
        Parameter(
            name=["--module", "-m"],
            help="Include all targets in a module (ingestion, graphs, analytics).",
        ),
    ] = None
    all_targets: Annotated[
        bool,
        Parameter(
            name=["--all", "-a"],
            help="Include all targets across all modules.",
        ),
    ] = False
    format: Annotated[
        str,
        Parameter(
            name=["--format", "-F"],
            help="Output format (json, mermaid).",
        ),
    ] = "json"
    output_file: Annotated[
        Path | None,
        Parameter(
            name=["--output", "-o"],
            help="Output file path (e.g., dag.json).",
        ),
    ] = None
```

#### 3. CLI Handler (`cli/handlers/build.py`)

```python
def build_graph_handler(ctx: CommandContext) -> CliResult[BuildGraphResult]:
    """Handle 'build graph' command to export Hamilton DAG."""
    from codeintel.build.hamilton.driver_factory import build_driver
    from codeintel.build.hamilton.observability import export_dag_json, get_dag_info

    # ... resolve goals from targets/module/all_targets ...

    hamilton_runtime = build_driver(mode="generated")
    dag_info = get_dag_info(hamilton_runtime, goals)
    dag_json = export_dag_json(hamilton_runtime, goals)

    if output_file:
        Path(output_file).write_text(dag_json, encoding="utf-8")

    return CliResult.ok(BuildGraphResult(
        dag_json=dag_json,
        node_count=dag_info["node_count"],
        edge_count=dag_info["edge_count"],
    ))
```

---

## Package Exports Update

### Main Package (`__init__.py`)

The package exports are now **alphabetically sorted** for consistency and include all Phase 1 additions:

```python
__all__ = [
    "BuildEnv",
    "CanonicalPluginMeta",
    "DatasetRef",
    "HamiltonBuildExecutor",
    "HamiltonBuildResult",
    "HamiltonNodeMode",
    "HamiltonRuntime",
    "IbisIOConfig",
    "TargetRunRecord",
    "build_driver",
    "dataset_node",
    "export_dag_json",
    "export_execution_json",
    "get_dag_info",
    "get_pandera_schema",
    "list_available_nodes",
    "list_execution_order",
    "list_execution_targets",
    "refs_from_target_result",
    "target_node",
    "target_to_node_name",
    "to_node_name",
    "validate_dataframe",
    "validate_dataset_ref",
    "with_contract",
]
```

### Updated Module Docstring

```python
"""Hamilton integration for the CodeIntel build system.

Phase 0 Implementation
----------------------
- Wraps existing target plugins as Hamilton nodes
- Reuses existing manifest/hashing infrastructure
- Provides skip-if-unchanged caching via manifest checks
- Explicit node definitions for the risk_factors chain

Phase 1 Implementation (Full Production Features)
-------------------------------------------------
- HamiltonNodeMode: Support for "phase0" and "generated" node modes
- HamiltonRuntime: Extended with target↔node mappings
- Closure execution: Full dependency closure computed and executed
- Upstream failure gating: Downstream skipped if upstream fails
- Force targets: --force flag bypasses skip checks
- Run tracking: Builds tracked in build.runs table
- DatasetRef: Type-safe dataset references populated on success
- Dataset nodes: d__* nodes generated for all contract tables
- Observability: DAG export and visualization via CLI
"""
```

---

## Code Quality Refinements

### Pandera Contract Hooks (`contracts/pandera_hook.py`)

The Pandera integration was hardened with explicit type guards and clearer error paths:

```python
def _ensure_dataframe(result: object, table_key: str) -> pd.DataFrame:
    """Ensure validation inputs are DataFrames.

    Raises
    ------
    TypeError
        If the provided result is not a pandas DataFrame.
    """
    if isinstance(result, pd.DataFrame):
        return result
    msg = f"Expected pandas.DataFrame for {table_key}, got {type(result).__name__}"
    raise TypeError(msg)
```

**Key improvements**:
- **Explicit type guard** (`_ensure_dataframe`) with clear error messages
- **Skip-on-missing-schema preserved** for `validate_dataset_ref` (returns `True, None` when no schema)
- **TypeError vs ValueError** distinction for clearer error categorization
- **Full NumPy-style docstrings** on all public functions

### Node Helper Semantics (`targets_phase0.py`)

Helper functions now use dataclasses for complex parameter sets:

```python
@dataclass(frozen=True)
class _SuccessRecordParams:
    """Parameters for building a success record."""
    env: BuildEnv
    target: OutputTarget
    target_name: str
    meta_name: str
    input_hash: str | None
    options_hash: str | None
    duration_ms: float
    row_counts: dict[str, int]
```

**Key improvements**:
- **Dataclass-based parameters** reduce function argument counts
- **Full NumPy-style return docs** on all helpers
- **Sorted `__all__` exports** for consistency

### DAG Edge Construction (`observability.py`)

```python
# Before: nested loop with append
for dep in target.dependencies:
    if dep in runtime.target_to_node:
        edges.append({"from": dep, "to": target_name})

# After: list.extend with generator comprehension
edges.extend(
    {"from": dep, "to": target_name}
    for dep in target.dependencies
    if dep in runtime.target_to_node
)
```

---

## Definition of Done Verification

| Criterion | Status |
|-----------|--------|
| 1. `codeintel build run --engine hamilton --all` executes for all targets | ✅ |
| 2. `--force` forces recompute of previously "fresh" targets | ✅ |
| 3. Build history/status shows Hamilton runs via `build.runs` | ✅ |
| 4. Downstream targets do not run when upstream fails | ✅ |
| 5. `TargetRunRecord.datasets` populated for successful targets | ✅ |
| 6. Dataset nodes generated for all contract tables | ✅ |

---

## CLI Usage Examples

```bash
# Execute all targets with Hamilton
codeintel build run --engine hamilton --all

# Force recompute specific targets
codeintel build run --engine hamilton risk_factors --force risk_factors

# Export DAG for risk_factors chain
codeintel build graph risk_factors --output dag.json

# Export DAG for all analytics targets
codeintel build graph --module analytics --format json

# Export full DAG
codeintel build graph --all --output full-dag.json
```

---

## References

- [Phase 0 Specification](Hamilton_apache_phase0.md)
- [Phase 1 Specification](Hamilton_apache_phase1.md)
- [Phase 0 Implementation Report](Hamilton_implementation_report.md)
- [Hamilton Documentation](https://hamilton.dagworks.io/)

