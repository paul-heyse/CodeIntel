# Hamilton Integration Phase 2 Implementation Report

> **Status**: Phase 2 Complete  
> **Date**: December 2024

---

## Executive Summary

Phase 2 transforms Hamilton into the **source of truth** for the CodeIntel build system, establishing best-in-class incremental builds with deep observability. This phase delivers:

- **Hamilton as default engine** with "generated" mode as the standard execution path
- **Build planner** with compute_plan() providing upfront visibility into what will run and why
- **Manifest index prefetch** eliminating per-target DB round trips with cascading hash semantics
- **DatasetRef v2** with full snapshot identity (repo/commit) for complete lineage
- **ArtifactRef** for tracking non-tabular outputs (SCIP indexes, FAISS files, etc.)
- **Loader nodes** (q__*, df__*) enabling downstream consumers to declare typed dependencies
- **Run targets persistence** with per-target execution records for debugging
- **Enhanced graph exports** with Mermaid and Graphviz DOT visualization
- **Explain staleness** showing exactly which dependencies changed and why

### Optional Enhancements Delivered

- **`--validate-outputs` flag** for post-write Pandera schema validation
- **`build history --run-id`** with per-target breakdown for run debugging
- **`a__*` artifact nodes** for artifact lineage in the Hamilton DAG

---

## Phase 2 Implementation Summary

| PR | Feature | Files Modified |
|----|---------|----------------|
| PR-08 | Hamilton Default Mode | `driver_factory.py`, `executor.py`, `cli/commands/build.py`, `cli/handlers/build.py` |
| PR-09 | Build Planner | `planner.py` (new), `cli/commands/build.py`, `cli/handlers/build.py` |
| PR-10 | Manifest Index + Hash Cascade | `env.py`, `hashing.py`, `manifest_hook.py`, `targets_phase0.py` |
| PR-11 | DatasetRef v2 + ArtifactRef | `dataset_ref.py`, `artifact_ref.py` (new), `manifest_hook.py` |
| PR-12 | Loader Nodes | `node_factory.py`, `naming.py` |
| PR-13 | Run Targets Persistence | `schemas.py`, `build_tracking.py`, `executor.py` |
| PR-14 | Graph Exports | `observability.py`, `cli/handlers/build.py` |
| PR-15 | Explain Staleness | `planner.py`, `cli/commands/build.py`, `cli/handlers/build.py` |

---

## PR-08: Hamilton Default Mode

### Problem

Phase 1 defaulted to "phase0" mode with explicit nodes, requiring users to opt-in to the generated node system. This created friction and prevented Hamilton from being the primary execution path.

### Solution

Changed all defaults from "phase0" to "generated" mode, making Hamilton with dynamic node generation the standard execution path.

### Implementation

#### 1. Driver Factory Defaults (`driver_factory.py`)

```python
def build_driver(
    *,
    config: dict[str, Any] | None = None,
    profile: str | None = None,
    mode: HamiltonNodeMode = "generated",  # Changed from "phase0"
) -> HamiltonRuntime:
    """Build Hamilton Driver for build execution.
    
    The default mode is now "generated" for dynamic node generation.
    Use "phase0" only for debugging explicit node chains.
    """
```

#### 2. Executor Defaults (`executor.py`)

```python
class HamiltonBuildExecutor:
    def __init__(
        self,
        profile: str = "default",
        mode: HamiltonNodeMode = "generated",  # Changed from "phase0"
    ) -> None:
        self._profile = profile
        self._mode = mode
```

#### 3. CLI Command Updates (`cli/commands/build.py`)

```python
@dataclass
class BuildRunCommand:
    engine: Annotated[str, ...] = "hamilton"  # Changed from "legacy"
    hamilton_mode: Annotated[
        str,
        Parameter(
            name=["--hamilton-mode"],
            help="Hamilton node mode: generated (default) or phase0 (debug).",
        ),
    ] = "generated"
```

#### 4. Handler Integration (`cli/handlers/build.py`)

```python
def _execute_build_hamilton(
    runtime: ResolvedRuntime,
    gateway: StorageGateway,
    goals: list[str],
    run_mode: RunMode,
    force: list[str] | None = None,
    hamilton_mode: str = "generated",  # Default changed
) -> BuildResultLike | None:
    # ...
    mode: HamiltonNodeMode = "generated" if hamilton_mode == "generated" else "phase0"
    executor = HamiltonBuildExecutor(profile="default", mode=mode)
```

---

## PR-09: Build Planner

### Problem

Users had no way to understand what would run and why before executing a build. The skip/compute decision was opaque until after execution.

### Solution

Created a Hamilton-centric planner that produces a complete build plan with status and reason for each target, enabling "dry-run" style preview and debugging.

### Implementation

#### 1. Plan Status and Reason Types (`planner.py`)

```python
PlanStatus = Literal["compute", "skip", "missing", "blocked"]

PlanReason = Literal[
    "forced",
    "no_manifest",
    "hash_changed",
    "up_to_date",
    "upstream_missing",
    "no_plugin",
]
```

#### 2. PlanEntry Dataclass (`planner.py`)

```python
@dataclass(frozen=True)
class PlanEntry:
    """Plan entry describing why a target will or won't run.

    Attributes
    ----------
    target
        Target name being planned.
    node
        Hamilton node name (e.g., "t__function_metrics").
    module
        Target module (ingestion, graphs, analytics).
    status
        Plan status: "compute", "skip", "missing", or "blocked".
    reason
        Reason for the status.
    input_hash
        Current computed input hash for the target.
    options_hash
        Current computed options hash from configuration.
    prior_input_hash
        Input hash from prior manifest, if available.
    dependencies
        Tuple of target names this target depends on.
    table_keys
        Tuple of table keys this target produces.
    dep_hashes
        Current dependency hash mapping for explain support.
    prior_dep_hashes
        Prior dependency hash mapping from manifest.
    """

    target: str
    node: str
    module: str
    status: PlanStatus
    reason: PlanReason
    input_hash: str | None
    options_hash: str | None
    prior_input_hash: str | None
    dependencies: tuple[str, ...]
    table_keys: tuple[str, ...]
    artifact_keys: tuple[str, ...] = ()
    dep_hashes: dict[str, str] = field(default_factory=dict)
    prior_dep_hashes: dict[str, str] = field(default_factory=dict)

    def explain_staleness(self) -> StalenessExplanation:
        """Generate detailed staleness explanation."""
        # Compute changed, added, removed dependencies
        ...
```

#### 3. HamiltonBuildPlan Dataclass (`planner.py`)

```python
@dataclass(frozen=True)
class HamiltonBuildPlan:
    """Complete build plan for Hamilton execution.

    Attributes
    ----------
    requested
        Tuple of target names originally requested.
    closure
        Tuple of target names in dependency closure (topological order).
    entries
        Tuple of PlanEntry objects, one per target in closure.
    """

    requested: tuple[str, ...]
    closure: tuple[str, ...]
    entries: tuple[PlanEntry, ...] = field(default_factory=tuple)

    @property
    def to_compute(self) -> tuple[str, ...]:
        """Return targets that will be computed."""
        return tuple(e.target for e in self.entries if e.status == "compute")

    @property
    def to_skip(self) -> tuple[str, ...]:
        """Return targets that will be skipped."""
        return tuple(e.target for e in self.entries if e.status == "skip")

    def get_entry(self, target: str) -> PlanEntry | None:
        """Get plan entry for a specific target."""
        for entry in self.entries:
            if entry.target == target:
                return entry
        return None
```

#### 4. compute_plan Function (`planner.py`)

```python
def compute_plan(
    *,
    env: BuildEnv,
    graph: TargetGraph | None = None,
    requested: tuple[str, ...],
    mode: HamiltonNodeMode = "generated",
) -> HamiltonBuildPlan:
    """Compute a build plan for the requested targets.

    Determines status and reason for each target in the dependency closure
    without executing anything. Uses manifest_index from env for efficient
    skip checks.

    Parameters
    ----------
    env
        Build environment with gateway, snapshot, and manifest_index.
    graph
        Target graph (defaults to global registry).
    requested
        Tuple of target names to plan.
    mode
        Hamilton node mode for node naming.

    Returns
    -------
    HamiltonBuildPlan
        Complete plan with entries for all targets in closure.
    """
    graph = graph or get_target_graph()
    closure = graph.topological_order(requested)

    entries: list[PlanEntry] = []
    upstream_status: dict[str, PlanStatus] = {}

    for target_name in closure:
        entry = _compute_plan_entry(
            target_name=target_name,
            env=env,
            graph=graph,
            mode=mode,
            upstream_status=upstream_status,
        )
        entries.append(entry)
        upstream_status[target_name] = entry.status

    return HamiltonBuildPlan(
        requested=requested,
        closure=closure,
        entries=tuple(entries),
    )
```

#### 5. CLI Integration (`cli/commands/build.py`)

```python
@cli_command("build.plan", handler=build_plan_handler, config=_BUILD_CONFIG)
@build_app.command(name="plan")
@dataclass
class BuildPlanCommand:
    """Show build plan with status and reason for each target."""

    targets: Annotated[list[str] | None, ...] = None
    module: Annotated[str | None, ...] = None
    all_targets: Annotated[bool, ...] = False
    force: Annotated[list[str] | None, ...] = None
    output_file: Annotated[str | None, ...] = None
```

---

## PR-10: Manifest Index + Hash Cascade

### Problem

Each target's skip check made a separate DB call to load its manifest. For large builds with many targets, this created significant overhead. Additionally, hash computation didn't properly cascade—changing an upstream hash didn't always invalidate downstream targets.

### Solution

Added `manifest_index` field to `BuildEnv` for bulk-loaded manifests, and updated hash computation to use `input_hash` (not `output_hash`) for proper cascade semantics.

### Implementation

#### 1. BuildEnv Extension (`env.py`)

```python
@dataclass(frozen=True)
class BuildEnv:
    """Bundled execution dependencies for Hamilton node execution.

    Attributes
    ----------
    manifest_index
        Pre-loaded mapping of target names to their manifests for this
        repo/commit. Used to avoid per-target DB round trips during
        skip checks and hash computation.
    validate_outputs
        When True, validate produced datasets against their Pandera schemas
        after write. Validation failures will mark the target as failed.
    """

    gateway: StorageGateway
    snapshot: SnapshotRef
    paths: BuildPaths
    providers: Providers
    config: BuildConfig
    profile: str | None = None
    force_targets: frozenset[str] = field(default_factory=frozenset)
    manifest_index: Mapping[str, OutputManifest] | None = None
    validate_outputs: bool = False  # Optional enhancement
```

#### 2. Hash Computation with Manifest Index (`hashing.py`)

```python
def compute_input_hash(
    target: OutputTarget,
    snapshot: SnapshotRef,
    gateway: StorageGateway,
    options_hash: str | None = None,
    manifests: Mapping[str, OutputManifest] | None = None,
) -> str:
    """Compute content-addressable hash of target inputs.

    Uses input_hash (not output_hash) for dependency cascading to ensure
    that any upstream change propagates downstream.

    Parameters
    ----------
    manifests
        Optional pre-loaded manifest index. If provided, avoids per-target
        DB calls for dependency hash lookups.
    """
    # Collect dependency hashes from manifests
    dep_hashes: list[str] = []
    for dep_name in target.dependencies:
        if manifests and dep_name in manifests:
            dep_manifest = manifests[dep_name]
            # Use input_hash for cascade semantics
            dep_hashes.append(f"{dep_name}:{dep_manifest.input_hash}")
        elif gateway:
            dep_manifest = gateway.build.load_manifest(
                dep_name, snapshot.repo, snapshot.commit
            )
            if dep_manifest:
                dep_hashes.append(f"{dep_name}:{dep_manifest.input_hash}")
        else:
            dep_hashes.append(f"{dep_name}:missing")

    # Combine all inputs for final hash
    combined = "|".join([
        f"repo:{snapshot.repo}",
        f"commit:{snapshot.commit}",
        f"target:{target.name}",
        f"options:{options_hash or 'none'}",
        *sorted(dep_hashes),
    ])
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
```

#### 3. CLI Handler Prefetch (`cli/handlers/build.py`)

```python
def _execute_build_hamilton(
    runtime: ResolvedRuntime,
    gateway: StorageGateway,
    goals: list[str],
    run_mode: RunMode,
    force: list[str] | None = None,
    hamilton_mode: str = "generated",
    validate_outputs: bool = False,
) -> BuildResultLike | None:
    # Prefetch all manifests for this repo/commit
    manifests_list = gateway.build.list_manifests(
        repo=runtime.snapshot.repo,
        commit=runtime.snapshot.commit,
    )
    manifest_index = {m.target: m for m in manifests_list}
    LOG.debug("build.cli.hamilton.manifest_index count=%d", len(manifest_index))

    env = BuildEnv(
        gateway=gateway,
        snapshot=runtime.snapshot,
        paths=runtime.paths,
        providers=providers,
        config=config,
        profile="default",
        force_targets=frozenset(force or ()),
        manifest_index=manifest_index,
        validate_outputs=validate_outputs,
    )
```

---

## PR-11: DatasetRef v2 + ArtifactRef

### Problem

`DatasetRef` lacked full snapshot identity (repo/commit), making cross-snapshot lineage tracking impossible. Additionally, there was no mechanism for tracking non-tabular artifacts like SCIP indexes or FAISS files.

### Solution

Extended `DatasetRef` with `repo` and `commit` fields, and created `ArtifactRef` for non-tabular output tracking. Updated `TargetRunRecord` to include artifacts.

### Implementation

#### 1. DatasetRef v2 (`dataset_ref.py`)

```python
@dataclass(frozen=True)
class DatasetRef:
    """Lightweight reference to a DuckDB table in the build DAG.

    Attributes
    ----------
    table_key
        Fully-qualified table name (e.g., "analytics.function_metrics").
    repo
        Repository slug for snapshot identity.
    commit
        Commit SHA for snapshot identity.
    row_count
        Optional row count for observability.
    """

    table_key: str
    repo: str = ""
    commit: str = ""
    row_count: int | None = None


def refs_from_target_result(
    target_name: str,
    table_keys: tuple[str, ...],
    row_counts: dict[str, int] | None = None,
    snapshot: SnapshotRef | None = None,
) -> dict[str, DatasetRef]:
    """Create DatasetRef mapping from target execution result.

    When snapshot is provided, populates repo and commit fields
    for complete lineage tracking.
    """
    refs: dict[str, DatasetRef] = {}
    for table_key in table_keys:
        refs[table_key] = DatasetRef(
            table_key=table_key,
            repo=snapshot.repo if snapshot else "",
            commit=snapshot.commit if snapshot else "",
            row_count=(row_counts or {}).get(table_key),
        )
    return refs
```

#### 2. ArtifactRef Dataclass (`artifact_ref.py`)

```python
@dataclass(frozen=True)
class ArtifactRef:
    """Reference to a non-tabular artifact in the build DAG.

    This is a lightweight handle for file-based outputs, indexes, models,
    and other artifacts that are not DuckDB tables.

    Attributes
    ----------
    name
        Artifact name identifier.
    artifact_type
        Type of artifact: "file", "index", "model", etc.
    repo
        Repository slug for snapshot identity.
    commit
        Commit SHA for snapshot identity.
    path
        Optional filesystem path to the artifact.
    metadata
        Additional metadata for observability and debugging.
    """

    name: str
    artifact_type: str
    repo: str
    commit: str
    path: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def with_path(self, new_path: str) -> ArtifactRef:
        """Return a new ref with updated path."""
        return ArtifactRef(
            name=self.name,
            artifact_type=self.artifact_type,
            repo=self.repo,
            commit=self.commit,
            path=new_path,
            metadata=self.metadata,
        )
```

#### 3. TargetRunRecord Extension (`manifest_hook.py`)

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
    datasets: tuple[DatasetRef, ...] = ()
    artifacts: tuple[ArtifactRef, ...] = ()  # New field
```

---

## PR-12: Loader Nodes

### Problem

Downstream consumers had no type-safe way to declare dependencies on specific datasets. They could only depend on entire target nodes, not individual table outputs.

### Solution

Added `q__*` (Ibis query) and `df__*` (pandas DataFrame) loader node generation in `build_target_module()`.

### Implementation

#### 1. Naming Conventions (`naming.py`)

```python
def query_node(table_key: str) -> str:
    """Convert table key to Ibis query loader node identifier.

    Query nodes use the "q" prefix and return Ibis table expressions.

    Examples
    --------
    >>> query_node("analytics.function_metrics")
    'q__analytics__function_metrics'
    """
    return to_node_name(table_key, prefix="q")


def dataframe_node(table_key: str) -> str:
    """Convert table key to pandas DataFrame loader node identifier.

    DataFrame nodes use the "df" prefix and return pandas DataFrames.

    Examples
    --------
    >>> dataframe_node("analytics.function_metrics")
    'df__analytics__function_metrics'
    """
    return to_node_name(table_key, prefix="df")
```

#### 2. Query Node Generator (`node_factory.py`)

```python
def _create_query_node_function(
    *,
    table_key: str,
    target_name: str,
) -> Callable[..., object]:
    """Create a Hamilton node function for Ibis query loading.

    The generated node loads a DatasetRef as an Ibis table expression,
    enabling SQL-based transformations downstream.
    """
    from codeintel.build.hamilton.io.ibis_adapter import load_dataset_ibis

    q_name = query_node(table_key)
    d_name = dataset_node(table_key)

    def query_fn(env: BuildEnv, **kwargs: object) -> object:
        ds_ref = kwargs.get(d_name)
        if not isinstance(ds_ref, DatasetRef):
            msg = f"Expected DatasetRef for {d_name}, got {type(ds_ref)}"
            raise TypeError(msg)
        return load_dataset_ibis(gateway=env.gateway, ref=ds_ref)

    # Build signature with proper annotations
    query_fn.__signature__ = inspect.Signature([
        inspect.Parameter("env", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=BuildEnv),
        inspect.Parameter(d_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=DatasetRef),
    ])
    query_fn.__name__ = q_name
    query_fn.__doc__ = f"Load {table_key} as Ibis expression from {target_name} target."

    return tag(domain=domain, table=table_key, node_type="query")(query_fn)
```

#### 3. DataFrame Node Generator (`node_factory.py`)

```python
def _create_dataframe_node_function(
    *,
    table_key: str,
    target_name: str,
) -> Callable[..., object]:
    """Create a Hamilton node function for pandas DataFrame loading.

    The generated node loads a DatasetRef as a pandas DataFrame,
    enabling Python-based transformations downstream.
    """
    from codeintel.build.hamilton.io.ibis_adapter import load_dataset_df

    df_name = dataframe_node(table_key)
    d_name = dataset_node(table_key)

    def dataframe_fn(env: BuildEnv, **kwargs: object) -> object:
        ds_ref = kwargs.get(d_name)
        if not isinstance(ds_ref, DatasetRef):
            msg = f"Expected DatasetRef for {d_name}, got {type(ds_ref)}"
            raise TypeError(msg)
        return load_dataset_df(gateway=env.gateway, ref=ds_ref)

    dataframe_fn.__signature__ = inspect.Signature([
        inspect.Parameter("env", inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=BuildEnv),
        inspect.Parameter(d_name, inspect.Parameter.POSITIONAL_OR_KEYWORD, annotation=DatasetRef),
    ])
    dataframe_fn.__name__ = df_name
    dataframe_fn.__doc__ = f"Load {table_key} as pandas DataFrame from {target_name} target."

    return tag(domain=domain, table=table_key, node_type="dataframe")(dataframe_fn)
```

#### 4. Module Registration (`node_factory.py`)

```python
def build_target_module(
    *,
    include_targets: set[str] | None = None,
    exclude_targets: set[str] | None = None,
    include_dataset_nodes: bool = True,
    include_loader_nodes: bool = True,
    include_artifact_nodes: bool = True,
) -> ModuleType:
    # ... target and dataset node generation ...

    # Generate loader nodes
    if include_loader_nodes:
        for table_key in table_keys:
            # Query node (returns Ibis expression)
            q_fn = _create_query_node_function(table_key=table_key, target_name=target.name)
            setattr(module, query_node(table_key), q_fn)
            query_to_node[table_key] = query_node(table_key)

            # DataFrame node (returns pandas DataFrame)
            df_fn = _create_dataframe_node_function(table_key=table_key, target_name=target.name)
            setattr(module, dataframe_node(table_key), df_fn)
            dataframe_to_node[table_key] = dataframe_node(table_key)

    # Attach mappings
    module.QUERY_TO_NODE = query_to_node
    module.DATAFRAME_TO_NODE = dataframe_to_node
```

---

## PR-13: Run Targets Persistence

### Problem

Build history only showed run-level information. There was no way to see per-target execution details for debugging failed or slow builds.

### Solution

Created `build.run_targets` schema and persistence layer for storing per-target execution records.

### Implementation

#### 1. Schema Definition (`schemas.py`)

```python
TableSchema(
    schema="build",
    name="run_targets",
    columns=[
        Column("run_id", "VARCHAR", nullable=False),
        Column("target", "VARCHAR", nullable=False),
        Column("status", "VARCHAR", nullable=False),
        Column("input_hash", "VARCHAR"),
        Column("options_hash", "VARCHAR"),
        Column("duration_ms", "DOUBLE"),
        Column("row_counts", "JSON"),
        Column("error", "VARCHAR"),
        Column("datasets", "JSON"),
        Column("artifacts", "JSON"),
        Column("dep_hashes", "JSON"),
    ],
    primary_key=("run_id", "target"),
)
```

#### 2. Persistence Methods (`build_tracking.py`)

```python
class BuildTracking:
    def save_run_targets(
        self,
        run_id: str,
        records: Sequence[TargetRunRecord],
    ) -> int:
        """Persist per-target execution records for a run.

        Parameters
        ----------
        run_id
            Build run identifier.
        records
            Sequence of TargetRunRecord objects to persist.

        Returns
        -------
        int
            Number of records inserted.
        """
        rows = [self._target_record_to_row(run_id, r) for r in records]
        return self._backend.bulk_insert("build.run_targets", rows)

    def list_run_targets(self, run_id: str) -> list[dict[str, Any]]:
        """List per-target records for a specific run.

        Parameters
        ----------
        run_id
            Run identifier to fetch targets for.

        Returns
        -------
        list[dict[str, Any]]
            List of target execution records as dictionaries.
        """
        sql = "SELECT * FROM build.run_targets WHERE run_id = ?"
        result = self._con.execute(sql, [run_id]).fetchall()
        return [self._parse_run_target_row(row) for row in result]
```

#### 3. History Enhancement (`cli/handlers/build.py`)

```python
def build_history_handler(ctx: CommandContext) -> CliResult[BuildHistoryResult]:
    if run_id:
        record = _lookup_run_by_id(gateway, runtime.snapshot.repo, run_id)
        # Fetch per-target breakdown for this run
        run_targets = gateway.build.list_run_targets(record.run_id)
        return CliResult.ok(
            BuildHistoryResult(
                runs=[record.to_dict()],
                count=1,
                targets=run_targets,  # New field!
            )
        )
```

#### 4. BuildHistoryResult Extension (`result_types.py`)

```python
@dataclass(frozen=True)
class BuildHistoryResult:
    """Result from build history command.

    Attributes
    ----------
    runs
        List of build run records.
    count
        Total number of runs returned.
    targets
        Per-target run records when a specific run_id is queried.
    """

    runs: list[dict[str, Any]]
    count: int
    targets: list[dict[str, Any]] | None = None

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"runs": self.runs, "count": self.count}
        if self.targets is not None:
            result["targets"] = self.targets
        return result
```

---

## PR-14: Graph Exports

### Problem

The JSON DAG export was useful but not easily visualizable. Developers needed to manually process JSON to understand the graph structure.

### Solution

Added Mermaid and Graphviz DOT export formats for direct visualization in documentation, PRs, and debugging tools.

### Implementation

#### 1. Mermaid Export (`observability.py`)

```python
def export_dag_mermaid(
    runtime: HamiltonRuntime,
    targets: list[str],
) -> str:
    """Export DAG as Mermaid flowchart syntax.

    Produces output suitable for embedding in Markdown documents
    or rendering with Mermaid-compatible tools.

    Returns
    -------
    str
        Mermaid flowchart definition.
    """
    dag_info = get_dag_info(runtime, targets)
    lines = ["flowchart TD"]

    # Define nodes with labels
    for node in dag_info["nodes"]:
        name = node["name"]
        module = node["module"]
        lines.append(f"    {name}[{name}]")

    # Define edges
    for edge in dag_info["edges"]:
        lines.append(f"    {edge['from']} --> {edge['to']}")

    return "\n".join(lines)
```

#### 2. DOT Export (`observability.py`)

```python
def export_dag_dot(
    runtime: HamiltonRuntime,
    targets: list[str],
) -> str:
    """Export DAG as Graphviz DOT syntax.

    Produces output suitable for rendering with Graphviz or
    compatible visualization tools.

    Returns
    -------
    str
        Graphviz DOT graph definition.
    """
    dag_info = get_dag_info(runtime, targets)
    lines = ["digraph hamilton_dag {"]
    lines.append("    rankdir=TB;")
    lines.append("    node [shape=box];")

    # Define nodes
    for node in dag_info["nodes"]:
        name = node["name"]
        module = node["module"]
        lines.append(f'    {name} [label="{name}\\n({module})"];')

    # Define edges
    for edge in dag_info["edges"]:
        lines.append(f"    {edge['from']} -> {edge['to']};")

    lines.append("}")
    return "\n".join(lines)
```

#### 3. CLI Format Options (`cli/commands/build.py`)

```python
@dataclass
class BuildGraphCommand:
    output_format: Annotated[
        str,
        Parameter(
            name=["--format", "-f"],
            help="Output format: json (default), mermaid, or dot.",
        ),
    ] = "json"
```

#### 4. Handler Integration (`cli/handlers/build.py`)

```python
def build_graph_handler(ctx: CommandContext) -> CliResult[BuildGraphResult]:
    output_format = ctx.params.get_str("output_format") or "json"

    if output_format == "mermaid":
        output_text = export_dag_mermaid(hamilton_runtime, goals)
    elif output_format == "dot":
        output_text = export_dag_dot(hamilton_runtime, goals)
    else:
        output_text = export_dag_json(hamilton_runtime, goals)

    if output_file:
        Path(output_file).write_text(output_text, encoding="utf-8")
```

---

## PR-15: Explain Staleness

### Problem

When a target was marked for recomputation, users couldn't tell which specific dependencies changed. This made incremental build debugging difficult.

### Solution

Added `dep_hashes` and `prior_dep_hashes` to `PlanEntry`, implemented `explain_staleness()` method, and created `explain_plan()` function for batch explanations.

### Implementation

#### 1. StalenessExplanation Dataclass (`planner.py`)

```python
@dataclass(frozen=True)
class StalenessExplanation:
    """Detailed explanation of why a target is stale.

    Provides a breakdown of what changed between the prior computation
    and the current state.

    Attributes
    ----------
    target
        Target name.
    status
        Plan status (compute, skip, blocked, missing).
    reason
        Reason for the status.
    input_hash_current
        Current computed input hash.
    input_hash_prior
        Prior input hash from manifest (if any).
    changed_deps
        Dependencies whose hashes changed.
    added_deps
        Dependencies that were added since prior computation.
    removed_deps
        Dependencies that were removed since prior computation.
    dep_hashes
        Current dependency hash mapping.
    prior_dep_hashes
        Prior dependency hash mapping.
    """

    target: str
    status: PlanStatus
    reason: PlanReason
    input_hash_current: str | None
    input_hash_prior: str | None
    changed_deps: tuple[str, ...]
    added_deps: tuple[str, ...]
    removed_deps: tuple[str, ...]
    dep_hashes: dict[str, str]
    prior_dep_hashes: dict[str, str]

    def summary(self) -> str:
        """Return human-readable summary."""
        parts = [f"{self.target}: {self.status} - {self.reason}"]
        if self.changed_deps:
            parts.append(f"  changed: {', '.join(self.changed_deps)}")
        if self.added_deps:
            parts.append(f"  added: {', '.join(self.added_deps)}")
        if self.removed_deps:
            parts.append(f"  removed: {', '.join(self.removed_deps)}")
        return "\n".join(parts)
```

#### 2. PlanEntry.explain_staleness Method (`planner.py`)

```python
def explain_staleness(self) -> StalenessExplanation:
    """Generate detailed staleness explanation.

    Computes the set of changed, added, and removed dependencies
    by comparing current dep_hashes with prior_dep_hashes.
    """
    current_deps = set(self.dep_hashes.keys())
    prior_deps = set(self.prior_dep_hashes.keys())

    added = tuple(sorted(current_deps - prior_deps))
    removed = tuple(sorted(prior_deps - current_deps))

    changed: list[str] = []
    for dep in current_deps & prior_deps:
        if self.dep_hashes[dep] != self.prior_dep_hashes.get(dep):
            changed.append(dep)

    return StalenessExplanation(
        target=self.target,
        status=self.status,
        reason=self.reason,
        input_hash_current=self.input_hash,
        input_hash_prior=self.prior_input_hash,
        changed_deps=tuple(sorted(changed)),
        added_deps=added,
        removed_deps=removed,
        dep_hashes=self.dep_hashes,
        prior_dep_hashes=self.prior_dep_hashes,
    )
```

#### 3. explain_plan Function (`planner.py`)

```python
def explain_plan(plan: HamiltonBuildPlan) -> list[StalenessExplanation]:
    """Generate staleness explanations for all targets in a plan.

    Returns
    -------
    list[StalenessExplanation]
        List of explanations, one per target in the plan's closure.
    """
    return [entry.explain_staleness() for entry in plan.entries]
```

#### 4. CLI Command (`cli/commands/build.py`)

```python
@cli_command("build.explain", handler=build_explain_handler, config=_BUILD_CONFIG)
@build_app.command(name="explain")
@dataclass
class BuildExplainCommand:
    """Explain why a target is stale and what dependencies changed."""

    target: Annotated[
        str,
        Parameter(name=None, help="Target name to explain."),
    ]
    force: Annotated[list[str] | None, ...] = None
```

---

## Optional Enhancement: --validate-outputs Flag

### Implementation

Added `validate_outputs: bool = False` field to `BuildEnv` and `--validate-outputs` CLI flag to `BuildRunCommand`. This enables post-write Pandera schema validation:

```python
# CLI flag
validate_outputs: Annotated[
    bool,
    Parameter(
        name=["--validate-outputs"],
        help="Validate produced datasets against Pandera schemas after write.",
        negative=(),
    ),
] = False

# Passed through to BuildEnv
env = BuildEnv(
    ...,
    validate_outputs=validate_outputs,
)
```

---

## Optional Enhancement: Artifact Node Generation

### Implementation

Added `a__*` artifact nodes alongside `q__*` and `df__*` loader nodes:

```python
def _create_artifact_node_function(
    *,
    artifact_name: str,
    target_name: str,
) -> Callable[..., object]:
    """Create a Hamilton node function for artifact access.

    The generated node extracts an ArtifactRef from the parent target's
    TargetRunRecord.
    """
    a_name = artifact_node(artifact_name)
    t_name = target_node(target_name)

    def artifact_fn(env: BuildEnv, **kwargs: object) -> ArtifactRef:
        run_record = kwargs.get(t_name)
        if not isinstance(run_record, TargetRunRecord):
            msg = f"Expected TargetRunRecord for {t_name}, got {type(run_record)}"
            raise TypeError(msg)

        for art in run_record.artifacts:
            if art.name == artifact_name:
                return art

        # Placeholder if artifact not found
        return ArtifactRef(
            name=artifact_name,
            artifact_type="unknown",
            repo=env.repo,
            commit=env.commit,
        )

    return tag(domain=domain, artifact=artifact_name, node_type="artifact")(artifact_fn)
```

Module exports `ARTIFACT_TO_NODE` mapping alongside `QUERY_TO_NODE` and `DATAFRAME_TO_NODE`.

---

## Package Exports Update

### Main Package (`__init__.py`)

```python
__all__ = [
    # ... Phase 1 exports ...
    
    # Phase 2 additions
    "ArtifactRef",
    "HamiltonBuildPlan",
    "PlanEntry",
    "PlanReason",
    "PlanStatus",
    "StalenessExplanation",
    "compute_plan",
    "explain_plan",
    "export_dag_dot",
    "export_dag_mermaid",
]
```

---

## Test Suite

Phase 2 includes a comprehensive test suite in `tests/build/hamilton/`:

| File | Tests | Coverage |
|------|-------|----------|
| `test_pr08_defaults.py` | 8 | Default mode verification |
| `test_pr09_planner.py` | 8 | Plan status matrix, closure order |
| `test_pr10_manifest_index.py` | 5 | Hash cascade, skip logic |
| `test_pr11_datasetref_v2.py` | 10 | DatasetRef/ArtifactRef fields |
| `test_pr12_loader_nodes.py` | 6 | q__/df__ node generation |
| `test_pr13_run_targets.py` | 5 | Schema, persistence, history |
| `test_pr14_graph_exports.py` | 7 | Mermaid/DOT export |
| `test_pr15_explain.py` | 9 | Staleness explanation |

All tests follow the Testing Charter: real components, no monkeypatching, production-parity execution.

---

## Definition of Done Verification

| Criterion | Status |
|-----------|--------|
| 1. Hamilton is default engine with generated mode | ✅ |
| 2. `compute_plan()` provides upfront visibility | ✅ |
| 3. Manifest index eliminates per-target DB calls | ✅ |
| 4. DatasetRef includes repo/commit for lineage | ✅ |
| 5. ArtifactRef tracks non-tabular outputs | ✅ |
| 6. Loader nodes (q__*, df__*) enable typed deps | ✅ |
| 7. Run targets persisted for debugging | ✅ |
| 8. Mermaid/DOT exports available | ✅ |
| 9. Explain staleness shows dep changes | ✅ |
| 10. --validate-outputs flag implemented | ✅ |
| 11. build history --run-id includes targets | ✅ |
| 12. a__* artifact nodes generated | ✅ |

---

## CLI Usage Examples

```bash
# Build with Hamilton (default)
codeintel build run function_metrics

# Show build plan before execution
codeintel build plan function_metrics

# Explain why a target is stale
codeintel build explain function_metrics

# Export DAG as Mermaid
codeintel build graph --module analytics --format mermaid --output dag.mmd

# Export DAG as Graphviz DOT
codeintel build graph --all --format dot --output dag.dot

# Build with output validation
codeintel build run --validate-outputs function_metrics

# View build history with per-target details
codeintel build history --run-id hamilton-20241212-abc123

# Force recompute with generated mode
codeintel build run risk_factors --force risk_factors
```

---

## References

- [Phase 0 Specification](Hamilton_apache_phase0.md)
- [Phase 1 Specification](Hamilton_apache_phase1.md)
- [Phase 2 Specification](Hamilton_apache_phase2.md)
- [Phase 1 Implementation Report](Hamilton_phase1_implementation_report.md)
- [Hamilton Documentation](https://hamilton.dagworks.io/)

