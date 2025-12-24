# Test Helpers Consolidation Implementation Plan

## Goals
- Make test setup production-aligned by default, with a single canonical entry point.
- Consolidate helper surfaces into a small, discoverable API with consistent behavior.
- Reduce bespoke helper code and eliminate duplicate semantics.
- Treat "golden" fixtures as configurable variants of standard fixtures.
- End state: zero compatibility shims; legacy helpers deleted.

## Non-goals
- Preserve backward compatibility or legacy import paths beyond migration.
- Maintain parallel helper stacks in perpetuity.

## Canonical Architecture (Target End State)

### Primary entry point
- `tests/_helpers/scenarios.py`
  - Canonical: `TestScenario` becomes the sole public entry for test setup.
  - `create_test_context` is internal-only and used by scenario internals.

### Fixture catalogs
- `tests/_helpers/fixtures/snapshots.py`
  - `SnapshotVariant` dataclass with normalized formatting.
  - `SnapshotCatalog` or `SnapshotVariants` container with presets.
  - `to_snapshot(repo_root: Path) -> SnapshotRef` uses production formatting rules.
- `tests/_helpers/fixtures/graphs.py`
  - `GraphFixtureSpec` + `GraphFixtureFactory` for all graph generation.
  - Presets: `standard`, `golden`, `layered`, `simple` that differ only by config.
- `tests/_helpers/fixtures/coverage.py`
  - `CoverageFixtureSpec` + `CoverageFixtureFactory` that owns:
    - Seeding test tables.
    - Building fake coverage objects.
    - Default metadata coercion.
- `tests/_helpers/fixtures/repos.py`
  - `RepoFixtureSpec` + `RepoFixtureWriter` unifying:
    - Canonical repo.
    - Callgraph alias repo.
    - Graph metrics repo.
    - Arbitrary tree writing.
- `tests/_helpers/fixtures/rows.py`
  - Unified row factory API:
    - `row_for(table_key: str, **fields)`
    - `blank_row(table_key: str)`
    - `row_list_for(table_key: str, count: int, **overrides)`
  - Supports dataclass rows or dicts consistently.

### Harness alignment
- Hamilton and serving harnesses take `TestScenario` or `ScenarioConfig` as inputs.
- Harness builders are thin wrappers over the scenario, not alternative setup systems.

### Baseline hardening (already applied)
- Canonical schema seeding helper is now `tests/_helpers/schemas.py::ensure_production_schemas`.
  - Uses production DDL (`create_schemas` + metadata DDL) and is idempotent.
  - All tests must call this helper instead of ad-hoc `CREATE SCHEMA` for production schemas.
- Hamilton harness record lookup now includes diagnostic context when records are missing.
  - Missing records surface build error, failed targets, and skipped targets.
- New regression coverage exists for schema seeding idempotency in
  `tests/_helpers/test_schema_seeding.py`.
- Serving snapshot helpers now seed production schemas before creating docs tables.

## Detailed Work Plan

### Phase 0: Design final APIs (no code changes)
- Finalize the canonical API signatures for:
  - Snapshot variants.
  - Scenario builder.
  - Graph fixtures.
  - Coverage fixtures.
  - Repo fixtures.
  - Row factories.
- Decide preset names and expected default values.
- Confirm standard docstrings and modules for each fixture catalog.
- Ensure schema seeding helper is preserved and aligned with the new fixture layout
  (keep in `tests/_helpers/schemas.py` or move to `tests/_helpers/fixtures/schemas.py`).

### Phase 1: Implement canonical fixture catalogs

#### 1. Snapshot variants
- Create `tests/_helpers/fixtures/snapshots.py`.
- Define `SnapshotVariant` with consistent fields (repo, commit, run_id, repo_root).
- Provide standardized variants: DEFAULT, GOLDEN, METRICS, SPAN.
- Ensure snapshot formatting matches production conventions.
- Remove all ad-hoc repo/commit constants after migration.

#### 2. Scenario canonicalization
- Make `TestScenario` the only public helper entry point.
- Move `create_test_context` to a private module or mark as internal.
- Ensure `TestScenario` accepts:
  - `snapshot_variant` (default: DEFAULT).
  - `repo_fixture` (default: canonical repo).
  - `seed_packs`.
  - `file_backed`.
  - `write_files`.
  - `extra` metadata.
- Provide scenario presets like `.minimal()`, `.with_graph()`, `.with_coverage()` but ensure they route through the same core config.

#### 3. Graph fixtures
- Introduce `GraphFixtureSpec` with parameters for:
  - Topology type (chain, star, layered, golden).
  - Size parameters (nodes, edges, layers, density).
  - Directed vs undirected.
  - Optional deterministic labels.
- Implement `GraphFixtureFactory` with presets that map to previous helpers.
- Route `GraphRuntimeDouble` and graph tests through the unified factory.

#### 4. Coverage fixtures
- Create `CoverageFixtureSpec` that controls:
  - Coverage ratios.
  - Test catalog entries.
  - Coverage line behavior.
  - Seeded GOID entries.
- Consolidate seeding and fake coverage creation in one place.
- Keep assertions in `tests/_helpers/assertions/coverage_assertions.py` but have them consume standardized fixture outputs.

#### 5. Repo fixtures
- Create `RepoFixtureSpec` and `RepoFixtureWriter` for all test repos.
- Include canonical, callgraph alias, and graph metrics variants as presets.
- Provide a single tree writer for arbitrary fixtures.
- Route `write_canonical_repo` and all repo writers through this API.

#### 6. Row factories
- Consolidate dataclass builders (`tests/_helpers/builders/*`), ad-hoc row helpers (`tests/_helpers/rows.py`), and schema dict factories (`tests/_helpers/factories/row_factories.py`) into one API.
- Ensure all seed packs use the same row factory API for consistency.

#### 7. Schema seeding helper alignment
- Keep `ensure_production_schemas` as the canonical API (do not regress or delete).
- If fixtures are re-homed, relocate to `tests/_helpers/fixtures/schemas.py` and update imports.

### Phase 2: Migrate all helper consumers

#### Environment + scenario migration
- Update all harnesses and orchestration helpers to accept `TestScenario` or `ScenarioConfig`.
- Replace direct calls to `create_test_context` in tests with `TestScenario` usage.

#### Graph migration
- Replace usage of `tests/_helpers/fakes/networkx_graphs.py`, `tests/_helpers/graphs.py`, and `tests/_helpers/factories/graph_factories.py` with the canonical graph fixture factory.
- Treat golden graphs as presets in the same factory.

#### Coverage migration
- Move all tests and seed packs to use the canonical coverage fixture module.
- Remove direct use of `tests/_helpers/coverage.py`, `tests/_helpers/seeds/coverage.py`, and `tests/_helpers/fakes/coverage.py` after migration.

#### Repo fixture migration
- Update ingestion tests to use `RepoFixtureWriter` with specs instead of bespoke repo setup helpers.

#### Row factory migration
- Update all seed packs and test helpers to emit rows via the unified row factory API.

### Phase 3: Decommission legacy helpers

#### Delete compatibility layers
- Remove `tests/_helpers/defaults.py` and any compatibility imports.
- Remove any temporary shims introduced during migration.

#### Delete legacy modules
- Remove legacy helper modules that are fully superseded, including:
  - `tests/_helpers/fakes/networkx_graphs.py`
  - `tests/_helpers/graphs.py`
  - `tests/_helpers/factories/graph_factories.py`
  - `tests/_helpers/coverage.py`
  - `tests/_helpers/fakes/coverage.py`
  - `tests/_helpers/seeds/coverage.py`
  - `tests/_helpers/orchestration/repo_writers.py`
  - `tests/_helpers/fakes/ingestion_context.py`
  - `tests/_helpers/rows.py`
  - `tests/_helpers/factories/row_factories.py`
  - `tests/_helpers/builders/*` (if replaced fully)

## Migration Matrix (Old -> New)

### Snapshots
- `tests/_helpers/constants.py` -> `tests/_helpers/fixtures/snapshots.py`
- `tests/_helpers/configs/graph_config.py` -> snapshot variants preset
- `tests/_helpers/configs/coverage_config.py` -> snapshot variants preset
- `tests/_helpers/seeds/span.py` -> snapshot variants preset

### Schemas
- Ad-hoc `CREATE SCHEMA` in tests -> `tests/_helpers/schemas.py::ensure_production_schemas`
- `tests/_helpers/schemas.py` -> keep as canonical or move to `fixtures/schemas.py`

### Environment
- `tests/_helpers/context.py` -> `TestScenario` (internal usage only)
- `tests/_helpers/env.py` -> `TestScenario` facade or deletion

### Graphs
- `tests/_helpers/fakes/networkx_graphs.py` -> `fixtures/graphs.py`
- `tests/_helpers/factories/graph_factories.py` -> `fixtures/graphs.py`
- `tests/_helpers/graphs.py` -> `fixtures/graphs.py`
- `tests/_helpers/seeds/golden_graphs.py` -> `fixtures/graphs.py` preset

### Coverage
- `tests/_helpers/coverage.py` -> `fixtures/coverage.py`
- `tests/_helpers/fakes/coverage.py` -> `fixtures/coverage.py`
- `tests/_helpers/seeds/coverage.py` -> `fixtures/coverage.py`

### Repo fixtures
- `tests/_helpers/repo.py` -> `fixtures/repos.py`
- `tests/_helpers/orchestration/repo_writers.py` -> `fixtures/repos.py`
- `tests/_helpers/fakes/ingestion_context.py` -> `fixtures/repos.py`

### Rows
- `tests/_helpers/builders/*` -> `fixtures/rows.py`
- `tests/_helpers/rows.py` -> `fixtures/rows.py`
- `tests/_helpers/factories/row_factories.py` -> `fixtures/rows.py`

## Acceptance Criteria
- All tests use `TestScenario` (or direct `ScenarioConfig`) for setup.
- All snapshot defaults come from `fixtures/snapshots.py` presets.
- All graph fixtures come from a single factory with golden presets.
- Coverage seeding and fake coverage creation live in one module.
- Repo fixture writing is defined by a single `RepoFixtureWriter` API.
- Row factories are centralized and used by all seed packs.
- No legacy helper modules remain in the repo.
- No ad-hoc `CREATE SCHEMA` statements for production schemas remain in tests.
- Missing `TargetRunRecord` errors include build error + target status context.

## Verification Plan
- Run the quality report:
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Run tests in segments based on impacted directories:
  - `uv run pytest -q tests/ingestion`
  - `uv run pytest -q tests/graphs`
  - `uv run pytest -q tests/serving`
  - `uv run pytest -q tests/analytics`

## Risks and Mitigations
- Risk: Large-scale API change leads to inconsistent updates across tests.
  - Mitigation: Use a migration checklist with a file-by-file pass; enforce no legacy helpers remain.
- Risk: Golden fixture parity breaks due to fixture refactor.
  - Mitigation: Encode golden presets in the same factory using existing parameters; keep deterministic seeds.
- Risk: Row factory unification breaks seed pack assumptions.
  - Mitigation: Build parity tests that compare old vs new row outputs for a sample pack during migration, then delete old modules.

## Execution Order Summary
1. Implement snapshot variants and update `TestScenario` to consume them.
2. Introduce unified graph fixture factory and migrate graph consumers.
3. Introduce unified coverage fixtures and migrate coverage consumers.
4. Introduce unified repo fixture writer and migrate ingestion helpers.
5. Introduce unified row factory and migrate seed packs.
6. Remove legacy helper modules and ensure no imports remain.

---

## Phase 1 Detailed Task Checklist (Exact File Edits)

### 1. Create snapshot variants catalog
- Add new module `tests/_helpers/fixtures/snapshots.py`
  - Define `SnapshotVariant` dataclass with fields:
    - `repo: str`
    - `commit: str`
    - `run_id: str | None = None`
    - `repo_root: Path | None = None`
  - Add method `to_snapshot(repo_root: Path | None = None) -> SnapshotRef`
    - Use production formatting (no ad-hoc string patterns).
    - `repo_root` argument overrides dataclass value when provided.
  - Add variant presets:
    - `DEFAULT_VARIANT`
    - `GOLDEN_VARIANT`
    - `METRICS_VARIANT`
    - `SPAN_VARIANT`
  - Add `__all__` with public exports.

### 2. Integrate snapshot variants into scenario configuration
- Update `tests/_helpers/scenarios.py`
  - Add import from `tests._helpers.fixtures.snapshots`.
  - Extend `ScenarioConfig`:
    - Add `snapshot_variant: SnapshotVariant = DEFAULT_VARIANT`
    - Remove `repo` and `commit` fields.
  - Update `TestScenario.with_repo` / `with_commit`:
    - Replace with `with_snapshot_variant(variant: SnapshotVariant) -> Self`.
    - Remove or deprecate `with_repo` and `with_commit`.
  - Update scenario presets to use `snapshot_variant` only.
- Update `tests/_helpers/context.py`
  - Keep `create_test_context` but make it accept:
    - `snapshot_variant: SnapshotVariant | None = None`
  - Replace manual repo/commit usage with `snapshot_variant.to_snapshot(...)`.
  - Ensure `BuildPaths` uses the same repo root as the variant snapshot.

### 3. Wire harnesses to scenario config (only direct wiring for Phase 1)
- Update `tests/_helpers/harnesses/hamilton_build.py`
  - In `HamiltonBuildHarness.open`, replace `EnvOptions`/repo/commit literals with `SnapshotVariant`.
  - Add `HarnessOpenOptions.snapshot_variant: SnapshotVariant | None = None`.
  - Use `snapshot_variant` to populate repo/commit consistently.
- Update `tests/_helpers/hamilton_execution.py`
  - Add builder method `with_snapshot_variant(variant: SnapshotVariant) -> HamiltonTestBuilder`.
  - Remove direct repo/commit setters, or mark them as internal and route through variants.
- Update `tests/_helpers/hamilton_fixtures.py`
  - Replace `snapshot_info` tuple with `snapshot_variant: SnapshotVariant`.

### 4. Consolidate default constants (remove old)
- Update `tests/_helpers/constants.py`
  - Remove `DEFAULT_REPO`, `DEFAULT_COMMIT`, `DEFAULT_RUN_ID`.
  - Keep non-snapshot constants (graph shapes, etc.).
  - Adjust `__all__` accordingly.
- Update `tests/_helpers/defaults.py`
  - Delete the module entirely (compat shim is forbidden).
- Update any imports in Phase 1 touched files to avoid `DEFAULT_REPO/DEFAULT_COMMIT`.

### 5. Update config modules to use snapshot variants
- Update `tests/_helpers/configs/graph_config.py`
  - Replace `REPO`/`COMMIT` constants with a `SnapshotVariant` preset import.
  - Update `GraphEngineSeed` to accept a `snapshot_variant` or `snapshot` object.
- Update `tests/_helpers/configs/coverage_config.py`
  - Replace `REPO`/`COMMIT` constants with a `SnapshotVariant` preset import.
  - Update `CoverageSeedConfig` to use `snapshot_variant` instead of `repo/commit`.

### 6. Update doc references and ensure no legacy references remain
- Update `docs/tests_refinement/helpers_consolidation_plan.md`
  - Add the new snapshot variant module to the architecture list.
- Update `docs/tests_refinement/helpers_wider_deployment.md`
  - Replace any references to `DEFAULT_REPO`/`DEFAULT_COMMIT` with snapshot variants.

### 7. Deletions in Phase 1
- Delete `tests/_helpers/defaults.py` after all imports are updated.
- Remove `DEFAULT_REPO`/`DEFAULT_COMMIT` from `tests/_helpers/constants.py`.

### Phase 1 Validation Steps
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Run targeted tests that exercise the scenario + harness layers:
  - `uv run pytest -q tests/ingestion`
  - `uv run pytest -q tests/serving`

---

## Phase 1 File-by-File Before/After Snippets

### `tests/_helpers/scenarios.py`

Before:
```python
@dataclass
class ScenarioConfig:
    repo: str = DEFAULT_REPO
    commit: str = DEFAULT_COMMIT
    seed_packs: list[SeedPack] = field(default_factory=list)
    file_backed: bool = False
    write_files: bool = False
    extra: dict[str, object] = field(default_factory=dict)

class TestScenario:
    def with_repo(self, repo: str) -> Self:
        self.config.repo = repo
        return self

    def with_commit(self, commit: str) -> Self:
        self.config.commit = commit
        return self
```

After:
```python
@dataclass
class ScenarioConfig:
    snapshot_variant: SnapshotVariant = DEFAULT_VARIANT
    seed_packs: list[SeedPack] = field(default_factory=list)
    file_backed: bool = False
    write_files: bool = False
    extra: dict[str, object] = field(default_factory=dict)

class TestScenario:
    def with_snapshot_variant(self, variant: SnapshotVariant) -> Self:
        self.config.snapshot_variant = variant
        return self
```

### `tests/_helpers/context.py`

Before:
```python
def create_test_context(
    tmp_path: Path,
    options: EnvOptions | None = None,
    *,
    gateway_options: GatewayOptions | None = None,
) -> TestContext:
    env_opts = options or EnvOptions()
    repo_root_path, build_dir_path, db_path = _prepare_paths(tmp_path, env_opts)
    snapshot = SnapshotRef(repo=env_opts.repo, commit=env_opts.commit, repo_root=repo_root_path)
```

After:
```python
def create_test_context(
    tmp_path: Path,
    options: EnvOptions | None = None,
    *,
    gateway_options: GatewayOptions | None = None,
    snapshot_variant: SnapshotVariant | None = None,
) -> TestContext:
    env_opts = options or EnvOptions()
    repo_root_path, build_dir_path, db_path = _prepare_paths(tmp_path, env_opts)
    variant = snapshot_variant or DEFAULT_VARIANT
    snapshot = variant.to_snapshot(repo_root=repo_root_path)
```

### `tests/_helpers/harnesses/hamilton_build.py`

Before:
```python
@dataclass(frozen=True)
class HarnessOpenOptions:
    repo_strategy: RepoStrategy = "canonical"
    repo_writer: RepoWriter | None = None
    seed_packs: Sequence[SeedPack] = ()
    gateway_options: GatewayOptions | None = None
    tools_config: ToolsConfig | None = None
    providers: Providers | None = None
    build_config: BuildConfig | None = None
```

After:
```python
@dataclass(frozen=True)
class HarnessOpenOptions:
    repo_strategy: RepoStrategy = "canonical"
    repo_writer: RepoWriter | None = None
    seed_packs: Sequence[SeedPack] = ()
    gateway_options: GatewayOptions | None = None
    tools_config: ToolsConfig | None = None
    providers: Providers | None = None
    build_config: BuildConfig | None = None
    snapshot_variant: SnapshotVariant | None = None
```

### `tests/_helpers/hamilton_execution.py`

Before:
```python
class HamiltonTestBuilder:
    repo_slug: str = DEFAULT_REPO
    commit_sha: str = DEFAULT_COMMIT

    def with_repo_info(self, repo: str, commit: str) -> HamiltonTestBuilder:
        self.repo_slug = repo
        self.commit_sha = commit
        return self
```

After:
```python
class HamiltonTestBuilder:
    snapshot_variant: SnapshotVariant = DEFAULT_VARIANT

    def with_snapshot_variant(self, variant: SnapshotVariant) -> HamiltonTestBuilder:
        self.snapshot_variant = variant
        return self
```

### `tests/_helpers/hamilton_fixtures.py`

Before:
```python
@dataclass(frozen=True)
class BuildEnvOptions:
    snapshot_info: tuple[str, str] = ("test/repo", "abc123")
```

After:
```python
@dataclass(frozen=True)
class BuildEnvOptions:
    snapshot_variant: SnapshotVariant = DEFAULT_VARIANT
```

### `tests/_helpers/constants.py`

Before:
```python
DEFAULT_REPO: str = "demo/repo"
DEFAULT_COMMIT: str = "deadbeef"
DEFAULT_RUN_ID: str = "test-run-001"
```

After:
```python
LAYERED_DAG_SHAPES: tuple[tuple[int, ...], ...] = ((2, 3, 2), (3, 3, 3), (2, 2, 2, 2))
BRIDGE_COUNTS: tuple[int, ...] = (1, 2, 3)
WEIGHTED_CYCLE_SIZES: tuple[int, ...] = (3, 4, 5)
```

### `tests/_helpers/configs/graph_config.py`

Before:
```python
REPO = "demo/repo"
COMMIT = "deadbeef"

@dataclass(frozen=True)
class GraphEngineSeed:
    repo: str = "test/metrics"
    commit: str = "metrics123"
```

After:
```python
from tests._helpers.fixtures.snapshots import METRICS_VARIANT

@dataclass(frozen=True)
class GraphEngineSeed:
    snapshot_variant: SnapshotVariant = METRICS_VARIANT
```

### `tests/_helpers/configs/coverage_config.py`

Before:
```python
REPO = "demo/repo"
COMMIT = "deadbeef"

@dataclass(frozen=True)
class CoverageSeedConfig:
    repo: str = REPO
    commit: str = COMMIT
```

After:
```python
from tests._helpers.fixtures.snapshots import DEFAULT_VARIANT

@dataclass(frozen=True)
class CoverageSeedConfig:
    snapshot_variant: SnapshotVariant = DEFAULT_VARIANT
```

### `tests/_helpers/defaults.py`

Before:
```python
from tests._helpers.constants import DEFAULT_COMMIT, DEFAULT_REPO, DEFAULT_RUN_ID
__all__ = ["DEFAULT_COMMIT", "DEFAULT_REPO", "DEFAULT_RUN_ID"]
```

After:
```python
# File deleted (compat shim removed).
```

---

## Phase 2 Detailed Task Checklist (Exact File Edits)

### 1. Create unified graph fixture factory
- Add new module `tests/_helpers/fixtures/graphs.py`
  - Define `GraphFixtureSpec` dataclass with fields:
    - `kind: Literal["chain", "star", "cycle", "layered", "golden", "custom"]`
    - `directed: bool = True`
    - `nodes: int | None = None`
    - `edges: int | None = None`
    - `layers: tuple[int, ...] | None = None`
    - `spokes: int | None = None`
    - `cycle_size: int | None = None`
    - `seed: int | None = None`
  - Define `GraphFixtureFactory` with:
    - `build(spec: GraphFixtureSpec) -> nx.Graph`
    - Presets: `STANDARD_CALL`, `STANDARD_IMPORT`, `GOLDEN_CALL`, `GOLDEN_IMPORT`.
  - Ensure golden fixtures are built via the same factory and specs as standard fixtures.

### 2. Migrate graph helper modules
- Update `tests/_helpers/graphs.py`
  - Replace direct NetworkX construction with `GraphFixtureFactory` usage.
  - Replace `standard_graph_fixtures` to return a bundle built from specs.
  - Remove `call_chain_graph`, `call_star_graph`, `import_cycle_graph`, `symbol_star_graph` or make them thin wrappers around the factory.
- Update `tests/_helpers/fakes/networkx_graphs.py`
  - Replace all implementations with calls to `GraphFixtureFactory`.
  - Keep constant names if still referenced, but remove duplicated logic.
- Update `tests/_helpers/factories/graph_factories.py`
  - Remove bespoke implementations and delegate to the unified factory.

### 3. Migrate golden graph seed module
- Update `tests/_helpers/seeds/golden_graphs.py`
  - Replace hardcoded graph construction with factory presets.
  - Ensure seeded stats align with the `GraphFixtureSpec` values.

### 4. Update graph runtime/test doubles
- Update `tests/_helpers/fakes/graph_runtime.py`
  - Change `from_fixtures` to accept `GraphFixtureSpec` or use presets.
  - Remove any duplicated graph defaults.

### 5. Remove legacy graph modules after migration
- Delete `tests/_helpers/fakes/networkx_graphs.py` once no longer referenced.
- Delete `tests/_helpers/factories/graph_factories.py` once no longer referenced.
- Reduce `tests/_helpers/graphs.py` to a thin facade or remove if unused.

### Phase 2 Validation Steps
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Run targeted graph tests:
  - `uv run pytest -q tests/graphs`
  - `uv run pytest -q tests/analytics`

---

## Phase 2 File-by-File Before/After Snippets

### `tests/_helpers/fakes/networkx_graphs.py`

Before:
```python
def chain_graph(length: int = DEFAULT_CHAIN_LENGTH) -> nx.DiGraph:
    g = nx.DiGraph()
    ...
    return g

def star_graph(spokes: int = DEFAULT_SPOKES, *, inward: bool = False) -> nx.DiGraph:
    g = nx.DiGraph()
    ...
    return g
```

After:
```python
def chain_graph(length: int = DEFAULT_CHAIN_LENGTH) -> nx.DiGraph:
    spec = GraphFixtureSpec(kind="chain", nodes=length, directed=True)
    return cast("nx.DiGraph", GraphFixtureFactory.build(spec))

def star_graph(spokes: int = DEFAULT_SPOKES, *, inward: bool = False) -> nx.DiGraph:
    spec = GraphFixtureSpec(kind="star", spokes=spokes, directed=True)
    graph = cast("nx.DiGraph", GraphFixtureFactory.build(spec))
    return graph.reverse(copy=True) if inward else graph
```

### `tests/_helpers/graphs.py`

Before:
```python
def standard_graph_fixtures(
    *,
    chain_length: int = DEFAULT_CHAIN_LENGTH,
    cycle_size: int = DEFAULT_CYCLE_SIZE,
    star_spokes: int = DEFAULT_SPOKES,
) -> GraphFixtures:
    return GraphFixtures(
        call_graph=call_chain_graph(chain_length),
        import_graph=import_cycle_graph(cycle_size),
        config_graph=nx.Graph(),
        symbol_module_graph=symbol_star_graph(star_spokes),
        symbol_function_graph=symbol_star_graph(star_spokes),
        cfg_graph=nx.DiGraph(),
    )
```

After:
```python
def standard_graph_fixtures(
    *,
    chain_length: int = DEFAULT_CHAIN_LENGTH,
    cycle_size: int = DEFAULT_CYCLE_SIZE,
    star_spokes: int = DEFAULT_SPOKES,
) -> GraphFixtures:
    call_spec = GraphFixtureSpec(kind="chain", nodes=chain_length, directed=True)
    import_spec = GraphFixtureSpec(kind="cycle", cycle_size=cycle_size, directed=True)
    symbol_spec = GraphFixtureSpec(kind="star", spokes=star_spokes, directed=False)
    return GraphFixtures(
        call_graph=cast("nx.DiGraph", GraphFixtureFactory.build(call_spec)),
        import_graph=cast("nx.DiGraph", GraphFixtureFactory.build(import_spec)),
        config_graph=nx.Graph(),
        symbol_module_graph=cast("nx.Graph", GraphFixtureFactory.build(symbol_spec)),
        symbol_function_graph=cast("nx.Graph", GraphFixtureFactory.build(symbol_spec)),
        cfg_graph=nx.DiGraph(),
    )
```

### `tests/_helpers/factories/graph_factories.py`

Before:
```python
def build_layered_call_graph() -> nx.DiGraph:
    g = nx.DiGraph()
    ...
    return g
```

After:
```python
def build_layered_call_graph() -> nx.DiGraph:
    spec = GraphFixtureSpec(kind="layered", directed=True, layers=(4, 5, 3, 2))
    return cast("nx.DiGraph", GraphFixtureFactory.build(spec))
```

### `tests/_helpers/seeds/golden_graphs.py`

Before:
```python
def _build_goids(repo: str, commit: str) -> list[GoidRow]:
    ...
```

After:
```python
def _build_goids(repo: str, commit: str) -> list[GoidRow]:
    call_spec = GraphFixtureSpec(kind="golden", directed=True)
    call_graph = cast("nx.DiGraph", GraphFixtureFactory.build(call_spec))
    ...
```

### `tests/_helpers/fakes/graph_runtime.py`

Before:
```python
@classmethod
def from_fixtures(
    cls,
    fixtures: GraphFixtures | None = None,
    *,
    gateway: StorageGateway | None = None,
    snapshot: SnapshotRef | None = None,
    backend: GraphBackendConfig | None = None,
    copy_graphs: bool = True,
) -> GraphRuntimeDouble:
    if fixtures is None:
        graphs_module = import_module("tests._helpers.graphs")
        graphs = graphs_module.standard_graph_fixtures()
    else:
        graphs = fixtures
```

---

## Phase 3 Detailed Task Checklist (Exact File Edits)

### 1. Create unified coverage fixture module
- Add new module `tests/_helpers/fixtures/coverage.py`
  - Define `CoverageFixtureSpec` dataclass with fields:
    - `include_catalog: bool = True`
    - `include_edges: bool = True`
    - `include_functions: bool = True`
    - `include_lines: bool = True`
    - `passing_ratio: float = 0.75`
    - `edge_meta: Mapping[str, object] | None = None`
    - `test_meta: Mapping[str, object] | None = None`
  - Define `CoverageFixtureFactory` with:
    - `seed(ctx: TestContext, spec: CoverageFixtureSpec) -> None`
    - `build_fake_coverage(ctx: TestContext) -> FakeCoverage`
  - Normalize coercion helpers (`_coerce_int`, `_coerce_float`) inside this module.

### 2. Migrate coverage helpers to unified module
- Update `tests/_helpers/coverage.py`
  - Replace seeding and coercion logic with calls into `fixtures/coverage.py`.
  - Keep only thin wrappers (if any) during migration, then delete.
- Update `tests/_helpers/fakes/coverage.py`
  - Remove direct implementation and use `CoverageFixtureFactory.build_fake_coverage`.
- Update `tests/_helpers/seeds/coverage.py`
  - Replace `CoveragePack` internals to call the unified fixture factory.
  - Ensure `CoveragePack` remains a simple adapter over the new factory.

### 3. Align coverage lines seeding
- Update `tests/_helpers/seeds/coverage_lines.py`
  - Route line seeding through `CoverageFixtureFactory` when possible.
  - Remove duplicate default line ranges.

### 4. Update coverage assertions to consume unified fixtures
- Update `tests/_helpers/assertions/coverage_assertions.py`
  - Ensure expectations match the new fixture defaults.
  - Remove any assumptions tied to legacy seed layouts.

### 5. Remove legacy coverage modules after migration
- Delete `tests/_helpers/coverage.py` once no longer referenced.
- Delete `tests/_helpers/fakes/coverage.py` once no longer referenced.
- Delete `tests/_helpers/seeds/coverage.py` once no longer referenced.

### Phase 3 Validation Steps
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Run targeted coverage tests:
  - `uv run pytest -q tests/coverage`
  - `uv run pytest -q tests/analytics`

---

## Phase 3 File-by-File Before/After Snippets

### `tests/_helpers/coverage.py`

Before:
```python
def seed_goid(
    con: DuckDBPyConnection,
    snapshot: SnapshotRef,
    data: GoidSeedData,
) -> None:
    con.execute(
        \"\"\"
        INSERT INTO core.goids (
            urn, repo, commit, rel_path, language, kind, qualname, goid_h128,
            start_line, end_line, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NOW())
        \"\"\",
        [
            data.urn,
            snapshot.repo,
            snapshot.commit,
            data.rel_path,
            data.language,
            data.kind,
            data.qualname,
            data.goid_h128,
            data.start_line,
            data.end_line,
        ],
    )
```

After:
```python
def seed_goid(
    con: DuckDBPyConnection,
    snapshot: SnapshotRef,
    data: GoidSeedData,
) -> None:
    CoverageFixtureFactory.seed_goid(con, snapshot, data)
```

### `tests/_helpers/fakes/coverage.py`

Before:
```python
def build_fake_coverage_from_gateway(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> FakeCoverage:
    ...
```

After:
```python
def build_fake_coverage_from_gateway(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> FakeCoverage:
    ctx = TestContext(snapshot=snapshot, gateway=gateway, build_paths=BuildPaths.from_explicit())
    return CoverageFixtureFactory.build_fake_coverage(ctx)
```

### `tests/_helpers/seeds/coverage.py`

Before:
```python
@dataclass
class CoveragePack:
    include_catalog: bool = True
    include_edges: bool = True
    include_functions: bool = True
    passing_ratio: float = 0.75

    def apply(self, ctx: TestContext) -> None:
        ...
```

After:
```python
@dataclass
class CoveragePack:
    include_catalog: bool = True
    include_edges: bool = True
    include_functions: bool = True
    passing_ratio: float = 0.75

    def apply(self, ctx: TestContext) -> None:
        spec = CoverageFixtureSpec(
            include_catalog=self.include_catalog,
            include_edges=self.include_edges,
            include_functions=self.include_functions,
            passing_ratio=self.passing_ratio,
        )
        CoverageFixtureFactory.seed(ctx, spec)
```

### `tests/_helpers/seeds/coverage_lines.py`

Before:
```python
def apply(self, ctx: TestContext) -> None:
    rows = [
        CoverageLineRow(...),
        CoverageLineRow(...),
    ]
    insert_rows(ctx.gateway, rows)
```

After:
```python
def apply(self, ctx: TestContext) -> None:
    spec = CoverageFixtureSpec(include_lines=True)
    CoverageFixtureFactory.seed(ctx, spec)
```

---

## Phase 4 Detailed Task Checklist (Exact File Edits)

### 1. Create unified repo fixture writer
- Add new module `tests/_helpers/fixtures/repos.py`
  - Define `RepoFixtureSpec` dataclass with fields:
    - `kind: Literal["canonical", "callgraph_alias", "graph_metrics", "custom"]`
    - `repo_root: Path | None = None`
    - `files: Mapping[str, str] | None = None` (for custom trees)
    - `module_map: Mapping[str, str] | None = None`
  - Define `RepoFixtureWriter` with:
    - `write(spec: RepoFixtureSpec) -> RepoFixture`
    - `write_tree(root: Path, files: Mapping[str, str]) -> RepoFixture`
  - Define `RepoFixture` dataclass:
    - `files: tuple[Path, ...]`
    - `module_map: dict[str, str]`
    - `module_paths() -> list[str]`

### 2. Migrate canonical repo helpers
- Update `tests/_helpers/repo.py`
  - Replace `write_canonical_repo` implementation with `RepoFixtureWriter`.
  - Move `CanonicalRepo` into `fixtures/repos.py` or replace with `RepoFixture`.
  - Keep GOID constants if still used by seed packs (remove after phase 5).

### 3. Migrate orchestration repo writers
- Update `tests/_helpers/orchestration/repo_writers.py`
  - Replace `write_sample_repo`, `write_callgraph_alias_repo`, and `write_graph_metrics_repo`
    with calls into `RepoFixtureWriter` presets.
  - Remove `_write_file` and any direct `Path.write_text` utilities.

### 4. Migrate ingestion test helpers
- Update `tests/_helpers/fakes/ingestion_context.py`
  - Replace `build_repo_tree` with `RepoFixtureWriter.write_tree`.
  - Remove file writing logic from this module.

### 5. Update module expectation helpers
- Update `tests/_helpers/modules_expectations.py`
  - Ensure module expectation helpers accept `RepoFixture` or `RepoFixtureSpec` where useful.
  - Remove duplicate module map assembly logic in favor of fixture output.

### 6. Remove legacy repo writer modules after migration
- Delete `tests/_helpers/orchestration/repo_writers.py` after all references are updated.
- Delete `tests/_helpers/fakes/ingestion_context.py` after all references are updated.
- Reduce `tests/_helpers/repo.py` to constants or delete if unused after phase 5.

### Phase 4 Validation Steps
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Run targeted ingestion and repo-related tests:
  - `uv run pytest -q tests/ingestion`
  - `uv run pytest -q tests/build`

---

## Phase 4 File-by-File Before/After Snippets

### `tests/_helpers/orchestration/repo_writers.py`

Before:
```python
def write_callgraph_alias_repo(repo_root: Path) -> list[Path]:
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    ...
    return files
```

After:
```python
def write_callgraph_alias_repo(repo_root: Path) -> list[Path]:
    spec = RepoFixtureSpec(kind="callgraph_alias", repo_root=repo_root)
    fixture = RepoFixtureWriter.write(spec)
    return list(fixture.files)
```

### `tests/_helpers/fakes/ingestion_context.py`

Before:
```python
def build_repo_tree(root: Path, files: Mapping[str, str]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for rel_path, content in files.items():
        target = root / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
    return root
```

After:
```python
def build_repo_tree(root: Path, files: Mapping[str, str]) -> Path:
    RepoFixtureWriter.write_tree(root, files)
    return root
```

### `tests/_helpers/repo.py`

Before:
```python
def write_canonical_repo(repo_root: Path) -> CanonicalRepo:
    pkg_dir = repo_root / "pkg"
    pkg_dir.mkdir(parents=True, exist_ok=True)
    ...
    return CanonicalRepo(...)
```

After:
```python
def write_canonical_repo(repo_root: Path) -> RepoFixture:
    spec = RepoFixtureSpec(kind="canonical", repo_root=repo_root)
    return RepoFixtureWriter.write(spec)
```

---

## Phase 5 Detailed Task Checklist (Exact File Edits)

### 1. Create unified row factory API
- Add new module `tests/_helpers/fixtures/rows.py`
  - Define `RowFactory` with:
    - `blank_row(table_key: str) -> Mapping[str, object]`
    - `row_for(table_key: str, **fields: object) -> Mapping[str, object]`
    - `rows_for(table_key: str, count: int, **overrides: object) -> list[Mapping[str, object]]`
  - Define `RowCoercions` helpers for common conversions (int, bool, JSON, timestamps).
  - Optionally provide adapters to emit dataclass rows for insertable tables.

### 2. Migrate dataclass builders
- Update `tests/_helpers/builders/__init__.py`
  - Keep dataclass rows but mark them internal (no direct usage in tests).
  - Add thin adapters to convert RowFactory output into dataclass rows when required.
- Update `tests/_helpers/builders/analytics.py`, `tests/_helpers/builders/core.py`,
  `tests/_helpers/builders/graph.py`
  - Remove row-construction helpers that duplicate `RowFactory` defaults.
  - Keep only row dataclasses and `insert_rows` protocol.

### 3. Migrate ad-hoc row helpers
- Update `tests/_helpers/rows.py`
  - Replace all row helper functions with wrappers around `RowFactory`.
  - Remove local coercion logic (use `RowCoercions`).
  - Delete the module once all consumers migrate to `fixtures/rows.py`.

### 4. Migrate schema-based row factories
- Update `tests/_helpers/factories/row_factories.py`
  - Replace `blank_*` and `sample_*` helpers with `RowFactory` presets.
  - Move any sample data generation into `RowFactory` presets.
  - Delete this module after migration is complete.

### 5. Migrate seed packs to RowFactory
- Update seed packs in `tests/_helpers/seeds/*` to use `RowFactory`.
  - Ensure each pack’s defaults map to `RowFactory` presets.
  - Remove pack-specific row builders in favor of shared defaults.

### 6. Remove legacy row modules after migration
- Delete `tests/_helpers/rows.py` after all references are updated.
- Delete `tests/_helpers/factories/row_factories.py` after all references are updated.
- Remove any unused row helper functions in `tests/_helpers/builders/*`.

### Phase 5 Validation Steps
- Run `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Run targeted tests that exercise row/seed behavior:
  - `uv run pytest -q tests/analytics`
  - `uv run pytest -q tests/build`

---

## Phase 5 File-by-File Before/After Snippets

### `tests/_helpers/rows.py`

Before:
```python
def function_metrics_row(
    *,
    goid: int,
    rel_path: str,
    qualname: str,
    snapshot: tuple[str, str] = (DEFAULT_REPO, DEFAULT_COMMIT),
    metrics: Mapping[str, int | str | bool | datetime | float | None] | None = None,
) -> FunctionMetricsRow:
    ...
```

After:
```python
def function_metrics_row(
    *,
    goid: int,
    rel_path: str,
    qualname: str,
    snapshot: SnapshotVariant = DEFAULT_VARIANT,
    metrics: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    return RowFactory.row_for(
        "analytics.function_metrics",
        function_goid_h128=goid,
        rel_path=rel_path,
        qualname=qualname,
        **(metrics or {}),
    )
```

### `tests/_helpers/factories/row_factories.py`

Before:
```python
def blank_file_profile_row() -> FileProfileRowModel:
    return cast("FileProfileRowModel", dict.fromkeys(FILE_PROFILE_COLUMNS))
```

After:
```python
def blank_file_profile_row() -> Mapping[str, object]:
    return RowFactory.blank_row("analytics.file_profile")
```

### `tests/_helpers/builders/analytics.py`

Before:
```python
@dataclass(frozen=True)
class FunctionMetricsRow:
    ...
    def to_tuple(self) -> tuple[...]:
        ...
```

After:
```python
@dataclass(frozen=True)
class FunctionMetricsRow:
    ...
    def to_tuple(self) -> tuple[...]:
        ...

def function_metrics_row_from_mapping(
    fields: Mapping[str, object],
) -> FunctionMetricsRow:
    return FunctionMetricsRow(**fields)
```

### `tests/_helpers/seeds/core.py`

Before:
```python
goid_rows.append(
    GoidRow(
        goid_h128=meta.goid,
        urn=f"urn:codeintel:{ctx.repo}:{ctx.commit}:{meta.rel_path}#{meta.qualname}",
        repo=ctx.repo,
        commit=ctx.commit,
        rel_path=meta.rel_path,
        kind="function",
        qualname=meta.qualname,
        start_line=meta.start_line,
        end_line=meta.end_line,
        language="python",
        created_at=now,
    )
)
```

After:
```python
row_data = RowFactory.row_for(
    "core.goids",
    goid_h128=meta.goid,
    rel_path=meta.rel_path,
    qualname=meta.qualname,
    start_line=meta.start_line,
    end_line=meta.end_line,
    language="python",
)
goid_rows.append(GoidRow(**row_data))
```

---

## Phase 6 Detailed Task Checklist (Legacy Removal + Final Verification)

### 1. Legacy removal sweep
- Remove any remaining compatibility shims or transitional wrappers.
- Delete legacy helper modules that were kept temporarily:
  - `tests/_helpers/defaults.py`
  - `tests/_helpers/coverage.py`
  - `tests/_helpers/fakes/coverage.py`
  - `tests/_helpers/seeds/coverage.py`
  - `tests/_helpers/fakes/networkx_graphs.py`
  - `tests/_helpers/factories/graph_factories.py`
  - `tests/_helpers/orchestration/repo_writers.py`
  - `tests/_helpers/fakes/ingestion_context.py`
  - `tests/_helpers/rows.py`
  - `tests/_helpers/factories/row_factories.py`
- Reduce or delete `tests/_helpers/graphs.py` and `tests/_helpers/repo.py` if unused.

### 2. Update package exports
- Update `tests/_helpers/__init__.py`
  - Remove all exports tied to deleted modules.
  - Ensure public exports only include canonical fixture modules and `TestScenario`.
- Update `tests/_helpers/seeds/__init__.py` if any pack names changed.
- Update `tests/_helpers/factories/__init__.py` to remove old exports.

### 3. Remove stray imports and references
- Run a repo-wide search for deleted module paths.
- Remove or replace any remaining imports in tests and helpers.
 - Confirm no ad-hoc `CREATE SCHEMA` for production schemas remains.

### 4. Final verification gates
- Run the quality report:
  - `uv run python -m tools.quality_report --output build/quality-results/quality_report.json`
- Run segmented tests in order of impact:
  - `uv run pytest -q tests/ingestion`
  - `uv run pytest -q tests/graphs`
  - `uv run pytest -q tests/coverage`
  - `uv run pytest -q tests/analytics`
  - `uv run pytest -q tests/serving`
  - `uv run pytest -q tests/build`

### 5. Documentation cleanup
- Update `docs/tests_refinement/helpers_consolidation_plan.md` to reflect the final module map.
- Remove references to legacy helpers from `docs/tests_refinement/helpers_wider_deployment.md`.

### Phase 6 Completion Notes
- Legacy modules removed (defaults/coverage/rows/graphs/repo shims, graph_factories, repo_writers).
- Canonical surfaces: `tests/_helpers/fixtures/*` + `tests/_helpers/scenarios.py`.
- Public exports: `tests/_helpers/__init__.py` and `tests/_helpers/seeds/__init__.py` now map to
  canonical fixtures and packs only.

---

## Phase 6 File-by-File Before/After Snippets

### `tests/_helpers/__init__.py`

Before:
```python
from tests._helpers.coverage import build_fake_coverage, seed_coverage_pack
from tests._helpers.graphs import standard_graph_fixtures
from tests._helpers.rows import function_metrics_row
```

After:
```python
from tests._helpers.fixtures.coverage import CoverageFixtureFactory, CoverageFixtureSpec
from tests._helpers.fixtures.graphs import GraphFixtureFactory, GraphFixtureSpec
from tests._helpers.fixtures.rows import RowFactory
from tests._helpers.scenarios import TestScenario
```

### `tests/_helpers/seeds/__init__.py`

Before:
```python
from tests._helpers.seeds.coverage import COVERAGE_PACK, CoveragePack
```

After:
```python
from tests._helpers.seeds.coverage import COVERAGE_PACK, CoveragePack
# Ensure coverage pack is the adapter that uses CoverageFixtureFactory.
```

### `docs/tests_refinement/helpers_wider_deployment.md`

Before:
```markdown
Use DEFAULT_REPO / DEFAULT_COMMIT and build rows manually.
```

After:
```markdown
Use snapshot variants via `tests._helpers.fixtures.snapshots`.
```

After:
```python
@classmethod
def from_fixtures(
    cls,
    fixtures: GraphFixtures | None = None,
    *,
    gateway: StorageGateway | None = None,
    snapshot: SnapshotRef | None = None,
    backend: GraphBackendConfig | None = None,
    copy_graphs: bool = True,
    call_spec: GraphFixtureSpec | None = None,
    import_spec: GraphFixtureSpec | None = None,
) -> GraphRuntimeDouble:
    if fixtures is None:
        graphs = standard_graph_fixtures(
            chain_length=call_spec.nodes if call_spec else DEFAULT_CHAIN_LENGTH,
            cycle_size=import_spec.cycle_size if import_spec else DEFAULT_CYCLE_SIZE,
        )
    else:
        graphs = fixtures
