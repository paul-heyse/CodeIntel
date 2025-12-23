## Context
Modules-first helpers improved ingestion test realism, but other target families still
run through ad-hoc or stubbed helpers. Tool-driven ingestion (pytest, coverage, scip)
needs deterministic outputs, and skip/recompute behavior needs manifest lifecycle
helpers. The test suite already uses a hexagonal architecture and Hamilton targets, so
new helpers should reuse production execution paths with deterministic I/O.

## Goals / Non-Goals
- Goals:
  - Execute Hamilton targets in tests via production `HamiltonBuildExecutor`.
  - Provide deterministic tool outputs, artifacts, and failure modes for tool plugins.
  - Provide harnesses per target family (graph, analytics, serving).
  - Provide manifest lifecycle helpers for skip and recompute validation.
  - Provide concise, schema-aware assertions for TargetRunRecord and datasets.
  - Provide deterministic repo fixtures and table snapshots for regression tests.
- Non-Goals:
  - No changes to production tool plugins or target logic.
  - No expansion of CLI behavior or runtime configuration.
  - No new external dependencies beyond current test stack.

## Current State
- `tests/_helpers/hamilton_execution.py` still relies on stubbed execution paths.
- Tool plugins are partially simulated (coverage/pyrefly only), limiting realistic
  ingestion tests.
- Manifest lifecycle tests are limited because helpers to prime/load manifests are
  missing.
- Graph, analytics, and serving targets have no dedicated harnesses or assertions.

## Architecture Overview
### 1) Base Hamilton Harness
Provide a single production-parity harness built around `TestContext`, `BuildEnv`,
and `HamiltonBuildExecutor`. It executes real Hamilton DAGs and returns
`TargetRunRecord`s.

Key characteristics:
- Multi-target execution runs a shared dependency closure once.
- `BuildEnv` overrides are handled via `dataclasses.replace` (frozen dataclass).
- Supports memory and on-disk gateway modes (on-disk required for threadpool tests).

### 2) Tool Realism Layer
Two complementary paths:
- **FakeToolRunner** (default): deterministic, fast file/stdout outputs and failure
  modes, injected via `ToolsConfig`/`Providers`.
- **ToolSandbox** (integration): stub binaries on PATH to exercise the real
  subprocess execution path.

### 3) Target-Family Harness Wrappers
Thin wrappers around the base harness to provide target sets and convenience
assertions per domain:
- Graph harness: call/import graph targets and graph metrics.
- Analytics harness: function metrics, risk/hotspot targets.
- Serving harness: snapshot publish + search index artifacts.

### 4) Manifest Lifecycle Helpers
Helpers to load and prime manifests and assert skip/recompute behavior. This
unlocks deterministic skip logic tests without expensive upstream execution.

### 5) Assertion Helpers
Reusable assertions for `TargetRunRecord` status, dataset/artifact presence,
row_counts consistency, and schema validation using `SCHEMA_REGISTRY`.

### 6) Deterministic Repo Fixtures
Repo writers that emit stable structures (monorepos, generated file noise, large
files, scope paths) with expected module inventories to compare against.

### 7) Table Snapshot Utilities
Deterministic snapshotting for tables with stable ordering and readable diffs.

### 8) Upstream Status Guard Helpers
Centralize dependency-status checks to avoid duplicated patterns and to standardize
rules for accepting `skipped` as cached success when appropriate.

### 9) Build Plan/Status Harness
Helpers to compute plans and status for a harness and compare results with compact
diffs for regression tests.

### 10) Config Override Helpers
Helpers that write build config sections and reload BuildConfig into a harness,
enabling deterministic options-hash tests.

### 11) Repo Fixture Registry
Registry mapping fixture tags to repo writers and expected module inventories so
tests can request fixtures by intent rather than by function name.

### 12) Failure Scenario Helpers
Assertions for partial/failed TargetRunRecord bundles and common error surfaces.

## Interfaces (Execution-Ready API Sketches)
### Base Harness (`tests/_helpers/harnesses/hamilton_build.py`)
```python
@dataclass
class HamiltonBuildHarness:
    ctx: TestContext
    env: BuildEnv
    executor: HamiltonBuildExecutor

    @classmethod
    def open(..., *, gateway_mode: Literal["memory", "disk"] = "memory") -> HamiltonBuildHarness
    @classmethod
    def wrap(ctx: TestContext, ...) -> HamiltonBuildHarness

    def run_targets(self, targets: Sequence[str]) -> dict[str, TargetRunRecord]
    def run(self, *targets: str) -> HamiltonBuildResult
    def record(self, target: str) -> TargetRunRecord

    def with_force_targets(self, *targets: str) -> HamiltonBuildHarness
    def with_profile(self, profile: str | None) -> HamiltonBuildHarness
    def with_config(self, config: BuildConfig) -> HamiltonBuildHarness

    @property
    def artifacts(self) -> HarnessArtifacts
    @property
    def manifests(self) -> ManifestHelpers
```

### Tool Realism
```python
class FakeToolRunner:
    payloads: dict[str, bytes | str]
    stdout_payloads: dict[str, str]
    returncodes: dict[str, int]
    raise_on: set[str]
    not_found: set[str]

class ToolSandbox:
    @classmethod
    def create(tmp_path: Path) -> ToolSandbox
    def install_stub(self, name: str, *, stdout: str = "", returncode: int = 0) -> None
    def tools_config(self) -> ToolsConfig
    def path_env(self) -> Mapping[str, str]
```

### Target-Family Harnesses
```python
class GraphHarness:
    def run_graph_targets(self) -> dict[str, TargetRunRecord]
    def assert_graph_tables(self, *table_keys: str) -> None

class AnalyticsHarness:
    def run_metrics_targets(self) -> dict[str, TargetRunRecord]
    def snapshot_tables(self, *table_keys: str) -> None

class ServingHarness:
    def publish_snapshot(self) -> ServingSnapshotManifest
    def assert_search_index(self) -> None
```

### Manifest Helpers (`tests/_helpers/manifests.py`)
```python
def load_manifest_index(gateway: StorageGateway, snapshot: SnapshotRef) -> dict[str, OutputManifest]

def prime_manifest(..., *, target: str, input_hash: str, options_hash: str | None) -> OutputManifest

def run_twice_and_assert_skip(harness: HamiltonBuildHarness, target: str) -> None

def assert_skipped(record: TargetRunRecord) -> None
```

### Target Record Assertions (`tests/_helpers/assertions/target_records.py`)
```python
def assert_target_ok(record: TargetRunRecord) -> None

def assert_record_has_datasets(record: TargetRunRecord, keys: Sequence[str]) -> None

def assert_record_has_artifacts(record: TargetRunRecord, names: Sequence[str]) -> None

def assert_record_row_counts(record: TargetRunRecord, expected: Mapping[str, int]) -> None

def assert_schema_valid(gateway: StorageGateway, table_key: str) -> None
```

### Repo Fixtures (`tests/_helpers/orchestration/repo_writers.py`)
```python
def write_monorepo_fixture(repo_root: Path, *, languages: Sequence[str]) -> RepoExpectations

def write_generated_noise_fixture(repo_root: Path, *, include_generated: bool) -> RepoExpectations

def write_large_file_fixture(repo_root: Path, *, max_bytes: int) -> RepoExpectations

def write_scoped_paths_fixture(repo_root: Path, *, scope_paths: Sequence[str]) -> RepoExpectations
```

### Table Snapshots (`tests/_helpers/snapshots/tables.py`)
```python
def snapshot_table(
    gateway: StorageGateway,
    table_key: str,
    *,
    order_by: Sequence[str],
    where: str | None = None,
    columns: Sequence[str] | None = None,
    hash_rows: bool = False,
) -> list[tuple[object, ...]]

def write_snapshot(path: Path, rows: Sequence[tuple[object, ...]]) -> None

def diff_snapshot(expected: Path, actual: Path) -> str
```

### Upstream Status Guards (`tests/_helpers/assertions/dependencies.py`)
```python
def require_upstream_ok(
    record: TargetRunRecord,
    *,
    target: str,
    allow_skipped: bool = True,
) -> None
```

### Plan/Status Harness (`tests/_helpers/harnesses/plan_status.py`)
```python
def compute_plan_summary(harness: HamiltonBuildHarness, targets: Sequence[str]) -> PlanSummary

def format_plan_diff(expected: PlanSummary, actual: PlanSummary) -> str
```

### Config Override Helpers (`tests/_helpers/build_config_overrides.py`)
```python
def write_build_config_sections(
    repo_root: Path,
    sections: Mapping[str, Mapping[str, object]],
) -> Path

def reload_build_config(harness: HamiltonBuildHarness) -> HamiltonBuildHarness
```

### Repo Fixture Registry (`tests/_helpers/orchestration/repo_registry.py`)
```python
def get_repo_fixture(tag: str) -> RepoFixture

def list_repo_fixtures() -> Mapping[str, RepoFixture]
```

### Failure Scenario Helpers (`tests/_helpers/assertions/target_failures.py`)
```python
def assert_partial_failure(
    records: Mapping[str, TargetRunRecord],
    *,
    failed: Sequence[str],
    succeeded: Sequence[str] = (),
    skipped: Sequence[str] = (),
) -> None
```

## Risks / Trade-offs
- Increased helper surface area may complicate onboarding.
  - Mitigation: keep the base harness minimal and expose concise wrappers.
- More realistic execution may increase test runtime.
  - Mitigation: FakeToolRunner remains the default; ToolSandbox reserved for
    integration-marked tests.

## Migration Plan
1. Add base harness, tool realism helpers, and manifest utilities.
2. Add assertion helpers and repo fixtures.
3. Add target-family harness wrappers and table snapshot utilities.
4. Add minimal tests and documentation updates to drive adoption.
