You’re absolutely right that doing this first pass of slicing will kill a ton of implicit complexity / regression risk. Let’s wire a plan that’s faithful to how things *actually* look in your repo right now.

Below I’ll walk through:

1. How to split `pipeline.orchestration.steps` into submodules + a shared core.
2. How to make `prefect_flow` a thin adapter over the step registry (and what to do with the existing tasks).
3. How to modularize `config.builder` into ingestion / graph / analytics / serving config components while keeping `ConfigBuilder` as the façade.

I’ll reference concrete classes / functions from your code (as they exist in the zips you attached).

---

## 1. Split `pipeline.orchestration.steps` into submodules

### 1.1. Introduce a small `core` module for shared orchestration primitives

Create **`codeintel/pipeline/orchestration/core.py`** and move the shared, cross-cutting pieces out of `steps.py`:

From current `steps.py` (line numbers approximate from the file you sent):

* `PipelineContext` dataclass (lines ~136–172 etc).
* `PipelineStep` protocol (lines 461–468).
* Helper functions:

  * `_log_step`
  * `_ingestion_ctx`
  * `_function_catalog`
  * `_analytics_context`
  * `_graph_engine`
  * `_graph_runtime`
  * `ensure_graph_runtime`

You also rely on some primitives in there:

* `ScanProfile` from `codeintel.ingestion.source_scanner`.
* `IngestionContext` from `codeintel.ingestion.context`.
* `FunctionCatalogProvider` / `FunctionCatalogService` from `codeintel.analytics.functions.catalog`.
* `GraphContext` / `GraphRuntimeOptions` from `codeintel.analytics.graph_runtime`.
* `NxGraphEngine` from `codeintel.graphs.engine_factory`.
* `AnalyticsContext`, `AnalyticsContextConfig`, `build_analytics_context` from `codeintel.analytics.context`.

**New `core.py` skeleton**:

```python
# codeintel/pipeline/orchestration/core.py

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from coverage import Coverage

from codeintel.analytics.context import (
    AnalyticsContext,
    AnalyticsContextConfig,
    build_analytics_context,
)
from codeintel.analytics.functions.catalog import (
    FunctionCatalogProvider,
    FunctionCatalogService,
)
from codeintel.analytics.graph_runtime import (
    GraphContext,
    GraphRuntimeOptions,
    build_graph_engine,
)
from codeintel.config import (
    ConfigBuilder,
    GraphBackendConfig,
    ScanProfiles,
    SnapshotRef,
    ToolsConfig,
)
from codeintel.ingestion.context import IngestionContext
from codeintel.ingestion.source_scanner import ScanProfile
from codeintel.ingestion.tool_runner import ToolRunner
from codeintel.ingestion.tool_service import ToolService
from codeintel.storage.gateway import StorageGateway

log = logging.getLogger(__name__)


def _log_step(name: str) -> None:
    log.debug("Running pipeline step: %s", name)


@dataclass
class PipelineContext:
    snapshot: SnapshotRef
    paths: ...  # BuildPaths
    gateway: StorageGateway
    tool_runner: ToolRunner | None = None
    tool_service: ToolService | None = None
    scip_runner: ... | None = None
    artifact_writer: ... | None = None
    function_catalog: FunctionCatalogProvider | None = None
    coverage: Coverage | None = None
    ast_collector: ... | None = None
    cst_collector: ... | None = None
    cfg_builder: ... | None = None
    code_profile_cfg: ScanProfile | None = None
    config_profile_cfg: ScanProfile | None = None
    graph_backend_cfg: GraphBackendConfig | None = None
    analytics_context: AnalyticsContext | None = None
    change_tracker: ... | None = None
    tools: ToolsConfig | None = None
    graph_engine: ... | None = None
    export_datasets: tuple[str, ...] | None = None
    extra: dict[str, object] = field(default_factory=dict)

    # property accessors: repo_root, repo, commit, db_path, build_dir,
    # code_profile, config_profile, tools_config, etc.
    # (just paste the existing implementations in here)

class PipelineStep(Protocol):
    name: str
    deps: Sequence[str]
    def run(self, ctx: PipelineContext) -> None: ...


def _ingestion_ctx(ctx: PipelineContext) -> IngestionContext:
    # move the existing implementation here
    ...


def _function_catalog(ctx: PipelineContext) -> FunctionCatalogProvider:
    # move existing implementation
    ...


def _analytics_context(ctx: PipelineContext) -> AnalyticsContext:
    # move existing implementation (build_analytics_context + caching)
    ...


def _graph_engine(ctx: PipelineContext, acx: AnalyticsContext | None = None):
    # move existing implementation using build_graph_engine
    ...


def _graph_runtime(
    ctx: PipelineContext,
    *,
    graph_ctx: GraphContext | None = None,
    acx: AnalyticsContext | None = None,
) -> GraphRuntimeOptions:
    # move existing implementation
    ...


def ensure_graph_runtime(
    ctx: PipelineContext,
    *,
    graph_ctx: GraphContext | None = None,
    acx: AnalyticsContext | None = None,
) -> GraphRuntimeOptions:
    return _graph_runtime(ctx, graph_ctx=graph_ctx, acx=acx)
```

> **Goal:** all step modules depend only on `PipelineContext`, `PipelineStep`, and these helpers via `core`, not on each other.

In `steps.py` you will subsequently **delete** those definitions and import them from `core`.

---

### 1.2. Create step submodules by responsibility

Add four new files:

* `codeintel/pipeline/orchestration/steps_ingestion.py`
* `codeintel/pipeline/orchestration/steps_graphs.py`
* `codeintel/pipeline/orchestration/steps_analytics.py`
* `codeintel/pipeline/orchestration/steps_export.py`

Each of these will:

* Import `PipelineContext`, `_log_step`, `_ingestion_ctx`, `_analytics_context`, `ensure_graph_runtime`, etc. from `core`.
* Define the subset of step dataclasses for that responsibility.
* Export a mapping `*_STEPS: dict[str, PipelineStep]` so the central registry can be composed easily.

#### 1.2.1. `steps_ingestion.py`

Move these classes from `steps.py`:

* `SchemaBootstrapStep`
* `RepoScanStep`
* `SCIPIngestStep`
* `CSTStep`
* `AstStep`
* `CoverageIngestStep`
* `TestsIngestStep`
* `TypingIngestStep`
* `DocstringsIngestStep`
* `ConfigIngestStep`

Plus the ingestion-specific imports:

* `run_repo_scan`, `run_scip_ingest`, `run_cst_extract`, `run_ast_extract`, `run_coverage_ingest`, `run_tests_ingest`, `run_typing_ingest`, `run_docstrings_ingest`, `run_config_ingest` (as in your current file).
* `ScipIngestResult`, `ScanProfile`, `ToolRunner`, `ToolService`, etc. already used in those steps.

Example skeleton:

```python
# codeintel/pipeline/orchestration/steps_ingestion.py

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from codeintel.ingestion.ast_ingest import run_ast_extract
from codeintel.ingestion.config_ingest import run_config_ingest
from codeintel.ingestion.coverage_ingest import run_coverage_ingest
from codeintel.ingestion.cst_extract import run_cst_extract
from codeintel.ingestion.docstrings_ingest import run_docstrings_ingest
from codeintel.ingestion.repo_scan import run_repo_scan
from codeintel.ingestion.scip_ingest import ScipIngestResult, run_scip_ingest
from codeintel.ingestion.tests_ingest import run_tests_ingest
from codeintel.ingestion.typing_ingest import run_typing_ingest

from .core import PipelineContext, PipelineStep, _ingestion_ctx, _log_step


@dataclass
class SchemaBootstrapStep:
    name: str = "schema_bootstrap"
    deps: Sequence[str] = ()

    def run(self, ctx: PipelineContext) -> None:
        # No-op; schema is applied by StorageGateway or Prefect task.
        _log_step(self.name)


@dataclass
class RepoScanStep:
    name: str = "repo_scan"
    deps: Sequence[str] = ("schema_bootstrap",)

    def run(self, ctx: PipelineContext) -> None:
        _log_step(self.name)
        tracker = run_repo_scan(_ingestion_ctx(ctx))
        ctx.change_tracker = tracker
        ctx.extra["repo_scan"] = tracker.summary()


@dataclass
class SCIPIngestStep:
    name: str = "scip_ingest"
    deps: Sequence[str] = ("repo_scan",)

    def run(self, ctx: PipelineContext) -> None:
        _log_step(self.name)
        ingest_ctx = _ingestion_ctx(ctx)
        result: ScipIngestResult = run_scip_ingest(ingest_ctx)
        ctx.extra["scip_ingest"] = result
        # keep your existing logging logic...


# ...CSTStep, AstStep, CoverageIngestStep, TestsIngestStep, TypingIngestStep,
# DocstringsIngestStep, ConfigIngestStep moved over verbatim...


INGESTION_STEPS: dict[str, PipelineStep] = {
    "schema_bootstrap": SchemaBootstrapStep(),
    "repo_scan": RepoScanStep(),
    "scip_ingest": SCIPIngestStep(),
    "cst_extract": CSTStep(),
    "ast_extract": AstStep(),
    "coverage_ingest": CoverageIngestStep(),
    "tests_ingest": TestsIngestStep(),
    "typing_ingest": TypingIngestStep(),
    "docstrings_ingest": DocstringsIngestStep(),
    "config_ingest": ConfigIngestStep(),
}
```

#### 1.2.2. `steps_graphs.py`

Move:

* `GoidsStep`
* `CallGraphStep`
* `CFGStep`
* `ImportGraphStep`
* `SymbolUsesStep`

Imports:

* `build_goids` from `analytics.functions.goids`.
* `build_call_graph` from `analytics.graphs.callgraph`.
* `build_cfg_and_dfg` from `analytics.cfg_dfg.materialize`.
* `build_import_graph` from `analytics.graphs.import_graph`.
* `build_symbol_use_edges` from `analytics.graphs.symbol_graph`.
* plus config builder usage: `ctx.config_builder().goid_builder()`, `call_graph()`, `cfg_builder()`, `import_graph()`, `symbol_uses()`.

Skeleton:

```python
# codeintel/pipeline/orchestration/steps_graphs.py

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from codeintel.analytics.cfg_dfg import build_cfg_and_dfg
from codeintel.analytics.graph_runtime import GraphContext
from codeintel.analytics.graphs import (
    build_call_graph,
    build_import_graph,
    build_symbol_use_edges,
)
from codeintel.analytics.functions.goids import build_goids

from .core import (
    PipelineContext,
    PipelineStep,
    _analytics_context,
    _function_catalog,
    _graph_runtime,
    _log_step,
)


@dataclass
class GoidsStep:
    name: str = "goids"
    deps: Sequence[str] = (
        "repo_scan",
        "scip_ingest",
        "cst_extract",
        "ast_extract",
        "config_ingest",
    )

    def run(self, ctx: PipelineContext) -> None:
        _log_step(self.name)
        cfg = ctx.config_builder().goid_builder()
        build_goids(ctx.gateway, cfg)


@dataclass
class CallGraphStep:
    name: str = "callgraph"
    deps: Sequence[str] = ("goids",)

    def run(self, ctx: PipelineContext) -> None:
        _log_step(self.name)
        gateway = ctx.gateway
        catalog = _function_catalog(ctx)
        cfg = ctx.config_builder().call_graph(
            cst_collector=ctx.cst_collector,
            ast_collector=ctx.ast_collector,
        )
        build_call_graph(gateway, cfg, catalog_provider=catalog)


# CFGStep, ImportGraphStep, SymbolUsesStep similarly

GRAPH_STEPS: dict[str, PipelineStep] = {
    "goids": GoidsStep(),
    "callgraph": CallGraphStep(),
    "cfg": CFGStep(),
    "import_graph": ImportGraphStep(),
    "symbol_uses": SymbolUsesStep(),
}
```

#### 1.2.3. `steps_analytics.py`

Move all “analytics” classes (as per the `# Analytics steps` marker):

* `HotspotsStep`
* `FunctionHistoryStep`
* `HistoryTimeseriesStep`
* `FunctionEffectsStep`
* `FunctionAnalyticsStep`
* `FunctionContractsStep`
* `DataModelsStep`
* `DataModelUsageStep`
* `ConfigDataFlowStep`
* `CoverageAnalyticsStep`
* `TestCoverageEdgesStep`
* `RiskFactorsStep`
* `GraphMetricsStep`
* `SemanticRolesStep`
* `SubsystemsStep`
* `TestProfileStep`
* `BehavioralCoverageStep`
* `EntryPointsStep`
* `ExternalDependenciesStep`
* `ProfilesStep`

Imports: existing ones from `codeintel.analytics.*`, `codeintel.analytics.graph_service`, `analytics.history`, etc., plus config builder methods like `.hotspots()`, `.function_history()`, `.history_timeseries()`, `.function_analytics()`, `.function_effects()`, `.function_contracts()`, `.data_models()`, `.data_model_usage()`, `.config_data_flow()`, `.coverage_analytics()`, `.test_coverage()`, `.test_profile()`, `.graph_metrics()`, `.entrypoints()`, `.external_dependencies()`, `.profiles_analytics()`, `.subsystems()`, etc.

You’ll also use helpers from `core`:

* `_analytics_context`
* `_graph_runtime` / `ensure_graph_runtime`
* `_function_catalog`

Skeleton fragment:

```python
# codeintel/pipeline/orchestration/steps_analytics.py

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from codeintel.analytics.ast_metrics import build_hotspots
from codeintel.analytics.coverage_analytics import compute_coverage_functions
from codeintel.analytics.data_models import compute_data_models
from codeintel.analytics.data_model_usage import compute_data_model_usage
from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.analytics.functions.effects import compute_function_effects
from codeintel.analytics.functions.history import (
    compute_function_history,
    compute_history_timeseries_gateways,
)
from codeintel.analytics.functions.metrics import compute_function_metrics
from codeintel.analytics.functions.contracts import compute_function_contracts
from codeintel.analytics.graphs import (
    compute_config_data_flow,
    compute_graph_metrics,
    compute_graph_stats,
    # etc...
)
from codeintel.analytics.profiles import (
    build_file_profile,
    build_function_profile,
    build_module_profile,
)
from codeintel.analytics.tests.profiles import build_test_profiles
# ... other analytics imports ...

from .core import (
    PipelineContext,
    PipelineStep,
    _analytics_context,
    _function_catalog,
    _graph_runtime,
    ensure_graph_runtime,
    _log_step,
)


@dataclass
class HotspotsStep:
    name: str = "hotspots"
    deps: Sequence[str] = ("ast_extract",)

    def run(self, ctx: PipelineContext) -> None:
        _log_step(self.name)
        acx = _analytics_context(ctx)
        cfg = ctx.config_builder().hotspots()
        build_hotspots(ctx.gateway, cfg, catalog_provider=acx.catalog, context=acx)


# ... all other analytics steps ...

ANALYTICS_STEPS: dict[str, PipelineStep] = {
    "hotspots": HotspotsStep(),
    "function_history": FunctionHistoryStep(),
    "history_timeseries": HistoryTimeseriesStep(),
    "function_effects": FunctionEffectsStep(),
    "function_metrics": FunctionAnalyticsStep(),
    "function_contracts": FunctionContractsStep(),
    "data_models": DataModelsStep(),
    "data_model_usage": DataModelUsageStep(),
    "config_data_flow": ConfigDataFlowStep(),
    "coverage_functions": CoverageAnalyticsStep(),
    "test_coverage_edges": TestCoverageEdgesStep(),
    "risk_factors": RiskFactorsStep(),
    "graph_metrics": GraphMetricsStep(),
    "semantic_roles": SemanticRolesStep(),
    "subsystems": SubsystemsStep(),
    "test_profile": TestProfileStep(),
    "behavioral_coverage": BehavioralCoverageStep(),
    "entrypoints": EntryPointsStep(),
    "external_dependencies": ExternalDependenciesStep(),
    "profiles": ProfilesStep(),
}
```

(The exact mapping of names/steps should mirror your existing `PIPELINE_STEPS` literal from `steps.py`.)

#### 1.2.4. `steps_export.py`

Move:

* `ExportDocsStep`

Imports:

* `ExportCallOptions`, `export_all_jsonl`, `export_all_parquet` from `codeintel.pipeline.export.*`.

Skeleton:

```python
# codeintel/pipeline/orchestration/steps_export.py

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from codeintel.pipeline.export.export_jsonl import ExportCallOptions, export_all_jsonl
from codeintel.pipeline.export.export_parquet import export_all_parquet

from .core import PipelineContext, PipelineStep, _log_step


@dataclass
class ExportDocsStep:
    name: str = "export_docs"
    deps: Sequence[str] = (
        "repo_scan",
        "scip_ingest",
        "cst_extract",
        "ast_extract",
        "coverage_ingest",
        "tests_ingest",
        "typing_ingest",
        "docstrings_ingest",
        "config_ingest",
        "function_metrics",
        "function_effects",
        "function_contracts",
        "data_models",
        "data_model_usage",
        "config_data_flow",
        "coverage_functions",
        "test_coverage_edges",
        "hotspots",
        "function_history",
        "subsystems",
        "profiles",
    )

    def run(self, ctx: PipelineContext) -> None:
        _log_step(self.name)
        datasets = ctx.export_datasets
        export_all_parquet(
            ctx.gateway,
            ctx.document_output_dir,
            options=ExportCallOptions(validate_exports=False, datasets=datasets),
        )
        export_all_jsonl(
            ctx.gateway,
            ctx.document_output_dir,
            options=ExportCallOptions(validate_exports=False, datasets=datasets),
        )


EXPORT_STEPS: dict[str, PipelineStep] = {
    "export_docs": ExportDocsStep(),
}
```

---

### 1.3. Rewrite `steps.py` as an aggregator + registry

Now shrink **`codeintel/pipeline/orchestration/steps.py`** down to:

* Imports of `PipelineContext`, `PipelineStep` and maybe helper functions from `core`.
* Imports of `INGESTION_STEPS`, `GRAPH_STEPS`, `ANALYTICS_STEPS`, `EXPORT_STEPS`.
* Construction of the global registry (`PIPELINE_STEPS`, `PIPELINE_DEPS`, `PIPELINE_SEQUENCE`).
* The `_topological_order`, `_expand_with_deps`, and `run_pipeline` logic.
* Re-exports of `PipelineContext`, `PipelineStep`, `ProfilesStep`, `RiskFactorsStep`, `ExportDocsStep` for compatibility with existing imports (e.g. `prefect_flow.py`).

Skeleton:

```python
# codeintel/pipeline/orchestration/steps.py

from __future__ import annotations

from collections.abc import Sequence

from .core import PipelineContext, PipelineStep
from .steps_ingestion import INGESTION_STEPS
from .steps_graphs import GRAPH_STEPS
from .steps_analytics import ANALYTICS_STEPS, ProfilesStep, RiskFactorsStep
from .steps_export import EXPORT_STEPS, ExportDocsStep

# Public registry ----------------------------------------------------------------

PIPELINE_STEPS: dict[str, PipelineStep] = {}
PIPELINE_STEPS.update(INGESTION_STEPS)
PIPELINE_STEPS.update(GRAPH_STEPS)
PIPELINE_STEPS.update(ANALYTICS_STEPS)
PIPELINE_STEPS.update(EXPORT_STEPS)

PIPELINE_STEPS_BY_NAME: dict[str, PipelineStep] = PIPELINE_STEPS
PIPELINE_DEPS: dict[str, tuple[str, ...]] = {
    name: tuple(step.deps) for name, step in PIPELINE_STEPS.items()
}
PIPELINE_SEQUENCE: tuple[str, ...] = tuple(PIPELINE_STEPS.keys())


def _topological_order(step_names: Sequence[str]) -> list[str]:
    # paste your existing implementation
    ...


def _expand_with_deps(name: str, expanded: set[str]) -> None:
    # paste your existing implementation
    ...


def run_pipeline(
    ctx: PipelineContext,
    *,
    selected_steps: Sequence[str] | None = None,
) -> None:
    # paste your existing implementation: expand deps, topo-sort, then run
    ...


__all__ = [
    "PipelineContext",
    "PipelineStep",
    "PIPELINE_STEPS",
    "PIPELINE_DEPS",
    "PIPELINE_SEQUENCE",
    "run_pipeline",
    # these are used by prefect_flow right now:
    "ProfilesStep",
    "RiskFactorsStep",
    "ExportDocsStep",
]
```

> This preserves all current import sites:
>
> * `prefect_flow.py` can still do `from codeintel.pipeline.orchestration.steps import PipelineContext, ProfilesStep, RiskFactorsStep, run_pipeline`.

---

## 2. Make `prefect_flow` a thin adapter over the registry

Your **current `prefect_flow.py`** already has the key piece:

```python
@flow(name="export_docs_flow")
def export_docs_flow(args: ExportArgs, targets: Iterable[str] | None = None) -> None:
    _configure_prefect_logging()
    run_logger = get_run_logger()
    graph_backend = args.graph_backend or GraphBackendConfig()
    _GRAPH_BACKEND_STATE["config"] = graph_backend
    maybe_enable_nx_gpu(graph_backend)

    ctx = _build_pipeline_context(args)
    selected = tuple(targets) if targets is not None else None
    try:
        run_logger.info("Starting pipeline for %s@%s", ctx.repo, ctx.commit)
        run_pipeline(ctx, selected_steps=selected)
        run_logger.info("Pipeline complete for %s@%s", ctx.repo, ctx.commit)
    ...
```

That’s already the “thin adapter over `PipelineContext` + registry” you want. The extra weight in this module is the set of **per-step Prefect tasks** (`t_repo_scan`, `t_callgraph`, `t_graph_metrics`, etc.) which:

* Call ingestion / analytics functions directly.
* Rebuild gateways and configs themselves rather than using `PipelineContext`.
* Are not used anywhere outside `prefect_flow.py` based on the code you attached.

### 2.1. Update imports to go through the new `core` / `steps` structure

After the steps refactor, adjust the imports at the top of `prefect_flow.py`:

```python
# OLD
from codeintel.pipeline.orchestration.steps import (
    PipelineContext,
    ProfilesStep,
    RiskFactorsStep,
    run_pipeline,
)

# NEW (still fine, because steps re-exports these)
from codeintel.pipeline.orchestration.steps import (
    PipelineContext,
    ProfilesStep,
    RiskFactorsStep,
    run_pipeline,
)
```

No change needed **as long as you re-export from `steps.py`** as above. If you decide to import directly from `core`, you can instead:

```python
from codeintel.pipeline.orchestration.core import PipelineContext
from codeintel.pipeline.orchestration.steps_analytics import ProfilesStep, RiskFactorsStep
from codeintel.pipeline.orchestration.steps import run_pipeline
```

I’d keep the simple version and let `steps.py` be the public façade.

### 2.2. Decide what to do with the unused per-step Prefect tasks

You have a long set of tasks like:

* `t_repo_scan`, `t_scip_ingest`, `t_cst_extract`, `t_ast_extract`, `t_coverage_ingest`, `t_tests_ingest`, `t_typing_ingest`, `t_config_ingest`, …
* `t_goids`, `t_callgraph`, `t_cfg`, `t_import_graph`, `t_symbol_uses`, …
* `t_hotspots`, `t_function_history`, `t_coverage_functions`, `t_test_coverage_edges`, `t_graph_metrics`, `t_subsystems`, `t_profiles`, etc.

From your repo snapshot:

* They are **not called** from `export_docs_flow`.
* I couldn’t find any calls to `t_repo_scan` / `t_callgraph` etc. outside `prefect_flow.py` and tests (there were none in the zips).

There are different ways to handle this, but this is by far the simplest:

#### Simplest, and matches your stated goal): delete them

1. Remove all `@task` definitions that are not used.
2. Keep only:

   * `_build_pipeline_context`
   * `_get_gateway`/`_ingest_gateway` cache logic
   * `export_docs_flow`

This makes `prefect_flow.py` *literally* just:

* Helpers to build `PipelineContext` from `ExportArgs`.
* A Prefect `@flow` that calls `run_pipeline`.

You then have exactly one orchestration story: `run_pipeline` + registry.


## 3. Modularize `config.builder` by step type

Right now `config/builder.py` contains:

* Shared helper types like `EntryPointToggles`.

* All 33 `*StepConfig` dataclasses:

  ```text
  BehavioralCoverageStepConfig
  CFGBuilderStepConfig
  CallGraphStepConfig
  ConfigDataFlowStepConfig
  ConfigIngestStepConfig
  CoverageAnalyticsStepConfig
  CoverageIngestStepConfig
  DataModelUsageStepConfig
  DataModelsStepConfig
  DocstringStepConfig
  EntryPointsStepConfig
  ExternalDependenciesStepConfig
  FunctionAnalyticsStepConfig
  FunctionContractsStepConfig
  FunctionEffectsStepConfig
  FunctionHistoryStepConfig
  GoidBuilderStepConfig
  GraphMetricsStepConfig
  HistoryTimeseriesStepConfig
  HotspotsStepConfig
  ImportGraphStepConfig
  ProfilesAnalyticsStepConfig
  PyAstIngestStepConfig
  RepoScanStepConfig
  ScipIngestStepConfig
  SemanticRolesStepConfig
  SubsystemsStepConfig
  SymbolUsesStepConfig
  TestCoverageStepConfig
  TestProfileStepConfig
  TestsIngestStepConfig
  TypingIngestStepConfig
  ```

* And the monolithic `ConfigBuilder` whose methods build each of these.

### 3.1. Step 1: Physically move step config dataclasses into new modules

Create:

* `codeintel/config/steps_ingestion.py`
* `codeintel/config/steps_graphs.py`
* `codeintel/config/steps_analytics.py`
* (optional now, but future-proof) `codeintel/config/steps_serving.py` if you want to centralize serving-related builders later.

**3.1.1. `steps_ingestion.py`**

Move these dataclasses (cut/paste from `builder.py`):

* `RepoScanStepConfig`
* `ScipIngestStepConfig`
* `PyAstIngestStepConfig`
* `CoverageIngestStepConfig`
* `TestsIngestStepConfig`
* `TypingIngestStepConfig`
* `DocstringStepConfig`
* `ConfigIngestStepConfig`

Preserve them as-is: same fields, docstrings, properties.

Skeleton:

```python
# codeintel/config/steps_ingestion.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from codeintel.config.primitives import SnapshotRef, BuildPaths, ScanProfiles, ToolBinaries

@dataclass(frozen=True)
class RepoScanStepConfig:
    snapshot: SnapshotRef
    paths: BuildPaths
    scan_profiles: ScanProfiles
    tags_index_path: Path | None = None
    # existing properties: repo, commit, repo_root, etc.


@dataclass(frozen=True)
class ScipIngestStepConfig:
    snapshot: SnapshotRef
    paths: BuildPaths
    binaries: ToolBinaries
    scip_runner: ... | None = None
    artifact_writer: ... | None = None
    # etc.

# ... all other ingestion-related StepConfig dataclasses ...
```

**3.1.2. `steps_graphs.py`**

Move:

* `GoidBuilderStepConfig`
* `CallGraphStepConfig`
* `CFGBuilderStepConfig`
* `ImportGraphStepConfig`
* `SymbolUsesStepConfig`
* `GraphMetricsStepConfig`
* `ConfigDataFlowStepConfig`
* `ExternalDependenciesStepConfig`

Imports will again be from `primitives` plus any graph-specific types you already refer to (e.g. `GraphBackendConfig` if used here, though mostly it’s in `ConfigBuilder`).

**3.1.3. `steps_analytics.py`**

Move:

* `HotspotsStepConfig`
* `FunctionHistoryStepConfig`
* `HistoryTimeseriesStepConfig`
* `CoverageAnalyticsStepConfig`
* `TestCoverageStepConfig`
* `TestProfileStepConfig`
* `BehavioralCoverageStepConfig`
* `FunctionAnalyticsStepConfig`
* `FunctionEffectsStepConfig`
* `FunctionContractsStepConfig`
* `SemanticRolesStepConfig`
* `DataModelsStepConfig`
* `DataModelUsageStepConfig`
* `EntryPointsStepConfig`
* `ProfilesAnalyticsStepConfig`
* `SubsystemsStepConfig`

Again, no behavioral change; just relocation.

**3.1.4. Update `config/__init__.py` to re-export from new modules**

Currently `config/__init__.py` re-exports the step configs from `builder`:

```python
from codeintel.config.builder import (
    BehavioralCoverageStepConfig,
    CallGraphStepConfig,
    ...
)

__all__ = [
    "BehavioralCoverageStepConfig",
    "CallGraphStepConfig",
    ...
]
```

Change these imports to:

```python
from codeintel.config.steps_ingestion import (
    RepoScanStepConfig,
    ScipIngestStepConfig,
    PyAstIngestStepConfig,
    CoverageIngestStepConfig,
    TestsIngestStepConfig,
    TypingIngestStepConfig,
    DocstringStepConfig,
    ConfigIngestStepConfig,
)
from codeintel.config.steps_graphs import (
    GoidBuilderStepConfig,
    CallGraphStepConfig,
    CFGBuilderStepConfig,
    ImportGraphStepConfig,
    SymbolUsesStepConfig,
    GraphMetricsStepConfig,
    ConfigDataFlowStepConfig,
    ExternalDependenciesStepConfig,
)
from codeintel.config.steps_analytics import (
    HotspotsStepConfig,
    FunctionHistoryStepConfig,
    HistoryTimeseriesStepConfig,
    CoverageAnalyticsStepConfig,
    TestCoverageStepConfig,
    TestProfileStepConfig,
    BehavioralCoverageStepConfig,
    FunctionAnalyticsStepConfig,
    FunctionEffectsStepConfig,
    FunctionContractsStepConfig,
    SemanticRolesStepConfig,
    DataModelsStepConfig,
    DataModelUsageStepConfig,
    EntryPointsStepConfig,
    ProfilesAnalyticsStepConfig,
    SubsystemsStepConfig,
)
```

Keep `__all__` identical so that *all existing imports*:

```python
from codeintel.config import GraphMetricsStepConfig
```

continue to work.

At this point, nothing outside `config` should notice the change.

---

### 3.2. Step 2: Extract small “step builder” components

Now we tackle the internal organization of `ConfigBuilder` so that it “just composes” smaller builders.

Create helper classes (no dataclasses needed) in each `steps_*` module:

#### 3.2.1. Ingestion step builder

In `steps_ingestion.py`:

```python
class IngestionStepBuilder:
    """
    Provides ingestion-related config builders used by ConfigBuilder.

    Expects an object with attributes:
      - snapshot: SnapshotRef
      - paths: BuildPaths
      - profiles: ScanProfiles
      - binaries: ToolBinaries
    """

    def __init__(self, owner: object) -> None:
        self._owner = owner

    @property
    def snapshot(self) -> SnapshotRef:  # type: ignore[override]
        return self._owner.snapshot

    @property
    def paths(self) -> BuildPaths:
        return self._owner.paths

    @property
    def profiles(self) -> ScanProfiles:
        return self._owner.profiles

    @property
    def binaries(self) -> ToolBinaries:
        return self._owner.binaries

    def scip_ingest(
        self,
        *,
        scip_runner: Callable[..., ScipIngestResult] | None = None,
        artifact_writer: Callable[[Path, Path, Path], None] | None = None,
    ) -> ScipIngestStepConfig:
        return ScipIngestStepConfig(
            snapshot=self.snapshot,
            paths=self.paths,
            binaries=self.binaries,
            scip_runner=scip_runner,
            artifact_writer=artifact_writer,
        )

    def repo_scan(
        self,
        *,
        tags_index_path: Path | None = None,
        tool_runner: object | None = None,
    ) -> RepoScanStepConfig:
        return RepoScanStepConfig(
            snapshot=self.snapshot,
            paths=self.paths,
            scan_profiles=self.profiles,
            tags_index_path=tags_index_path,
            tool_runner=tool_runner,
        )

    # coverage_ingest, tests_ingest, typing_ingest, docstring, config_ingest,
    # py_ast_ingest — move the method bodies from ConfigBuilder here.
```

Do the same pattern in:

* `steps_graphs.py` → `GraphStepBuilder`
* `steps_analytics.py` → `AnalyticsStepBuilder`

Each builder takes an `owner` (the `ConfigBuilder` instance) and just reads its attributes.

#### 3.2.2. Refactor `ConfigBuilder` to delegate

In `config/builder.py`, reduce `ConfigBuilder` to:

* A dataclass with fields: `snapshot`, `paths`, `binaries`, `profiles`, `graph_backend`, `function_parser`, etc. (whatever you have now).
* Two constructors: `from_snapshot`, `from_primitives` (as you already do).
* Thin delegating methods that forward into the small builders.

Add properties:

```python
from codeintel.config.steps_ingestion import IngestionStepBuilder
from codeintel.config.steps_graphs import GraphStepBuilder
from codeintel.config.steps_analytics import AnalyticsStepBuilder

@dataclass
class ConfigBuilder:
    snapshot: SnapshotRef
    paths: BuildPaths
    binaries: ToolBinaries
    profiles: ScanProfiles
    graph_backend: GraphBackendConfig = field(default_factory=GraphBackendConfig)
    function_parser: FunctionParserKind | None = None
    # ... any other current fields ...

    # existing from_snapshot / from_primitives stay here

    @property
    def ingestion(self) -> IngestionStepBuilder:
        return IngestionStepBuilder(self)

    @property
    def graphs(self) -> GraphStepBuilder:
        return GraphStepBuilder(self)

    @property
    def analytics(self) -> AnalyticsStepBuilder:
        return AnalyticsStepBuilder(self)
```

Then, for each current step method, change implementation to delegate:

```python
    # Ingestion ----------------------------------------------------------------

    def scip_ingest(
        self,
        *,
        scip_runner: Callable[..., ScipIngestResult] | None = None,
        artifact_writer: Callable[[Path, Path, Path], None] | None = None,
    ) -> ScipIngestStepConfig:
        return self.ingestion.scip_ingest(
            scip_runner=scip_runner,
            artifact_writer=artifact_writer,
        )

    def repo_scan(
        self,
        *,
        tags_index_path: Path | None = None,
        tool_runner: object | None = None,
    ) -> RepoScanStepConfig:
        return self.ingestion.repo_scan(
            tags_index_path=tags_index_path,
            tool_runner=tool_runner,
        )

    # Graphs -------------------------------------------------------------------

    def call_graph(
        self,
        *,
        cst_collector: Callable[..., list[CallGraphEdgeRow]] | None = None,
        ast_collector: Callable[..., list[CallGraphEdgeRow]] | None = None,
    ) -> CallGraphStepConfig:
        return self.graphs.call_graph(
            cst_collector=cst_collector,
            ast_collector=ast_collector,
        )

    def cfg_builder(
        self,
        *,
        cfg_builder: Callable[..., tuple[list[CFGBlockRow], list[CFGEdgeRow], list[DFGEdgeRow]]] | None = None,
    ) -> CFGBuilderStepConfig:
        return self.graphs.cfg_builder(cfg_builder=cfg_builder)

    # Analytics ----------------------------------------------------------------

    def hotspots(self, *, max_commits: int = 2000) -> HotspotsStepConfig:
        return self.analytics.hotspots(max_commits=max_commits)

    def function_history(
        self,
        *,
        max_history_days: int | None = 365,
        min_lines_threshold: int = 1,
        default_branch: str = "HEAD",
    ) -> FunctionHistoryStepConfig:
        return self.analytics.function_history(
            max_history_days=max_history_days,
            min_lines_threshold=min_lines_threshold,
            default_branch=default_branch,
        )

    # ... and so on for all analytics config methods ...
```

This gives you:

* A **single façade** (`ConfigBuilder`) that external code keeps using unchanged.
* Internals grouped by concern (`IngestionStepBuilder`, `GraphStepBuilder`, `AnalyticsStepBuilder`) in separate modules.
* An obvious place for an LLM agent to introspect: e.g. `dir(builder.analytics)` vs. `dir(builder.graphs)`.

You *can* later expose those builders publicly if you want:

```python
# config/__init__.py
__all__ += ["IngestionStepBuilder", "GraphStepBuilder", "AnalyticsStepBuilder"]
```

…but that’s optional.

---

### 3.3. (Optional) `steps_serving.py`

Right now, serving-specific config (FastAPI, MCP) lives in `config/serving_models.py` and is used directly by `serving.http.fastapi` / `serving.mcp.*`.

If you want parity with the other step builders, you could:

* Add a small `ServingConfigBuilder` in `steps_serving.py` that:

  * Uses `ServingConfig.from_env()` / similar factories you already have.
  * Validates that the gateway DB’s `core.repo_map` matches the requested repo/commit (you already have helpers for that in `serving_models.py`).

Then `ConfigBuilder` can expose:

```python
@property
def serving(self) -> ServingConfigBuilder:
    return ServingConfigBuilder(self)
```

This can wait until you actually feel pain around serving config, though.

---

## Sanity checks and migration strategy

1. **Incremental approach**
   Do these in order:

   1. Add `core.py` and move shared helpers; update `steps.py` imports but change nothing else.
   2. Add `steps_ingestion.py` / `steps_graphs.py` / `steps_analytics.py` / `steps_export.py`; copy classes over; build `INGESTION_STEPS` etc.; change `steps.py` to assemble `PIPELINE_STEPS` from those. Run tests.
   3. Move step config dataclasses into `config/steps_*` modules; update `config/__init__.py` re-exports; keep `ConfigBuilder` methods intact. Run tests.
   4. Introduce `IngestionStepBuilder` / `GraphStepBuilder` / `AnalyticsStepBuilder`; switch `ConfigBuilder` methods to delegate. Run tests.
   5. Finally, clean up `prefect_flow.py`:

      * Confirm `export_docs_flow` still works.
      * Decide whether to remove or wrap the per-step tasks.

2. **Type-check and introspection**

   * Run `pyright` or `mypy` to catch any import cycles or missing attributes on the builder components.
   * `dir(ConfigBuilder().analytics)` will now show you all analytics knobs, which is exactly what your agents will love.

3. **Behavior parity**

   * Pipeline step names, dependencies and config defaults stay identical.
   * `run_pipeline(ctx, selected_steps=...)` semantics are unchanged.
   * `from codeintel.config import GraphMetricsStepConfig` and `from codeintel.config import ConfigBuilder` semantics are unchanged.

