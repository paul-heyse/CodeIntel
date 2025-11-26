


You’ve already done a *lot* of the heavy lifting in `analytics` (parsing/, cfg_dfg/, subsystems/, tests/, the `span_resolver` shim, etc.). So this plan is about **finishing the “shape” of the package**: turning it into a set of clearly-bounded domains with a stable API layer.

I’ll focus on:

* Function-level analytics
* Graph-level analytics
* History / temporal analytics
* How callers (Prefect, graphs, server, etc.) should import things
* Tests / guardrails so it doesn’t drift again

I’ll assume these zips reflect your current state (they do: I can see e.g. `analytics/parsing/__init__.py`, `analytics/subsystems/__init__.py`, `analytics/tests/__init__.py`, `analytics/span_resolver.py` already acting as shims).

---

## 0. Goals & constraints

**Goal for this refactor:**

* `analytics` becomes a set of **obvious subdomains**, each with a small public API:

  * `analytics.functions` – all function-level metrics & typedness
  * `analytics.graphs` – graph-level metrics at function/module/subsystem scale
  * `analytics.history` – git history & temporal metrics
  * `analytics.parsing`, `analytics.cfg_dfg`, `analytics.subsystems`, `analytics.tests` – already good
  * `analytics.context`, `analytics.profiles`, `analytics.graph_service` – orchestration helpers on top of the domains

* **Callers** (Prefect, graphs, etc.):

  * Import from `codeintel.analytics.<domain>` rather than deep module paths.
  * Don’t need to know which file implements what; they just call the domain API.

* **Backwards compatibility**:

  * We *don’t* break existing imports immediately — we add domain APIs first.
  * If you later want to physically move modules (e.g. into `analytics/graphs/`), you can do that behind the domain API without touching call sites.

---

## 1. Inventory of current `analytics` structure (what we’re shaping)

From the `analytics.zip` you attached, the package currently looks (simplified) like:

**Top-level modules**

```text
analytics/__init__.py
analytics/ast_metrics.py
analytics/ast_utils.py
analytics/config_data_flow.py
analytics/config_graph_metrics.py
analytics/context.py
analytics/coverage_analytics.py
analytics/data_model_usage.py
analytics/data_models.py
analytics/dependencies.py
analytics/entrypoint_detectors.py
analytics/entrypoints.py
analytics/evidence.py
analytics/function_ast_cache.py
analytics/function_contracts.py
analytics/function_effects.py
analytics/function_history.py
analytics/git_history.py
analytics/graph_metrics.py
analytics/graph_metrics_ext.py
analytics/graph_runtime.py
analytics/graph_service.py
analytics/graph_stats.py
analytics/history_timeseries.py
analytics/module_graph_metrics_ext.py
analytics/profiles.py
analytics/semantic_roles.py
analytics/span_resolver.py  # shim to analytics.parsing.span_resolver
analytics/subsystem_agreement.py
analytics/subsystem_graph_metrics.py
analytics/symbol_graph_metrics.py
```

**Subpackages (already nicely done)**

```text
analytics/cfg_dfg/            # CFG/DFG metrics; exports compute_cfg_metrics / compute_dfg_metrics
analytics/functions/          # functions.config, functions.metrics, functions.parsing, typedness
analytics/parsing/            # parse_python_module, span_resolver, validation, registry
analytics/subsystems/         # build_subsystems etc
analytics/tests/              # test coverage, test graph metrics, behavioral profiles
```

You’ve already done great work with:

* `analytics.parsing.__init__` exposing a nice, centralized parsing API.
* `analytics.cfg_dfg.__init__` exporting `compute_cfg_metrics`/`compute_dfg_metrics`.
* `analytics.subsystems.__init__` exporting `build_subsystems`.
* `analytics.tests.__init__` exporting the test analytics API.
* `analytics.span_resolver` as a compatibility shim around `analytics.parsing.span_resolver`.

So the missing piece is mainly:

> The *top-level* analytics modules are a bit of a grab-bag of functions/graphs/history. We want those to be grouped into explicit domain APIs.

---

## 2. Step 1 – Make `analytics.functions` a real domain API

### 2.1: Decide what “function analytics” includes

Function-centric modules today:

* Under `analytics/functions/`:

  * `config.py` – `FunctionAnalyticsConfig` helpers & `FunctionAnalyticsOptions`
  * `metrics.py` – `compute_function_metrics_and_types`
  * `typedness.py` – typedness flags & helpers
  * `parsing.py` – parse Python files for function-level analysis

* Top-level modules that are clearly function-oriented:

  * `function_ast_cache.py`
  * `function_effects.py`
  * `function_contracts.py`
  * `function_history.py` (also history-ish, but per-function)
  * `ast_metrics.py` (function/module hotspots)
  * `ast_utils.py` (AST helpers used by analytics)

We won’t move all files yet; instead we’ll **expose the main entrypoints** through a cleaned-up API.

### 2.2: Expand `analytics/functions/__init__.py` into the public surface

Current file:

```python
"""Function-level analytics package."""

from __future__ import annotations

__all__: list[str] = []
```

Proposed new `analytics/functions/__init__.py`:

```python
"""Function-level analytics public API.

This module centralizes all the main entrypoints for per-function analytics, so
callers do not need to reach into submodules.
"""

from __future__ import annotations

from codeintel.analytics.functions.config import FunctionAnalyticsOptions
from codeintel.config.models import FunctionAnalyticsConfig

# Core metrics entrypoints
from codeintel.analytics.functions.metrics import compute_function_metrics_and_types
from codeintel.analytics.function_effects import compute_function_effects
from codeintel.analytics.function_contracts import compute_function_contracts
from codeintel.analytics.function_history import compute_function_history

# Optional helpers that some callers may want
from codeintel.analytics.functions.typedness import TypednessFlags  # type: ignore[attr-defined]

__all__ = [
    "FunctionAnalyticsConfig",
    "FunctionAnalyticsOptions",
    "compute_function_metrics_and_types",
    "compute_function_effects",
    "compute_function_contracts",
    "compute_function_history",
    "TypednessFlags",
]
```

Notes:

* We import `FunctionAnalyticsConfig` from `codeintel.config.models` because that’s where it lives now; it’s still nice to surface it via `analytics.functions` for convenience.
* If `TypednessFlags` isn’t exported from `functions.typedness`, we can either:

  * Export it there; or
  * Omit it from `__all__` for now and only surface the main `compute_*` functions.

### 2.3: Update callers to use the domain API

**Prefect flow** currently imports function analytics like this (from your zip):

```python
from codeintel.analytics.function_contracts import compute_function_contracts
from codeintel.analytics.function_effects import compute_function_effects
from codeintel.analytics.function_history import compute_function_history
from codeintel.analytics.functions.config import FunctionAnalyticsOptions
from codeintel.analytics.functions.metrics import compute_function_metrics_and_types
```

Change to:

```python
from codeintel.analytics.functions import (
    FunctionAnalyticsConfig,          # if you want it here
    FunctionAnalyticsOptions,
    compute_function_contracts,
    compute_function_effects,
    compute_function_history,
    compute_function_metrics_and_types,
)
```

Then gradually:

* **Stop importing** `analytics.function_*` everywhere outside analytics.
* Treat `analytics.functions` as the canonical API.

### 2.4: (Optional Phase 2) Physically move function-level modules

Once imports are cleaned up, you *can* move top-level modules like:

* `function_ast_cache.py` → `functions/ast_cache.py`
* `function_effects.py` → `functions/effects.py`
* `function_contracts.py` → `functions/contracts.py`
* `function_history.py` → `functions/history.py`
* `ast_utils.py` → perhaps into `functions/ast_utils.py` or into a shared `common` module

If you do, add compatibility shims at the old paths, similar to your existing `analytics.span_resolver`:

```python
# analytics/function_effects.py
"""Compatibility shim; use codeintel.analytics.functions instead."""

from __future__ import annotations
from codeintel.analytics.functions.effects import compute_function_effects

__all__ = ["compute_function_effects"]
```

That way, existing imports remain valid, but new code uses `analytics.functions`.

---

## 3. Step 2 – Introduce `analytics.graphs` as the graph analytics domain

### 3.1: Decide what belongs to “graph analytics”

Graph-oriented modules are currently:

* Top-level:

  ```text
  graph_metrics.py
  graph_metrics_ext.py
  module_graph_metrics_ext.py
  graph_stats.py
  symbol_graph_metrics.py
  config_graph_metrics.py
  config_data_flow.py
  subsystem_graph_metrics.py
  subsystem_agreement.py
  graph_service.py
  graph_runtime.py
  ```

* Plus CFG/DFG and subsystems packages that already have their own subpackages:

  * `analytics/cfg_dfg/*`
  * `analytics/subsystems/*`
  * `analytics/tests/graph_metrics.py` (already within tests domain)

We’ll keep `cfg_dfg` & `subsystems` as their own subdomains, but we want a top-level **“graph analytics” entrypoint** for:

* `compute_graph_metrics`
* function/module graph metrics extensions
* symbol graph metrics
* config graph metrics / dataflow
* subsystem graph metrics & agreement

### 3.2: Create `analytics/graphs/__init__.py`

New file:

```python
# analytics/graphs/__init__.py
"""Graph-level analytics public API.

This module exposes graph metrics across functions, modules, symbols, configs,
and subsystems, so callers do not need to import individual modules.
"""

from __future__ import annotations

from codeintel.analytics.graphs.graph_metrics import compute_graph_metrics
from codeintel.analytics.graphs.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.analytics.graph_stats import compute_graph_stats
from codeintel.analytics.symbol_graph_metrics import compute_symbol_graph_metrics
from codeintel.analytics.config_graph_metrics import compute_config_graph_metrics
from codeintel.analytics.config_data_flow import compute_config_data_flow
from codeintel.analytics.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.analytics.subsystem_agreement import compute_subsystem_agreement

__all__ = [
    "compute_graph_metrics",
    "compute_graph_metrics_functions_ext",
    "compute_graph_metrics_modules_ext",
    "compute_graph_stats",
    "compute_symbol_graph_metrics",
    "compute_config_graph_metrics",
    "compute_config_data_flow",
    "compute_subsystem_graph_metrics",
    "compute_subsystem_agreement",
]
```

This *doesn’t* move any code yet; it just centralizes the exports.

If you want to include `cfg_dfg` and `subsystems` as part of this API, you can add:

```python
from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.subsystems import build_subsystems

__all__ += ["compute_cfg_metrics", "compute_dfg_metrics", "build_subsystems"]
```

### 3.3: Update callers to use `analytics.graphs`

In Prefect (and anywhere else), replace:

```python
from codeintel.analytics.graphs.graph_metrics import compute_graph_metrics
from codeintel.analytics.graphs.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.analytics.graph_stats import compute_graph_stats
from codeintel.analytics.config_graph_metrics import compute_config_graph_metrics
from codeintel.analytics.config_data_flow import compute_config_data_flow
from codeintel.analytics.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.analytics.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.symbol_graph_metrics import (
    compute_symbol_graph_metrics,
)
```

with:

```python
from codeintel.analytics.graphs import (
    compute_graph_metrics,
    compute_graph_metrics_functions_ext,
    compute_graph_metrics_modules_ext,
    compute_graph_stats,
    compute_config_graph_metrics,
    compute_config_data_flow,
    compute_subsystem_graph_metrics,
    compute_subsystem_agreement,
    compute_symbol_graph_metrics,
)
```

Also consider:

* Using `analytics.cfg_dfg` (already an API) where CFG/DFG are needed.
* Using `analytics.subsystems` (already an API) instead of `analytics.subsystems.materialize`.

### 3.4: (Optional Phase 2) Physically move graph modules under `analytics/graphs`

Later, you *could* move these files:

* `graph_metrics.py` → `graphs/metrics.py`
* `graph_metrics_ext.py` → `graphs/metrics_ext.py`
* `module_graph_metrics_ext.py` → `graphs/module_metrics_ext.py`
* `graph_stats.py` → `graphs/stats.py`
* `symbol_graph_metrics.py` → `graphs/symbol_metrics.py`
* `config_graph_metrics.py` → `graphs/config_metrics.py`
* `config_data_flow.py` → `graphs/config_data_flow.py`
* `subsystem_graph_metrics.py` → `graphs/subsystem_metrics.py`
* `subsystem_agreement.py` → `graphs/subsystem_agreement.py`

and add compatibility shims at the old paths, exactly like you did for `analytics.span_resolver`.

But thanks to the domain API, **call sites won’t care**.

---

## 4. Step 3 – Introduce `analytics.history` for temporal analytics

Temporal/history-oriented modules:

* `function_history.py`
* `history_timeseries.py`
* `git_history.py`

They’re currently at package root.

### 4.1: Add `analytics/history/__init__.py`

New directory: `analytics/history/` with:

```python
# analytics/history/__init__.py
"""History-aware analytics (git + temporal metrics)."""

from __future__ import annotations

from codeintel.analytics.function_history import compute_function_history
from codeintel.analytics.history_timeseries import compute_history_timeseries_gateways
from codeintel.analytics.git_history import build_git_history  # or whatever its entrypoint is named

__all__ = [
    "compute_function_history",
    "compute_history_timeseries_gateways",
    "build_git_history",
]
```

### 4.2: Update callers

From Prefect:

```python
from codeintel.analytics.function_history import compute_function_history
from codeintel.analytics.history_timeseries import compute_history_timeseries_gateways
```

change to:

```python
from codeintel.analytics.history import (
    compute_function_history,
    compute_history_timeseries_gateways,
)
```

If anything calls into `git_history.py` directly, consider switching those imports to `analytics.history`.

### 4.3: (Optional Phase 2) Move history files under `analytics/history`

You can later:

* Move `function_history.py` to `history/function_history.py` etc.
* Provide thin shims at the old paths, same pattern as above.

---

## 5. Step 4 – Keep orchestration modules as “roots”

Some modules are already good as *root-level orchestrators* rather than domain packages:

* `analytics/context.py` – seeds `AnalyticsContext`, invokes various domain analytics, builds temp tables, etc.
* `analytics/profiles.py` – builds denormalized function/module/file profiles from underlying analytics tables.
* `analytics/graph_service.py`, `analytics/graph_runtime.py` – central graph context + GPU preferences.
* `analytics/coverage_analytics.py`, `analytics/tests/*` – coverage and test analytics; tests already has its own subpackage.
* `analytics/entrypoints.py` / `entrypoint_detectors.py` / `dependencies.py` – entrypoint / dependency semantics.
* `analytics/semantic_roles.py`, `analytics/evidence.py` – more advanced/experimental semantics.

For these, the plan is:

* **Do not move them** (they’re high-level orchestration).
* Where possible, have them import **domain APIs** (e.g., `analytics.functions`, `analytics.graphs`, `analytics.history`) instead of individual files; but that’s an optional follow-up.

This keeps `analytics` root focused on:

* Orchestration (`context`, `profiles`, `graph_service`, `graph_runtime`)
* Cross-cutting domain helpers (entrypoints, dependencies, semantic roles, evidence)

---

## 6. Step 5 – Update Prefect and other call sites

Using your `orchestration/prefect_flow.py` as the canonical consumer, we want to normalize imports to:

* `codeintel.analytics.functions`
* `codeintel.analytics.graphs`
* `codeintel.analytics.history`
* `codeintel.analytics.tests`
* `codeintel.analytics.cfg_dfg`
* `codeintel.analytics.subsystems`
* `codeintel.analytics.parsing` (already nice)
* `codeintel.analytics.coverage_analytics`, `analytics.profiles`, etc. as needed

### 6.1: Concrete import rewrite

Current analytics imports in `prefect_flow.py` (from the zip):

```python
from codeintel.analytics.ast_metrics import build_hotspots
from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.config_data_flow import compute_config_data_flow
from codeintel.analytics.config_graph_metrics import compute_config_graph_metrics
from codeintel.analytics.context import (
    AnalyticsContext,
    PipelineAnalyticsOptions,
)
from codeintel.analytics.coverage_analytics import compute_coverage_functions
from codeintel.analytics.data_model_usage import compute_data_model_usage
from codeintel.analytics.data_models import compute_data_models
from codeintel.analytics.dependencies import (
    compute_dependency_graph_metrics,
    compute_dependency_summary,
)
from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.analytics.function_contracts import compute_function_contracts
from codeintel.analytics.function_effects import compute_function_effects
from codeintel.analytics.function_history import compute_function_history
from codeintel.analytics.functions.config import FunctionAnalyticsOptions
from codeintel.analytics.functions.metrics import compute_function_metrics_and_types
from codeintel.analytics.graphs.graph_metrics import compute_graph_metrics
from codeintel.analytics.graphs.graph_metrics_ext import compute_graph_metrics_functions_ext
from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import build_graph_context
from codeintel.analytics.graph_stats import compute_graph_stats
from codeintel.analytics.history_timeseries import compute_history_timeseries_gateways
from codeintel.analytics.module_graph_metrics_ext import compute_graph_metrics_modules_ext
from codeintel.analytics.parsing.validation import FunctionValidationReporter
from codeintel.analytics.semantic_roles import compute_semantic_roles
from codeintel.analytics.subsystem_agreement import compute_subsystem_agreement
from codeintel.analytics.subsystem_graph_metrics import compute_subsystem_graph_metrics
from codeintel.analytics.subsystems import build_subsystems
from codeintel.analytics.symbol_graph_metrics import (
    compute_symbol_graph_metrics,
)
from codeintel.analytics.tests import (
    build_behavioral_coverage,
    build_test_profile,
    compute_test_coverage_edges,
    compute_test_graph_metrics,
)
```

**After the new domain APIs** you want:

```python
from codeintel.analytics.ast_metrics import build_hotspots
from codeintel.analytics.cfg_dfg import compute_cfg_metrics, compute_dfg_metrics
from codeintel.analytics.context import (
    AnalyticsContext,
    PipelineAnalyticsOptions,
)
from codeintel.analytics.coverage_analytics import compute_coverage_functions
from codeintel.analytics.data_model_usage import compute_data_model_usage
from codeintel.analytics.data_models import compute_data_models
from codeintel.analytics.dependencies import (
    compute_dependency_graph_metrics,
    compute_dependency_summary,
)
from codeintel.analytics.entrypoints import build_entrypoints
from codeintel.analytics.functions import (
    FunctionAnalyticsConfig,            # optional convenience
    FunctionAnalyticsOptions,
    compute_function_contracts,
    compute_function_effects,
    compute_function_history,
    compute_function_metrics_and_types,
)
from codeintel.analytics.graphs import (
    compute_config_data_flow,
    compute_config_graph_metrics,
    compute_graph_metrics,
    compute_graph_metrics_functions_ext,
    compute_graph_metrics_modules_ext,
    compute_graph_stats,
    compute_subsystem_agreement,
    compute_subsystem_graph_metrics,
    compute_symbol_graph_metrics,
)
from codeintel.analytics.graph_runtime import GraphRuntimeOptions
from codeintel.analytics.graph_service import build_graph_context
from codeintel.analytics.history import (
    compute_history_timeseries_gateways,
    # compute_function_history also available here if you want
)
from codeintel.analytics.parsing.validation import FunctionValidationReporter
from codeintel.analytics.semantic_roles import compute_semantic_roles
from codeintel.analytics.subsystems import build_subsystems
from codeintel.analytics.tests import (
    build_behavioral_coverage,
    build_test_profile,
    compute_test_coverage_edges,
    compute_test_graph_metrics,
)
```

The changes are:

* Function analytics consolidated under `analytics.functions`.
* Graph analytics consolidated under `analytics.graphs`.
* History under `analytics.history`.

This makes Prefect (and any other orchestration) much easier to read: you can literally see the shape of the analytics domains from the imports.

---

## 7. Step 6 – Tests and architecture guardrails

Finally, add some small tests to keep this structure from drifting again.

### 7.1: Test that domain APIs expose expected symbols

Example: `tests/analytics/test_functions_api.py`:

```python
from codeintel.analytics.functions import (
    FunctionAnalyticsOptions,
    compute_function_contracts,
    compute_function_effects,
    compute_function_history,
    compute_function_metrics_and_types,
)


def test_functions_api_exports_expected_symbols() -> None:
    # The import is the test: if these symbols disappear, we want a clear failure.
    assert callable(compute_function_metrics_and_types)
    assert callable(compute_function_effects)
    assert callable(compute_function_contracts)
    assert callable(compute_function_history)
    assert FunctionAnalyticsOptions is not None
```

Similar for `analytics.graphs` and `analytics.history`.

### 7.2: Optional “no deep imports from analytics internals” test

If you want to enforce that *outside* of `analytics/`, nobody imports internal modules like `analytics.graph_metrics_ext` directly, add a lightweight architecture test:

```python
# tests/architecture/test_analytics_imports.py

from pathlib import Path

FORBIDDEN_SUFFIXES = (
    "graph_metrics_ext",
    "module_graph_metrics_ext",
    "symbol_graph_metrics",
    "config_graph_metrics",
    "config_data_flow",
    "function_contracts",
    "function_effects",
    "function_history",
)


def test_external_imports_use_domain_apis() -> None:
    root = Path("src/codeintel")
    for path in root.rglob("*.py"):
        # Allow imports inside analytics package itself
        if "analytics" in path.parts:
            continue

        text = path.read_text(encoding="utf-8")
        if "codeintel.analytics." not in text:
            continue

        for suffix in FORBIDDEN_SUFFIXES:
            bad = f"codeintel.analytics.{suffix}"
            assert bad not in text, f"{bad} should be imported via domain APIs (file: {path})"
```

This doesn’t impact functionality but gives you a nice “red light” if a new call site tries to bypass `analytics.functions` or `analytics.graphs`.

---

