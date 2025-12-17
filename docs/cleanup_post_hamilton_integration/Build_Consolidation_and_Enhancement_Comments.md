Below are the **key changes + additions I’d make to your `BUILD_CONSOLIDATION_AND_ENHANCEMENT_PLAN.md`** after reviewing your current `src/codeintel/build` (in `CodeIntel_Plugins_Removed.zip`), your Hamilton advanced guide, and current Hamilton docs.

I’ll focus on items that materially improve **extensibility, hardness, maintainability**, and “Hamilton-native-ness” *now that plugins are removed*—and I’ll also call out where your current plan is slightly optimistic (esp. around `@cache`) and how to tighten it into a truly best-in-class architecture.

---

## 1) Add a “Module Overrides” strategy to eliminate a lot of complexity in auto/native unification

### Why this matters

Right now your “auto mode” requires **exclusion lists** / “don’t generate nodes for targets that have native modules” logic (and historically, plugin wrappers made this worse). With plugins gone, you have a huge simplification available:

**Use `Builder.allow_module_overrides()` as the core consolidation mechanism.** Hamilton will pick the function from the later imported module when names collide. ([Hamilton][1])

That lets you implement:

* a **single “templates module”** (or small set of template modules) that defines nodes for *all* targets (even if they are “generic wrappers”)
* a set of **native modules** that override the template nodes for the targets you’ve migrated
* **no manual “exclude native targets from generation”** logic in driver construction

### What changes in your plan

Add a new “Architecture consolidation” sub-step:

> **Auto mode = template DAG + native overrides**, relying on Hamilton module override semantics rather than your own registry-driven filtering.

### Representative code sketch

```python
# src/codeintel/build/hamilton/driver_factory.py (refactor direction)

from hamilton import driver
from codeintel.build.hamilton.templates import all_targets as template_mod
from codeintel.build.hamilton.native import analytics, graphs, ingestion, export

def build_driver(config: dict, adapters: list):
    # Order matters: later modules override earlier ones.
    return (
        driver.Builder()
        .with_config(config)
        .with_modules(
            template_mod,
            analytics,
            graphs,
            ingestion,
            export,
        )
        .allow_module_overrides()   # <- key consolidation lever
        .with_adapters(*adapters)
        .build()
    )
```

### What this lets you delete/simplify (sooner than later)

Once templates exist and overrides work, you can collapse/remove a lot of bespoke “mode” machinery and registry filtering in:

* `src/codeintel/build/hamilton/driver_factory.py` (simplify modes)
* `src/codeintel/build/hamilton/nodes/node_factory.py` (eventually retire)
* any “exclude_target_nodes_for_targets” style logic (no longer needed)

---

## 2) Upgrade your plan from “@datasaver only” to “Materializers + Data Adapters as the canonical IO layer”

You already planned “I/O standardization with `@datasaver`”, which is good.

But to make this *best-in-class*, I’d strengthen it like this:

### Best-in-class direction

**Use Hamilton’s materialization system (`with_materializers`) plus data adapters (`DataLoader` / `DataSaver`) as the canonical way to express IO in the DAG.** ([Hamilton][1]) ([Hamilton][2])

This buys you:

* IO becomes **visible in the Hamilton DAG** (better graph exports, better debugging)
* IO becomes **portable** (DuckDB today, Parquet/S3 tomorrow)
* lifecycle hooks (including lineage emitters) can “see” IO operations

Hamilton explicitly supports adding loaders/savers into the graph via `.with_materializers()` ([Hamilton][1]).

### Why it matters beyond cleanup: lineage + serving

If you eventually want best-in-class lineage (OpenLineage or similar), Hamilton’s OpenLineage adapter requires materializers/data loader+saver metadata to emit dataset lineage. ([Hamilton][3])

So: **materializers aren’t just cleanup—they unlock ecosystem-grade lineage**.

### Concrete plan change

In your plan, promote IO standardization to a first-class pillar:

> “All reads/writes are expressed through Hamilton IO primitives (materializers, data adapters, dataloader/datasaver decorators). StorageGateway becomes an implementation detail behind these.”

### Representative code sketch (DuckDB saver expressed as a Hamilton “saver node”)

This is one viable “bridge pattern” (keeps your `StorageGateway` while aligning with Hamilton):

```python
# src/codeintel/build/hamilton/io/duckdb_savers.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict

import ibis.expr.types as ir
from hamilton.function_modifiers import datasaver, tag

from codeintel.build.hamilton.env import BuildEnv

@datasaver()
@tag(io_kind="duckdb", io_direction="write")
def a__duckdb_table_write(
    table: ir.Table,
    env: BuildEnv,
    table_key: str,   # passed via parameterization or subdag inputs
) -> Dict[str, Any]:
    # Centralize the one blessed “write ibis table to duckdb” path
    env.gateway.write_ibis_table(
        conn_name=env.snapshot.conn_name,
        table_key=table_key,
        table=table,
        if_exists="replace",
    )
    return {"table_key": table_key, "conn_name": env.snapshot.conn_name}
```

Then you generate per-target saver nodes *via parameterization/subdag* rather than hand-writing 50+ materializers.

---

## 3) Re-scope the `@cache` ambition: use caching **selectively**, not as the primary “skip build” mechanism

Your plan suggests eliminating the many `should_skip()` / skip-logic call sites by using `@cache`.

That’s *directionally attractive*, but here’s the catch in your architecture:

### The core issue

Many of your “compute nodes” return **Ibis expressions** (`ibis.expr.types.Table`). Caching those objects is not the same thing as caching the *materialized data*. Even if Hamilton can serialize them, the cached value may be:

* unstable across versions
* not representative of underlying data changes
* not sufficient to guarantee correctness of “skip writing this table”

Hamilton’s caching system is powerful, but it’s meant to store and reuse **intermediate computed results** (`with_cache()` enables this), not automatically replace your manifest-driven artifact-level incremental build semantics. ([Hamilton][1])

Also: caching logic has nuanced behavior around dependencies and invalidation. ([Hamilton][4])

### Best-in-class recommendation

Amend your plan to:

#### Strategy

* Keep your **manifest-driven incremental build** as the correctness authority for *materialized artifacts/tables*.
* Use Hamilton caching for what it’s best at:

  * expensive pure-Python nodes (AST parsing, file enumeration, symbol extraction, metadata normalization)
  * small/medium intermediate objects that are deterministic and safe to persist

#### How to encode this in the DAG

* Only apply `@cache` to nodes whose outputs are:

  * deterministic
  * not side-effecting
  * represent “compute”, not “write”

### Plan change to write down explicitly

> “`@cache` is an optimization for deterministic compute nodes; artifact/tables skip behavior remains manifest-backed until/unless we can prove a deterministic, artifact-aware caching strategy.”

This preserves “best-in-class hardness” while still leveraging Hamilton’s caching.

---

## 4) Add `@subdag` / `@parameterized_subdag` as a primary consolidation tool for repeatable target patterns

Your plan leans heavily on `@parameterize`. Good—but you can go further *and cleaner* with subDAGs:

* `@subdag` lets you embed and namespace reusable pipelines inside the same DAG. ([Hamilton][5])
* `@parameterized_subdag` is “syntactic sugar” to create *multiple* subDAG instances (but Hamilton cautions it’s advanced; still appropriate for your scale). ([Hamilton][6])

### Why this is uniquely valuable for your build system

Your repeated pattern isn’t just “one function repeated”—it’s often a **mini-pipeline**:

> q__ loaders → compute → validate → materialize → run_record

Subdags are the Hamilton-native way to express that pattern *once*, then stamp it out per target with consistent tags, consistent IO, consistent validation.

### Representative sketch (pattern stamping)

```python
# src/codeintel/build/hamilton/templates/target_subdag.py
from hamilton.function_modifiers import subdag, source, value

# imagine `target_pipeline` module defines:
#   - load inputs
#   - compute
#   - validate
#   - materialize
# and exposes final node "target_run_record"

@subdag(
    target_pipeline,
    namespace=source("target_name"),
    inputs={
        "table_key": source("table_key"),
        "target_name": source("target_name"),
    },
)
def t__target_run(target_run_record):
    return target_run_record
```

This is illustrative, but the key is: **subdag gives you a reusable “target pipeline”** and keeps the overall DAG visible. ([Hamilton][5])

---

## 5) Add the “Pipe family” to your plan for making complex Ibis transforms DAG-visible and maintainable

You have many “large” compute functions (Ibis expressions that do multiple joins, filters, aggregations).

Hamilton’s **pipe family** is exactly meant to turn those sequential redefinitions into DAG nodes (for visibility, testability, and optional reuse). ([Hamilton][7])

This is a *big* best-in-class win because:

* your graph exports become genuinely readable
* intermediate steps can be unit tested
* schema inference can be more granular (and easier to debug)
* it reduces “monolithic compute function” complexity without making everything a separate file

### Representative sketch (breaking an Ibis transformation into named steps)

```python
from hamilton.function_modifiers import pipe_input, step, source

def _filter_active_functions(t):
    return t.filter(t.is_active == True)

def _join_module_info(t, modules):
    return t.join(modules, t.module_id == modules.id)

@pipe_input(
    step(_filter_active_functions).named("functions__active"),
    step(_join_module_info, modules=source("q__modules")).named("functions__with_modules"),
)
def t__some_target__compute(q__functions):
    # final node just returns the final piped value
    return q__functions
```

Pipe docs explicitly call out that pipe is useful when you want the transformations to appear as DAG nodes, and notes about using `named` for stable node names. ([Hamilton][7])

---

## 6) Strengthen your “Parallel execution” section: start with GraphAdapters, then use Dynamic DAGs only where they shine

Your plan mentions dynamic DAGs and parallel execution. I’d refine it into a best-practice sequence:

### Best-in-class sequence

1. **GraphAdapter-based concurrency** (simpler; good for independent nodes)
2. **Dynamic DAG execution** only for true “map/reduce” patterns (per-file, per-module, etc.)

Hamilton’s builder supports enabling dynamic execution and configuring executors. ([Hamilton][1])

### Practical guidance for your system

* For “parse N files” tasks: dynamic DAGs are excellent.
* For “run many targets” tasks: concurrency can help, but be careful with DuckDB write contention and connection/thread safety (your StorageGateway/pooling layer becomes the control point).

### Plan change

Add an explicit rule:

> “No parallel writes to the same DuckDB connection/file; parallelism happens at ‘independent target groups’ or ‘per-file parse’ where the IO backend can support it.”

---

## 7) Add “OpenLineage-ready” as an explicit design goal (even if optional)

Even if you don’t ship OpenLineage immediately, making the system OpenLineage-ready is a *best-in-class* architecture marker—and it aligns perfectly with your long-term “metadata for agents” objective.

Hamilton’s OpenLineage adapter exists, and it explicitly notes that to emit lineage you must use Hamilton’s materializer abstraction (datasaver/dataloader, save_to/load_from, or with_materializers). ([Hamilton][3])

### Plan change

Add a small optional milestone:

* “IO expressed through Hamilton materializers”
* “OpenLineage adapter can run in ‘file transport’ mode for local validation”
* “Later: emit to Marquez or similar”

This gives you a standards-based lineage channel essentially “for free” once the IO layer is Hamilton-native.

---

## 8) One more consolidation target: retire the remaining “plugin-era” scaffolding aggressively

Since plugins are gone, your plan should explicitly include deleting/retiring the plugin-shaped architecture remnants that still exist in `src/codeintel/build`.

Based on the build code I inspected, the highest ROI deletions (once templates/overrides/subdags are in place) are:

### Candidates to deprecate/delete (as you consolidate)

* `src/codeintel/build/plugin.py` + `src/codeintel/build/plugins/` (even if mostly empty now)
* `src/codeintel/build/hamilton/nodes/node_factory.py` (dynamic node generation that still references plugin concepts)
* `src/codeintel/build/context.py` (plugin-style TargetExecutionContext/ContextResources shape)
* `src/codeintel/build/unified_registry.py` and `src/codeintel/build/targets.py` **once** BuildSpec/DAG-derived inventories fully replace them

### Replace with

* Hamilton DAG introspection + BuildSpec compile
* module overrides for auto/native
* subdags for repeated patterns
* materializers for IO
* lifecycle adapters for manifest/run recording

---

## Summary: the “best-in-class” deltas I’d add to your plan

If I were editing your consolidation plan, I’d add these as **top-level pillars** (not footnotes):

1. **Builder-first driver construction** (not `Driver(...)`), because it unlocks:

   * module overrides ([Hamilton][1])
   * materializers ([Hamilton][1])
   * caching ([Hamilton][1])
   * dynamic execution configs ([Hamilton][1])

2. **Module override strategy** to simplify auto/native unification (biggest consolidation win). ([Hamilton][1])

3. **Materializers + Data Adapters** as the canonical IO layer (enables lineage + simplifies code). ([Hamilton][2])

4. **Selective caching** (compute-only), keep manifest skip semantics authoritative for artifacts. ([Hamilton][4])

5. **Subdag + Parameterized_subdag** for stamping repeated “target pipeline” structures. ([Hamilton][5])

6. **Pipe family** to make complex Ibis transformations DAG-visible and testable. ([Hamilton][7])

7. **OpenLineage-ready** architecture as an explicit goal (optional milestone, but huge “best-in-class” signal). ([Hamilton][3])

---


[1]: https://hamilton.apache.org/concepts/builder/ "Builder - Hamilton"
[2]: https://hamilton.apache.org/reference/io/adapter-documentation/ "Data Adapters - Hamilton"
[3]: https://hamilton.staged.apache.org/reference/lifecycle-hooks/OpenLineageAdapter/ "plugins.h_openlineage.OpenLineageAdapter - Hamilton"
[4]: https://hamilton.staged.apache.org/reference/caching/caching-logic/ "Caching logic - Hamilton"
[5]: https://hamilton.apache.org/reference/decorators/subdag/ "subdag - Hamilton"
[6]: https://hamilton.apache.org/reference/decorators/parameterize_subdag/ "parameterized_subdag - Hamilton"
[7]: https://hamilton.staged.apache.org/reference/decorators/pipe/ "pipe family - Hamilton"
