

## #1 I’m implementing here: make “what a target outputs” truly DAG-derived (from Hamilton materialization nodes), not contract-self-referential

Right now, your architecture clearly separates:

* **Compute nodes** (tagged compute) that produce table rows / artifact paths, and
* **Saver metadata nodes** produced by `SaveToObjectMetadataDecorator`, named via `materialize_node(...)`, which are the *actual* “materializations exist in the DAG” signal. 

But your *current* `derive_target_outputs()` derives outputs from **support nodes** (dataset/artifact nodes) — and those support nodes are generated from the **target contract**. That makes “DAG output validation” self-fulfilling: the DAG “outputs” will always match the contract because they were generated from it.

This is exactly the kind of coupling inversion you want to remove if your design goal is “Hamilton DAG derived outputs determine orchestration/IO/etc.”

Hamilton’s intended pattern here is: **attach tags/metadata to nodes**, then query them (e.g., via `list_available_variables(tag_filter=...)`). ([Hamilton][1])
And materializers/savers *are* nodes in the graph (materialization adds nodes to the dataflow), so they are the perfect source-of-truth for “what gets produced.” ([hamilton.staged.apache.org][2])

So #1 is:

> **Derive each target’s dataset/artifact outputs from the *materialization metadata nodes* (the SaveTo-created nodes), and then drive downstream registries/support nodes/validation from that derived view.**

---

## Exact introspection algorithm

### Inputs

* `HamiltonRuntime runtime` (you already have this abstraction)
* `runtime.dr` (Hamilton Driver)
* `runtime.dr.graph` (function graph; you currently inspect nodes directly in other places)
* Tag keys you already standardize on:

  * `TAG_NODE_TYPE`, `TAG_TARGET`, `TAG_TABLE_KEY`, `TAG_ARTIFACT` (from `codeintel.core.hamilton.tags`)
* **New node types** we’ll introduce for materialization metadata nodes:

  * `NODE_TYPE_MATERIALIZATION_DATASET`
  * `NODE_TYPE_MATERIALIZATION_ARTIFACT`

### Required invariant

Every **SaveTo-produced metadata node** must carry enough tags to answer:

* “Which target owns me?”
* “What dataset table_key / artifact_name do I represent?”

You already have the information at decorator-build time (e.g., `target_name=value(...)`, `table_key=value(...)`, `artifact_name=value(...)`) in your native modules. 

### Algorithm (deterministic, tag-first)

**Step 0: Query nodes by tag** (preferred Hamilton surface)
Use `Driver.list_available_variables(tag_filter={...})` rather than scanning raw `graph.nodes`, because tag queries are a first-class Hamilton pattern. ([Hamilton][1])

**Step 1: Collect dataset materializations**

* `dataset_mats = dr.list_available_variables(tag_filter={TAG_NODE_TYPE: NODE_TYPE_MATERIALIZATION_DATASET})`
* For each node `n` in `dataset_mats`:

  * `target = n.tags[TAG_TARGET]` (must be non-empty `str`)
  * `table_key = n.tags[TAG_TABLE_KEY]` (must be non-empty `str`)
  * `datasets_by_target[target].add(table_key)`

**Step 2: Collect artifact materializations**

* `artifact_mats = dr.list_available_variables(tag_filter={TAG_NODE_TYPE: NODE_TYPE_MATERIALIZATION_ARTIFACT})`
* For each node `n` in `artifact_mats`:

  * `target = n.tags[TAG_TARGET]` (must be non-empty `str`)
  * `artifact = n.tags[TAG_ARTIFACT]` (must be non-empty `str`)
  * `artifacts_by_target[target].add(artifact)`

**Step 3: Normalize**

* Convert each set to a sorted tuple for stable outputs:

  * `datasets_by_target = {t: tuple(sorted(keys)) ...}`
  * `artifacts_by_target = {t: tuple(sorted(names)) ...}`

**Step 4: Validate + diff vs contracts**
Given a `TargetGraph` with `OutputTarget.contract`:

* For each target `T`:

  * expected tables = `set(T.contract.table_keys)`
  * observed tables = `set(datasets_by_target.get(T.name, ()))`
  * expected artifacts = `set(T.contract.artifact_names)`
  * observed artifacts = `set(artifacts_by_target.get(T.name, ()))`
  * record:

    * **missing** = expected − observed
    * **extra** = observed − expected
* Also record:

  * observed outputs for unknown targets (tag target not in graph)
  * targets that have no observed outputs (fine)

This yields a real, meaningful “contract ↔ DAG materialization” verification.

---

## Minimal code edits to implement #1 end-to-end

I’m listing edits in the order that lets you do a **safe compare-then-flip**.

### 1) Add node-type constants for materialization metadata nodes

**File:** `src/codeintel/core/hamilton/tags.py`
**Edit:** add two constants (string values are up to you; keep them stable)

```python
# New node-type values (do NOT reuse NODE_TYPE_DATASET / NODE_TYPE_ARTIFACT)
NODE_TYPE_MATERIALIZATION_DATASET = "materialization.dataset"
NODE_TYPE_MATERIALIZATION_ARTIFACT = "materialization.artifact"
```

Why: you cannot tag SaveTo metadata nodes as `dataset/artifact` today without breaking your current `derive_target_outputs()` logic (it assumes dataset/artifact nodes are support nodes with a producing-target dependency).

---

### 2) Tag SaveTo-produced metadata nodes with (target, table_key/artifact_name, node_type)

**File:** `src/codeintel/build/hamilton/save_to.py`
**Function:** `SaveToObjectMetadataDecorator.create_saver_node(...)`

You already build the metadata node and set Hamilton saver tags. 

**Add imports:**

```python
from codeintel.core.hamilton.tags import (
    TAG_NODE_TYPE, TAG_TARGET, TAG_TABLE_KEY, TAG_ARTIFACT,
    NODE_TYPE_MATERIALIZATION_DATASET, NODE_TYPE_MATERIALIZATION_ARTIFACT,
)
```

**Then, after you compute `resolved_kwargs`:**

* Pull constants out if present:

  * `target_name = resolved_kwargs.get("target_name")`
  * `table_key = resolved_kwargs.get("table_key")`
  * `artifact_name = resolved_kwargs.get("artifact_name")`

**Patch-like sketch:**

```python
tags = {
    "hamilton.data_saver": True,
    "hamilton.data_saver.sink": self.sink,
    "hamilton.data_saver.classname": self.get_saver_classes()[0].name(),
}

target_name = resolved_kwargs_typed.get("target_name")
if isinstance(target_name, str) and target_name:
    tags[TAG_TARGET] = target_name

table_key = resolved_kwargs_typed.get("table_key")
artifact_name = resolved_kwargs_typed.get("artifact_name")

if isinstance(table_key, str) and table_key:
    tags[TAG_NODE_TYPE] = NODE_TYPE_MATERIALIZATION_DATASET
    tags[TAG_TABLE_KEY] = table_key
elif isinstance(artifact_name, str) and artifact_name:
    tags[TAG_NODE_TYPE] = NODE_TYPE_MATERIALIZATION_ARTIFACT
    tags[TAG_ARTIFACT] = artifact_name
# else: leave untyped (some savers might not map to “dataset/artifact”)
```

Result: every SaveTo metadata node becomes self-describing and queryable via tag filters, matching Hamilton’s best-practice usage of tags for compilation/introspection. ([Hamilton][3])

---

### 3) Add a new “materialization-derived outputs” introspector

**File:** `src/codeintel/build/hamilton/introspect.py`

**Add imports from core tags:**

```python
from codeintel.core.hamilton.tags import (
    NODE_TYPE_MATERIALIZATION_DATASET,
    NODE_TYPE_MATERIALIZATION_ARTIFACT,
)
```

**Add a new function (do not change existing one yet):**

```python
def derive_target_outputs_from_materializations(runtime: HamiltonRuntime) -> DerivedTargetOutputs:
    dr = runtime.dr

    datasets: dict[str, set[str]] = {}
    artifacts: dict[str, set[str]] = {}

    dataset_nodes = dr.list_available_variables(
        tag_filter={TAG_NODE_TYPE: NODE_TYPE_MATERIALIZATION_DATASET}
    )
    for n in dataset_nodes:
        tags = n.tags or {}
        target = tags.get(TAG_TARGET)
        table_key = tags.get(TAG_TABLE_KEY)
        if isinstance(target, str) and target and isinstance(table_key, str) and table_key:
            datasets.setdefault(target, set()).add(table_key)
        else:
            raise RuntimeError(f"Bad dataset materialization tags on {n.name}: {tags}")

    artifact_nodes = dr.list_available_variables(
        tag_filter={TAG_NODE_TYPE: NODE_TYPE_MATERIALIZATION_ARTIFACT}
    )
    for n in artifact_nodes:
        tags = n.tags or {}
        target = tags.get(TAG_TARGET)
        artifact = tags.get(TAG_ARTIFACT)
        if isinstance(target, str) and target and isinstance(artifact, str) and artifact:
            artifacts.setdefault(target, set()).add(artifact)
        else:
            raise RuntimeError(f"Bad artifact materialization tags on {n.name}: {tags}")

    return DerivedTargetOutputs(
        datasets_by_target={k: tuple(sorted(v)) for k, v in datasets.items()},
        artifacts_by_target={k: tuple(sorted(v)) for k, v in artifacts.items()},
    )
```

Why `list_available_variables(tag_filter=...)`? Hamilton explicitly documents tag-based discovery + `tag_filter` semantics (exact match, tag-exists, AND queries). ([Hamilton][1])

---

### 4) “Compare” plumbing: compute BOTH old + new outputs and diff them

**File:** `src/codeintel/build/target_metadata.py`
**Location:** `TargetMetadataService.outputs` cached_property

Currently it does:

```python
outputs = derive_target_outputs(system.runtime)
```

Change to:

* compute:

  * `support_outputs = derive_target_outputs(system.runtime)`  (existing behavior)
  * `mat_outputs = derive_target_outputs_from_materializations(system.runtime)` (new)
* diff them and either:

  * log warnings (default)
  * raise (strict mode / CI)
* pick which to return based on a feature flag (see migration)

This is the safe “compare-then-flip” core.

---

### 5) Make the contract checker actually check the DAG

**File:** `src/codeintel/build/hamilton/contracts/check_target_contracts.py`

This checker intends to verify contract-vs-DAG output agreement, but as discussed it’s currently self-referential if `service.outputs` comes from support nodes.

**Minimal change for the compare phase:**

* Have it call the *materialization-derived* function directly for “observed,” while leaving everything else untouched:

```python
observed = derive_target_outputs_from_materializations(service.system.runtime)
```

Now it truly checks that your `SaveToObjectMetadataDecorator` nodes exist for each contract output. This aligns with your architecture where these metadata nodes are the real materialization surface. 

---

### 6) “Flip” plumbing: drive support-node generation from materializations (optional but completes the E2E loop)

This is what makes the DAG *seamlessly adapt* when you add a new materialization node: support nodes appear because the DAG says the output exists, not because a parallel contract list was updated.

**Files:**

* `src/codeintel/build/hamilton/driver_factory.py`
* `src/codeintel/build/hamilton/nodes/support_factory.py`

**Edits:**

1. In `driver_factory._build_support_graph_and_module(...)`:

* Build the native driver (you already do)
* Compute `mat_outputs = derive_target_outputs_from_materializations(native_runtime)`
* Pass `mat_outputs` into support module generation when the flag is enabled.

2. In `support_factory.build_support_module(...)` / `_populate_for_target(...)`:

* Add optional parameter `derived_outputs: DerivedTargetOutputs | None = None`
* If provided, use `derived_outputs.datasets_by_target[target.name]` instead of `target.contract.table_keys`, similarly for artifacts.

This is directly relevant to your stated support-node runtime gating (dataset/loader nodes enforce upstream success) —you want those nodes to exist because the DAG says the materialization exists, not because a second registry said so.

---

## Compare-then-flip migration plan (safe, incremental)

### Phase A — “Compare only” (no behavior change)

Goal: prove the new derivation matches reality before changing orchestration.

* Ship steps **(1)–(5)**:

  * new tags on SaveTo metadata nodes
  * new `derive_target_outputs_from_materializations`
  * contract checker uses materialization-derived outputs
  * `TargetMetadataService` logs diffs, but still returns old `support_outputs`

**What you get immediately:** a meaningful CI/quality check that your DAG’s actual materializations match declared contracts.

### Phase B — “Flip outputs source behind a flag”

Add a config/env flag, e.g.:

* `CODEINTEL_OUTPUTS_SOURCE=materializations|support` (default `support`)

When `materializations`:

* `TargetMetadataService.outputs` returns `mat_outputs`
* still compute and diff `support_outputs`; if mismatch, raise in strict mode

### Phase C — “Flip support generation behind the same flag”

Add:

* `CODEINTEL_SUPPORT_NODES_FROM=materializations|contracts` (or reuse same flag)

When enabled:

* support nodes generated from `mat_outputs`

### Phase D — Make it the default, then delete old path

Once stable:

* default to materializations
* remove the support-derived output introspection path (or keep only as debug)

---

## Why this is “best-in-class Hamilton + DAG-centric”

* It treats materializers/savers as **first-class DAG nodes** (Hamilton explicitly models materialization as nodes in the dataflow). ([hamilton.staged.apache.org][2])
* It uses **tags + tag filtering** as the “semantic ABI” for discovery/registry compilation, which Hamilton docs recommend and support directly. ([Hamilton][3])
* It eliminates a major source of drift: “contract says X, DAG actually materializes Y.”

---

## Small “extras” I’d include while you’re touching this code (still minimal)

1. **Enforce tag completeness at decorator-build time**

   * If a saver node has `table_key` but no `target_name`, raise early in `create_saver_node()`.
2. **Include saver class + sink in tags** (you already do with `hamilton.data_saver.*`)

   * Later you can compile operational registries like: “all DuckDB materializations” vs “artifact writes”.

---

If you meant a *different* “#1” from my earlier list (e.g., “auto-generate tool/serving operations from DAG tags”), tell me which one and I’ll produce the same level of: **algorithm + file/function edit list + compare-then-flip plan** for that target as well.

[1]: https://hamilton.apache.org/reference/drivers/Driver/?utm_source=chatgpt.com "Builder - Hamilton - The Apache Software Foundation"
[2]: https://hamilton.staged.apache.org/concepts/materialization/?utm_source=chatgpt.com "Materialization - Hamilton - The Apache Software Foundation"
[3]: https://hamilton.apache.org/reference/decorators/tag/?utm_source=chatgpt.com "tag* - Hamilton - The Apache Software Foundation"



# enforce tag completeness and Include saver class + sink in tags #

Below are the “small extras” in **comprehensive, implementation-level detail** (with ready-to-apply diffs + a couple of usage/test snippets). Everything here is designed to be **minimal**, **fail-fast at DAG-build time**, and to make saver/materialization I/O nodes **first-class, tag-indexable DAG citizens**.

---

## 1) Enforce tag completeness at decorator-build time

### What “tag completeness” means here

For any node your `SaveToObjectMetadataDecorator` creates (i.e., the **DataSaver metadata node**):

* It must **always** have:

  * `hamilton.data_saver = True` (write-node detection, schema inference exclusions, etc.)
  * `hamilton.data_saver.sink = <stable sink name>` (operational grouping)
  * `hamilton.data_saver.classname = <class qualname>` (debuggability)
  * `target = <target_name>` (**critical** for contract enforcement + telemetry attribution)

* And when applicable, it should also include:

  * `table_key = <table_key>` for table materializations
  * `artifact = <artifact_name>` for artifact writes
  * `domain = <domain>` (optional but extremely useful for grouping; can be copied from upstream node tags)

### Why enforce at decorator-build time (graph construction)?

Because this gives you:

* **Immediate feedback** when someone adds a new computation/materialization and mis-wires it.
* A guarantee that any tooling relying on tags (telemetry hooks, contract enforcement, registries, UI, etc.) won’t silently degrade.

---

## 2) “If a saver node has table_key but no target_name, raise early in create_saver_node()”

This is the “pairing invariant” you called out:

* `table_key` strongly implies “this write belongs to a particular target”
* and your DataSavers rely on `target_name` for **manifest hashing**, **skip decisions**, and correct attribution.

So we fail fast during Driver construction.

(I also added the symmetric rule for `artifact_name` → requires `target_name`, because it’s the same class of failure and stays minimal.)

---

## 3) “Include saver class + sink in tags”

You already did this with:

* `hamilton.data_saver.sink = saver_cls.name()`
* `hamilton.data_saver.classname = saver_cls.__qualname__`

The diff below **keeps that**, but adds:

* checks that `name()` returns a non-empty string
* tag completeness validation (including `target`)
* canonical tags (`target`, `table_key`, `artifact`, and optionally `domain`)

---

## 4) Patch: `SaveToObjectMetadataDecorator` fail-fast invariants + canonical saver tags

### ✅ Diff (ready to apply)

```diff
--- a/src/codeintel/build/hamilton/save_to.py
+++ b/src/codeintel/build/hamilton/save_to.py
@@ -20,6 +20,7 @@
 from hamilton.node import DependencyType
 
 from codeintel.build.hamilton.boundary_types import MaterializationMetadata
+from codeintel.core.hamilton import tags as ht
 
 if TYPE_CHECKING:
     from collections.abc import Callable, Collection, Sequence
@@ -28,6 +29,11 @@
     from hamilton.io.data_adapters import AdapterCommon, DataSaver
 
 
+_TAG_DATA_SAVER = "hamilton.data_saver"
+_TAG_DATA_SAVER_SINK = "hamilton.data_saver.sink"
+_TAG_DATA_SAVER_CLASSNAME = "hamilton.data_saver.classname"
+
+
 class SaveToObjectMetadataDecorator(SingleNodeNodeTransformer):
     """Save-to decorator that types metadata nodes as ``MaterializationMetadata``.
 
@@ -61,6 +67,69 @@
         self.kwargs = kwargs
         self.target = target_
 
+    @staticmethod
+    def _require_resolved_str(
+        resolved_kwargs: dict[str, object],
+        *,
+        key: str,
+        fn_qualname: str,
+    ) -> str:
+        """Return a required resolved kwarg as a non-empty string.
+
+        This decorator is used to make I/O *DAG-visible*. In this codebase we rely
+        on stable, graph-introspectable tags for operational indexing (e.g.
+        grouping materializations by sink) and enforcement hooks (e.g. strict
+        contract enforcement based on ``target`` tag).
+
+        Therefore key identity fields like ``target_name``, ``table_key``, and
+        ``artifact_name`` are expected to be provided as fixed ``value(...)``
+        dependencies so they are available at graph construction time.
+        """
+        value = resolved_kwargs.get(key)
+        if not isinstance(value, str) or not value.strip():
+            msg = (
+                f"{fn_qualname}: SaveToObjectMetadataDecorator requires {key}=value(<non-empty str>) "
+                f"so saver tags can be derived at DAG-build time. Got: {value!r}"
+            )
+            raise InvalidDecoratorException(msg)
+        return value
+
+    @staticmethod
+    def _validate_saver_tags(
+        *,
+        tags: dict[str, object],
+        fn_qualname: str,
+        saver_node_name: str,
+    ) -> None:
+        """Fail fast if the generated saver node tags are incomplete."""
+        required: tuple[str, ...] = (
+            _TAG_DATA_SAVER,
+            _TAG_DATA_SAVER_SINK,
+            _TAG_DATA_SAVER_CLASSNAME,
+            ht.TAG_TARGET,
+        )
+        missing = [k for k in required if k not in tags]
+        if missing:
+            msg = (
+                f"{fn_qualname}: saver node '{saver_node_name}' is missing required tags: "
+                + ", ".join(missing)
+            )
+            raise InvalidDecoratorException(msg)
+
+        if tags.get(_TAG_DATA_SAVER) is not True:
+            msg = (
+                f"{fn_qualname}: saver node '{saver_node_name}' must set '{_TAG_DATA_SAVER}' to True"
+            )
+            raise InvalidDecoratorException(msg)
+
+        for key in (_TAG_DATA_SAVER_SINK, _TAG_DATA_SAVER_CLASSNAME, ht.TAG_TARGET):
+            value = tags.get(key)
+            if not isinstance(value, str) or not value.strip():
+                msg = (
+                    f"{fn_qualname}: saver node '{saver_node_name}' has empty/invalid tag '{key}': {value!r}"
+                )
+                raise InvalidDecoratorException(msg)
+
     def create_saver_node(
         self,
         node_: h_node.Node,
@@ -108,6 +177,51 @@
         dependencies_inverted = {v: k for k, v in dependencies.items()}
         resolved_kwargs_typed = cast("dict[str, object]", resolved_kwargs)
 
+        # --- Build-time invariants (fail fast, keep tags consistent) ---
+        # NOTE: We intentionally validate at graph-build time so misconfigured
+        # saver nodes fail during Driver construction rather than during a run.
+
+        if "table_key" in self.kwargs and "target_name" not in self.kwargs:
+            msg = (
+                f"{fn.__qualname__}: SaveToObjectMetadataDecorator specifies table_key but is "
+                "missing target_name. Table materializations require target_name=value(<target>) "
+                "so manifest hashing + contract enforcement are attributed to the correct target."
+            )
+            raise InvalidDecoratorException(msg)
+
+        if "artifact_name" in self.kwargs and "target_name" not in self.kwargs:
+            msg = (
+                f"{fn.__qualname__}: SaveToObjectMetadataDecorator specifies artifact_name but is "
+                "missing target_name. Artifact writes require target_name=value(<target>) so "
+                "manifest hashing + contract enforcement are attributed to the correct target."
+            )
+            raise InvalidDecoratorException(msg)
+
+        # Enforce target_name as a resolved string value.
+        # This is used by DataSavers for manifest hashing and by lifecycle hooks
+        # (contract enforcement + telemetry) via the `target` tag.
+        target_name = self._require_resolved_str(
+            resolved_kwargs_typed,
+            key="target_name",
+            fn_qualname=fn.__qualname__,
+        )
+
+        table_key: str | None = None
+        if "table_key" in self.kwargs:
+            table_key = self._require_resolved_str(
+                resolved_kwargs_typed,
+                key="table_key",
+                fn_qualname=fn.__qualname__,
+            )
+
+        artifact_tag: str | None = None
+        if "artifact_name" in self.kwargs:
+            artifact_tag = self._require_resolved_str(
+                resolved_kwargs_typed,
+                key="artifact_name",
+                fn_qualname=fn.__qualname__,
+            )
+
         def save_data(
             __adapter_factory: AdapterFactory = adapter_factory,
             __dependencies: dict[str, str] = dependencies_inverted,
@@ -145,17 +259,47 @@
         }
         input_types[node_to_save_str] = (node_.type, DependencyType.REQUIRED)
 
+        # Canonical tags for operational indexing + contract enforcement.
+        #
+        # IMPORTANT: Do NOT set ht.TAG_NODE_TYPE=materialize here.
+        # Target/materialize nodes (t__*) are the source of target discovery and
+        # dependency derivation. Saver nodes are operational I/O boundaries.
+        sink = saver_cls.name()
+        if not isinstance(sink, str) or not sink:
+            msg = f"{fn.__qualname__}: DataSaver.name() must return a non-empty string"
+            raise InvalidDecoratorException(msg)
+
+        tags: dict[str, object] = {
+            _TAG_DATA_SAVER: True,
+            _TAG_DATA_SAVER_SINK: sink,
+            _TAG_DATA_SAVER_CLASSNAME: f"{saver_cls.__qualname__}",
+            ht.TAG_TARGET: target_name,
+        }
+        if table_key is not None:
+            tags[ht.TAG_TABLE_KEY] = table_key
+        if artifact_tag is not None:
+            tags[ht.TAG_ARTIFACT] = artifact_tag
+
+        # Copy domain tag when present on the upstream node to keep telemetry
+        # grouping consistent (optional, but highly useful).
+        upstream_tags = node_.tags if isinstance(node_.tags, dict) else {}
+        domain = upstream_tags.get(ht.TAG_DOMAIN)
+        if isinstance(domain, str) and domain:
+            tags[ht.TAG_DOMAIN] = domain
+
+        self._validate_saver_tags(
+            tags=tags,
+            fn_qualname=fn.__qualname__,
+            saver_node_name=artifact_name_str,
+        )
+
         return h_node.Node(
             name=artifact_name_str,
             callabl=save_data,
             typ=cast("type[object]", MaterializationMetadata),
             input_types=input_types,
             namespace=artifact_namespace,
-            tags={
-                "hamilton.data_saver": True,
-                "hamilton.data_saver.sink": f"{saver_cls.name()}",
-                "hamilton.data_saver.classname": f"{saver_cls.__qualname__}",
-            },
+            tags=tags,
         )
```

### Net effect

* If someone adds a new saver node like:

```py
@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_="m__core__something",
    env=source("env"),
    graph=source("graph"),
    table_key=value("core.something"),
    # target_name=...  (oops)
)
def something_rows() -> tuple[tuple[object, ...], ...]:
    ...
```

…they’ll get a **Driver-build-time** exception, not a late runtime failure.

* Saver nodes now carry:

  * `target` (so contract enforcement + telemetry hook attribution works)
  * `table_key`/`artifact` when present (so registries can be built without string-parsing node names)

---

## 5) Operational registries: “all DuckDB materializations” vs “artifact writes”

You can do this without any new “registry framework” just by indexing tags.

### Minimal addition: extend `TagIndex` with saver node views

✅ Diff:

```diff
--- a/src/codeintel/build/hamilton/tag_index.py
+++ b/src/codeintel/build/hamilton/tag_index.py
@@ -61,6 +61,14 @@
     return {str(k): _stringify_tag_value(v) for k, v in tags_raw.items()}
 
+def _truthy(value: str | None) -> bool:
+    """Return True if a normalized tag value should be treated as truthy."""
+    if value is None:
+        return False
+    lowered = value.strip().lower()
+    return lowered in {"1", "true", "t", "yes", "y"}
+
@@ -162,5 +170,35 @@
             if tags.get(ht.TAG_NODE_TYPE) == ht.NODE_TYPE_ARTIFACT
         }
 
+    def data_saver_nodes(self) -> dict[str, dict[str, str]]:
+        """Return Hamilton DataSaver nodes keyed by node name."""
+        return {
+            name: tags
+            for name, tags in self.tags_by_node.items()
+            if _truthy(tags.get("hamilton.data_saver"))
+        }
+
+    def saver_nodes_by_sink(self) -> dict[str, dict[str, dict[str, str]]]:
+        """Group DataSaver nodes by sink."""
+        grouped: dict[str, dict[str, dict[str, str]]] = {}
+        for node_name, tags in self.data_saver_nodes().items():
+            sink = tags.get("hamilton.data_saver.sink") or "unknown"
+            grouped.setdefault(sink, {})[node_name] = tags
+        return grouped
```

### Example: compile “DuckDB vs artifact” registries

```py
from codeintel.build.hamilton.tag_index import TagIndex

def compile_io_registries(tag_index: TagIndex) -> dict[str, dict[str, dict[str, str]]]:
    savers_by_sink = tag_index.saver_nodes_by_sink()

    duckdb = {
        sink: nodes
        for sink, nodes in savers_by_sink.items()
        if sink.startswith("codeintel.duckdb")  # e.g. codeintel.duckdb_rows, codeintel.duckdb_table
    }
    artifacts = {
        sink: nodes
        for sink, nodes in savers_by_sink.items()
        if sink == "codeintel.file_artifact"
    }

    return {"duckdb": duckdb, "artifacts": artifacts}
```

### Example: build a “table_key → sink” index

Now that saver nodes carry `table_key` tags:

```py
def table_writes_index(tag_index: TagIndex) -> dict[str, str]:
    out: dict[str, str] = {}
    for node_name, tags in tag_index.data_saver_nodes().items():
        table_key = tags.get("table_key")
        sink = tags.get("hamilton.data_saver.sink")
        if table_key and sink:
            out[table_key] = sink
    return out
```

---

## 6) Tiny unit test snippet (optional but recommended)

This validates the “table_key requires target_name” invariant triggers during graph build:

```py
import types
import pytest
from hamilton.driver import Driver
from hamilton.function_modifiers.dependencies import source, value
from hamilton.function_modifiers.base import InvalidDecoratorException

from codeintel.build.hamilton.save_to import SaveToObjectMetadataDecorator
from codeintel.build.hamilton.materializers.duckdb_rows_saver import DuckDBRowsSaver

def test_save_to_requires_target_name_when_table_key_present():
    mod = types.ModuleType("dummy_mod")

    @SaveToObjectMetadataDecorator(
        [DuckDBRowsSaver],
        output_name_="m__core__bad",
        env=source("env"),
        graph=source("graph"),
        table_key=value("core.bad"),
        columns=value(("col1",)),
        # target_name is intentionally missing
    )
    def bad_rows() -> tuple[tuple[object, ...], ...]:
        return ((1,),)

    mod.bad_rows = bad_rows

    with pytest.raises(InvalidDecoratorException):
        Driver({}, mod)
```

---

## Summary of what you gain (for very little code)

* **Fail-fast** materialization wiring errors during DAG build.
* Saver nodes become **properly attributable** (target/domain), which improves:

  * contract enforcement correctness
  * telemetry utility
* You can build **operational registries** purely via tags:

  * “all DuckDB materializations”
  * “all artifact writes”
  * “table_key → sink”
  * “target → all writes”

If you want next: I can extend this same pattern to **loaders** (q__/df__ nodes) so you can compile a complete “read/write surface map” per target strictly from DAG tags (no extra registries), but I kept the above strictly within the “small extras / minimal touch” scope you asked for.


# introspection algorithm #

Below is a concrete, “drop-in” sketch of the **exact introspection algorithm** + the **minimal code edits (by file/function)** to implement the end-to-end “compare-then-flip” migration, plus the “small extras” (tag completeness + saver tags), and then the **loader (q__/df__) extension** to produce a **complete read/write surface map per target derived purely from DAG tags**.

I’m going to be very explicit about:

* **What tags become the single source of truth**
* **Where to enforce them**
* **How to derive inventories + surfaces from the DAG**
* **How to roll out safely (compare → warn → flip)**

---

## 0) The tag contract we want (small but decisive)

### Write boundary nodes (Hamilton DataSaver metadata nodes, your `m__...` nodes)

A DataSaver node must be *self-describing* enough that you can answer, from tags alone:

* “Which target owns this write?”
* “Is it writing a table or an artifact?”
* “What table_key or artifact_name is it writing?”
* “What sink/adapter is doing the writing?”

**Required tags on every `hamilton.data_saver` node:**

* `hamilton.data_saver = True`
* `hamilton.data_saver.sink = saver_cls.name()` (already present)
* `hamilton.data_saver.classname = saver_cls.__qualname__` (already present)
* `target = <owning target name>`  ✅ **new**
* exactly one of:

  * `table_key = <schema.table>` ✅ **new**, OR
  * `artifact = <artifact_name>` ✅ **new**
* (optional but very useful): propagate `domain = <module>` from the decorated compute node ✅ **new**

This single change unlocks:

* **Strict contract enforcement actually applying to savers** (because your `ContractEnforcementHook` activates by `target` tag)
* **DAG-derived output inventory** that reflects real IO, not “declared contract graph”
* Operational registries like “all DuckDB table writes” vs “all artifact writes” without any extra registries

### Read boundary nodes (your loaders `q__/df__`)

These are already *very close* to what we need. Your `support_factory` tags loader nodes with:

* `node_type = loader.query | loader.dataframe`
* `table_key = ...`
* `target = <producer target>` (producer is good—lets you map “who owns the data?”)
* `domain = ...`

That’s enough to derive target read surfaces by traversing upstream from `t__...` nodes.

---

## 1) “Small extras / minimal touch” — implement tag completeness + richer saver tags

### 1A) Edit `SaveToObjectMetadataDecorator.create_saver_node()` to enforce and tag

**File:** `src/codeintel/build/hamilton/save_to.py`
**Function:** `SaveToObjectMetadataDecorator.create_saver_node`

Here’s a patch-style snippet that does exactly what you requested:

* If `table_key` exists but `target_name` doesn’t → raise early
* Require `target_name/table_key/artifact_name` to be **static** `value(...)` (taggable)
* Copy `domain` from the decorated node’s tags if present
* Add `target`, `table_key`, `artifact` tags onto the saver node

```diff
diff --git a/src/codeintel/build/hamilton/save_to.py b/src/codeintel/build/hamilton/save_to.py
index 0000000..0000000 100644
--- a/src/codeintel/build/hamilton/save_to.py
+++ b/src/codeintel/build/hamilton/save_to.py
@@
 from hamilton.node import DependencyType

 from codeintel.build.hamilton.boundary_types import MaterializationMetadata
+from codeintel.core.hamilton import tags as ht

@@ class SaveToObjectMetadataDecorator(SingleNodeNodeTransformer):
     def create_saver_node(
         self,
         node_: h_node.Node,
         _config: dict[str, object],
         fn: Callable[..., object],
     ) -> h_node.Node:
@@
         saver_cls = resolve_adapter_class(node_.type, list(self.saver_classes))
         if saver_cls is None:
             msg = f"No saver class found for type: {node_.type!r} (fn={fn.__qualname__})"
             raise InvalidDecoratorException(msg)

         adapter_factory = AdapterFactory(saver_cls, **self.kwargs)
         dependencies, resolved_kwargs = resolve_kwargs(self.kwargs)
         dependencies_inverted = {v: k for k, v in dependencies.items()}
         resolved_kwargs_typed = cast("dict[str, object]", resolved_kwargs)

+        # ------------------------------
+        # CodeIntel "tag completeness" invariants for saver nodes
+        # ------------------------------
+        def _require_static_str(key: str, *, reason: str) -> str:
+            """Require that key is provided via value(...) so it exists in resolved_kwargs."""
+            value = resolved_kwargs_typed.get(key)
+            if not isinstance(value, str) or not value.strip():
+                msg = (
+                    f"SaveToObjectMetadataDecorator: {reason}. "
+                    f"Expected `{key}=value(...)` with a non-empty string "
+                    f"(node={node_.name!r}, saver={saver_cls.__qualname__}, fn={fn.__qualname__})."
+                )
+                raise InvalidDecoratorException(msg)
+            return value.strip()
+
+        has_table_key_kw = "table_key" in self.kwargs
+        has_artifact_name_kw = "artifact_name" in self.kwargs
+
+        # Your requested hard fail:
+        if has_table_key_kw and "target_name" not in self.kwargs:
+            raise InvalidDecoratorException(
+                "SaveToObjectMetadataDecorator: `table_key` requires `target_name` "
+                f"(node={node_.name!r}, saver={saver_cls.__qualname__}, fn={fn.__qualname__})."
+            )
+
+        # Symmetric enforcement (recommended—same reason):
+        if has_artifact_name_kw and "target_name" not in self.kwargs:
+            raise InvalidDecoratorException(
+                "SaveToObjectMetadataDecorator: `artifact_name` requires `target_name` "
+                f"(node={node_.name!r}, saver={saver_cls.__qualname__}, fn={fn.__qualname__})."
+            )
+
+        if has_table_key_kw and has_artifact_name_kw:
+            raise InvalidDecoratorException(
+                "SaveToObjectMetadataDecorator: expected exactly one of `table_key` or `artifact_name`, got both "
+                f"(node={node_.name!r}, saver={saver_cls.__qualname__}, fn={fn.__qualname__})."
+            )
+
+        # If the saver requires target_name (your savers do), require it be taggable/static:
+        requires_target_name = "target_name" in saver_cls.get_required_arguments()
+        target_name_value: str | None = None
+        if requires_target_name or ("target_name" in self.kwargs):
+            target_name_value = _require_static_str(
+                "target_name",
+                reason="saver nodes must be attributable to an owning target for contract enforcement + DAG IO introspection",
+            )
+
+        table_key_value: str | None = None
+        if has_table_key_kw:
+            table_key_value = _require_static_str(
+                "table_key",
+                reason="table writes must declare a stable table_key for DAG-derived IO inventory",
+            )
+
+        artifact_name_value: str | None = None
+        if has_artifact_name_kw:
+            artifact_name_value = _require_static_str(
+                "artifact_name",
+                reason="artifact writes must declare a stable artifact_name for DAG-derived IO inventory",
+            )
+
+        if table_key_value is None and artifact_name_value is None:
+            # Optional: if you ever add other savers that don't map to table/artifact, loosen this.
+            raise InvalidDecoratorException(
+                "SaveToObjectMetadataDecorator: saver nodes must declare either table_key or artifact_name "
+                f"(node={node_.name!r}, saver={saver_cls.__qualname__}, fn={fn.__qualname__})."
+            )
+
+        # Build tags: keep your existing saver tags + add CodeIntel IO identity tags.
+        node_tags = node_.tags if isinstance(node_.tags, dict) else {}
+        saver_tags: dict[str, object] = {
+            "hamilton.data_saver": True,
+            "hamilton.data_saver.sink": f"{saver_cls.name()}",
+            "hamilton.data_saver.classname": f"{saver_cls.__qualname__}",
+        }
+        # propagate domain if present (helps ops filtering)
+        domain_value = node_tags.get(ht.TAG_DOMAIN)
+        if isinstance(domain_value, str) and domain_value:
+            saver_tags[ht.TAG_DOMAIN] = domain_value
+        if target_name_value is not None:
+            saver_tags[ht.TAG_TARGET] = target_name_value
+        if table_key_value is not None:
+            saver_tags[ht.TAG_TABLE_KEY] = table_key_value
+        if artifact_name_value is not None:
+            saver_tags[ht.TAG_ARTIFACT] = artifact_name_value

@@
         return h_node.Node(
             name=artifact_name_str,
             callabl=save_data,
             typ=cast("type[object]", MaterializationMetadata),
             input_types=input_types,
             namespace=artifact_namespace,
-            tags={
-                "hamilton.data_saver": True,
-                "hamilton.data_saver.sink": f"{saver_cls.name()}",
-                "hamilton.data_saver.classname": f"{saver_cls.__qualname__}",
-            },
+            tags=saver_tags,
         )
```

### Why this matters immediately

1. **Strict contract enforcement finally applies to savers**
   Your `ContractEnforcementHook` activates `ContractEnforcer.activate()` only when it sees `tags["target"]`. Without this change, calls like `ContractEnforcer.validate_table_write(...)` in `DuckDBRowsSaver`/`DuckDBIbisTableSaver` often won’t enforce anything (because the saver node wasn’t tagged as belonging to a target).

2. You can now compile an authoritative IO registry from the DAG.

---

## 2) “Operational registries” from tags (no extra registries)

Once the saver nodes have `target/table_key/artifact/sink`, you can build the exact registries you mentioned.

Example helper (can live in `introspect.py` or a small `io_registry.py`):

```python
from collections import defaultdict
from typing import Any

from codeintel.core.hamilton import tags as ht
from codeintel.build.hamilton.runtime import HamiltonRuntime

def compile_write_registry(runtime: HamiltonRuntime) -> dict[str, list[dict[str, Any]]]:
    """Group saver nodes by sink, with enough fields for ops registries."""
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for node in runtime.dr.graph.nodes.values():
        tags = node.tags if isinstance(node.tags, dict) else {}
        if tags.get("hamilton.data_saver") is not True:
            continue

        sink = tags.get("hamilton.data_saver.sink")
        if not isinstance(sink, str):
            sink = "unknown"

        grouped[sink].append(
            {
                "node": node.name,
                "domain": tags.get(ht.TAG_DOMAIN),
                "target": tags.get(ht.TAG_TARGET),
                "table_key": tags.get(ht.TAG_TABLE_KEY),
                "artifact": tags.get(ht.TAG_ARTIFACT),
            }
        )

    # stable ordering for deterministic output
    for sink in list(grouped):
        grouped[sink] = sorted(grouped[sink], key=lambda r: str(r["node"]))

    return dict(grouped)

def duckdb_materializations(runtime: HamiltonRuntime) -> list[dict[str, Any]]:
    reg = compile_write_registry(runtime)
    duckdb_sinks = {"codeintel.duckdb_rows", "codeintel.duckdb_table"}
    out: list[dict[str, Any]] = []
    for sink in duckdb_sinks:
        out.extend(reg.get(sink, []))
    return out

def artifact_writes(runtime: HamiltonRuntime) -> list[dict[str, Any]]:
    reg = compile_write_registry(runtime)
    return reg.get("codeintel.file_artifact", [])
```

That’s the “all DuckDB materializations vs artifact writes” registry you described, built purely from DAG tags.

---

## 3) The “#1 end-to-end” piece: derive output inventory from **real writes**, safely

### The core problem today

Your current “derived outputs” are derived from `d__`/`a__` nodes, but those nodes are generated from the **declared contract graph**, so they’re not truly “DAG-derived IO.”

### The fix

Derive target output inventory from **saver nodes** (`hamilton.data_saver=True`) instead.

### Minimal code addition: `derive_target_outputs_from_savers(runtime)`

**File:** `src/codeintel/build/hamilton/introspect.py`

Add imports at the top:

```python
from codeintel.core.hamilton.tags import (
    NODE_TYPE_LOADER_QUERY,
    NODE_TYPE_LOADER_DATAFRAME,
    # already imported: TAG_TARGET, TAG_TABLE_KEY, TAG_ARTIFACT, TAG_NODE_TYPE, ...
)
```

Add this function (and export it):

```python
def derive_target_outputs_from_savers(runtime: HamiltonRuntime) -> DerivedTargetOutputs:
    """Derive outputs from DataSaver nodes (m__...) instead of contract-derived d__/a__ nodes."""
    nodes: Mapping[str, Node] = runtime.dr.graph.nodes

    datasets: dict[str, set[str]] = {}
    artifacts: dict[str, set[str]] = {}

    for node in nodes.values():
        tags = node.tags
        if not isinstance(tags, dict):
            continue
        if tags.get("hamilton.data_saver") is not True:
            continue

        target = tags.get(TAG_TARGET)
        if not isinstance(target, str) or not target:
            raise RuntimeError(f"DataSaver node {node.name} missing target tag")

        table_key = tags.get(TAG_TABLE_KEY)
        artifact_name = tags.get(TAG_ARTIFACT)

        if isinstance(table_key, str) and table_key:
            datasets.setdefault(target, set()).add(table_key)
        if isinstance(artifact_name, str) and artifact_name:
            artifacts.setdefault(target, set()).add(artifact_name)

        if (not table_key) and (not artifact_name):
            raise RuntimeError(
                f"DataSaver node {node.name} missing both table_key and artifact tags"
            )

    datasets_by_target = {k: tuple(sorted(v)) for k, v in datasets.items()}
    artifacts_by_target = {k: tuple(sorted(v)) for k, v in artifacts.items()}
    return DerivedTargetOutputs(
        datasets_by_target=datasets_by_target,
        artifacts_by_target=artifacts_by_target,
    )
```

---

## 4) Safe “compare-then-flip” migration: exact algorithm + minimal integration edits

### 4A) Compare algorithm (deterministic)

Given:

* `base_graph: TargetGraph` (declared contracts)
* `runtime: HamiltonRuntime` (full DAG)

Do:

1. **Declared outputs** per target:

   * tables = `set(target.contract.table_keys)`
   * artifacts = `set(target.contract.artifact_names)`

2. **DAG outputs** per target:

   * `dag_outputs = derive_target_outputs_from_savers(runtime)`
   * tables = `set(dag_outputs.datasets_by_target.get(target, ()))`
   * artifacts = `set(dag_outputs.artifacts_by_target.get(target, ()))`

3. Compute diffs:

   * Missing in contract: `dag - declared`
   * Missing in DAG: `declared - dag`

4. Emit issues:

   * In “compare” mode: warnings
   * In “strict compare” mode: errors
   * In “flip” mode: use DAG inventory as authoritative

### 4B) Minimal integration point #1: make `validate_graph()` actually validate real IO

**File:** `src/codeintel/build/hamilton/validate.py`

Add a saver-output collector (mirrors your dataset/artifact collectors, but uses the saver tags):

```python
def _collect_saver_outputs(
    nodes: Mapping[str, NodeLike],
) -> tuple[dict[str, str], dict[str, str], list[GraphValidationIssue]]:
    produced_table_to_target: dict[str, str] = {}
    produced_artifact_to_target: dict[str, str] = {}
    issues: list[GraphValidationIssue] = []

    for node_name in sorted(nodes):
        node = nodes[node_name]
        tags = _tags_mapping(node)
        if tags is None:
            continue
        if tags.get("hamilton.data_saver") is not True:
            continue

        target = tags.get(TAG_TARGET)
        if not isinstance(target, str) or not target:
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_tag",
                    message="DataSaver node missing target tag",
                    node=node.name,
                )
            )
            continue

        table_key = tags.get(TAG_TABLE_KEY)
        artifact = tags.get(TAG_ARTIFACT)

        if isinstance(table_key, str) and table_key:
            existing = produced_table_to_target.get(table_key)
            if existing is not None and existing != target:
                issues.append(
                    GraphValidationIssue(
                        severity="error",
                        code="duplicate_table_key",
                        message=f"table_key produced by multiple targets: {existing}, {target}",
                        node=node.name,
                        target=target,
                        table_key=table_key,
                    )
                )
            produced_table_to_target[table_key] = target

        if isinstance(artifact, str) and artifact:
            existing = produced_artifact_to_target.get(artifact)
            if existing is not None and existing != target:
                issues.append(
                    GraphValidationIssue(
                        severity="error",
                        code="duplicate_artifact",
                        message=f"artifact produced by multiple targets: {existing}, {target}",
                        node=node.name,
                        target=target,
                        artifact=artifact,
                    )
                )
            produced_artifact_to_target[artifact] = target

        if (not table_key) and (not artifact):
            issues.append(
                GraphValidationIssue(
                    severity="error",
                    code="missing_tag",
                    message="DataSaver node missing both table_key and artifact tags",
                    node=node.name,
                    target=target,
                )
            )

    return produced_table_to_target, produced_artifact_to_target, issues
```

Then, in `validate_graph(...)`, call it and feed it into your mismatch checker instead of the contract-derived dataset/artifact nodes.

Where you currently do:

* `_collect_produced_tables(...)`
* `_collect_produced_artifacts(...)`
* `_derived_outputs_mismatch_issues(...)`

Add:

```python
saver_table_to_target, saver_artifact_to_target, saver_issues = _collect_saver_outputs(nodes)
issues.extend(saver_issues)

issues.extend(
    _derived_outputs_mismatch_issues(
        base_graph,
        produced_table_to_target=saver_table_to_target,
        produced_artifact_to_target=saver_artifact_to_target,
    )
)
```

Now your “derived outputs mismatch” validation becomes **real**: it compares declared contract vs actual DAG IO boundaries.

### 4C) Minimal integration point #2: flip output inventory source behind a mode flag

If you want the full end-to-end effect (use DAG inventory for runtime behavior), do this:

#### Step 1: Add a build setting (3 modes)

**File:** `src/codeintel/core/config/settings.py` (BuildSettings)

```python
@dataclass(frozen=True, slots=True)
class BuildSettings:
    engine_version: str
    export_audit: ExportAuditSettings = field(default_factory=ExportAuditSettings)

    # "declared" (today), "compare" (warn), "dag" (flip)
    output_inventory_source: str = "declared"
```

#### Step 2: Load it from env

**File:** `src/codeintel/core/runtime/loader.py` in `_load_build_settings()`:

```python
def _load_build_settings() -> BuildSettings:
    source = os.environ.get("CODEINTEL_OUTPUT_INVENTORY_SOURCE", "declared").strip().lower()
    if source not in {"declared", "compare", "dag"}:
        source = "declared"
    return BuildSettings(
        engine_version=_resolve_engine_version(),
        export_audit=ExportAuditSettings(
            log_path=_resolve_export_audit_log_path(),
            table_enabled=_resolve_export_audit_table_enabled(),
        ),
        output_inventory_source=source,
    )
```

#### Step 3: Compute “dag inventory” once and compare/log

**File:** `src/codeintel/build/target_metadata.py`

Change import:

```diff
-from codeintel.build.hamilton.introspect import derive_target_outputs
+from codeintel.build.hamilton.introspect import derive_target_outputs, derive_target_outputs_from_savers
```

Where you currently build `TargetSystemOutputs`, compute both:

* `declared_like = derive_target_outputs(runtime)` (today’s behavior)
* `dag_like = derive_target_outputs_from_savers(runtime)` (new)

Then:

* If mode is `"compare"`: log diffs but continue returning declared
* If mode is `"dag"`: return dag-derived

(Exact wiring depends on where you instantiate `TargetMetadataService`, but the intent is clear and the change is localized.)

#### Step 4: BuildEnv uses the chosen inventory automatically

**File:** `src/codeintel/build/run_context.py` in `BuildRunContext.build_env()`

Change:

```diff
-from codeintel.build.target_inventory import get_output_inventory
+from codeintel.build.target_inventory import get_output_inventory
+from codeintel.build.target_metadata import get_target_metadata_service
```

Then:

```python
output_inventory = self.output_inventory
if output_inventory is None:
    source = getattr(self.settings, "output_inventory_source", "declared")
    if source in {"compare", "dag"}:
        # outputs now reflect whatever mode target_metadata_service is running in
        output_inventory = get_target_metadata_service().outputs
    else:
        output_inventory = get_output_inventory()
```

That’s the “flip” lever.

---

## 5) The “next” piece you asked for: extend the pattern to loaders to get a complete read/write surface map

Once saver nodes are tagged and loader nodes already are, the remaining work is an **introspection traversal**.

### 5A) What “surface map” means in concrete terms

For each target `T`, we want:

* **Writes**

  * Tables written: from saver nodes with `target=T` and `table_key=...`
  * Artifacts written: from saver nodes with `target=T` and `artifact=...`
* **Reads**

  * Tables read: any `loader.query` / `loader.dataframe` node reachable upstream from `t__T`

    * include `table_key`
    * include *producer target* (already on loader tags as `target`)
    * include loader kind (query vs dataframe)

All of that comes from DAG tags + edges, with no other registries.

### 5B) Add a new introspection helper (recommended location: `introspect.py`)

**File:** `src/codeintel/build/hamilton/introspect.py`

Add dataclasses (small but makes downstream usage clean):

```python
from dataclasses import dataclass
from collections import defaultdict, deque

from codeintel.core.hamilton.tags import (
    NODE_TYPE_LOADER_QUERY,
    NODE_TYPE_LOADER_DATAFRAME,
    TAG_DOMAIN,
    # already imported: TAG_NODE_TYPE, TAG_TABLE_KEY, TAG_TARGET, TAG_ARTIFACT ...
)

@dataclass(frozen=True)
class TableRead:
    table_key: str
    producer_target: str | None
    loader_node: str
    loader_type: str  # loader.query | loader.dataframe

@dataclass(frozen=True)
class TableWrite:
    table_key: str
    sink: str
    saver_node: str

@dataclass(frozen=True)
class ArtifactWrite:
    artifact_name: str
    sink: str
    saver_node: str

@dataclass(frozen=True)
class TargetIOSurface:
    target: str
    reads: tuple[TableRead, ...]
    table_writes: tuple[TableWrite, ...]
    artifact_writes: tuple[ArtifactWrite, ...]
```

Now the function itself:

```python
def derive_target_io_surface(runtime: HamiltonRuntime) -> dict[str, TargetIOSurface]:
    nodes: Mapping[str, Node] = runtime.dr.graph.nodes

    # 1) Index saver writes by owning target
    table_writes: dict[str, list[TableWrite]] = defaultdict(list)
    artifact_writes: dict[str, list[ArtifactWrite]] = defaultdict(list)

    for node in nodes.values():
        tags = node.tags
        if not isinstance(tags, dict):
            continue
        if tags.get("hamilton.data_saver") is not True:
            continue

        target = tags.get(TAG_TARGET)
        if not isinstance(target, str) or not target:
            continue  # should be impossible after the save_to.py enforcement

        sink = tags.get("hamilton.data_saver.sink")
        sink_str = sink if isinstance(sink, str) else "unknown"

        tk = tags.get(TAG_TABLE_KEY)
        if isinstance(tk, str) and tk:
            table_writes[target].append(TableWrite(table_key=tk, sink=sink_str, saver_node=node.name))

        art = tags.get(TAG_ARTIFACT)
        if isinstance(art, str) and art:
            artifact_writes[target].append(
                ArtifactWrite(artifact_name=art, sink=sink_str, saver_node=node.name)
            )

    # 2) For each target, traverse upstream and collect loader reads.
    surfaces: dict[str, TargetIOSurface] = {}

    for target_name, target_node_name in runtime.target_to_node.items():
        root = nodes.get(target_node_name)
        if root is None:
            continue

        reads: list[TableRead] = []
        seen: set[str] = set()
        q: deque[Node] = deque(root.dependencies)

        while q:
            cur = q.popleft()
            if cur.name in seen:
                continue
            seen.add(cur.name)

            tags = cur.tags if isinstance(cur.tags, dict) else {}
            node_type = tags.get(TAG_NODE_TYPE)

            # Stop at loader boundary nodes: these represent a real table read surface.
            if node_type in {NODE_TYPE_LOADER_QUERY, NODE_TYPE_LOADER_DATAFRAME}:
                table_key = tags.get(TAG_TABLE_KEY)
                if isinstance(table_key, str) and table_key:
                    producer = tags.get(TAG_TARGET)
                    producer_str = producer if isinstance(producer, str) else None
                    reads.append(
                        TableRead(
                            table_key=table_key,
                            producer_target=producer_str,
                            loader_node=cur.name,
                            loader_type=str(node_type),
                        )
                    )
                continue  # do not traverse past read boundary

            # Optionally: stop at other target nodes to avoid cross-target explosion.
            # (Your DAG usually doesn't embed other t__ nodes upstream anyway,
            # because the interface is q__/df__.)
            if tags.get(TAG_NODE_TYPE) == NODE_TYPE_MATERIALIZE and cur.name != root.name:
                continue

            # Otherwise, keep walking
            for dep in cur.dependencies:
                q.append(dep)

        # stable + dedup by table_key+loader_type+producer
        reads_sorted = tuple(
            sorted(
                { (r.table_key, r.loader_type, r.producer_target, r.loader_node): r for r in reads }.values(),
                key=lambda r: (r.table_key, r.loader_type, r.producer_target or "", r.loader_node),
            )
        )
        writes_sorted = tuple(sorted(table_writes.get(target_name, []), key=lambda w: (w.table_key, w.sink, w.saver_node)))
        artifact_sorted = tuple(sorted(artifact_writes.get(target_name, []), key=lambda w: (w.artifact_name, w.sink, w.saver_node)))

        surfaces[target_name] = TargetIOSurface(
            target=target_name,
            reads=reads_sorted,
            table_writes=writes_sorted,
            artifact_writes=artifact_sorted,
        )

    return surfaces
```

### What you get out of this immediately

* A **complete IO map per target**:

  * reads → which tables, from which producer targets, via which loader kind
  * writes → which tables/artifacts, via which saver sink
* No registries. No separate “inventory” tables. Just DAG tags and structure.

### 5C) Optional: use this surface map to generate “IO policy” automatically

Once you have `TargetIOSurface`, you can do DAG-derived policies like:

* “Targets that write DuckDB tables must run with `parallel_backend=...`”
* “Artifact writers must serialize” (already done by your `ThreadPoolAdapter`, but now you can target it precisely)
* “Targets that read from `graphs.*` must depend on graph materializations” (you can validate that read table_keys belong to some producer target, etc.)

---

## 6) Summary: minimal change set (by file/function)

### Mandatory (core extras)

1. **`src/codeintel/build/hamilton/save_to.py`**

   * `SaveToObjectMetadataDecorator.create_saver_node`
   * Add:

     * early failures for missing `target_name` when `table_key` present (your request)
     * add `target/table_key/artifact/domain` tags to saver nodes

### Minimal for “#1 compare-then-flip”

2. **`src/codeintel/build/hamilton/introspect.py`**

   * add `derive_target_outputs_from_savers(runtime)`

3. **`src/codeintel/build/hamilton/validate.py`**

   * add `_collect_saver_outputs(...)`
   * feed saver-derived outputs into `_derived_outputs_mismatch_issues(...)`

### Optional but recommended to fully “flip”

4. **`src/codeintel/core/config/settings.py` + `src/codeintel/core/runtime/loader.py`**

   * add `BuildSettings.output_inventory_source = {"declared","compare","dag"}`

5. **`src/codeintel/build/run_context.py`**

   * `BuildRunContext.build_env` chooses output inventory source

### Loader extension (surface map)

6. **`src/codeintel/build/hamilton/introspect.py`**

   * add `derive_target_io_surface(runtime)` and the surface dataclasses

---

If you want, I can also:

* Wire `derive_target_io_surface()` into your CLI “lineage” or “explain” handlers so you can print the IO surface map per target in one command, **and**
* Add one “golden” test that asserts saver nodes always have `{target, (table_key|artifact)}` tags, so this never regresses again.


# CLI output and golden test enforcing behaviors #

Below is an end-to-end, “drop-in” implementation of what you asked for:

1. **Wire `derive_target_io_surface()` into the CLI `build explain` command** so you can print a per-target IO surface (reads/writes) in one command, and
2. **Add a “golden” test** that enforces saver nodes always have `{target, (table_key|artifact)}` tags.

This assumes (as discussed earlier) that saver metadata nodes are created via your `SaveToObjectMetadataDecorator` and that you want the IO surface to be **DAG-derived + boundary-aware** (i.e., don’t attribute upstream targets’ internal reads/writes to the current target).

---

## 1) Minimal prerequisite: make saver nodes carry canonical tags

Your IO-surface derivation is only as good as the tags on the saver metadata nodes (`m__*`). The smallest safe change is:

* Add canonical tags to saver nodes when they’re created:

  * `target` (from `target_name`)
  * `table_key` (from `table_key`) **or** `artifact` (from `artifact_name`)
* Add a guardrail: **if a saver has `table_key`/`artifact` but no `target_name`, raise early**.

### File: `src/codeintel/build/hamilton/save_to.py`

Add the canonical tags import:

```py
from codeintel.core.hamilton.tags import TAG_ARTIFACT, TAG_TABLE_KEY, TAG_TARGET
```

Then inside `SaveToObjectMetadataDecorator.create_saver_node(...)`, after:

```py
dependencies, resolved_kwargs = resolve_kwargs(self.kwargs)
resolved_kwargs_typed = cast("dict[str, object]", resolved_kwargs)
```

add:

```py
# --- Canonical tags for DAG-derived IO surface mapping ---
target_tag = resolved_kwargs_typed.get("target_name")
table_key_tag = resolved_kwargs_typed.get("table_key")
artifact_tag = resolved_kwargs_typed.get("artifact_name")
if artifact_tag is None:
    # Some savers may use the generic name "artifact".
    artifact_tag = resolved_kwargs_typed.get("artifact")

tag_target = target_tag if isinstance(target_tag, str) and target_tag else None
tag_table_key = table_key_tag if isinstance(table_key_tag, str) and table_key_tag else None
tag_artifact = artifact_tag if isinstance(artifact_tag, str) and artifact_tag else None

# Guardrail: if a saver declares a table_key/artifact, it must also declare a target.
if (tag_table_key or tag_artifact) and not tag_target:
    msg = (
        "Saver node has table_key/artifact but no target_name. "
        "Pass target_name=value(<target>) into SaveToObjectMetadataDecorator so tags are complete. "
        f"(fn={fn.__qualname__}, saver={saver_cls.__qualname__})"
    )
    raise InvalidDecoratorException(msg)
```

Finally, in the saver node’s `tags=...`, extend with:

```py
tags={
    "hamilton.data_saver": True,
    "hamilton.data_saver.sink": f"{saver_cls.name()}",
    "hamilton.data_saver.classname": f"{saver_cls.__qualname__}",
    **({TAG_TARGET: tag_target} if tag_target else {}),
    **({TAG_TABLE_KEY: tag_table_key} if tag_table_key else {}),
    **({TAG_ARTIFACT: tag_artifact} if tag_artifact else {}),
},
```

This is the critical “plumbing” that makes the IO surface derivable *strictly from DAG tags*.

---

## 2) Implement `derive_target_io_surface()` (boundary-aware)

You want **per-target** read/write mapping **without** accidentally including upstream targets’ internal I/O. The key is:

* Traverse upstream from `t__<target>` (materialize node),
* **Stop traversal when you hit another target’s materialize node**, and
* Collect:

  * Reads: loader/query + loader/dataframe (+ dataset-ref if desired)
  * Writes: saver (`hamilton.data_saver`) nodes *tagged with this target*

### File: `src/codeintel/build/hamilton/introspect.py`

#### 2.1 Import loader node-type constants

At the top where you import tags, add:

```py
from codeintel.core.hamilton.tags import (
    NODE_TYPE_ARTIFACT,
    NODE_TYPE_DATASET,
    NODE_TYPE_LOADER_DATAFRAME,
    NODE_TYPE_LOADER_QUERY,
    NODE_TYPE_MATERIALIZE,
    TAG_ARTIFACT,
    TAG_NODE_TYPE,
    TAG_TABLE_KEY,
    TAG_TARGET,
)
```

#### 2.2 Add the function

Place this right after `derive_target_dependencies(...)` (or near other “derive_…” functions):

```py
def derive_target_io_surface(
    runtime: HamiltonRuntime,
    *,
    include_targets: Iterable[str] | None = None,
) -> dict[str, dict[str, object]]:
    """Derive per-target read/write IO surface strictly from Hamilton DAG tags.

    Boundary-aware: stops traversal when encountering upstream target materialize nodes,
    so upstream targets' internal reads/writes are not attributed to the current target.
    """

    nodes: Mapping[str, Node] = runtime.dr.graph.nodes
    node_to_target = _target_node_index(nodes)

    # Build unique mapping of target -> root materialize node
    target_to_node: dict[str, str] = {}
    for node_name, target_name in node_to_target.items():
        if target_name in target_to_node:
            msg = f"Duplicate materialize nodes for target '{target_name}'"
            raise RuntimeError(msg)
        target_to_node[target_name] = node_name

    targets = set(target_to_node)
    if include_targets is not None:
        targets = targets.intersection(set(include_targets))

    def _as_sorted_list(values: set[str]) -> list[str]:
        return sorted(v for v in values if v)

    surfaces: dict[str, dict[str, object]] = {}

    for target_name in sorted(targets):
        root = nodes[target_to_node[target_name]]

        reads_ibis: set[str] = set()
        reads_df: set[str] = set()
        reads_ds: set[str] = set()

        write_tables: set[str] = set()
        write_artifacts: set[str] = set()
        write_ops: list[dict[str, object]] = []
        by_sink: dict[str, dict[str, list[str]]] = {}

        visited: set[str] = set()
        stack: list[Node] = list(root.dependencies)

        while stack:
            node = stack.pop()
            if node.name in visited:
                continue
            visited.add(node.name)

            upstream_target = node_to_target.get(node.name)
            if upstream_target is not None and upstream_target != target_name:
                # Stop at upstream target boundary.
                continue

            tags = node.tags
            if not isinstance(tags, dict):
                stack.extend(node.dependencies)
                continue

            node_type = tags.get(TAG_NODE_TYPE)

            # --- Reads (loader/dataset nodes) ---
            if node_type in {NODE_TYPE_LOADER_QUERY, NODE_TYPE_LOADER_DATAFRAME, NODE_TYPE_DATASET}:
                table_key = tags.get(TAG_TABLE_KEY)
                if isinstance(table_key, str) and table_key:
                    if node_type == NODE_TYPE_LOADER_QUERY:
                        reads_ibis.add(table_key)
                    elif node_type == NODE_TYPE_LOADER_DATAFRAME:
                        reads_df.add(table_key)
                    elif node_type == NODE_TYPE_DATASET:
                        reads_ds.add(table_key)

            # --- Writes (data_saver nodes) ---
            if tags.get("hamilton.data_saver") is True and tags.get(TAG_TARGET) == target_name:
                sink = tags.get("hamilton.data_saver.sink")
                saver_class = tags.get("hamilton.data_saver.classname")

                table_key = tags.get(TAG_TABLE_KEY)
                artifact = tags.get(TAG_ARTIFACT)

                if isinstance(table_key, str) and table_key:
                    write_tables.add(table_key)
                    write_ops.append(
                        {
                            "kind": "table",
                            "key": table_key,
                            "node": node.name,
                            "sink": sink,
                            "saver_class": saver_class,
                        }
                    )
                    if isinstance(sink, str) and sink:
                        bucket = by_sink.setdefault(sink, {"tables": [], "artifacts": []})
                        bucket["tables"].append(table_key)

                if isinstance(artifact, str) and artifact:
                    write_artifacts.add(artifact)
                    write_ops.append(
                        {
                            "kind": "artifact",
                            "key": artifact,
                            "node": node.name,
                            "sink": sink,
                            "saver_class": saver_class,
                        }
                    )
                    if isinstance(sink, str) and sink:
                        bucket = by_sink.setdefault(sink, {"tables": [], "artifacts": []})
                        bucket["artifacts"].append(artifact)

            stack.extend(node.dependencies)

        reads_all = reads_ibis | reads_df | reads_ds

        # De-duplicate within by_sink buckets
        for sink, bucket in by_sink.items():
            bucket["tables"] = sorted(set(bucket.get("tables", [])))
            bucket["artifacts"] = sorted(set(bucket.get("artifacts", [])))

        # Stable ordering for ops output
        write_ops = sorted(write_ops, key=lambda op: (str(op.get("kind")), str(op.get("key"))))

        surfaces[target_name] = {
            "reads": {
                "tables": _as_sorted_list(reads_all),
                "ibis": _as_sorted_list(reads_ibis),
                "dataframe": _as_sorted_list(reads_df),
                "dataset_ref": _as_sorted_list(reads_ds),
            },
            "writes": {
                "tables": _as_sorted_list(write_tables),
                "artifacts": _as_sorted_list(write_artifacts),
                "ops": write_ops,
                "by_sink": by_sink,
            },
        }

    return surfaces
```

#### 2.3 Export it

At the bottom `__all__`, add:

```py
"derive_target_io_surface",
```

---

## 3) Wire it into `codeintel build explain` (one-command output)

### 3.1 Add a flag to the command

File: `src/codeintel/cli/commands/build.py`

In `BuildExplainCommand`, add:

```py
io_surface: Annotated[
    bool,
    Parameter(
        name=["--io-surface"],
        help=(
            "Include a per-target IO surface (reads/writes) derived strictly from Hamilton DAG tags. "
            "Useful for quickly seeing what a target reads and materializes."
        ),
        negative=(),
    ),
] = False
```

### 3.2 Extend the result type to optionally include it

File: `src/codeintel/cli/core/result_types.py`

In `BuildExplainResult`, add a new optional field:

```py
io_surface: dict[str, object] | None = None
```

And in `to_dict()`, include it only when present:

```py
**({"io_surface": self.io_surface} if self.io_surface is not None else {}),
```

### 3.3 Compute it in the handler only when requested

File: `src/codeintel/cli/handlers/build.py`

Add the import:

```py
from codeintel.build.hamilton.introspect import derive_target_io_surface
```

Then in `build_explain_handler`, right after:

```py
explanation = entry.explain_staleness()
```

add:

```py
io_surface: dict[str, object] | None = None
if ctx.params.get_bool("io_surface"):
    # Derive operational read/write surface strictly from Hamilton DAG tags.
    # Note: we intentionally build a fresh runtime here to avoid coupling
    # explain-plan computation to DAG introspection.
    h_runtime = build_driver()
    io_surface = derive_target_io_surface(
        h_runtime,
        include_targets=(params.target,),
    ).get(params.target)
```

And pass it into `BuildExplainResult(...)`:

```py
result = BuildExplainResult(
    ...
    summary=explanation.summary(),
    io_surface=io_surface,
)
```

### 3.4 Example usage

```bash
codeintel build explain function_metrics --io-surface
```

If you want structured output:

```bash
codeintel build explain function_metrics --io-surface --output-format json
```

You’ll get something like:

* `reads.tables`: all table keys read (union)
* `reads.ibis`: those read via `q__*`
* `reads.dataframe`: those read via `df__*`
* `writes.tables`: table keys written by saver nodes tagged with this target
* `writes.by_sink`: grouped by saver sink (e.g., DuckDB vs file artifacts)
* `writes.ops`: per-write operational details (kind/key/node/sink/class)

---

## 4) Add the “golden” saver-tag regression test

This ensures the whole surface-map mechanism can’t silently regress.

### New file: `tests/build/hamilton/test_prXX_saver_nodes_have_canonical_tags.py`

```py
"""PR-XX: Saver nodes have canonical tags for IO surface introspection.

Required tags:
- target
- table_key OR artifact
"""

from __future__ import annotations

import pytest

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.core.hamilton import tags as ht


def test_prXX_saver_nodes_have_target_and_output_tags() -> None:
    runtime = build_driver()

    missing: list[str] = []
    for node_name, node in runtime.dr.graph.nodes.items():
        tags = node.tags
        if not isinstance(tags, dict):
            continue
        if tags.get("hamilton.data_saver") is not True:
            continue

        target = tags.get(ht.TAG_TARGET)
        table_key = tags.get(ht.TAG_TABLE_KEY)
        artifact = tags.get(ht.TAG_ARTIFACT)

        if not isinstance(target, str) or not target:
            missing.append(f"{node_name}: missing target tag")
            continue

        has_table = isinstance(table_key, str) and bool(table_key)
        has_artifact = isinstance(artifact, str) and bool(artifact)
        if not (has_table or has_artifact):
            missing.append(f"{node_name}: missing table_key/artifact tag")

    if missing:
        pytest.fail("Saver nodes missing canonical tags:\n" + "\n".join(sorted(missing)))
```

This is intentionally strict and simple: if a node is a saver metadata node, it must be attributable to a target and an output identity.

---

## Why I wired this into `build explain` (vs `build lineage`)

Your current `build lineage` handler is **warehouse/history oriented** (it walks persisted lineage tables). It’s great for *historical asset lineage*, but it’s not the cleanest entry point for **DAG-introspection at build time**.

`build explain <target>` is already a **target-scoped** diagnostic command, so adding `--io-surface` keeps the mental model tight:

* “Why will this run?” + “What will it touch?”

If you later want IO surface in lineage too, the clean design is usually:

* keep “historical lineage” and “DAG surface” as separate concepts, and
* optionally provide a join/bridge command that shows both.

---

