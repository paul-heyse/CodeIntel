
Below is a **comprehensive, implementation-ready plan (with code snippets)** to deliver:

> **Eliminate (or demote) the target metadata registry** by deriving `OutputTarget` from **DAG tags + docstrings** (with a **small override layer** for the truly non-derivable bits).

This plan is designed to be **safe and incremental**: you can land the new DAG-native path first, then migrate targets module-by-module, then delete the registry once the diff is at zero.

---

# North-star behavior

After this change:

* The **existence** of a build target is defined by the presence of exactly one `t__<target>` node tagged `node_type="materialize"` + `target="<name>"`.
* Most metadata is derived from the **`t__` node**:

  * `OutputTarget.name` ← `target` tag
  * `OutputTarget.module` ← `domain` tag (ingestion/graphs/analytics/export)
  * `OutputTarget.description` ← `t__` function docstring (first paragraph/line)
  * `OutputTarget.resources/execution/parameters` ← structured tags on `t__` node (JSON)
* `OutputTarget.contract` is derived from **materialization nodes** (your `m__*` saver nodes) via tags:

  * tables ← saver nodes tagged with `table_key` and `output_role="contract"`
  * artifacts ← saver nodes tagged with `artifact` (+ `artifact_path_template`) and `output_role="contract"`
* Registry becomes optional:

  * either removed entirely
  * or kept as a minimal override map (schema overrides etc.), *not* a “must match DAG” gate.

---

# PR-by-PR implementation plan

## PR-00: Prerequisite alignment with DAG-derived runtime outputs

Before changing how `OutputTarget` is compiled, ensure runtime records and artifact paths are already
derived from DAG saver tags (see `centralization_big_move_2.md` PR-05). This avoids a split-brain
state where targets are DAG-derived but artifact paths still rely on registry contracts.

Cross-link: align this prerequisite with the integrated migration runway in
`centralization_big_move_2.md` section “Integrated migration timeline (DAG-first runway)”.

## PR-01: Add canonical tag keys for target spec metadata + a `@codeintel_target(...)` decorator

### 1.1 Add spec tag keys in `codeintel/core/hamilton/tags.py`

```py
# src/codeintel/core/hamilton/tags.py

TAG_TARGET_RESOURCES = "target_resources"
TAG_TARGET_EXECUTION = "target_execution"
TAG_TARGET_PARAMETERS = "target_parameters"
TAG_TARGET_ESTIMATED_DURATION_MS = "target_estimated_duration_ms"
TAG_TARGET_SPEC_VERSION = "target_spec_version"

__all__ += [
    "TAG_TARGET_RESOURCES",
    "TAG_TARGET_EXECUTION",
    "TAG_TARGET_PARAMETERS",
    "TAG_TARGET_ESTIMATED_DURATION_MS",
    "TAG_TARGET_SPEC_VERSION",
]
```

### 1.2 Extend `build/hamilton/tagging.py` typing to allow these keys (optional but recommended)

Add them to `TagKey` and `_HamiltonTagKwargs` so your tagging helpers remain type-safe.

```py
# src/codeintel/build/hamilton/tagging.py
TagKey = Literal[
    ...
    "target_resources",
    "target_execution",
    "target_parameters",
    "target_estimated_duration_ms",
    "target_spec_version",
]
```

…and add cases in `_set_tag_secondary(...)` or `_set_tag_primary(...)` (I’d treat them as “secondary”).

### 1.3 Implement a single “target spec” decorator

Create:

`src/codeintel/build/hamilton/native/target_decorators.py`

```py
from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Mapping

from codeintel.build.hamilton.tagging import tag_materialize, TagKey, TagValue
from codeintel.build.parameters import EMPTY_PARAMETERS, TargetParameters
from codeintel.build.resources import DEFAULT_EXECUTION, DEFAULT_RESOURCES, TargetExecution, TargetResources
from codeintel.core.hamilton import tags as ht

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")

Decorator = Callable[[Callable[P, R]], Callable[P, R]]

def _json_dumps(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))

def codeintel_target(
    *,
    domain: str,
    target: str,
    resources: TargetResources = DEFAULT_RESOURCES,
    execution: TargetExecution = DEFAULT_EXECUTION,
    parameters: TargetParameters = EMPTY_PARAMETERS,
    estimated_duration_ms: int | None = None,
    spec_version: str = "1",
    extra_tags: Mapping[TagKey, TagValue] | None = None,
) -> Decorator[P, R]:
    """Canonical target anchor decorator.

    Attaches:
      - node_type=materialize, domain, target
      - JSON-encoded resources/execution/parameters in stable tag keys
    """
    spec_tags: dict[TagKey, TagValue] = {
        ht.TAG_TARGET_RESOURCES: _json_dumps(asdict(resources)),
        ht.TAG_TARGET_EXECUTION: _json_dumps(asdict(execution)),
        ht.TAG_TARGET_PARAMETERS: _json_dumps(parameters.as_dict()),
        ht.TAG_TARGET_SPEC_VERSION: spec_version,
    }
    if estimated_duration_ms is not None:
        spec_tags[ht.TAG_TARGET_ESTIMATED_DURATION_MS] = str(estimated_duration_ms)

    merged = dict(extra_tags or {})
    merged.update(spec_tags)

    return tag_materialize(domain=domain, target=target, extra_tags=merged)
```

**Why this matters:** it becomes the single “spec lives next to the target node” surface, and you stop needing a separate registration call.

**Hardness:** make these tags mandatory in DAG compilation (missing tags should raise), so every target
declares its own execution metadata explicitly.

---

## PR-02: Verification gates for saver tags (contract inference prerequisite)

`SaveToObjectMetadataDecorator` already tags saver nodes with `target` + `table_key`/`artifact`. This PR
converts that expectation into **hard verification**, so DAG-derived contracts cannot silently drift.

### 2.1 Add a DAG tag validation test

Add a test that asserts every `hamilton.data_saver` node has:

* `TAG_TARGET`
* exactly one of `TAG_TABLE_KEY` or `TAG_ARTIFACT`
* `output_role` in `{contract, internal}`

### 2.2 Add a DAG artifact-path validation test

Add a test that asserts **contract** artifact saver nodes include `TAG_ARTIFACT_PATH_TEMPLATE`.

This keeps internal saver nodes excluded, and enforces that artifact paths are fully DAG-derived.

---

## PR-03: Implement a DAG-native target spec compiler that produces `OutputTarget` from `Driver.graph.nodes`

Create a new module:

`src/codeintel/build/hamilton/target_spec_compiler.py`

### 3.1 Core types: overrides are allowed but minimal

```py
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Mapping
from types import MappingProxyType

from codeintel.build.contracts import ArtifactSpec, OutputContract, placeholder_table_schema
from codeintel.build.hamilton.introspect import derive_target_outputs_from_savers
from codeintel.build.hamilton.validate import validate_graph
from codeintel.build.parameters import TargetParameters, EMPTY_PARAMETERS
from codeintel.build.resources import TargetExecution, TargetResources, DEFAULT_EXECUTION, DEFAULT_RESOURCES
from codeintel.build.targets import OutputTarget, TargetGraph
from codeintel.core.hamilton import tags as ht
from codeintel.storage.helpers.table_key import validate_table_key

if TYPE_CHECKING:
    from hamilton.driver import Driver
    from hamilton.node import Node
    from codeintel.core.schemas.primitives import TableSchema
    from codeintel.build.hamilton.runtime import HamiltonRuntime

@dataclass(frozen=True, slots=True)
class TargetSpecOverride:
    """Small override layer for truly non-derivable settings."""
    override_tables: tuple["TableSchema", ...] = ()
    # Optional metadata overrides (rare)
    resources: TargetResources | None = None
    execution: TargetExecution | None = None
    parameters: TargetParameters | None = None
    description: str | None = None
```

### 3.2 Robust docstring extraction (don’t guess Hamilton internals)

```py
def _node_docstring(node: "Node") -> str:
    # Be defensive: sf-hamilton Node exposes docstrings via a stable field,
    # but we don't want to hard-code one name.
    for attr in ("documentation", "doc_string", "doc", "description"):
        val = getattr(node, attr, None)
        if isinstance(val, str) and val.strip():
            return val.strip()

    fn = getattr(node, "callable", None) or getattr(node, "func", None)
    if callable(fn) and getattr(fn, "__doc__", None):
        doc = fn.__doc__ or ""
        if doc.strip():
            return doc.strip()

    return ""

def _summary(doc: str) -> str:
    for line in doc.splitlines():
        if line.strip():
            return line.strip()
    return ""
```

### 3.3 Parse JSON-encoded spec tags from `t__` nodes

```py
import json

def _parse_json_tag(tags: Mapping[str, object], key: str) -> dict[str, object] | None:
    raw = tags.get(key)
    if raw is None:
        return None
    if isinstance(raw, dict):
        return dict(raw)
    if isinstance(raw, str) and raw.strip():
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    return None

def _resources_from_tags(tags: Mapping[str, object]) -> TargetResources:
    obj = _parse_json_tag(tags, ht.TAG_TARGET_RESOURCES)
    if obj is None:
        return DEFAULT_RESOURCES
    tools = obj.get("tools")
    if isinstance(tools, list):
        obj = {**obj, "tools": tuple(str(item) for item in tools)}
    return TargetResources(**obj)

def _execution_from_tags(tags: Mapping[str, object]) -> TargetExecution:
    obj = _parse_json_tag(tags, ht.TAG_TARGET_EXECUTION)
    return DEFAULT_EXECUTION if obj is None else TargetExecution(**obj)

def _parameters_from_tags(tags: Mapping[str, object]) -> TargetParameters:
    obj = _parse_json_tag(tags, ht.TAG_TARGET_PARAMETERS)
    return EMPTY_PARAMETERS if obj is None else TargetParameters(obj)
```

**Hardness:** treat missing or malformed `TAG_TARGET_RESOURCES`, `TAG_TARGET_EXECUTION`, or
`TAG_TARGET_PARAMETERS` as compilation errors when `strict=True`. This prevents “implicit defaults”
from masking missing metadata.

### 3.4 Derive produced outputs from saver nodes (DAG materializations)

This is the key: contracts no longer define produced outputs; the DAG does.

**Canonical source:** use the existing `derive_target_outputs_from_savers` helper and extend it
to include `artifact_path_template` for contract saver nodes. All consumers (compiler, runtime,
validation) must read from this single DAG-derived view.

**Policy:** internal saver nodes (`output_role="internal"`) are excluded from contract derivation.
Missing tags on contract saver nodes are fatal.

### 3.5 Compile `OutputTarget` objects from target anchors (`t__*` nodes)

```py
def _is_target_anchor(node: "Node") -> bool:
    tags = getattr(node, "tags", None)
    if not isinstance(tags, dict):
        return False
    return (
        tags.get(ht.TAG_NODE_TYPE) == ht.NODE_TYPE_MATERIALIZE
        and isinstance(tags.get(ht.TAG_TARGET), str)
        and isinstance(tags.get(ht.TAG_DOMAIN), str)
    )

def compile_output_targets_from_driver(
    driver: "Driver",
    *,
    overrides_by_target: Mapping[str, TargetSpecOverride] | None = None,
    strict: bool = True,
) -> tuple[OutputTarget, ...]:
    nodes = driver.graph.nodes
    overrides_by_target = overrides_by_target or MappingProxyType({})
    runtime = HamiltonRuntime(dr=driver, graph=TargetGraph())
    validation = validate_graph(runtime.dr)
    if strict and validation.errors:
        msg = "Graph validation errors:\n" + "\n".join(
            f"- {issue.message}" for issue in validation.errors
        )
        raise RuntimeError(msg)
    derived_outputs = derive_target_outputs_from_savers(runtime)

    anchors: dict[str, "Node"] = {}
    for node in nodes.values():
        if not _is_target_anchor(node):
            continue
        tags = node.tags
        target_name = tags[ht.TAG_TARGET]
        if target_name in anchors:
            raise RuntimeError(f"Duplicate t__ materialize node for target {target_name}")
        anchors[target_name] = node

    if strict and not anchors:
        raise RuntimeError("No build targets discovered in Hamilton graph (missing t__ nodes?)")

    results: list[OutputTarget] = []
    for target_name in sorted(anchors):
        node = anchors[target_name]
        tags = node.tags

        domain = tags.get(ht.TAG_DOMAIN)
        if not isinstance(domain, str) or not domain:
            raise RuntimeError(f"Target {target_name} missing domain tag")

        doc = _node_docstring(node)
        description = _summary(doc)

        override = overrides_by_target.get(target_name)
        if override and override.description:
            description = override.description

        if strict and not description:
            raise RuntimeError(f"Target {target_name} must have a docstring summary (or override)")

        if strict and domain not in {"ingestion", "graphs", "analytics", "export"}:
            raise RuntimeError(f"Target {target_name} has invalid domain tag: {domain!r}")

        spec_version = tags.get(ht.TAG_TARGET_SPEC_VERSION)
        if strict and spec_version != "1":
            raise RuntimeError(
                f"Target {target_name} missing/invalid spec version tag: {spec_version!r}"
            )

        # Resources/execution/parameters from tags, optionally overridden
        resources = _resources_from_tags(tags)
        execution = _execution_from_tags(tags)
        parameters = _parameters_from_tags(tags)

        if override and override.resources is not None:
            resources = override.resources
        if override and override.execution is not None:
            execution = override.execution
        if override and override.parameters is not None:
            parameters = override.parameters

        table_keys = derived_outputs.table_keys_by_target.get(target_name, ())
        override_tables = override.override_tables if override else ()
        tables = _resolve_table_schemas(table_keys, override_tables)  # reuse existing helper below

        artifacts = derived_outputs.artifacts_by_target.get(target_name, ())
        contract = OutputContract(tables=tables, artifacts=artifacts)

        results.append(
            OutputTarget(
                name=target_name,
                module=domain,  # validated by TargetModule Literal at runtime
                contract=contract,
                dependencies=(),  # patched later by target_graph_from_hamilton()
                resources=resources,
                execution=execution,
                parameters=parameters,
                description=description,
            )
        )

    return tuple(results)

def _resolve_table_schemas(
    table_keys: tuple[str, ...],
    override_tables: tuple["TableSchema", ...],
) -> tuple["TableSchema", ...]:
    overrides: dict[str, TableSchema] = {}
    for table_schema in override_tables:
        validate_table_key(table_schema.table_key)
        if table_schema.table_key in overrides:
            msg = f"Duplicate override table schema: {table_schema.table_key}"
            raise ValueError(msg)
        overrides[table_schema.table_key] = table_schema

    tables: list[TableSchema] = []
    seen: set[str] = set()
    for key in table_keys:
        validate_table_key(key)
        if key in seen:
            msg = f"Duplicate table_key in target spec: {key}"
            raise ValueError(msg)
        seen.add(key)
        tables.append(overrides.get(key) or placeholder_table_schema(key))

    extra_overrides = sorted(set(overrides) - seen)
    if extra_overrides:
        msg = f"Override tables not declared in table_keys: {extra_overrides}"
        raise ValueError(msg)

    return tuple(tables)
```

This compiler is now your replacement for the registry.

**Hardness (optional):** when `strict=True`, assert the materialize node name matches
`target_node(<target_name>)` to prevent accidental tag/name mismatches.

---

## PR-04: Wire the compiler into `driver_factory.py` and `support_factory.py` (remove `resolve_registered_targets` dependency)

### 4.1 Update `driver_factory._build_base_graph(...)`

**Before:**

* discover target names from nodes
* `resolve_registered_targets(target_names)`
* register those OutputTargets into `TargetGraph`

**After:**

* build native driver
* compile OutputTargets directly from the driver nodes
* build base graph from those targets

```py
# src/codeintel/build/hamilton/driver_factory.py
from codeintel.build.hamilton.target_spec_compiler import (
    compile_output_targets_from_driver,
)
from codeintel.build.hamilton.native.target_overrides import TARGET_SPEC_OVERRIDES

def _build_base_graph(*, config: dict[str, Any] | None) -> tuple[TargetGraph, h_driver.Driver]:
    native_mods = load_native_modules()
    driver = h_driver.Builder().with_config(config or {}).with_modules(*native_mods).build()

    targets = compile_output_targets_from_driver(
        driver,
        overrides_by_target=TARGET_SPEC_OVERRIDES,
        strict=True,
    )

    base_graph = TargetGraph()
    for t in targets:
        base_graph.register(t)

    return base_graph, driver
```

Delete the import and use of `resolve_registered_targets` entirely.

### 4.2 Update `support_factory._build_contract_graph()` similarly

```py
# src/codeintel/build/hamilton/nodes/support_factory.py
from codeintel.build.hamilton.target_spec_compiler import compile_output_targets_from_driver
from codeintel.build.hamilton.native.target_overrides import TARGET_SPEC_OVERRIDES

def _build_contract_graph() -> TargetGraph:
    native_mods = load_native_modules()
    dr = h_driver.Builder().with_modules(*native_mods).build()

    targets = compile_output_targets_from_driver(
        dr,
        overrides_by_target=TARGET_SPEC_OVERRIDES,
        strict=True,
    )

    base_graph = TargetGraph()
    for t in targets:
        base_graph.register(t)

    runtime = HamiltonRuntime(dr=dr, graph=base_graph)
    derived = derive_target_dependencies(runtime)
    return target_graph_from_hamilton(runtime, base_graph=base_graph, derived_deps=derived, strict=False)
```

---

## PR-05: Create the “small override layer” (schemas, rare execution overrides)

Add:

`src/codeintel/build/hamilton/native/target_overrides.py`

Use this to keep the override layer minimal and explicit.

```py
from __future__ import annotations

from types import MappingProxyType
from codeintel.build.hamilton.target_spec_compiler import TargetSpecOverride
from codeintel.build.hamilton.native import target_override_tables as ot
from codeintel.build.resources import TOOL_EXECUTION, TargetResources
from codeintel.build.parameters import TargetParameters

TARGET_SPEC_OVERRIDES = MappingProxyType({
    # Schema overrides (non-inferable)
    "scip": TargetSpecOverride(override_tables=ot.SCIP_OVERRIDE_TABLES),
    "modules": TargetSpecOverride(override_tables=ot.MODULES_OVERRIDE_TABLES),
    "ast": TargetSpecOverride(override_tables=ot.AST_OVERRIDE_TABLES),
    "cst": TargetSpecOverride(override_tables=ot.CST_OVERRIDE_TABLES),
    "docstrings": TargetSpecOverride(override_tables=ot.DOCSTRINGS_OVERRIDE_TABLES),
    "goids": TargetSpecOverride(override_tables=ot.GOIDS_OVERRIDE_TABLES),

    # You can add rare policy overrides here if truly needed:
    # "scip": TargetSpecOverride(
    #     override_tables=ot.SCIP_OVERRIDE_TABLES,
    #     execution=TOOL_EXECUTION,
    #     resources=TargetResources(tracker=True, modules=True, tools=("scip-python","scip")),
    # ),
})
```

**Important:** in the end state, you should prefer `@codeintel_target(...)` for resources/execution/parameters, and keep this file mostly about **schema overrides**.

---

## PR-06: Migrate native target modules from `register_output_targets(...)` to `@codeintel_target(...)` + docstrings

This is the big, mechanical migration.

### 6.1 Pattern to apply to every module

* Delete the `register_output_targets(make_output_target(...))` block entirely.
* Replace `@tag_materialize(domain=..., target=...)` with `@codeintel_target(...)` and move resources/execution/parameters into that decorator.
* Ensure the `t__<target>` function has a docstring summary line.
* Do not override `spec_version` unless you are intentionally bumping the target-spec schema (default `"1"`).

### 6.2 Example: migrate `scip.py`

**Before** (registry block at top + `@tag_materialize`):

```py
register_output_targets(
    make_output_target(
        name=SCIP_TARGET_NAME,
        module="ingestion",
        description="SCIP index ingestion and GOID generation.",
        options=TargetSpecOptions(...),
    )
)

@tag_materialize(domain="ingestion", target=SCIP_TARGET_NAME)
def t__scip(...):
    ...
```

**After**:

```py
from codeintel.build.hamilton.native.target_decorators import codeintel_target
from codeintel.build.resources import TOOL_EXECUTION, TargetResources

@codeintel_target(
    domain="ingestion",
    target=SCIP_TARGET_NAME,
    resources=TargetResources(
        tracker=True,
        modules=True,
        tools=("scip-python", "scip"),
    ),
    execution=TOOL_EXECUTION,
    # spec_version defaults to "1" and should not be overridden unless you bump the tag schema.
)
def t__scip(...):
    """SCIP index ingestion and GOID generation."""
    ...
```

And you no longer need:

* `SCIP_TABLE_KEYS`
* `SCIP_ARTIFACT_SPECS`
* `TargetSpecOptions`, `make_output_target`, `register_output_targets`

Because:

* tables are derived from saver nodes tagged with `table_key`
* artifacts are derived from saver nodes tagged `artifact` + `artifact_path_template` (contract-only, from PR-02)

### 6.3 Migration notes

* Do this per domain: ingestion → graphs → analytics → export.
* Keep the override tables in `TARGET_SPEC_OVERRIDES` until you later unify schema inference.

---

## PR-07: “No-registry” cleanup (delete or demote the registry module)

Once PR-06 lands and all targets compile from DAG tags:

### Option A (recommended): delete registry entirely

* Remove `src/codeintel/build/hamilton/native/target_spec_helpers.py`
* Remove unused imports in all native modules
* Remove any “missing/extra registry entries” logic

### Option B: keep only a tiny helper module

If you still want some utilities (e.g., placeholder schema helpers), keep a renamed module:

* `target_spec_helpers.py` → `target_spec_utils.py`
* remove `_TARGET_REGISTRY`, `register_*`, `resolve_registered_targets`

---

## PR-08: Tests (prevent drift forever)

Add tests that enforce the new invariants.

### 8.1 Target discovery + compilation must be complete

```py
def test_all_targets_compile_from_dag(build_driver_runtime):
    runtime = build_driver_runtime()  # or driver_factory.build_driver()
    graph = runtime.graph
    assert len(graph.all_targets) > 0
```

### 8.2 Every target anchor must have a docstring summary

```py
def test_target_anchors_have_docstrings(runtime):
    for t in runtime.graph.all_targets:
        # description is compiled from docstring/override
        assert t.description.strip(), f"Target {t.name} missing description"
```

### 8.3 Graph validation must be clean

Prefer centralized validation over bespoke tests. Validate:

* saver tags (`target`, `output_role`, `table_key`/`artifact`)
* artifact saver tags include `artifact_path_template`
* materialize nodes include `target_spec_version`

```py
def test_graph_validation_is_clean(runtime):
    result = validate_graph(runtime.dr)
    assert not result.errors
```

### 8.4 Every target anchor must carry a spec version tag

```py
def test_target_anchors_have_spec_version(runtime):
    for node in runtime.dr.graph.nodes.values():
        tags = getattr(node, "tags", None)
        if not isinstance(tags, dict):
            continue
        if tags.get("node_type") != "materialize":
            continue
        assert tags.get("target_spec_version") == "1"
```

### 8.5 Every saver node must be attributable + have output tags

This prevents regressions where a new materializer forgets `target_name/table_key/artifact`.

```py
def test_materialization_nodes_have_target_and_output_tags(runtime):
    for node in runtime.dr.graph.nodes.values():
        tags = getattr(node, "tags", None)
        if not isinstance(tags, dict):
            continue
        if tags.get("hamilton.data_saver") not in (True, "True", 1, "1"):
            continue
        assert "target" in tags
        assert ("table_key" in tags) or ("artifact" in tags)
        if "artifact" in tags:
            assert "artifact_path_template" in tags
```

---

# “Strict mode” knobs (recommended)

During migration, run `compile_output_targets_from_driver(strict=False)` until PR-06.5 flips strict
mode. After that, keep `strict=True` as the default so you fail fast when:

* a target anchor is missing `domain/target` tags
* a target anchor has no docstring summary
* a target anchor is missing `target_spec_version`
* an artifact saver lacks `artifact_path_template`
* duplicate `t__` nodes exist for one target

This gives you the same correctness posture as the registry, but **with the DAG as the authority**.

Cross-link: the strictness flip here corresponds to Phase 3 in the integrated runway in
`centralization_big_move_2.md` (“Strictness flip (hardness on)”).

---

# Summary of concrete file changes

**New files**

* `src/codeintel/build/hamilton/native/target_decorators.py`  (adds `@codeintel_target`)
* `src/codeintel/build/hamilton/target_spec_compiler.py`      (DAG → OutputTarget compiler)
* `src/codeintel/build/hamilton/native/target_overrides.py`   (small override layer)

**Modified files**

* `src/codeintel/core/hamilton/tags.py`                       (new spec tag keys)
* `src/codeintel/build/hamilton/tagging.py`                   (optional typing support)
* `src/codeintel/build/hamilton/driver_factory.py`            (use compiler, remove registry dependency)
* `src/codeintel/build/hamilton/nodes/support_factory.py`     (same)
* `tests/build/hamilton/test_target_spec_tags.py`             (new tag verification tests)
* All native target modules: remove `register_output_targets(...)`, add `@codeintel_target`, add `t__` docstrings

**Deleted (end state)**

* `src/codeintel/build/hamilton/native/target_spec_helpers.py` (registry)

---

If you want, I can also include a **mechanical migration checklist** per native module (file-by-file) showing exactly what to delete/add (imports, constants, decorator arguments) so this becomes a straightforward Codex/Cursor batch refactor.


# file by file checklist #

Below is a **mechanical, file-by-file migration checklist** to remove the out-of-band target registry and make each target’s **`t__<target>` node** the canonical “spec anchor” via **`@codeintel_target(...)` + docstring**.

I’m assuming you’ve already landed (or are landing) the prerequisites from the plan:

* `codeintel.build.hamilton.native.target_decorators.codeintel_target`
* `codeintel.build.hamilton.target_spec_compiler.compile_output_targets_from_driver`
* `codeintel.build.hamilton.native.target_overrides.TARGET_SPEC_OVERRIDES` (schema override layer)
* `SaveToObjectMetadataDecorator` emits `TAG_TARGET`, `TAG_TABLE_KEY`, `TAG_ARTIFACT`, `TAG_ARTIFACT_PATH_TEMPLATE` on saver nodes
* `TAG_TARGET_SPEC_VERSION` is enforced on `t__` materialize nodes (default "1")

If any of those aren’t merged yet, keep this checklist as “what to do once they are.”

---

## Global batch refactor recipe (Codex/Cursor-friendly)

Apply these repo-wide edits first (they make the per-file work mostly mechanical):

### G1) Remove registry resolution plumbing (core wiring)

* `src/codeintel/build/hamilton/driver_factory.py`

  * Remove import: `resolve_registered_targets`
  * Replace the “resolve registered targets” path with “compile targets from driver graph”
* `src/codeintel/build/hamilton/nodes/support_factory.py`

  * Same as above

### G2) Create schema override map (central place)

Create / update:

* `src/codeintel/build/hamilton/native/target_overrides.py`

It should contain entries for every target that currently passes `override_tables=...` in `TargetSpecOptions`. After the migration, **native target modules should not import override tables** just to feed registration.

### G3) Repo-wide delete/replace patterns

You can do most of the “native module cleanup” with these systematic steps:

1. Delete all blocks starting at `register_output_targets(` up to the matching `)`.
2. Remove imports from `codeintel.build.hamilton.native.target_spec_helpers`:

   * `TargetSpecOptions`, `make_output_target`, `register_output_targets`
3. Replace `@tag_materialize(domain="X", target=Y_TARGET_NAME)` with:

   * `@codeintel_target(domain="X", target=Y_TARGET_NAME, ...)`
4. Ensure each `t__...` target anchor has a **docstring first line** that is the old registry description.

---

# File-by-file checklist (native modules)

I’m listing each module, the targets it defines, and the exact mechanical edits: **imports, constants, decorator swaps, and override-map updates**.

---

## Ingestion

### 1) `src/codeintel/build/hamilton/native/ingestion/scip.py`

**Targets**

* `SCIP_TARGET_NAME = "scip"` → anchor: `t__scip`

**Edits**

1. **Imports**

   * Remove:

     * `from codeintel.build.contracts import ArtifactSpec` (only used for registry constants)
     * `from codeintel.build.hamilton.native.target_override_tables import SCIP_OVERRIDE_TABLES` (only used for registry)
     * `from codeintel.build.hamilton.native.target_spec_helpers import TargetSpecOptions, make_output_target, register_output_targets`
   * Keep:

     * `TargetResources`, `TOOL_EXECUTION` (needed for `@codeintel_target`)
   * Add:

     * `from codeintel.build.hamilton.native.target_decorators import codeintel_target`

2. **Constants**

   * Delete (registry-only):

     * `SCIP_ARTIFACT_SPECS = (...)`
   * Keep (still used elsewhere in module):

     * `SCIP_TABLE_KEYS` (used for counts/loops)
     * artifact name constants (`SCIP_ARTIFACT_INDEX`, `SCIP_ARTIFACT_JSON`)
     * table key constants

3. **Delete registry block**

   * Delete the entire `register_output_targets(make_output_target(...))` block.

4. **Decorator swap**

   * Replace:

     ```py
     @tag_materialize(domain="ingestion", target=SCIP_TARGET_NAME)
     def t__scip(...):
     ```
   * With:

     ```py
     @codeintel_target(
         domain="ingestion",
         target=SCIP_TARGET_NAME,
         resources=TargetResources(
             tracker=True,
             modules=True,
             tools=("scip-python", "scip"),
         ),
         execution=TOOL_EXECUTION,
     )
     def t__scip(...):
         """SCIP index ingestion and GOID generation."""
         ...
     ```
   * Keep the rest of the docstring below the first line if you want; just ensure the **first line** matches.

5. **Override map**

   * In `native/target_overrides.py`, add:

     ```py
     "scip": TargetSpecOverride(override_tables=ot.SCIP_OVERRIDE_TABLES),
     ```

---

### 2) `src/codeintel/build/hamilton/native/ingestion/ingest_targets.py`

**Targets & anchors**

* `modules` → `t__modules`
* `config_ingest` → `t__config_ingest`
* `coverage_ingest` → `t__coverage_ingest`
* `tests_ingest` → `t__tests_ingest`
* `typing` → `t__typing`

**Edits**

1. **Imports**

   * Remove:

     * `from codeintel.build.hamilton.native.target_override_tables import (...)` (these are registry-only after migration)

       * `MODULES_OVERRIDE_TABLES`, `CONFIG_INGEST_OVERRIDE_TABLES`, `COVERAGE_INGEST_OVERRIDE_TABLES`, `TESTS_INGEST_OVERRIDE_TABLES`, `TYPING_OVERRIDE_TABLES`
     * `from codeintel.build.hamilton.native.target_spec_helpers import TargetSpecOptions, make_output_target, register_output_targets`
   * Add:

     * `from codeintel.build.hamilton.native.target_decorators import codeintel_target`
   * Keep:

     * `TargetResources`, `TOOL_EXECUTION` (needed for typing target decorator)

2. **Delete registry block**

   * Delete the entire `register_output_targets(...)` block with the 5 `make_output_target(...)` entries.

3. **Decorator swaps (5 places)**

   * For each anchor, replace `@tag_materialize(...)` with `@codeintel_target(...)` and add docstring first line:

   **t__modules**

   ```py
   @codeintel_target(domain="ingestion", target=MODULES_TARGET_NAME)
   def t__modules(...):
       """Repository module and file index from scanning."""
   ```

   **t__config_ingest**

   ```py
   @codeintel_target(domain="ingestion", target=CONFIG_INGEST_TARGET_NAME)
   def t__config_ingest(...):
       """Configuration file parsing and reference tracking."""
   ```

   **t__coverage_ingest**

   ```py
   @codeintel_target(domain="ingestion", target=COVERAGE_INGEST_TARGET_NAME)
   def t__coverage_ingest(...):
       """Line-level test coverage ingestion."""
   ```

   **t__tests_ingest**

   ```py
   @codeintel_target(domain="ingestion", target=TESTS_INGEST_TARGET_NAME)
   def t__tests_ingest(...):
       """Test catalog ingestion from pytest."""
   ```

   **t__typing** (this one carries resources/execution)

   ```py
   @codeintel_target(
       domain="ingestion",
       target=TYPING_TARGET_NAME,
       resources=TargetResources(
           tracker=True,
           modules=True,
           tools=("pyright", "pyrefly", "ruff"),
       ),
       execution=TOOL_EXECUTION,
   )
   def t__typing(...):
       """Type annotation analysis and static diagnostics."""
   ```

4. **Override map**
   Add entries (in `native/target_overrides.py`):

   ```py
   "modules": TargetSpecOverride(override_tables=ot.MODULES_OVERRIDE_TABLES),
   "config_ingest": TargetSpecOverride(override_tables=ot.CONFIG_INGEST_OVERRIDE_TABLES),
   "coverage_ingest": TargetSpecOverride(override_tables=ot.COVERAGE_INGEST_OVERRIDE_TABLES),
   "tests_ingest": TargetSpecOverride(override_tables=ot.TESTS_INGEST_OVERRIDE_TABLES),
   "typing": TargetSpecOverride(override_tables=ot.TYPING_OVERRIDE_TABLES),
   ```

---

### 3) `src/codeintel/build/hamilton/native/ingestion/extraction_targets.py`

**Targets & anchors**

* `ast` → `t__ast`
* `cst` → `t__cst`
* `docstrings` → `t__docstrings`

**Edits**

1. **Imports**

   * Remove:

     * `from codeintel.build.hamilton.native.target_override_tables import AST_OVERRIDE_TABLES, CST_OVERRIDE_TABLES, DOCSTRINGS_OVERRIDE_TABLES` (registry-only after migration)
     * `from codeintel.build.hamilton.native.target_spec_helpers import TargetSpecOptions, make_output_target, register_output_targets`
   * Add:

     * `from codeintel.build.hamilton.native.target_decorators import codeintel_target`
   * Keep:

     * `TargetResources`, `CPU_INTENSIVE_EXECUTION` (needed for AST decorator)

2. **Delete registry block**

   * Delete the `register_output_targets(...)` block with 3 entries.

3. **Decorator swaps**
   **t__ast** (resources/execution)

   ```py
   @codeintel_target(
       domain="ingestion",
       target=AST_TARGET_NAME,
       resources=TargetResources(tracker=True, modules=True),
       execution=CPU_INTENSIVE_EXECUTION,
   )
   def t__ast(...):
       """Python AST extraction and metrics."""
   ```

   **t__cst**

   ```py
   @codeintel_target(domain="ingestion", target=CST_TARGET_NAME)
   def t__cst(...):
       """Concrete syntax tree extraction."""
   ```

   **t__docstrings**

   ```py
   @codeintel_target(domain="ingestion", target=DOCSTRINGS_TARGET_NAME)
   def t__docstrings(...):
       """Docstring extraction and parsing."""
   ```

4. **Override map**

   ```py
   "ast": TargetSpecOverride(override_tables=ot.AST_OVERRIDE_TABLES),
   "cst": TargetSpecOverride(override_tables=ot.CST_OVERRIDE_TABLES),
   "docstrings": TargetSpecOverride(override_tables=ot.DOCSTRINGS_OVERRIDE_TABLES),
   ```

---

## Graphs

### 4) `src/codeintel/build/hamilton/native/graphs/import_graph.py`

**Targets**

* `import_graph` → `t__import_graph`

**Edits**

1. Remove imports:

   * `IMPORT_GRAPH_OVERRIDE_TABLES` (registry-only)
   * `TargetSpecOptions/make_output_target/register_output_targets`
2. Add:

   * `codeintel_target` import
3. Delete registry block
4. Decorator swap:

   ```py
   @codeintel_target(domain="graphs", target=IMPORT_GRAPH_TARGET_NAME)
   def t__import_graph(...):
       """Module import graph construction."""
   ```
5. Override map:

   ```py
   "import_graph": TargetSpecOverride(override_tables=ot.IMPORT_GRAPH_OVERRIDE_TABLES),
   ```

---

### 5) `src/codeintel/build/hamilton/native/graphs/call_graph.py`

**Targets**

* `call_graph` → `t__call_graph`

**Edits**

1. Remove imports:

   * `CALL_GRAPH_OVERRIDE_TABLES` (registry-only)
   * `TargetSpecOptions/make_output_target/register_output_targets`
2. Add `codeintel_target`
3. Delete registry block
4. Decorator swap:

   ```py
   @codeintel_target(domain="graphs", target=CALL_GRAPH_TARGET_NAME)
   def t__call_graph(...):
       """Function call graph construction."""
   ```
5. Override map:

   ```py
   "call_graph": TargetSpecOverride(override_tables=ot.CALL_GRAPH_OVERRIDE_TABLES),
   ```

---

### 6) `src/codeintel/build/hamilton/native/graphs/cfg_dfg.py`

**Targets & anchors**

* `cfg` → `t__cfg`
* `dfg` → `t__dfg`

**Edits**

1. Remove imports:

   * `CFG_OVERRIDE_TABLES`, `DFG_OVERRIDE_TABLES` (registry-only)
   * `TargetSpecOptions/make_output_target/register_output_targets`
2. Add `codeintel_target`
3. Delete registry block
4. Decorator swaps:

   ```py
   @codeintel_target(domain="graphs", target=CFG_TARGET_NAME)
   def t__cfg(...):
       """Control flow graph construction per function."""
   ```

   ```py
   @codeintel_target(domain="graphs", target=DFG_TARGET_NAME)
   def t__dfg(...):
       """Data flow graph construction per function."""
   ```
5. Override map:

   ```py
   "cfg": TargetSpecOverride(override_tables=ot.CFG_OVERRIDE_TABLES),
   "dfg": TargetSpecOverride(override_tables=ot.DFG_OVERRIDE_TABLES),
   ```

---

### 7) `src/codeintel/build/hamilton/native/graphs/graph_targets.py`

**Targets & anchors**

* `goids` → `t__goids`
* `symbol_uses` → `t__symbol_uses`
* `call_graph_views` → `t__call_graph_views`
* `graph_metrics` → `t__graph_metrics`
* `graph_validation` → `t__graph_validation`

**Edits**

1. Remove imports:

   * `GOIDS_OVERRIDE_TABLES`, `SYMBOL_USES_OVERRIDE_TABLES`, `CALL_GRAPH_VIEWS_OVERRIDE_TABLES`,
     `GRAPH_METRICS_OVERRIDE_TABLES`, `GRAPH_VALIDATION_OVERRIDE_TABLES` (registry-only)
   * `TargetSpecOptions/make_output_target/register_output_targets`

2. Add `codeintel_target`

3. Delete registry block

4. Decorator swaps + docstring first lines:

   ```py
   @codeintel_target(domain="graphs", target=GOIDS_TARGET_NAME)
   def t__goids(...):
       """GOID resolution and crosswalk construction."""
   ```

   ```py
   @codeintel_target(domain="graphs", target=SYMBOL_USES_TARGET_NAME)
   def t__symbol_uses(...):
       """Symbol definition-to-use edge extraction."""
   ```

   ```py
   @codeintel_target(domain="graphs", target=CALL_GRAPH_VIEWS_TARGET_NAME)
   def t__call_graph_views(...):
       """Derived views over call graph for analytics."""
   ```

   ```py
   @codeintel_target(domain="graphs", target=GRAPH_METRICS_TARGET_NAME)
   def t__graph_metrics(...):
       """Graph topology metrics for functions and modules."""
   ```

   ```py
   @codeintel_target(domain="graphs", target=GRAPH_VALIDATION_TARGET_NAME)
   def t__graph_validation(...):
       """Graph integrity validation checks."""
   ```

5. Override map:

   ```py
   "goids": TargetSpecOverride(override_tables=ot.GOIDS_OVERRIDE_TABLES),
   "symbol_uses": TargetSpecOverride(override_tables=ot.SYMBOL_USES_OVERRIDE_TABLES),
   "call_graph_views": TargetSpecOverride(override_tables=ot.CALL_GRAPH_VIEWS_OVERRIDE_TABLES),
   "graph_metrics": TargetSpecOverride(override_tables=ot.GRAPH_METRICS_OVERRIDE_TABLES),
   "graph_validation": TargetSpecOverride(override_tables=ot.GRAPH_VALIDATION_OVERRIDE_TABLES),
   ```

---

## Analytics

### 8) `src/codeintel/build/hamilton/native/analytics/function_metrics.py`

**Targets**

* `function_metrics` → `t__function_metrics`

**Edits**

1. Remove imports:

   * `FUNCTION_METRICS_OVERRIDE_TABLES` (registry-only)
   * `TargetSpecOptions/make_output_target/register_output_targets`
2. Add `codeintel_target`
3. Delete registry block
4. Decorator swap:

   ```py
   @codeintel_target(domain="analytics", target=FUNCTION_METRICS_TARGET_NAME)
   def t__function_metrics(...):
       """Function structural metrics and type annotations."""
   ```
5. Override map:

   ```py
   "function_metrics": TargetSpecOverride(override_tables=ot.FUNCTION_METRICS_OVERRIDE_TABLES),
   ```

---

### 9) `src/codeintel/build/hamilton/native/analytics/function_detail_targets.py`

**Targets & anchors**

* `function_contracts` → `t__function_contracts`
* `function_effects` → `t__function_effects`

**Edits**

1. Remove imports:

   * `FUNCTION_CONTRACTS_OVERRIDE_TABLES`, `FUNCTION_EFFECTS_OVERRIDE_TABLES` (registry-only)
   * `TargetSpecOptions/make_output_target/register_output_targets`

2. Add `codeintel_target`

3. Delete registry block

4. Decorator swaps:

   ```py
   @codeintel_target(domain="analytics", target=FUNCTION_CONTRACTS_TARGET_NAME)
   def t__function_contracts(...):
       """Inferred function pre/postconditions."""
   ```

   ```py
   @codeintel_target(domain="analytics", target=FUNCTION_EFFECTS_TARGET_NAME)
   def t__function_effects(...):
       """Function purity and side-effect analysis."""
   ```

5. Override map:

   ```py
   "function_contracts": TargetSpecOverride(override_tables=ot.FUNCTION_CONTRACTS_OVERRIDE_TABLES),
   "function_effects": TargetSpecOverride(override_tables=ot.FUNCTION_EFFECTS_OVERRIDE_TABLES),
   ```

---

### 10) `src/codeintel/build/hamilton/native/analytics/coverage_targets.py`

**Targets & anchors**

* `coverage_functions` → `t__coverage_functions`
* `coverage_test_edges` → `t__coverage_test_edges`
* `behavioral_coverage` → `t__behavioral_coverage`

**Edits**

1. Remove imports:

   * `TargetSpecOptions/make_output_target/register_output_targets`
   * (only if registry-only) `COVERAGE_TEST_EDGES_OVERRIDE_TABLES`, `BEHAVIORAL_COVERAGE_OVERRIDE_TABLES`

2. Add `codeintel_target`

3. Delete registry block

4. Decorator swaps:

   ```py
   @codeintel_target(domain="analytics", target=COVERAGE_FUNCTIONS_TARGET_NAME)
   def t__coverage_functions(...):
       """Per-function coverage aggregation."""
   ```

   ```py
   @codeintel_target(domain="analytics", target=COVERAGE_TEST_EDGES_TARGET_NAME)
   def t__coverage_test_edges(...):
       """Test-to-function coverage edge aggregation."""
   ```

   ```py
   @codeintel_target(domain="analytics", target=BEHAVIORAL_COVERAGE_TARGET_NAME)
   def t__behavioral_coverage(...):
       """Behavioral coverage inference (coverage + call graph)."""
   ```

   (Use the exact description string from the registry block in your file.)

5. Override map (only for the targets that had override tables):

   ```py
   "coverage_test_edges": TargetSpecOverride(override_tables=ot.COVERAGE_TEST_EDGES_OVERRIDE_TABLES),
   "behavioral_coverage": TargetSpecOverride(override_tables=ot.BEHAVIORAL_COVERAGE_OVERRIDE_TABLES),
   ```

---

### 11) `src/codeintel/build/hamilton/native/analytics/dependency_targets.py`

De-scoped. The `analytics.dependency_targets` output is no longer planned for
the Hamilton DAG and has been removed from the active migration scope.

   ```py
   @codeintel_target(domain="analytics", target=EXTERNAL_DEPS_TARGET_NAME)
   def t__external_deps(...):
       """External library dependency analysis."""
   ```

   ```py
   @codeintel_target(domain="analytics", target=ENTRYPOINTS_TARGET_NAME)
   def t__entrypoints(...):
       """Entrypoint detection and top-level invocation surfaces."""
   ```
5. Override map:

   ```py
   "external_deps": TargetSpecOverride(override_tables=ot.EXTERNAL_DEPS_OVERRIDE_TABLES),
   "entrypoints": TargetSpecOverride(override_tables=ot.ENTRYPOINTS_OVERRIDE_TABLES),
   ```

---

### 12) `src/codeintel/build/hamilton/native/analytics/metrics_targets.py`

**Targets & anchors**

* `function_history` → `t__function_history`
* `history_timeseries` → `t__history_timeseries`
* `subsystem_graph_metrics` → `t__subsystem_graph_metrics`
* `symbol_graph_metrics` → `t__symbol_graph_metrics`
* `subsystem_agreement` → `t__subsystem_agreement`
* `test_graph_metrics` → `t__test_graph_metrics`

**Edits**

1. Remove imports:

   * all `*_OVERRIDE_TABLES` referenced only in registry:

     * `FUNCTION_HISTORY_OVERRIDE_TABLES`
     * `HISTORY_TIMESERIES_OVERRIDE_TABLES`
     * `SUBSYSTEM_GRAPH_METRICS_OVERRIDE_TABLES`
     * `SYMBOL_GRAPH_METRICS_OVERRIDE_TABLES`
     * `SUBSYSTEM_AGREEMENT_OVERRIDE_TABLES`
     * `TEST_GRAPH_METRICS_OVERRIDE_TABLES`
   * `TargetSpecOptions/make_output_target/register_output_targets`
2. Add `codeintel_target`
3. Delete registry block
4. Decorator swaps: each `@tag_materialize` → `@codeintel_target` and docstring first line from registry.
5. Override map entries:

   ```py
   "function_history": TargetSpecOverride(override_tables=ot.FUNCTION_HISTORY_OVERRIDE_TABLES),
   "history_timeseries": TargetSpecOverride(override_tables=ot.HISTORY_TIMESERIES_OVERRIDE_TABLES),
   "subsystem_graph_metrics": TargetSpecOverride(override_tables=ot.SUBSYSTEM_GRAPH_METRICS_OVERRIDE_TABLES),
   "symbol_graph_metrics": TargetSpecOverride(override_tables=ot.SYMBOL_GRAPH_METRICS_OVERRIDE_TABLES),
   "subsystem_agreement": TargetSpecOverride(override_tables=ot.SUBSYSTEM_AGREEMENT_OVERRIDE_TABLES),
   "test_graph_metrics": TargetSpecOverride(override_tables=ot.TEST_GRAPH_METRICS_OVERRIDE_TABLES),
   ```

---

### 13) `src/codeintel/build/hamilton/native/analytics/metadata_targets.py`

**Targets & anchors**

* `data_models` → `t__data_models`
* `data_model_usage` → `t__data_model_usage`
* `function_ast_features` → `t__function_ast_features`
* `profiles` → `t__profiles`

**Edits**

1. Remove imports:

   * `DATA_MODELS_OVERRIDE_TABLES`, `DATA_MODEL_USAGE_OVERRIDE_TABLES`,
     `FUNCTION_AST_FEATURES_OVERRIDE_TABLES`, `PROFILES_OVERRIDE_TABLES` (registry-only)
   * `TargetSpecOptions/make_output_target/register_output_targets`
2. Add `codeintel_target`
3. Delete registry block
4. Decorator swaps + docstring first line from registry:

   * `"Data model extraction (dataclasses, Pydantic, etc.)."`
   * `"Function-level data model usage tracking."`
   * `"AST-derived semantic features for functions."`
   * `"Denormalized profile tables for querying."`
5. Override map:

   ```py
   "data_models": TargetSpecOverride(override_tables=ot.DATA_MODELS_OVERRIDE_TABLES),
   "data_model_usage": TargetSpecOverride(override_tables=ot.DATA_MODEL_USAGE_OVERRIDE_TABLES),
   "function_ast_features": TargetSpecOverride(override_tables=ot.FUNCTION_AST_FEATURES_OVERRIDE_TABLES),
   "profiles": TargetSpecOverride(override_tables=ot.PROFILES_OVERRIDE_TABLES),
   ```

---

### 14) `src/codeintel/build/hamilton/native/analytics/hotspots.py`

**Targets**

* `hotspots` → `t__hotspots`

**Edits**

1. Remove imports:

   * `HOTSPOTS_OVERRIDE_TABLES` (registry-only)
   * `TargetSpecOptions/make_output_target/register_output_targets`
2. Add `codeintel_target`
3. Delete registry block
4. Decorator swap:

   ```py
   @codeintel_target(domain="analytics", target=HOTSPOTS_TARGET_NAME)
   def t__hotspots(...):
       """File hotspot analysis based on churn."""
   ```
5. Override map:

   ```py
   "hotspots": TargetSpecOverride(override_tables=ot.HOTSPOTS_OVERRIDE_TABLES),
   ```

---

### 15) `src/codeintel/build/hamilton/native/analytics/risk_factors.py`

**Targets**

* `risk_factors` → `t__risk_factors`

**Edits**

* Remove `TargetSpecOptions/make_output_target/register_output_targets`
* Add `codeintel_target`
* Delete registry block
* Decorator swap:

  ```py
  @codeintel_target(domain="analytics", target=RISK_FACTORS_TARGET_NAME)
  def t__risk_factors(...):
      """Composite risk factors per function."""
  ```
* (If it has override tables in the file, add it in override map; in Phase3 it registers override tables.)

---

### 16) `src/codeintel/build/hamilton/native/analytics/subsystem_targets.py`

**Targets**

* `subsystems` → `t__subsystems`

**Edits**

* Remove `SUBSYSTEMS_OVERRIDE_TABLES` import (registry-only)
* Remove `TargetSpecOptions/make_output_target/register_output_targets`
* Add `codeintel_target`
* Delete registry block
* Decorator swap:

  ```py
  @codeintel_target(domain="analytics", target=SUBSYSTEMS_TARGET_NAME)
  def t__subsystems(...):
      """Architectural subsystem inference."""
  ```
* Override map:

  ```py
  "subsystems": TargetSpecOverride(override_tables=ot.SUBSYSTEMS_OVERRIDE_TABLES),
  ```

---

### 17) `src/codeintel/build/hamilton/native/analytics/subsystem_cache_targets.py`

**Targets**

* `subsystem_caches` → `t__subsystem_caches`

**Edits**

* Remove `SUBSYSTEM_CACHE_OVERRIDE_TABLES` import (registry-only)
* Remove `TargetSpecOptions/make_output_target/register_output_targets`
* Add `codeintel_target`
* Delete registry block
* Decorator swap:

  ```py
  @codeintel_target(domain="analytics", target=SUBSYSTEM_CACHES_TARGET_NAME)
  def t__subsystem_caches(...):
      """Cached subsystem profile and coverage tables."""
  ```
* Override map:

  ```py
  "subsystem_caches": TargetSpecOverride(override_tables=ot.SUBSYSTEM_CACHE_OVERRIDE_TABLES),
  ```

---

### 18) `src/codeintel/build/hamilton/native/analytics/classification_targets.py`

**Targets & anchors**

* `semantic_roles` → `t__semantic_roles`
* `test_profile` → `t__test_profile`

**Edits**

* Remove `SEMANTIC_ROLES_OVERRIDE_TABLES`, `TEST_PROFILE_OVERRIDE_TABLES` imports (registry-only)
* Remove `TargetSpecOptions/make_output_target/register_output_targets`
* Add `codeintel_target`
* Delete registry block
* Decorator swaps:

  ```py
  @codeintel_target(domain="analytics", target=SEMANTIC_ROLES_TARGET_NAME)
  def t__semantic_roles(...):
      """Semantic role classification (handler, utility, etc.)."""
  ```

  ```py
  @codeintel_target(domain="analytics", target=TEST_PROFILE_TARGET_NAME)
  def t__test_profile(...):
      """Test profiling and test suite characterization."""
  ```
* Override map:

  ```py
  "semantic_roles": TargetSpecOverride(override_tables=ot.SEMANTIC_ROLES_OVERRIDE_TABLES),
  "test_profile": TargetSpecOverride(override_tables=ot.TEST_PROFILE_OVERRIDE_TABLES),
  ```

---

### 19) `src/codeintel/build/hamilton/native/analytics/config_graph_targets.py`

**Targets & anchors**

* `config_data_flow` → `t__config_data_flow`
* `cfg_dfg_metrics` → `t__cfg_dfg_metrics`

**Edits**

* Remove `CONFIG_DATA_FLOW_OVERRIDE_TABLES`, `CFG_DFG_METRICS_OVERRIDE_TABLES` imports (registry-only)
* Remove `TargetSpecOptions/make_output_target/register_output_targets`
* Add `codeintel_target`
* Delete registry block
* Decorator swaps (docstring first line from registry)
* Override map:

  ```py
  "config_data_flow": TargetSpecOverride(override_tables=ot.CONFIG_DATA_FLOW_OVERRIDE_TABLES),
  "cfg_dfg_metrics": TargetSpecOverride(override_tables=ot.CFG_DFG_METRICS_OVERRIDE_TABLES),
  ```

---

## Export

### 20) `src/codeintel/build/hamilton/native/export/serving_artifacts.py`

**Targets**

* `serving_artifacts` → `t__serving_artifacts`

**Edits**

1. Remove imports:

   * `from codeintel.build.contracts import ArtifactSpec` (registry-only)
   * `from codeintel.build.hamilton.native.target_spec_helpers import TargetSpecOptions, make_output_target, register_output_targets`
2. Remove constants:

   * `SERVING_ARTIFACT_SPECS` (registry-only)
3. Add `codeintel_target`
4. Delete registry block
5. Decorator swap:

   ```py
   @codeintel_target(domain="export", target=SERVING_ARTIFACTS_TARGET_NAME)
   def t__serving_artifacts(...):
       """Compile deterministic serving artifacts (semantic registry, schema manifest, buildspec)."""
   ```

**Note (artifact inference dependency):** this module should already have `FileArtifactSaver` nodes that pass `artifact_name=value(...)` and (after item #2) `path_template=value(...)`.

---

### 21) `src/codeintel/build/hamilton/native/export/export_targets.py`

**Targets & anchors**

* `export_jsonl` → `t__export_jsonl`
* `export_parquet` → `t__export_parquet`

**Edits**

1. Imports

   * Remove:

     * `from codeintel.build.contracts import ArtifactSpec` (registry-only)
     * `from codeintel.build.hamilton.materializers.artifact_saver import resolve_artifact_path` (precheck becomes unnecessary once DAG path_template is authoritative)
     * `from codeintel.build.hamilton.native.target_spec_helpers import TargetSpecOptions, make_output_target, register_output_targets`
   * Add:

     * `from codeintel.build.hamilton.native.target_decorators import codeintel_target`

2. Constants

   * Delete:

     * `EXPORT_JSONL_ARTIFACT_SPECS`
     * `EXPORT_PARQUET_ARTIFACT_SPECS`

3. Delete registry block (two make_output_target entries)

4. Remove contract-coupled precheck in `_export_manifest_plan(...)`

   * Delete the `resolve_artifact_path(...) is None` block entirely.

5. Decorator swaps + docstring first line:

   ```py
   @codeintel_target(domain="export", target=EXPORT_JSONL_TARGET_NAME)
   def t__export_jsonl(...):
       """Export datasets to JSONL format for Document Output."""
   ```

   ```py
   @codeintel_target(domain="export", target=EXPORT_PARQUET_TARGET_NAME)
   def t__export_parquet(...):
       """Export datasets to Parquet format for Document Output."""
   ```

---

# After all file edits: the “make it compile” sweep

Once you apply the above:

1. Run your lints to clear dead imports/constants:

   * `ruff check --fix`
   * `pyright` / `pyrefly` as applicable
2. Confirm no registry references remain:

   * `grep -R "register_output_targets" -n src/codeintel/build/hamilton | head`
   * `grep -R "make_output_target" -n src/codeintel/build/hamilton | head`
3. Confirm the only remaining `target_spec_helpers.py` usage is… none.
4. Ensure `native/target_overrides.py` includes all `override_tables` mappings listed above (that’s the most common “oops” in a batch refactor).

---
## PR-06.5: Strictness toggle (staged gate)

During the module-by-module migration, run `compile_output_targets_from_driver(strict=False)` so
incomplete conversion does not break the DAG. After PR-06 completes, flip to `strict=True` and
enforce:

* `target_spec_version` on all materialize nodes
* docstring summaries for all targets
* no missing saver tags for contract outputs

This makes the strictness flip a single, intentional step.
