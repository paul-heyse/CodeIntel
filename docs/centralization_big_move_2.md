Below is a **comprehensive implementation plan (with concrete code snippets)** for:

> **Move artifact path templates into DAG metadata** so `FileArtifactSaver` can resolve output paths from **DAG-attached metadata** (tags / adapter params) instead of consulting **contract-derived expected artifacts**.

This is directly aligned with your documented pattern that `SaveToObjectMetadataDecorator` creates deterministic `m__*` materialization nodes and is the “DAG-visible I/O” boundary, and with the fact that artifact path-template invariants + placeholder enforcement are currently part of the contract layer.

---

## 0) Current state (what we’re changing)

### Where the contract dependency lives today

In your repo, `FileArtifactSaver` resolves the output path by calling `_resolve_artifact_path()`, which:

1. looks up the target in the `TargetGraph`, then
2. calls `expected_artifacts(...)` (contract-driven), then
3. formats `ArtifactSpec.path_template` using `build_dir/scip_dir/export_dir/repo_root`, and returns the matching `ArtifactRef.path` as a `Path`.

That contract dependency is here (today):

```py
# src/codeintel/build/hamilton/materializers/artifact_saver.py
artifacts = expected_artifacts(
    target,
    env.snapshot,
    output_inventory=env.output_inventory,
    path_formatter={...},
)
...
return Path(art.path)
```

### Your placeholder enforcement is also contract-centric

Allowed placeholder enforcement is currently done in `target_spec_helpers._validate_artifact_specs()` by parsing `ArtifactSpec.path_template` and rejecting unknown keys.

---

## 1) Design target

### New rule

**Every artifact materialization node (`m__artifact__…`) must carry:**

* `artifact_name` (already passed to the saver today)
* `path_template` **(new)**

And `FileArtifactSaver` must resolve paths from:

* `artifact_name` + `path_template` provided as **adapter params** (and mirrored into **node tags**),
* plus the same allowed-placeholder enforcement you already do.

### Design principles (tightened for DAG-first correctness)

* **DAG saver tags are the authoritative runtime source** for artifact paths and artifact existence.
* **Contracts become derived views** (for BuildSpec/docs) and are not required for execution.
* **Strict enforcement uses DAG-derived contracts**, not registry contracts.
* **No fallback for path resolution** once call sites pass `path_template` (fail fast).

### What stays the same (for this PR)

* You can keep `ArtifactSpec` in contracts for now (BuildSpec / external docs), but **artifact materialization should not need it**.
* This change is localized to the artifact I/O boundary and does not require rewriting catalogs/spec compilation in the same PR.

---

## 2) Integrated migration timeline (DAG-first runway)

This plan is designed to land quickly while preventing split-brain behavior between registry
contracts and DAG-derived outputs. It stitches this document (runtime alignment + artifact paths)
with `centralization_big_move_3.md` (target compilation + registry removal).

### Phase 0: Runtime alignment (DAG outputs become the only runtime source)

**Goal:** ensure runtime records and artifact paths are DAG-derived before changing target
compilation or removing the registry.

**Deliverables**

* Extend `derive_target_outputs_from_savers` (existing helper) to include
  `artifact_path_template` for contract saver nodes.
* Update `OutputInventory` to carry:
  * `datasets_by_target`
  * `artifacts_by_target`
  * `artifact_templates_by_target` (name -> template)
* Update `expected_artifacts` and `create_run_record` to use DAG-derived templates from
  `OutputInventory` (no contract lookups).
* Extend output inventory diffing (`target_inventory`) to detect template mismatches.
* Enforce `output_role="internal"` for non-contract saver nodes (so internal outputs are excluded
  from contract derivation).

**Gate**

* `TargetRunRecord.artifacts` paths resolve from DAG templates.
* Missing `artifact_path_template` on contract saver nodes is a hard error.
* Inventory diffing surfaces template mismatches.

### Phase 1: DAG output derivation (single source for compiler + runtime)

**Goal:** use a single DAG derivation path for all consumers (compiler, runtime, validation).

**Deliverables**

* Reuse the extended `derive_target_outputs_from_savers` in:
  * target compilation (big_move_3 compiler)
  * runtime records (this plan)
  * graph validation (validate_graph should enforce missing templates)
* Ensure derivation filters to `output_role="contract"` only.

### Phase 2: Per-module migration (targets become DAG-native specs)

**Goal:** migrate each target module to `@codeintel_target` and remove registry specs.

**Deliverables**

* Replace `register_output_targets` blocks with `@codeintel_target` per module.
* Keep override tables in a small override map (schema-only).
* Add docstring summaries for all `t__` nodes.

### Phase 3: Strictness flip (hardness on)

**Goal:** enforce DAG as the sole authority.

**Deliverables**

* Set compiler strict mode to default on (after Phase 2 complete).
* Require `target_spec_version="1"` on all target anchors.
* Fail fast on missing tags or missing artifact templates.

### Phase 4: Registry removal (cleanup)

**Goal:** delete registry helpers and remove dependency on contract specs for runtime behavior.

**Deliverables**

* Remove `target_spec_helpers` registry logic.
* Ensure canonical target catalog is DAG-derived only.

---

## 3) Implementation plan (PR-by-PR)

### PR-01: Add canonical tag keys + path-template validation helper

#### 2.1 Add tag key for artifact path templates

In `src/codeintel/core/hamilton/tags.py` add:

```py
# src/codeintel/core/hamilton/tags.py

TAG_ARTIFACT_PATH_TEMPLATE = "artifact_path_template"

__all__ += [
    "TAG_ARTIFACT_PATH_TEMPLATE",
]
```

Why: you already have `TAG_ARTIFACT` for the artifact name, but you need a stable tag key for the template that tools/UI/introspection can rely on.

#### 2.2 Create a shared validator for `{placeholder}` keys

Make a small dependency-light helper:

```py
# src/codeintel/build/hamilton/materializers/path_templates.py
from __future__ import annotations

from string import Formatter

_ALLOWED_KEYS: frozenset[str] = frozenset({"build_dir", "export_dir", "repo_root", "scip_dir"})
_fmt = Formatter()

class PathTemplateError(ValueError):
    """Raised when an artifact path template uses unsupported placeholders."""

def validate_path_template(template: str) -> None:
    for _, field_name, _, _ in _fmt.parse(template):
        if field_name is None:
            continue
        if field_name not in _ALLOWED_KEYS:
            raise PathTemplateError(
                f"Unsupported artifact path_template placeholder {field_name!r} "
                f"(allowed={sorted(_ALLOWED_KEYS)})"
            )

def format_path_template(template: str, *, formatter: dict[str, str]) -> str:
    # validate first so missing keys and illegal keys are cleanly separated
    validate_path_template(template)
    return template.format(**formatter)

def default_formatter(*, build_dir: str, scip_dir: str, export_dir: str, repo_root: str) -> dict[str, str]:
    return {
        "build_dir": build_dir,
        "scip_dir": scip_dir,
        "export_dir": export_dir,
        "repo_root": repo_root,
    }
```

This mirrors your existing `_ALLOWED_ARTIFACT_TEMPLATE_KEYS` behavior, but moves it into a reusable “artifact materialization boundary” helper.

**Hardening:** replace `_ALLOWED_ARTIFACT_TEMPLATE_KEYS` usage with this helper so there is a single source of truth.

---

### PR-02: Extend `SaveToObjectMetadataDecorator` to attach artifact metadata (tags + adapter param plumbing)

We want the `m__artifact__…` node to advertise:

* `target` (so contract enforcement can activate, and telemetry can attribute)
* `artifact` (artifact name)
* `artifact_path_template` (required for contract outputs)

#### 2.3 Update `SaveToObjectMetadataDecorator.create_saver_node(...)`

In `src/codeintel/build/hamilton/save_to.py`:

1. import the canonical tag constants:

```py
from codeintel.core.hamilton import tags as ht
from codeintel.build.hamilton.materializers.path_templates import validate_path_template
```

2. after `resolved_kwargs_typed` is computed, build tags:

```py
tags = {
    "hamilton.data_saver": True,
    "hamilton.data_saver.sink": f"{saver_cls.name()}",
    "hamilton.data_saver.classname": f"{saver_cls.__qualname__}",
}

# Promote strongly-typed build tags when present in saver kwargs.
target_name = resolved_kwargs_typed.get("target_name")
if isinstance(target_name, str) and target_name:
    tags[ht.TAG_TARGET] = target_name

artifact_name = resolved_kwargs_typed.get("artifact_name")
if isinstance(artifact_name, str) and artifact_name:
    tags[ht.TAG_ARTIFACT] = artifact_name

path_template = resolved_kwargs_typed.get("path_template")
if isinstance(path_template, str) and path_template:
    validate_path_template(path_template)  # early, decorator-build-time failure
    tags[ht.TAG_ARTIFACT_PATH_TEMPLATE] = path_template
```

3. pass `tags=tags` into the Node constructor (replacing the current literal dict).

**Important:** this gives you exactly what you want: artifact path templates become **DAG metadata** (tags) at graph build time, not a side registry.

4. **Require `path_template` for contract artifact savers.** This mirrors the existing `output_role`/`target_name` guardrails and prevents silent fallback behavior:

```py
if output_role != "internal" and isinstance(artifact_name, str) and artifact_name:
    path_template = resolved_kwargs_typed.get("path_template")
    if not isinstance(path_template, str) or not path_template:
        msg = (
            f"{fn.__qualname__}: contract artifact saver nodes must provide "
            "path_template=value(<str>) so artifact paths are DAG-derived."
        )
        raise InvalidDecoratorException(msg)
```

**Note:** saver nodes already carry `target` in tags today (via `target_name=value(...)`), so enforcement is already target-scoped. The real gain here is adding `artifact_path_template` as a first-class tag.

---

### PR-03: Modify `FileArtifactSaver` to resolve paths from DAG-provided template (and stop consulting contracts)

#### 2.4 Add `path_template` to `FileArtifactSaver`

In `src/codeintel/build/hamilton/materializers/artifact_saver.py`, update the dataclass:

```py
@dataclass(frozen=True)
class FileArtifactSaver(DataSaver):
    env: BuildEnv
    graph: TargetGraph
    target_name: str
    artifact_name: str
    path_template: str | None = None   # NEW (required for contract outputs)
    hash_options: InputHashOptions | None = None
```

Because it has a default, it will be treated as an optional saver arg by Hamilton’s adapter machinery.

#### 2.5 Replace `_resolve_artifact_path(...)` with a DAG-template path resolver

Add:

```py
from codeintel.build.hamilton.materializers.path_templates import (
    default_formatter,
    format_path_template,
)

def _resolve_artifact_path_from_template(env: BuildEnv, template: str) -> Path:
    fmt = default_formatter(
        build_dir=str(env.paths.build_dir),
        scip_dir=str(env.paths.scip_dir),
        export_dir=str(env.paths.document_output_dir),
        repo_root=str(env.snapshot.repo_root),
    )
    return Path(format_path_template(template, formatter=fmt))
```

Then change `_resolve_artifact_path(...)` logic to:

```py
def _resolve_artifact_path(
    env: BuildEnv,
    graph: TargetGraph,
    target_name: str,
    artifact_name: str,
    *,
    path_template: str | None,
) -> Path | None:
    # DAG-provided template is the authoritative path source.
    if not path_template:
        raise ValueError(
            f"Missing artifact path_template for {target_name}.{artifact_name} "
            "on a contract output node."
        )
    return _resolve_artifact_path_from_template(env, path_template)
```

#### 2.6 Update `save_data()` to pass the template

Change both calls:

```py
resolved = _resolve_artifact_path(self.env, self.graph, self.target_name, self.artifact_name, path_template=self.path_template)
output_path = _resolve_artifact_path(self.env, self.graph, self.target_name, self.artifact_name, path_template=self.path_template)
```

Now `FileArtifactSaver` resolves the path from **DAG adapter params**, and **does not need `expected_artifacts` / contracts**.

---

### PR-04: Update all artifact SaveTo call sites to pass `path_template=value("...")`

You already pass `artifact_name=value(...)` at every artifact saver node; this PR adds `path_template=value(...)` next to it.

#### 2.7 Update SCIP artifact nodes

In `src/codeintel/build/hamilton/native/ingestion/scip.py`:

```py
@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node("artifact.scip_index"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_TARGET_NAME),
    artifact_name=value(SCIP_ARTIFACT_INDEX),
    path_template=value("{scip_dir}/index.scip"),   # NEW
)
...
```

and similarly:

```py
path_template=value("{scip_dir}/index.json")
```

#### 2.8 Update `serving_artifacts` templates (mirror existing contract spec strings)

In `src/codeintel/build/hamilton/native/export/serving_artifacts.py`:

```py
@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(f"artifact.{SERVING_ARTIFACT_SEMANTIC_REGISTRY}"),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SERVING_ARTIFACTS_TARGET_NAME),
    artifact_name=value(SERVING_ARTIFACT_SEMANTIC_REGISTRY),
    path_template=value("{build_dir}/serving/artifacts/semantic_registry.json"),  # NEW
)
```

…and repeat for each artifact:

* schema_manifest.json
* buildspec.json
* environment.json
* views_sql.json
* views_sql_diff.json

#### 2.9 Update `export_targets` (manifest artifacts)

In `src/codeintel/build/hamilton/native/export/export_targets.py`, wherever `FileArtifactSaver` is used, add the correct template (from its contract specs).

Also: remove the contract-based pre-check:

```py
# DELETE this whole block:
if resolve_artifact_path(env, graph, target_name=..., artifact_name=...) is None:
    raise ValueError(...)
```

Because with DAG templates, the saver can always resolve (or error) deterministically, and the early check is no longer buying correctness—it’s reintroducing contract coupling.

---

### PR-05: Eliminate remaining contract coupling in runtime records

To fully remove the runtime dependency on contract-derived artifacts, update the record creation path to use DAG-derived artifact specs.

#### 2.10 Make `expected_artifacts` DAG-derived

Update `expected_artifacts` to accept a DAG-derived inventory that includes path templates:

* add `artifact_templates_by_target: dict[str, dict[str, str]]` to `OutputInventory`
* populate it via saver tags (from `artifact_path_template`)
* build `ArtifactRef.path` directly from the template formatter

This keeps `TargetRunRecord.artifacts` consistent even when contracts are absent.

#### 2.11 Update `create_run_record` to use DAG-derived artifacts

In `src/codeintel/build/hamilton/run_records.py`, replace the `expected_artifacts(...)` call with a DAG-derived equivalent. This removes the contract dependency for path resolution in both succeeded and skipped cases.

#### 2.12 Update materialization record validation to use DAG-derived contract

In `src/codeintel/build/hamilton/native/materialization_records.py`, validate artifact names and existence against the **DAG-derived** contract (compiled from saver tags), not the registry contract.

---

### PR-06: Tests

#### 2.13 Golden tag test: saver nodes must carry the path template tag

Add a test that builds a runtime and asserts:

* each `m__artifact__*` saver node has:

  * `TAG_TARGET`
  * `TAG_ARTIFACT`
  * `TAG_ARTIFACT_PATH_TEMPLATE`

Example (sketch):

```py
# tests/build/hamilton/test_artifact_path_templates.py

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.runtime import HamiltonRuntime
from codeintel.core.hamilton import tags as ht

def test_artifact_materialization_nodes_have_path_templates(build_env, target_graph):
    runtime: HamiltonRuntime = build_driver(env=build_env, graph=target_graph)
    nodes = runtime.dr.graph.nodes

    artifact_savers = [
        (name, node)
        for name, node in nodes.items()
        if isinstance(node.tags, dict)
        and node.tags.get("hamilton.data_saver") is True
        and node.tags.get(ht.TAG_ARTIFACT)  # artifact saver nodes
    ]
    assert artifact_savers, "expected at least one artifact saver node"

    for name, node in artifact_savers:
        tags = node.tags
        assert ht.TAG_TARGET in tags
        assert ht.TAG_ARTIFACT in tags
        assert ht.TAG_ARTIFACT_PATH_TEMPLATE in tags
```

#### 2.14 Unit test: `FileArtifactSaver` resolves path from template without calling contracts

A simple “tripwire” test:

* construct a `FileArtifactSaver` with `path_template="{build_dir}/x.json"`
* ensure it returns a path under `env.paths.build_dir`
* verify it raises `ValueError` if `path_template` is missing (contract outputs must be explicit)

---

## 4) Backward compatibility and rollout knobs

### 3.1 Temporary compatibility window (optional)

If you need to land the change incrementally, allow a short compatibility window where missing
`path_template` is tolerated **only for `output_role="internal"`** saver nodes (non-contract
outputs). Contract outputs must always provide `path_template`.

Once you’ve migrated all call sites (PR-04), you should:

* remove any fallback path resolution entirely, and
* delete the contract-based `resolve_artifact_path(...)` helper (or keep it only for legacy read paths).

### 3.2 “Single source of truth” policy (post-merge)

After this lands, your “source of truth” for artifact paths becomes:

* **DAG saver node tag**: `artifact_path_template`
* **DAG saver node tag**: `artifact`
* (optionally) contract still stores it for BuildSpec/export docs, but it becomes “derived/checked” rather than “required for execution.”

A clean follow-on (recommended) is to add a strict check that:

* contract path templates (if present) must match DAG templates for the same `(target, artifact)`.
* missing DAG templates fail fast at graph build time.

---

## 5) Why this satisfies the goal

With this plan:

* Artifact path resolution no longer depends on `expected_artifacts(...)` / contract lookup.
* The artifact output node (`m__artifact__…`) carries the metadata needed to:

  * resolve output paths,
  * validate placeholder keys,
  * show up in tagging/introspection/telemetry as a first-class artifact write boundary.
* Adding a new artifact becomes exactly:

  * “add a new `SaveToObjectMetadataDecorator([FileArtifactSaver], ...)` node with `artifact_name` + `path_template`”
  * no other registry needed at execution time.

This is exactly the kind of “DAG boundary collapse” you described: declaration + resolution rules + enforcement attach to the DAG materialization node itself.

---

## Appendix: Minimal patch snippets (copy/paste ready)

### A) `SaveToObjectMetadataDecorator` tag augmentation (core excerpt)

```py
# src/codeintel/build/hamilton/save_to.py
from codeintel.core.hamilton import tags as ht
from codeintel.build.hamilton.materializers.path_templates import validate_path_template

...

tags = {
    "hamilton.data_saver": True,
    "hamilton.data_saver.sink": f"{saver_cls.name()}",
    "hamilton.data_saver.classname": f"{saver_cls.__qualname__}",
}

target_name = resolved_kwargs_typed.get("target_name")
if isinstance(target_name, str) and target_name:
    tags[ht.TAG_TARGET] = target_name

artifact_name = resolved_kwargs_typed.get("artifact_name")
if isinstance(artifact_name, str) and artifact_name:
    tags[ht.TAG_ARTIFACT] = artifact_name

path_template = resolved_kwargs_typed.get("path_template")
if isinstance(path_template, str) and path_template:
    validate_path_template(path_template)
    tags[ht.TAG_ARTIFACT_PATH_TEMPLATE] = path_template

return h_node.Node(
    ...,
    tags=tags,
)
```

### B) `FileArtifactSaver` template-first path resolution (core excerpt)

```py
# src/codeintel/build/hamilton/materializers/artifact_saver.py
from codeintel.build.hamilton.materializers.path_templates import (
    default_formatter,
    format_path_template,
)

def _resolve_artifact_path_from_template(env: BuildEnv, template: str) -> Path:
    fmt = default_formatter(
        build_dir=str(env.paths.build_dir),
        scip_dir=str(env.paths.scip_dir),
        export_dir=str(env.paths.document_output_dir),
        repo_root=str(env.snapshot.repo_root),
    )
    return Path(format_path_template(template, formatter=fmt))

def _resolve_artifact_path(..., *, path_template: str | None) -> Path | None:
    if not path_template:
        raise ValueError(
            f"Missing artifact path_template for {target_name}.{artifact_name} "
            "on a contract output node."
        )
    return _resolve_artifact_path_from_template(env, path_template)
```

---

### References (uploaded artifacts)

    
