Here’s a concrete, “rip it out now” plan to remove the **SCIP config legacy path** and leave only a clean, explicit configuration API around `_scip_resolver`. This builds directly on the earlier analysis of that file and its `cfg_source="legacy" | "explicit"` split. 

I’ll structure it as:

1. What exists today (in `_scip_resolver`)
2. Target architecture (what we want instead)
3. Step‑by‑step implementation with code sketches
4. What tests / callers need to change

---

## 1. Current state (SCIP config path)

Right now `_scip_resolver.py` has:

* **Dataclasses**

  ```python
  @dataclass(frozen=True)
  class ResolvedScipConfig:
      repo: str
      commit: str
      repo_root: Path
      build_dir: Path
      document_output_dir: Path
      scip_python_bin: str | None
      scip_bin: str | None
      modules: list[ModuleRecord]
      cfg_source: Literal["legacy", "explicit"]
      cfg: ScipIngestStepConfig | None
  ```

  ```python
  @dataclass(frozen=True)
  class ScipResolverInput:
      cfg: ScipIngestStepConfig | None = None
      repo: str | None = None
      commit: str | None = None
      repo_root: Path | None = None
      build_dir: Path | None = None
      document_output_dir: Path | None = None
      scip_python_bin: str | None = None
      scip_bin: str | None = None
      modules: Sequence[ModuleRecord] | None = None

      @classmethod
      def build(..., paths: ScipPathConfig | None = None, modules: Sequence[ModuleRecord] | None = None) -> "ScipResolverInput": ...
  ```

* **Resolver function** with a dual path:

  ```python
  def resolve_scip_inputs_from(
      gateway: StorageGateway,
      modules_or_cfg: Sequence[ModuleRecord] | object,
      inputs: ScipResolverInput,
  ) -> ResolvedScipConfig:
      cfg = inputs.cfg
      repo = inputs.repo
      ...
      modules = inputs.modules

      # LEGACY CONFIG OBJECT PATH
      if cfg is not None or isinstance(modules_or_cfg, ScipIngestStepConfig):
          actual_cfg = cfg or modules_or_cfg
          ...  # validate type
          module_map = load_module_map(gateway, actual_cfg.repo, actual_cfg.commit, language="python", logger=None)
          filesystem_adapter = import_module(...).FilesystemDiscoveryAdapter
          module_list = list(
              filesystem_adapter.iter_modules(
                  module_map,
                  actual_cfg.repo_root,
                  logger=None,
                  scan_profile=None,
              )
          )
          return ResolvedScipConfig(
              repo=actual_cfg.repo,
              commit=actual_cfg.commit,
              repo_root=actual_cfg.repo_root,
              build_dir=actual_cfg.build_dir,
              document_output_dir=actual_cfg.document_output_dir,
              scip_python_bin=actual_cfg.scip_python_bin,
              scip_bin=actual_cfg.scip_bin,
              modules=module_list,
              cfg_source="legacy",
              cfg=actual_cfg,
          )

      # EXPLICIT PATH (no ScipIngestStepConfig)
      module_list = list(modules) if modules is not None else []
      if not module_list and isinstance(modules_or_cfg, Sequence):
          module_list = list(modules_or_cfg)

      if repo is None or commit is None or repo_root is None or build_dir is None or document_output_dir is None:
          raise ValueError(...)

      return ResolvedScipConfig(
          repo=repo,
          commit=commit,
          repo_root=repo_root,
          build_dir=build_dir,
          document_output_dir=document_output_dir,
          scip_python_bin=scip_python_bin,
          scip_bin=scip_bin,
          modules=module_list,
          cfg_source="explicit",
          cfg=None,
      )
  ```

* Thin wrapper:

  ```python
  def resolve_scip_inputs(
      gateway: StorageGateway,
      modules_or_cfg: Sequence[ModuleRecord] | object,
      inputs: ScipResolverInput,
  ) -> ResolvedScipConfig:
      return resolve_scip_inputs_from(gateway, modules_or_cfg, inputs)
  ```

**Important observation:** inside this codebase, nothing in `ingestion` or `config` actually calls `resolve_scip_inputs` today; it’s effectively a utility that’s only used (if at all) from tests. The ingest plugin goes straight from `ctx` → `ScipIngestConfig` → `ScipIngestStep`, bypassing `_scip_resolver` entirely. 

That means we’re free to **hard‑cut the legacy path and simplify the API** without breaking runtime code — only tests / external tooling (which you control) need updates.

---

## 2. Target architecture

Let’s define what “go‑forward” looks like:

* **No knowledge of `ScipIngestStepConfig`** inside `_scip_resolver`.

* **No `cfg_source`, no `cfg`** fields in `ResolvedScipConfig` or `ScipResolverInput`.

* **No `modules_or_cfg` union param** and no `StorageGateway` dependency.

* **Single explicit path**:

  ```python
  def resolve_scip_inputs(
      modules: Sequence[ModuleRecord] | None,
      inputs: ScipResolverInput,
  ) -> ResolvedScipConfig:
      # normalize + validate fields
  ```

* `ScipResolverInput` is just a convenience struct (optionally built via `.build`) for carrying:

  * repo, commit
  * repo_root, build_dir, document_output_dir
  * scip_python_bin, scip_bin
  * modules (optional; can be passed separately as an argument, your choice)

In other words:

> **Only explicit, value‑based configuration. No object‑typed “config” path, no “legacy vs explicit” branching.**

---

## 3. Step‑by‑step implementation (with code sketches)

### Step 0 – (Sanity) confirm no runtime call sites

You already know, but here’s the check to run locally:

```bash
rg "resolve_scip_inputs" -n src/
rg "ScipResolverInput" -n src/
```

You should see only `_scip_resolver.py` and its package `__init__`, not plugins/steps. If that holds, everything we change here affects only tests / future utilities.

---

### Step 1 – Simplify `ResolvedScipConfig`

**File:** `ingestion/infrastructure_utilities/_scip_resolver.py`

1. Drop the `cfg_source` and `cfg` fields:

```python
from dataclasses import dataclass
from pathlib import Path
from collections.abc import Sequence

from codeintel.ingestion.ports.discovery import ModuleRecord

# Remove: from codeintel.config import ScipIngestStepConfig
# Remove: from codeintel.storage.gateway import StorageGateway
# Remove: from codeintel.storage.module_index import load_module_map
# Remove: from importlib import import_module

@dataclass(frozen=True)
class ResolvedScipConfig:
    """Normalized SCIP configuration (explicit-only path)."""

    repo: str
    commit: str
    repo_root: Path
    build_dir: Path
    document_output_dir: Path
    scip_python_bin: str | None
    scip_bin: str | None
    modules: list[ModuleRecord]
```

No more `cfg_source`, no more `cfg`.

---

### Step 2 – Simplify `ScipResolverInput`

Change it from the “legacy bridging” shape to a pure explicit‑config shape:

```python
@dataclass(frozen=True)
class ScipResolverInput:
    """Explicit SCIP input values used for normalization."""

    repo: str | None = None
    commit: str | None = None
    repo_root: Path | None = None
    build_dir: Path | None = None
    document_output_dir: Path | None = None
    scip_python_bin: str | None = None
    scip_bin: str | None = None
    modules: Sequence[ModuleRecord] | None = None

    @classmethod
    def build(
        cls,
        *,
        repo: str,
        commit: str,
        paths: ScipPathConfig,
        modules: Sequence[ModuleRecord] | None = None,
    ) -> "ScipResolverInput":
        """Convenience constructor from path config and module list."""
        return cls(
            repo=repo,
            commit=commit,
            repo_root=paths.repo_root,
            build_dir=paths.build_dir,
            document_output_dir=paths.document_output_dir,
            scip_python_bin=paths.scip_python_bin,
            scip_bin=paths.scip_bin,
            modules=modules,
        )
```

Remove these from the dataclass and its `.build`:

* `cfg: ScipIngestStepConfig | None`
* Any branches or docstrings referring to “legacy config object path.”

---

### Step 3 – Rewrite `resolve_scip_inputs_from` as explicit‑only

You can either keep the two‑function structure (`resolve_scip_inputs_from` + `resolve_scip_inputs`) or collapse to a single function.

I’d suggest **collapsing** to one public helper and deleting `_from` entirely, since nothing uses it today.

Replace the existing functions with:

```python
def resolve_scip_inputs(
    modules: Sequence[ModuleRecord] | None,
    inputs: ScipResolverInput,
) -> ResolvedScipConfig:
    """Normalize SCIP inputs into a required, typed config.

    This explicit-only version does not accept ScipIngestStepConfig.
    All values are taken from ScipResolverInput and the modules sequence.

    Raises
    ------
    ValueError
        If required parameters are missing or invalid.
    """
    repo = inputs.repo
    commit = inputs.commit
    repo_root = inputs.repo_root
    build_dir = inputs.build_dir
    document_output_dir = inputs.document_output_dir
    scip_python_bin = inputs.scip_python_bin
    scip_bin = inputs.scip_bin

    # Prefer an explicit modules argument; fall back to inputs.modules
    modules_seq = modules or inputs.modules or ()

    module_list = list(modules_seq)

    # Strict explicit-only semantics: every required field present, and at
    # least one module, or we treat it as programmer error.
    if (
        repo is None
        or commit is None
        or repo_root is None
        or build_dir is None
        or document_output_dir is None
    ):
        msg = "repo, commit, repo_root, build_dir, and document_output_dir are required"
        raise ValueError(msg)

    if not module_list:
        msg = "At least one ModuleRecord is required to resolve SCIP inputs"
        raise ValueError(msg)

    return ResolvedScipConfig(
        repo=repo,
        commit=commit,
        repo_root=repo_root,
        build_dir=build_dir,
        document_output_dir=document_output_dir,
        scip_python_bin=scip_python_bin,
        scip_bin=scip_bin,
        modules=module_list,
    )
```

Then **delete** the old `resolve_scip_inputs_from` entirely.

If you prefer to keep a helper, you can have:

```python
def _resolve_scip_inputs_from(
    modules: Sequence[ModuleRecord] | None,
    inputs: ScipResolverInput,
) -> ResolvedScipConfig:
    ...

def resolve_scip_inputs(
    modules: Sequence[ModuleRecord] | None,
    inputs: ScipResolverInput,
) -> ResolvedScipConfig:
    return _resolve_scip_inputs_from(modules, inputs)
```

but there’s no strong reason to, since there are no existing callers to preserve.

**Critically:** the entire `if cfg is not None or isinstance(modules_or_cfg, ScipIngestStepConfig): ...` block, plus all `modules_or_cfg` parameters, should be deleted. That’s the actual legacy path.

---

### Step 4 – Drop unused imports and references

At the top of `_scip_resolver.py`, remove:

```python
from importlib import import_module
from codeintel.config import ScipIngestStepConfig
from codeintel.storage.gateway import StorageGateway
from codeintel.storage.module_index import load_module_map
```

After that, the file should only import:

* `Sequence`, `dataclass`, `Path`
* `ModuleRecord`, `ScipPathConfig`, `ScipResolverInput`, `ResolvedScipConfig` as defined locally.

If you had any docstrings that mention “legacy config object path” or `cfg_source`, scrub or rephrase them to describe only the explicit, value‑based interface.

---

### Step 5 – Update re‑exports in `ingestion.infrastructure_utilities.__init__`

`ingestion/infrastructure_utilities/__init__.py` has:

```python
from codeintel.ingestion.infrastructure_utilities._scip_resolver import (
    ResolvedScipConfig,
    ScipPathConfig,
    ScipResolverInput,
    resolve_scip_inputs,
)
```

You can leave this as is; just ensure the `resolve_scip_inputs` import still matches the new function signature `(modules, inputs)` and not `(gateway, modules_or_cfg, inputs)`.

Also, if `__all__` in that module lists `"resolve_scip_inputs"`, no change needed besides correctly exporting the simplified function.

---

## 4. Tests & callers to update

Because nothing in `ingestion` or `config` calls the resolver today, the only code you’ll need to update is:

* **Tests in this repo** (which you said you’ll handle)
* Any external scripts that might have been using `_scip_resolver` directly

Here’s what to look for and how to change it.

### 4.1 Old signature vs new

**Old (legacy‑aware) signature:**

```python
resolve_scip_inputs(
    gateway: StorageGateway,
    modules_or_cfg: Sequence[ModuleRecord] | ScipIngestStepConfig,
    inputs: ScipResolverInput,
) -> ResolvedScipConfig
```

**New explicit‑only signature:**

```python
resolve_scip_inputs(
    modules: Sequence[ModuleRecord] | None,
    inputs: ScipResolverInput,
) -> ResolvedScipConfig
```

So:

* Remove `gateway` argument everywhere.
* Replace any `modules_or_cfg` that might be a `ScipIngestStepConfig` with a real `modules` list (or move that config into building `ScipResolverInput` + `modules` directly).

### 4.2 Example migration: legacy config object call

If you had something like:

```python
cfg = ScipIngestStepConfig(...)  # legacy style
inputs = ScipResolverInput(cfg=cfg)
resolved = resolve_scip_inputs(gateway, cfg, inputs)
```

Replace it with:

```python
paths = ScipPathConfig(
    repo_root=cfg.repo_root,
    build_dir=cfg.build_dir,
    document_output_dir=cfg.document_output_dir,
    scip_python_bin=cfg.binaries.scip_python,
    scip_bin=cfg.binaries.scip,
)
modules: Sequence[ModuleRecord] = my_module_list  # from wherever tests used to get it

inputs = ScipResolverInput.build(
    repo=cfg.repo,
    commit=cfg.commit,
    paths=paths,
    modules=modules,
)
resolved = resolve_scip_inputs(modules, inputs)
```

You can of course inline this; the important bit is:

* `ScipIngestStepConfig` no longer flows into `_scip_resolver`.
* Everything passed is just plain values and module records.

### 4.3 Example migration: explicit path caller

If tests already used the explicit path but via the old signature:

```python
modules = [...]
inputs = ScipResolverInput.build(repo=repo, commit=commit, paths=paths, modules=None)

resolved = resolve_scip_inputs(gateway, modules, inputs)
```

Becomes:

```python
modules = [...]
inputs = ScipResolverInput.build(repo=repo, commit=commit, paths=paths, modules=None)

resolved = resolve_scip_inputs(modules, inputs)
```

(Simple: drop `gateway` and keep the rest.)

### 4.4 Update expectations about `ResolvedScipConfig`

If tests assert on `cfg_source` or `cfg`, they’ll need to change:

* Remove assertions like `resolved.cfg_source == "legacy"` or `"explicit"`.
* Remove assertions on `resolved.cfg` being an instance of `ScipIngestStepConfig`.

Instead, assert on:

* Structural fields: `resolved.repo`, `resolved.commit`, `resolved.repo_root`, etc.
* `resolved.modules` containing the modules you passed.

---

## Final checklist

You’re “fully migrated away from legacy SCIP config path” when:

* [ ] `_scip_resolver.py` no longer imports `ScipIngestStepConfig`, `StorageGateway`, or `load_module_map`.
* [ ] `ResolvedScipConfig` has **no** `cfg_source` or `cfg` fields.
* [ ] `ScipResolverInput` has **no** `cfg` field.
* [ ] `resolve_scip_inputs` takes only `(modules, inputs)` and **never** branches on `ScipIngestStepConfig`.
* [ ] There is no string `"legacy"` in `_scip_resolver.py` other than maybe in a historical doc comment that you intentionally kept.
* [ ] Grep for `ScipIngestStepConfig` shows it only in `config.steps_ingestion`, `config.builder`, and `config.__init__` (where it’s the step config), not in ingestion infrastructure.
* [ ] All resolver‑using tests are updated to pass explicit values + module lists.

Once that’s done, your SCIP ingestion world is:

* `ConfigBuilder` → `ScipIngestStepConfig` (for pipeline configuration)
* Plugin → `ScipIngestConfig` + `ScipIngestStep` for runtime
* `ScipResolverInput` / `resolve_scip_inputs` as a **pure helper** that only understands explicit value inputs — **no legacy branch at all**.

If you’d like, we can next do a similarly aggressive plan for either:

* The **CLI argparse compatibility** layer, or
* The **analytics/graphs `from_kwargs` compat constructors**

…whichever you want to kill off next.
