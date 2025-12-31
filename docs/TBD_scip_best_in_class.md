## Cross-repo dependency linking playbook (Python + scip-python + SCIP): make external package “Go to definition” work end-to-end

### The core invariant (what must match for cross-repo to work)

SCIP cross-repo navigation ultimately hinges on **symbol identity**, and for external deps that identity includes a **package name + package version** component. The scip-python team explicitly frames cross-repo as “stable references to your dependencies” and gives the canonical example symbol: `scip-python python PyYAML 6.0 yaml/dump().`—then states navigation works **if you also indexed the PyYAML repo/package for PyYAML 6.0**. ([Sourcegraph][1])

More generally, Sourcegraph’s indexer team states the invariant bluntly:

1. “For every symbol, we need to identify the project name + version.”
2. “Every repo needs to be consistently indexed with the same name + version pair.” ([GitHub][2])

For Python, scip-python normally infers `(package name, version)` from `pip` metadata; it also supports an explicit `--environment` manifest if you don’t want pip calls. ([GitHub][3])

So the playbook is: **extract the exact dependency package triples scip-python will emit, then ensure you have indexes for those exact triples (name+version) for the dependency source repos you want to navigate into.**

---

# 1) Understand what scip-python emits for dependencies (your “ground truth”)

scip-python:

* uses `pip` to determine package names/versions unless you provide `--environment` JSON ([GitHub][3])
* warns that distribution name ≠ import module name (PyYAML vs `import yaml`) ([GitHub][3])
* states the **dependency `version` is used to generate stable references** to external packages ([GitHub][3])

### Practical interpretation

If your consumer repo is indexed in an environment where `PyYAML` metadata says version `"6.0"`, then the consumer index will contain symbols with `(manager="python", name="PyYAML", version="6.0")`. If you index the PyYAML source repo but set `--project-name pyyaml` or `--project-version 6` you will *not* match; cross-repo won’t resolve.

---

# 2) Step-by-step “make it work” pipeline

## Step 2.1 — Produce a dependency manifest from the *exact* consumer environment

Treat the consumer environment as the **source of truth** for:

* exact distribution name string,
* exact version string.

You can generate an scip-python-compatible `--environment` file directly from `importlib.metadata`. This doubles as:

* a reproducibility artifact, and
* the input to your dependency-indexing scheduler.

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from importlib.metadata import distributions
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class PyPkg:
    name: str        # distribution name, e.g. "PyYAML"
    version: str     # exact metadata string, e.g. "6.0"
    files: list[str] # relative paths inside site-packages

def build_environment_manifest() -> list[dict[str, Any]]:
    pkgs: list[PyPkg] = []
    for dist in distributions():
        name = dist.metadata.get("Name")
        if not name:
            continue
        version = dist.version  # exact metadata string
        files = [str(f) for f in (dist.files or [])]
        pkgs.append(PyPkg(name=name, version=version, files=files))

    # scip-python expects list[{name, version, files}] :contentReference[oaicite:6]{index=6}
    return [{"name": p.name, "version": p.version, "files": p.files} for p in sorted(pkgs, key=lambda x: x.name)]

def write_env_json(path: Path) -> None:
    data = build_environment_manifest()
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")

write_env_json(Path("env.json"))
```

This matches the documented environment schema (`name`, `version`, `files`) and semantics. ([GitHub][3])

### Why you should do this even if you’ll let scip-python use pip

Because now you can:

* diff dependency sets across builds,
* guarantee stable name/version strings for scheduling,
* run the consumer index with `--environment env.json` to eliminate “pip drift” as a variable. ([GitHub][3])

---

## Step 2.2 — Index the consumer repo using the frozen environment manifest

```bash
# consumer repo
scip-python index . \
  --project-name "github.com/acme/myrepo" \
  --project-version "$(git rev-parse HEAD)" \
  --environment env.json
```

`--environment` is explicitly supported to avoid pip calls. ([GitHub][3])

---

## Step 2.3 — For each dependency you care about, obtain its *source* at that exact version

You need a **source tree** that corresponds to the distribution you installed. Options:

### A) Map distribution → upstream VCS repo + tag (most common)

Maintain a mapping like:

```python
PYPI_TO_GIT = {
    "PyYAML": ("https://github.com/yaml/pyyaml.git", "6.0"),
    "requests": ("https://github.com/psf/requests.git", "v2.32.3"),
    # ...
}
```

### B) Vendor sdists into an internal “pypi-sources” repo

If you don’t want to depend on upstream VCS availability, download sdists and store them in a mono-repo you control.

### C) For internal/private packages, index your internal source repo directly

This is usually easiest—just ensure the project’s packaging metadata (and thus installed distribution name/version) matches the scip-python `--project-name`/`--project-version` you use when indexing that repo.

---

## Step 2.4 — Index the dependency repo with **project-name = distribution name** and **project-version = distribution version**

This is the single most common failure point.

From scip-python’s perspective, your dependency repo index must present itself as the package triple it expects. The scip-clang cross-repo checklist explicitly says every symbol needs name+version and repos must be indexed consistently under that pair. ([GitHub][2])

Example for PyYAML:

```bash
# repo checked out at the PyYAML 6.0 sources
scip-python index . \
  --project-name "PyYAML" \
  --project-version "6.0"
```

Now, the consumer symbol `scip-python python PyYAML 6.0 yaml/dump().` has a matching definition symbol in the PyYAML index, enabling navigation. ([Sourcegraph][1])

### When `--project-namespace` matters

If you index “dependency source bundles” that contain multiple packages, use `--project-namespace` to isolate symbol spaces. scip-python documents this explicitly as a cross-repo aid. ([GitHub][3])

---

## Step 2.5 — Upload both indexes into your code intelligence system

For Sourcegraph-style ingestion, scip-python’s README uses:

```bash
src code-intel upload
```

([GitHub][3])

Key idea: **repository name in the code host is not the same thing as SCIP project-name.**
You can host PyYAML sources in a repo called `github.com/acme/pypi/PyYAML`, but you still must index it with `--project-name PyYAML --project-version 6.0` to satisfy symbol identity.

---

# 3) “Do I need to index *every* dependency?” (a pragmatic answer)

Not necessarily. Index dependencies you actually want to navigate into.

However, note:

* scip-python will happily emit stable refs for *all* installed packages it can resolve. ([GitHub][3])
* if you don’t upload matching indexes for those dependencies, navigation will stop at “external reference without definition.”

This is similar in spirit to scip-clang’s cross-repo story: you must provide package name+version info so the system can identify which symbol belongs to which package. ([Sourcegraph][4])

A common rollout:

1. Start with top-level deps (from lockfile), then
2. add transitive deps when navigation gaps appear.

---

# 4) Automated verification: prove symbol compatibility before you ever open Sourcegraph

## 4.1 Extract the set of referenced external package triples from a consumer `index.scip`

This uses only symbol string tokenization (cheap).

```python
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator, Optional, Tuple

# Minimal SCIP symbol tokenizer: scheme manager name version are space-delimited,
# but spaces inside tokens are encoded as double spaces (SCIP symbol format).
def split_tokens_double_space(s: str) -> list[str]:
    out, cur = [], []
    i = 0
    while i < len(s):
        if s[i] == " ":
            if i + 1 < len(s) and s[i + 1] == " ":
                cur.append(" ")
                i += 2
            else:
                out.append("".join(cur))
                cur = []
                i += 1
        else:
            cur.append(s[i])
            i += 1
    out.append("".join(cur))
    return out

def symbol_package_triple(symbol: str) -> Optional[Tuple[str, str, str]]:
    if symbol.startswith("local "):
        return None
    toks = split_tokens_double_space(symbol)
    if len(toks) < 4:
        return None
    _, manager, name, version = toks[:4]  # scheme is toks[0]
    return (manager, name, version)

def load_index_symbols(index) -> Iterator[str]:
    # index: scip.Index parsed via protobuf bindings
    for doc in index.documents:
        for occ in doc.occurrences:
            if occ.symbol:
                yield occ.symbol

def external_packages_in_consumer(index, consumer_project_name: str, consumer_project_version: str) -> set[Tuple[str, str, str]]:
    consumer_pkg = ("python", consumer_project_name, consumer_project_version)
    out: set[Tuple[str, str, str]] = set()
    for sym in load_index_symbols(index):
        pkg = symbol_package_triple(sym)
        if pkg and pkg != consumer_pkg:
            out.add(pkg)
    return out
```

You can now compare this set against the set of dependency indexes you’ve actually uploaded/produced.

## 4.2 Detect “will never link” mismatches early

If consumer wants `("python","PyYAML","6.0")` but your dependency repo was indexed as `("python","pyyaml","6.0")` or version `("python","PyYAML","6")`, you can flag it immediately.

---

# 5) The three recurring failure modes (and how to harden)

## 5.1 Distribution name mismatches (PyYAML vs yaml)

scip-python explicitly documents that `name` is the **package/distribution name**, which may not equal the import module name. ([GitHub][3])
**Fix:** always use the distribution name from metadata (`importlib.metadata.version(...)` / `dist.metadata["Name"]`) as the canonical `--project-name` for dependency indexing.

## 5.2 Name normalization differences (case / hyphen / underscore)

Python packaging ecosystems treat names as equivalent under normalization rules (lowercase and replace runs of `[-_.]` with `-`). ([Python Enhancement Proposals (PEPs)][5])
**But SCIP matching is string-based.** If scip-python emits `PyYAML`, don’t index the dependency repo as `pyyaml`.
**Hardening tactic:** store *both*:

* `dist_name_raw` (exact metadata “Name”) → used for `--project-name`
* `dist_name_norm` (PEP 503 normalize) → used for lookup keys in your own mapping tables

```python
import re
def pep503_normalize(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()  # from PEP 503 :contentReference[oaicite:18]{index=18}
```

## 5.3 Version string mismatches (“1” vs “1.0”)

Python’s version scheme allows multiple equivalent forms (PEP 440 / packaging spec), but string forms can differ. ([packaging.python.org][6])
**Fix:** use the exact version string from installed metadata for:

* the consumer’s dependency references (automatically),
* the dependency repo’s `--project-version` (you control this—match it exactly).

Do *not* reformat versions unless you reformat them *everywhere*.

---

# 6) Scaling strategy: a “dependency index farm”

If you run a code indexing service, the robust architecture looks like:

1. **Consumer job**:

   * create env (`env.json`) and index consumer repo with that env ([GitHub][3])
   * extract package triples referenced externally
2. **Resolver job**:

   * for each `(python, dist_name, dist_version)` triple:

     * find a source mapping (repo URL + ref, or sdist artifact)
     * check if you already have an index for that triple
3. **Dependency job**:

   * checkout source at matching version
   * run `scip-python index` with `--project-name dist_name --project-version dist_version`
   * upload
4. **Validation job**:

   * run the “symbol triple mismatch” scan above
   * optionally, UI smoke test via tool queries

This directly operationalizes the “consistent name+version pairs” requirement. ([GitHub][2])

---

# 7) Why “auto detection” matters (and why you shouldn’t wait for it)

The scip-python team explicitly tracks improving inference of project name/version from `pyproject.toml` / `setup.py`, noting it would enable cross-repo navigation “out-of-the-box” rather than local-only. ([GitHub][7])
Until that’s fully automatic, your pipeline should **explicitly set** project-name/version for any repo you want to function as a dependency target.

---

## If you want next

I can turn this into a repo-ready “dependency indexing orchestrator” module that:

* extracts `env.json`,
* produces a deterministic list of package triples,
* resolves packages to sources (configurable mapping strategy),
* indexes + uploads missing ones,
* and emits a compact report: “X deps referenced, Y indexed, Z mismatched by name/version.”

[1]: https://sourcegraph.com/blog/scip-python "scip-python: a precise Python indexer | Sourcegraph Blog"
[2]: https://github.com/sourcegraph/scip-clang/issues/184 "Cross-repository code navigation · Issue #184 · sourcegraph/scip-clang · GitHub"
[3]: https://github.com/sourcegraph/scip-python "GitHub - sourcegraph/scip-python: SCIP indexer for Python"
[4]: https://sourcegraph.com/blog/c-cpp-cross-repo "Cross-repository precise code navigation for C and C++ | Sourcegraph Blog"
[5]: https://peps.python.org/pep-0503/?utm_source=chatgpt.com "PEP 503 – Simple Repository API | peps.python.org"
[6]: https://packaging.python.org/specifications/version-specifiers/?utm_source=chatgpt.com "Version specifiers"
[7]: https://github.com/sourcegraph/scip-python/issues/109 "Improve project name and version detection · Issue #109 · sourcegraph/scip-python · GitHub"


Below is a codebase-specific, “best‑in‑class” opportunity inventory focused on **how you invoke scip-python, how you persist/parse its outputs, and how you exploit SCIP data downstream**—with explicit callouts to the concrete places in your Phase4 `src/` implementation where changes slot in cleanly.

I’ll assume your intent is: **SCIP is a first-class symbol indexing substrate** (not just a one-off export), and Hamilton is your orchestration spine.

---

## 1) Make scip-python invocation *explicitly “production‑grade”* (project identity, determinism, scope, env)

### 1.1 Always set `--project-name` (and make it stable)

**Current state in your code:** your scip-python invocation (via `ToolService.run_scip_full()` → `ScipIndexerPlugin._run_scip_python()`) does **not** pass `--project-name` today (it builds `["index", <target_base>, "--output", <output>]`).

**Why it matters:**
SCIP’s entire cross-index “addressing” story is anchored on the package identity portion of symbols and metadata. scip-python’s canonical usage includes passing `--project-name`. ([Gitee][1])

**Best-in-class improvement:**

* Define a single “project identity resolver”:

  * `project_name = env.snapshot.repo` (or a normalized slug)
  * optionally `project_namespace` (see 1.3)
* Ensure this identity is consistent across:

  * indexing,
  * artifact naming/location,
  * DB rows,
  * downstream resolvers.

**Where to implement:**

* `src/codeintel/ingestion/engine/scip.py` → `_run_scip_python()` should accept `project_name` and append `--project-name=<...>` (or `--project-name`, `<...>`).
* `src/codeintel/ingestion/engine/service.py` → `run_scip_full()` should accept/forward `project_name`.

---

### 1.2 Always set `--project-version` (commit SHA is the obvious default)

**Current state:** no `--project-version`.

**Why it matters:**

* Cross-repo linking and “which revision did this symbol come from?” require versioning.
* Even for *your* internal pipeline, it’s the cleanest join key between:

  * `index.scip`,
  * `core.scip_*` rows,
  * any artifact store,
  * any “index registry” you build later.

scip-python’s autoindex example explicitly uses `--project-version`. ([Gitee][1])

**Best-in-class improvement:**

* Default `project_version = env.snapshot.commit` (full SHA).
* Optionally support semver-ish or “monorepo subproject” versions, but commit SHA is the rock-solid baseline.

**Where:**

* Same places as 1.1.

---

### 1.3 Support `--project-namespace` for monorepos / path-prefixed projects

**Current state:** not used.

**Why it matters:**
If you index multiple “logical projects” from one repo, or you mount repos into a larger namespace, symbol collisions and ambiguous resolution become real. scip-python supports `--project-namespace` to prefix all generated symbols. ([Gitee][1])

**Best-in-class improvement:**

* Introduce a config field like `scip.project_namespace` (optional).
* Use it when:

  * the repo is indexed under a mount path that doesn’t match its internal package layout,
  * multiple repos share symbol space (your “centralizing” architecture suggests you’ll get here).

**Where:**

* Add to a target options dataclass (see §2), then forward into `_run_scip_python()`.

---

### 1.4 Align scip-python’s *indexing scope* with your Hamilton scan scope (use `--target-only`)

**Current state:**

* Your repo scanning defaults to a “src/ root” profile (`default_code_profile`) and your `core.modules` table is derived from that.
* Your scip indexing runs at `repo_root` and will happily emit documents for *any* Python under the root unless constrained.

**Why it matters:**
If scip emits occurrences for files your other targets never ingest, you’ll get:

* “orphaned” symbols/occurrences,
* mismatched module maps,
* graph targets that silently underperform (because their joins/filtering don’t align).

scip-python supports `--target-only` to restrict indexing to a directory. ([Gitee][1])

**Best-in-class improvement:**

* Derive `target_only` directly from your scan profile (or `core.snapshots.source_root` if you treat that as canonical).
* For your repo, it’s likely `--target-only=src` (and maybe `--target-only=tests` depending on goals).

**Where:**

* You already have the concept in `ScipIngestOptions.scope_paths` (currently unused).
* Implement: `t__scip__run()` loads scip options, converts scope paths into `--target-only` args.

---

### 1.5 Make dependency identity stable with `--environment` (stop relying on “whatever pip sees”)

**Current state:** no environment file flow.

**Why it matters:**
scip-python uses `pip` to infer package names/versions/files for external dependency symbol stability; if the environment doesn’t match the repo’s lockfile (or pip can’t see packages), external symbols degrade. scip-python supports `--environment` to supply an explicit package inventory and skip pip inspection. ([Gitee][1])

**Best-in-class improvement:**
Add a Hamilton branch that produces **an environment manifest** for scip-python:

* `env.json` containing `[{name, version, files:[...]}]` entries. ([Gitee][1])
* This becomes:

  * a reproducibility anchor (same repo+commit yields same external symbol package IDs),
  * a cache key input (see §3.2),
  * a stepping stone to “dependency index store” (see §7).

**Where:**

* New target or subnodes under `scip`:

  * `scip__environment_manifest_path`
  * `scip__environment_manifest_artifact`
* Then pass `--environment=<path>` into `_run_scip_python()`.

---

### 1.6 Bake in memory controls for large repos (`NODE_OPTIONS`)

**Current state:** no automatic Node memory tuning.

**Why it matters:**
scip-python is Node-based and can OOM on large projects. The official guidance is to increase heap via `NODE_OPTIONS="--max-old-space-size=8192"` when needed. ([Gitee][1])

**Best-in-class improvement:**

* Introduce config for:

  * `scip.node_options` (default empty),
  * or “auto-scale by repo size” heuristics (coarser but practical).
* Apply in the environment you pass to ToolRunner:

  * `env = {**tools_config.env, "NODE_OPTIONS": "..."}`
  * and record it in run metadata.

---

## 2) Actually *use* your scip target options surface (right now it’s effectively dead)

### 2.1 Load and hash a real `Scip*Options` object inside `t__scip__run`

**Current state:**
`t__scip__run()` does not call `load_target_options()` or include a scip options hash in its “skip” logic. You *do* have `ScipIngestOptions` defined (`build/hamilton/native/options/ingestion.py`) but it isn’t wired into scip execution.

**Why it matters:**
A best-in-class index pipeline must be able to vary (and cache correctly) across:

* scope,
* environment mode,
* identity/versioning,
* timeouts,
* debug outputs,
* parsing strategy.

**Best-in-class improvement:**

* In `t__scip__run()`, do:

  * `opts = load_target_options(env, "scip", ScipIngestOptions)`
  * compute `options_hash_for_target("scip", opts)`
  * incorporate into skip/hashing and into tool invocation

**Follow-on:**
Evolve `ScipIngestOptions` into **two** dataclasses:

* `ScipIndexOptions` (CLI flags + env + identity)
* `ScipParseOptions` (protobuf vs json, field selection, diagnostics, etc.)

---

### 2.2 Extend your options to include the real scip-python knobs

Given scip-python’s documented surface, the options you’re missing that you should model explicitly:

* `project_name` ([Gitee][1])
* `project_version` ([Gitee][1])
* `project_namespace` ([Gitee][1])
* `target_only: list[str]` ([Gitee][1])
* `environment_file: Path | None` ([Gitee][1])
* `node_options: str | None` ([Gitee][1])

Then: make the Hamilton scip target *authoritative* for constructing scip-python args.

---

## 3) Fix (or harden) caching + rebuild correctness around scip artifacts

### 3.1 Remove the “if index.scip and index.json exist, skip run” short-circuit

**Current state:** in `t__scip__run()` you do:

```python
if output_scip.exists() and output_json.exists():
    return ScipRunResult(success=True, index_path=..., json_path=...)
```

This happens **after** `executor.should_skip()` has returned False.

**Why it matters:**
This introduces a correctness hole:

* manifest says “recompute” (inputs changed),
* but stale artifacts exist from an old run,
* you silently reuse them.

**Best-in-class improvement:**

* Delete this existence-based early return.
* If you want “resume semantics”, make it explicit:

  * verify the artifact’s embedded metadata (project/version/tool info) matches current run identity before reuse,
  * otherwise rebuild.

---

### 3.2 Include toolchain + config fingerprints in your cache keys

Right now your build input hash strategy is mostly driven by commit + dependencies, not by “scip-python version / Node version / pyrightconfig / env.json”.

**Why it matters:**
If you upgrade scip-python, indices can change without repo code changing. A deterministic pipeline treats the *indexer toolchain* as part of the input.

**Best-in-class improvement:**
Create a Hamilton node (or preflight step) that captures:

* scip-python version (and ideally its git SHA, if available),
* Node version,
* scip CLI version (if you keep using it),
* hash of `pyrightconfig.json` (and/or other config files),
* hash of `env.json` (if you use `--environment`).

Then fold that fingerprint into:

* `options_hash_for_target("scip", ...)` or
* `file_state_hash` for the target input hash.

---

## 4) Stop round-tripping through `scip print --json` for production ingestion (protobuf-first)

### 4.1 Parse `index.scip` directly via protobuf in Python (make JSON export optional)

**Current state:**

* You always run `scip print --json index.scip` to create `index.json`.
* Then you parse JSON (currently via `parse_scip_json_file`).

**Why it matters:**

* JSON export for large repos is expensive (time + disk + memory).
* You’re throwing away a lot of schema richness (relationships, diagnostics, syntax kinds, etc.).
* You’re also carrying an extra binary dependency (`scip` CLI) when you could decode with protobuf in-process.

**How to do it cleanly:**

* Vendor or fetch `scip.proto` and generate Python bindings once.
* Then parse:

  * `Index.ParseFromString(open("index.scip","rb").read())`
  * iterate `index.documents`, `doc.occurrences`, `doc.symbols`, `index.metadata`

Sourcegraph documents decoding a SCIP index using `protoc --decode` against `scip.proto`, which is the same schema you’d generate Python bindings from. ([help.sourcegraph.com][2])

**Best-in-class structure in your Hamilton scip target:**

* `scip__index_artifact` → always exists
* `scip__decoded_index` → protobuf decode (no JSON)
* `scip__json_artifact` → optional debug/inspection path (behind config)

---

### 4.2 If you keep JSON: make parsing streaming + schema-correct

If you must keep `index.json`:

* avoid `json.loads(read_text(...))` for giant files
* consider streaming parse (`ijson`) or chunked decode

Also: fix the correctness issues you currently have in the parsers:

* `ingestion/engine/results.py::ScipIndexResult.from_json_documents()` expects snake_case keys like `relative_path` and `symbol_roles`, but scip JSON commonly uses camelCase like `relativePath` and `symbolRoles` (your *other* parser handles both). This is why `t__scip__run()` often returns zero parsed documents and forces `t__scip__ingest()` to re-parse from disk.

---

## 5) Ingest more SCIP schema surface area (you’re leaving capability on the floor)

Right now you only persist:

* `core.scip_symbols`: (rel_path, symbol, documentation)
* `core.scip_occurrences`: (rel_path, symbol, range, roles)

**Best-in-class additions (tables or columns):**

### 5.1 Metadata / toolchain provenance

SCIP metadata includes tool info (name/version) and other run-level settings. Sourcegraph shows `metadata.tool_info.*` in decoded output. ([help.sourcegraph.com][2])

Persist it as:

* `core.scip_metadata` keyed by (repo, commit)

  * tool name/version
  * text encoding
  * project root, etc.

### 5.2 SymbolInformation richness

Persist (or at least optionally persist):

* `symbolInformation.kind`
* `signatureDocumentation`
* `relationships`

These are essential if you want:

* “show signature”
* “implements / implemented by”
* “type hierarchy / related symbol” navigation

### 5.3 Occurrence fields beyond range/roles

If you parse protobuf, you can ingest:

* `syntax_kind` (identifier vs keyword etc)
* `diagnostics` (if present)
* `enclosing_range`

These are key for:

* better UI highlighting,
* filtering to “real identifiers” only,
* surfacing index-time issues as quality signals.

---

## 6) Fix + deepen downstream exploitation in your graph targets

### 6.1 Fix symbol use extraction: you’re selecting a non-existent column

In `build/hamilton/native/graphs/graph_targets.py::_load_symbol_occurrences()` you select:

```python
.select(scip_tbl.symbol, scip_tbl.rel_path, scip_tbl.line, scip_tbl.roles)
```

But your `core.scip_occurrences` schema is `start_line`, not `line`. This likely causes the loader to throw and return `[]`, which means the entire symbol_uses target becomes a no-op.

**Best-in-class improvement:**

* Select `start_line` (and ideally `start_col`) and update `SymbolOccurrence` to carry both.
* Use `(rel_path, start_line, start_col)` as the stable position identity.

This is not just a bugfix—carrying columns unlocks the GOID crosswalk mapping in §6.3.

---

### 6.2 Use scip occurrences to upgrade call graph resolution (you already have a placeholder)

Your call graph code has scaffolding for scip-based resolution (e.g., `scip_candidates_by_use_path` is currently empty in `call_graph.py`).

**Best-in-class improvement:**

* Use scip definition occurrences to build a **definition map** keyed by symbol → (rel_path, range)
* Use scip reference occurrences at call sites to resolve the callee symbol, then map to GOID via crosswalk (§6.3)
* That yields a graph that is vastly better than pure CST heuristics in dynamic Python.

---

### 6.3 Populate the `goid_crosswalk.scip_symbol` column (it’s currently always None)

Your schema explicitly anticipates linking GOIDs to SCIP symbols, but your current GOID extraction writes `scip_symbol=None`.

**Best-in-class improvement:**
Add a Hamilton target (or subgraph) that links:

* GOID definitions (file path + def range)
  to
* scip definition occurrences (rel_path + range + roles “Definition”)

This is the single highest-leverage join in your system because it lets every GOID-native graph/view become SCIP-addressable—and vice versa.

---

## 7) Turn cross-repo dependency linking into an *actual pipeline*, not just a theory

The scip-python docs + blog make it clear: scip-python can generate stable external package symbol references using package name+version, which is the prerequisite for dependency linking. ([Gitee][1])

**But**: stable references only become *go-to-definition* across repos/packages if you can resolve those external symbols to an index for that package+version.

### 7.1 Build (or integrate) a “package index registry”

Best-in-class architecture:

* Maintain a store keyed by `(package_name, package_version)` → `index.scip` artifact pointer + metadata.
* When indexing a repo:

  * extract all external package references from scip symbols,
  * ensure the corresponding package indices exist (build if missing),
  * resolve cross-package definitions by lookup.

### 7.2 Make the environment manifest the contract boundary

If you adopt `--environment`, your env.json becomes the canonical input describing:

* which packages exist,
* which versions,
* which files belong to each.

That means you can drive “index these dependencies” deterministically from that manifest.

---

## 8) Intensify Hamilton alignment: make scip indexing a richer DAG (and more cacheable)

Your scip target already has a good start (run → ingest → materialize). To push it “best-in-class Hamilton”:

### 8.1 Decompose tool execution into explicit nodes

Instead of a monolithic `run_scip_full()`:

* `scip__args` (pure)
* `scip__run_index` (tool node → index.scip)
* `scip__decode_index` (pure / compute)
* `scip__emit_json` (optional tool/debug node)
* `scip__rows_symbols`, `scip__rows_occurrences`, `scip__rows_relationships`, etc.

This gives you:

* clearer caching boundaries,
* better telemetry,
* simpler retries.

### 8.2 Use profile-driven parameterization

You already have the concept of execution profiles/options in the system. Extend it so “scip-fast” vs “scip-full” changes:

* target-only scope,
* whether env.json is required,
* whether JSON artifact is emitted,
* ingestion depth (base vs full schema).

---

## 9) Testing + quality gates for SCIP integration (what you’re missing for “best-in-class”)

### 9.1 Golden index tests across scip-python versions

* Keep a tiny fixture repo.
* Run scip indexing in CI (containerized).
* Assert:

  * document counts,
  * stable symbol identities,
  * presence of key metadata/toolinfo,
  * your DB tables aren’t empty.

### 9.2 Schema correctness tests for edge cases

Specifically validate your parsers against:

* range encodings (3 vs 4 element)
* symbolRoles vs symbol_roles
* metadata present/absent

### 9.3 Performance regression gates

Track:

* index time,
* JSON size (if emitted),
* decode+ingest time,
* row counts,
* peak memory.

---

## Environment assumptions you should formalize (and preflight)

Based on scip-python’s documented prerequisites and typical SCIP decoding workflows:

* Linux runtime (your assumption)
* **Python ≥ 3.10** for scip-python usage expectations ([Gitee][1])
* **Node ≥ 16** (scip-python requirement) ([Gitee][1])
* `scip-python` installed (npm global or pinned in toolchain) ([Gitee][1])
* If you parse protobuf in Python:

  * protobuf runtime (`google.protobuf`) and generated `scip_pb2`
  * optionally `protoc` available for regeneration/debug decode workflows ([help.sourcegraph.com][2])
* If you keep `scip print --json`:

  * `scip` CLI installed and version-pinned (or at least version-recorded)

---

## The “top 5” changes I’d do first in *your* repo (highest ROI)

1. **Add `--project-name` + `--project-version`** to scip-python invocation (commit SHA version). ([Gitee][1])
2. **Delete the “existing index files == success” early return** in `t__scip__run()` (stale artifact hazard).
3. **Fix `_load_symbol_occurrences()` to select `start_line` (and add `start_col`)** so symbol_uses actually works.
4. **Move ingestion to protobuf-first decode of `index.scip`**, keep JSON export optional. ([help.sourcegraph.com][2])
5. **Implement `--environment` support via a Hamilton env-manifest node**, then you have the foundation for real cross-dependency linking. ([Gitee][1])

---

If you want, my next step can be: propose an explicit **Hamilton sub-DAG design** for “scip indexing + decode + ingestion + goid crosswalk enrichment + dependency registry hooks”, with the exact node signatures and where they plug into your existing targets (and what tables/artifacts get produced).

[1]: https://gitee.com/mirrors_sourcegraph/scip-python?skip_mobile=true "scip-python: SCIP indexer for Python"
[2]: https://help.sourcegraph.com/hc/en-us/articles/15045932124941-Decoding-SCIP-index-file "Decoding SCIP index file – Sourcegraph Help Center"



Below is an explicit **Hamilton sub-DAG design** that cleanly composes the full workflow you described:

* **SCIP indexing** (scip-python + scip)
* **Decode** (JSON + optional protobuf)
* **Ingest** into DuckDB
* **GOID crosswalk enrichment** (attach GOIDs ⇄ SCIP symbols deterministically)
* **Dependency registry hooks** (so external package symbol resolution can work end-to-end)

I’m going to express this in the idioms your repo already uses (native targets, `TargetRunRecord` dependencies, `SaveToObjectMetadataDecorator`, `DuckDBRowsSaver` / `DuckDBIbisTableSaver`, file artifacts, and “inputs bundle” structs).

---

## Assumptions (explicit)

### Runtime / platform

* Linux runtime (POSIX paths, subprocess behavior assumed).
* Python runtime supports `tomllib` (Py ≥ 3.11). Your repo appears to be Py 3.13 based on compiled artifacts in the zip.

### Tooling available on PATH (or configured via `ToolsConfig`)

* `scip-python` (the Python indexer) accessible via `ToolsConfig.scip_python_bin` (default `"scip-python"`).
* `scip` CLI accessible via `ToolsConfig.scip_bin` (default `"scip"`).

### Optional: protobuf decode path

If you want a **protobuf-native decode** (instead of `scip print --json`), assume:

* `protoc` installed (or `buf`, but `protoc` is sufficient).
* Python package `protobuf` installed.
* The SCIP protobuf schema (`scip.proto`) is available (vendored or fetched during build) and compiled into something like `scip_pb2.py`.

This is optional; the sub-DAG below works fully with `scip print --json`.

---

## Where this plugs into your existing targets

### Current relevant targets in your codebase

* `t__modules` (core module tracking)
* `t__scip` (SCIP artifacts + `core.scip_symbols`, `core.scip_occurrences`)
* `t__goids` (GOIDs + base `core.goid_crosswalk` with `scip_symbol` currently `NULL`)
* `t__symbol_uses` (derives `graph.symbol_use_edges` from SCIP occurrences)

### Proposed additions (minimal but “best-in-class”)

1. **New target:** `t__dependency_registry`

   * Produces `core.dependency_registry` (+ manifest artifact)
   * Optionally ensures dependency indexes exist under `build/scip/deps/...`

2. **Enhance existing target:** `t__scip`

   * Use the dependency registry as an upstream dependency (to ensure consistent package/version metadata is available).
   * Expand scip-python invocation surface: `--project-version`, pyright config pass-through, etc.
   * Emit a “hook” dataset: `core.scip_external_symbol_uses` (what external projects are referenced), so you can validate dependency coverage without creating DAG cycles.

3. **New target:** `t__scip_goid_crosswalk`

   * Depends on `t__scip` + `t__goids`
   * Produces a join artifact/table that makes SCIP symbols resolvable to GOIDs:

     * `core.scip_symbol_goids` (recommended)
     * optionally `core.goid_crosswalk_enriched` (same as crosswalk but `scip_symbol` populated)

### Target-level dependency shape

```text
t__modules
  ├── t__dependency_registry
  ├── t__scip            (depends on modules + dependency_registry)
  └── t__goids           (depends on modules)

t__scip_goid_crosswalk   (depends on scip + goids)

(optional)
t__symbol_uses           (can additionally depend on scip_goid_crosswalk to fill goid columns)
```

---

## Sub-DAG design (node-by-node), with exact signatures

I’m showing these as **concrete function signatures** you can drop into new/modified native modules, following your repo’s established patterns.

---

# 1) Dependency registry hooks target

## Target contract

**Target name:** `dependency_registry`
**Outputs:**

* Table: `core.dependency_registry`
* Artifact: `dependency_registry_manifest.json`
* (Side artifact directory, not necessarily contract-tracked): `build/scip/deps/<pkg>/<ver>/index.scip`

### Suggested table schema (high-level)

`core.dependency_registry` columns (suggested):

* `repo, commit` (snapshot keys)
* `package_name` (normalized)
* `package_version` (resolved/pinned)
* `ecosystem` (e.g., `"pypi"`)
* `dist_path` (where code was indexed from, if available)
* `scip_index_path` (path to index.scip in your artifact tree)
* `scip_json_path` (path to index.json if you also export)
* `indexed_at`

---

## `native/ingestion/dependency_registry.py` (new)

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
import json
import re
import tomllib

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.execution_result import ExecutionResult
from codeintel.build.hamilton.native.executor import NativeTargetExecutor, TOOL_EXECUTION
from codeintel.build.hamilton.native.materialization_records import (
    record_from_duckdb_materializations,
    record_from_file_artifact_materializations,
)
from codeintel.build.hamilton.materializers import DuckDBRowsSaver, FileArtifactSaver
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.target_decorators import codeintel_target, TargetSpecDescriptor
from codeintel.build.hamilton.tagging import tag_tool, tag_compute, tag_helper
from codeintel.build.schemas.column_resolution import deferred_columns_for_table_key
from hamilton.function_modifiers import SaveToObjectMetadataDecorator, source, value

from codeintel.build.targets import TargetGraph
from codeintel.build.hamilton.run_records import TargetRunRecord


DEPENDENCY_REGISTRY_TARGET = "dependency_registry"
DEPENDENCY_REGISTRY_TABLE_KEY = "core.dependency_registry"
DEPENDENCY_REGISTRY_MANIFEST_ARTIFACT = "dependency_registry_manifest"


@dataclass(frozen=True)
class DependencySpec:
    name: str
    version: str | None  # if unpinned, this may be None until lock-resolution


@dataclass(frozen=True)
class DependencyRegistryResult:
    result: ExecutionResult
    deps: tuple[DependencySpec, ...] = ()
    manifest_path: Path | None = None


@tag_tool(domain="ingestion", target=DEPENDENCY_REGISTRY_TARGET)
def t__dependency_registry__resolve(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
) -> DependencyRegistryResult:
    """
    Resolve pinned dependency set for the snapshot.

    Must produce a versioned set if you want cross-repo symbol linking to work
    (project_name + project_version must match dependency index metadata).
    """
    if t__modules.status != "succeeded":
        return DependencyRegistryResult(
            result=ExecutionResult.failed(f"Upstream modules failed: {t__modules.error}")
        )

    executor = NativeTargetExecutor.for_target(env, graph, DEPENDENCY_REGISTRY_TARGET)
    if executor.should_skip():
        return DependencyRegistryResult(result=ExecutionResult.skip("dependency_registry skipped"))

    repo_root = env.snapshot.repo_root
    if repo_root is None:
        return DependencyRegistryResult(result=ExecutionResult.failed("Missing repo_root"))

    # Minimal example: parse pyproject.toml [project] dependencies.
    # Best-in-class: prefer lock files (poetry.lock, uv.lock, pdm.lock, requirements.lock).
    pyproject = repo_root / "pyproject.toml"
    if not pyproject.exists():
        return DependencyRegistryResult(
            result=ExecutionResult.ok(table_counts={DEPENDENCY_REGISTRY_TABLE_KEY: 0}),
            deps=(),
        )

    raw = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    deps_raw = (raw.get("project", {}) or {}).get("dependencies", []) or []

    # Super-light requirement parsing. Replace with packaging.requirements.Requirement if allowed.
    req_name = re.compile(r"^([A-Za-z0-9_.-]+)")
    parsed: list[DependencySpec] = []
    for s in deps_raw:
        m = req_name.match(str(s).strip())
        if not m:
            continue
        name = m.group(1).lower().replace("_", "-")
        # version remains None unless you resolve via lock.
        parsed.append(DependencySpec(name=name, version=None))

    return DependencyRegistryResult(
        result=ExecutionResult.ok(table_counts={}),
        deps=tuple(parsed),
    )


@tag_helper(domain="ingestion", target=DEPENDENCY_REGISTRY_TARGET)
def dependency_registry__manifest_payload(
    env: BuildEnv,
    t__dependency_registry__resolve: DependencyRegistryResult,
) -> str:
    """
    Deterministic manifest artifact for downstream tooling.

    Even if you later add indexing, this becomes the stable “control plane”
    artifact other targets/services can consume.
    """
    payload = {
        "repo": env.snapshot.repo,
        "commit": env.snapshot.commit,
        "generated_at": datetime.now(UTC).isoformat(),
        "dependencies": [
            {"name": d.name, "version": d.version} for d in t__dependency_registry__resolve.deps
        ],
    }
    return json.dumps(payload, sort_keys=True, indent=2)


@tag_helper(domain="ingestion", target=DEPENDENCY_REGISTRY_TARGET)
def dependency_registry__manifest_path(env: BuildEnv) -> Path:
    return env.paths.build_dir / "dependency_registry_manifest.json"


@SaveToObjectMetadataDecorator(
    [FileArtifactSaver],
    output_name_=materialize_node(DEPENDENCY_REGISTRY_MANIFEST_ARTIFACT),
    env=source("env"),
    graph=source("graph"),
    target_name=value(DEPENDENCY_REGISTRY_TARGET),
    artifact_name=value(DEPENDENCY_REGISTRY_MANIFEST_ARTIFACT),
    output_path=source("dependency_registry__manifest_path"),
)
@tag_compute(domain="ingestion", target=DEPENDENCY_REGISTRY_TARGET)
def dependency_registry_manifest_artifact(
    dependency_registry__manifest_payload: str,
) -> str:
    return dependency_registry__manifest_payload


@SaveToObjectMetadataDecorator(
    [DuckDBRowsSaver],
    output_name_=materialize_node(DEPENDENCY_REGISTRY_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(DEPENDENCY_REGISTRY_TARGET),
    table_key=value(DEPENDENCY_REGISTRY_TABLE_KEY),
    columns=value(deferred_columns_for_table_key(DEPENDENCY_REGISTRY_TABLE_KEY)),
)
@tag_compute(domain="ingestion", target=DEPENDENCY_REGISTRY_TARGET)
def dependency_registry__rows(
    env: BuildEnv,
    t__dependency_registry__resolve: DependencyRegistryResult,
) -> tuple[tuple[object, ...], ...] | None:
    if not t__dependency_registry__resolve.result.success:
        return None

    now = datetime.now(UTC)
    rows: list[tuple[object, ...]] = []
    for dep in t__dependency_registry__resolve.deps:
        rows.append(
            (
                env.snapshot.repo,
                env.snapshot.commit,
                "pypi",
                dep.name,
                dep.version,
                None,  # dist_path (optional)
                None,  # scip_index_path (optional, filled if you index deps)
                None,  # scip_json_path (optional)
                now,
            )
        )
    return tuple(rows)


@tag_helper(domain="ingestion", target=DEPENDENCY_REGISTRY_TARGET)
def dependency_registry__materializations(
    m__artifact__dependency_registry_manifest: MaterializationMetadata,
    m__core__dependency_registry: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    return {
        DEPENDENCY_REGISTRY_MANIFEST_ARTIFACT: m__artifact__dependency_registry_manifest,
        DEPENDENCY_REGISTRY_TABLE_KEY: m__core__dependency_registry,
    }


@codeintel_target(
    domain="ingestion",
    target=DEPENDENCY_REGISTRY_TARGET,
    spec=TargetSpecDescriptor(execution=TOOL_EXECUTION),
)
def t__dependency_registry(
    env: BuildEnv,
    graph: TargetGraph,
    dependency_registry__materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """
    Finalize run record from the saver metadata.
    """
    # If you want a single record function: use record_from_duckdb_materializations
    # for table-only, or record_from_file_artifact_materializations for artifact-only.
    # For mixed outputs, follow your scip pattern (one file-artifact record w/ row_counts).
    return record_from_file_artifact_materializations(
        env=env,
        graph=graph,
        target_name=DEPENDENCY_REGISTRY_TARGET,
        materializations={DEPENDENCY_REGISTRY_MANIFEST_ARTIFACT: dependency_registry__materializations[
            DEPENDENCY_REGISTRY_MANIFEST_ARTIFACT
        ]},
        row_counts={DEPENDENCY_REGISTRY_TABLE_KEY: 0},  # or parse from duckdb metadata
    )
```

### Best-in-class extension point

Add a second tool node:

```python
@tag_tool(domain="ingestion", target=DEPENDENCY_REGISTRY_TARGET)
def t__dependency_registry__ensure_indexes(
    env: BuildEnv,
    graph: TargetGraph,
    t__dependency_registry__resolve: DependencyRegistryResult,
) -> DependencyRegistryResult:
    """
    For each resolved dep(name, version), ensure a dependency index exists at:
      build/scip/deps/<name>/<version>/index.scip

    This node is where you call scip-python against installed distributions
    or downloaded source/wheels.
    """
    ...
```

That node is the “hook” where you turn registry metadata into *actual* external SCIP indexes.

---

# 2) SCIP target upgrades (index + decode + ingest + dependency hooks)

You already have:

* `t__scip__run(env, graph, t__modules) -> ScipRunResult`
* `t__scip__ingest(env, t__modules, t__scip__run) -> ScipIngestResult`
* `scip__symbol_rows(...) -> rows | None` (materialized)
* `scip__occurrence_rows(...) -> rows | None` (materialized)
* file artifacts for `index.scip`, `index.json`

## The “best-in-class” modifications to **node signatures**

### Add dependency registry and options into the run node:

```python
@tag_tool(domain="ingestion", target=SCIP_TARGET_NAME)
def t__scip__run(
    env: BuildEnv,
    graph: TargetGraph,
    t__modules: TargetRunRecord,
    t__dependency_registry: TargetRunRecord,     # NEW: upstream dep registry
    scip__project_name: str,                     # NEW
    scip__project_version: str,                  # NEW
    scip__pyright_config_path: Path | None,      # NEW
    scip__extra_args: tuple[str, ...],           # NEW
) -> ScipRunResult:
    ...
```

### Add option computation nodes (pure helpers)

These should live in `native/ingestion/scip.py`:

```python
@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__project_name(env: BuildEnv) -> str:
    # Best practice: normalize to what you use as “project” in SCIP symbol format.
    # Common: repo slug or canonical package name.
    return env.snapshot.repo

@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__project_version(env: BuildEnv) -> str:
    # Best practice: commit SHA for repos, semantic version for packages.
    return env.snapshot.commit

@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__pyright_config_path(env: BuildEnv) -> Path | None:
    root = env.snapshot.repo_root
    if not root:
        return None
    p = root / "pyrightconfig.json"
    return p if p.exists() else None

@tag_helper(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__extra_args() -> tuple[str, ...]:
    # Hook for config-driven scip-python / pyright pass-through
    return ()
```

### Run node logic (key detail)

Inside `t__scip__run`, you need to pass these into your tool runner.

Right now, `ToolService.run_scip_full(...)` doesn’t accept them. Best-in-class is to extend:

```python
# codeintel/ingestion/engine/service.py
async def run_scip_full(
    self,
    repo_root: Path,
    output_scip: Path,
    *,
    project_name: str | None = None,
    project_version: str | None = None,
    pyright_config: Path | None = None,
    extra_args: Sequence[str] = (),
    rel_paths: Sequence[str] | None = None,
) -> ScipIndexResult:
    ...
```

…and thread those into `_run_scip_python(...)` args composition.

---

## Dependency “hooks” output from SCIP (no DAG cycle)

Add a *derived dataset* capturing what external projects appear in symbol strings:

**New table**: `core.scip_external_symbol_uses`

* `repo, commit`
* `external_project_name`
* `external_project_version` (if encoded/known)
* `count_occurrences`
* `first_seen_rel_path`

This is crucial because it tells you: “the repo references symbols from X@Y — do we have that dependency indexed?”

Node signatures:

```python
import ibis
import ibis.expr.types as ir
from codeintel.build.hamilton.native.ibis_helpers import filter_for_snapshot
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from hamilton.function_modifiers import SaveToObjectMetadataDecorator, source, value

SCIP_EXTERNAL_USES_TABLE_KEY = "core.scip_external_symbol_uses"

@SaveToObjectMetadataDecorator(
    [DuckDBIbisTableSaver],
    output_name_=materialize_node(SCIP_EXTERNAL_USES_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_TARGET_NAME),
    table_key=value(SCIP_EXTERNAL_USES_TABLE_KEY),
)
@tag_compute(domain="ingestion", target=SCIP_TARGET_NAME)
def scip__external_symbol_uses(
    env: BuildEnv,
    q__core__scip_occurrences: ir.Table,
) -> ir.Table:
    occ = filter_for_snapshot(q__core__scip_occurrences, env.snapshot)

    # TODO: replace with a proper SCIP symbol parser.
    # Heuristic placeholder: extract "project" prefix from symbol string.
    # Example if symbol looks like: "scip-python python <project> <version> ..."
    parts = occ.symbol.split(" ")
    project = ibis.ifelse(ibis.array_length(parts) >= 3, parts[2], ibis.literal("unknown"))
    version = ibis.ifelse(ibis.array_length(parts) >= 4, parts[3], ibis.literal(None))

    ext = occ.select(
        repo=occ.repo,
        commit=occ.commit,
        external_project_name=project,
        external_project_version=version,
        rel_path=occ.rel_path,
    )
    return (
        ext.group_by(["repo", "commit", "external_project_name", "external_project_version"])
           .aggregate(
               count_occurrences=ext.count(),
               first_seen_rel_path=ext.rel_path.min(),
           )
    )
```

This table is the “hook” that lets you measure whether dependency indexes exist (or need building) without forcing a cycle where the dependency registry depends on SCIP.

---

# 3) GOID crosswalk enrichment target (SCIP ⇄ GOID linkage)

This is the missing piece for **best-in-class symbol resolution** in your storage layer.

## Target contract

**Target name:** `scip_goid_crosswalk`
**Depends on:** `t__scip`, `t__goids` (implicitly via loader nodes if you use `q__*`)
**Outputs:**

* Table `core.scip_symbol_goids` (recommended)
* Optionally table `core.goid_crosswalk_enriched`

### Recommended output: `core.scip_symbol_goids`

Columns (suggested):

* `repo, commit`
* `scip_symbol`
* `goid_urn`
* `goid_h128`
* `file_path`
* `def_line_1b` (or `start_line_0b` + `start_line_1b`)
* `confidence` (float or enum: exact_line_match / fuzzy / none)
* `updated_at`

### Key algorithm (deterministic)

1. Filter SCIP occurrences to **definitions** (`roles & 1 != 0`)
2. Convert SCIP’s 0-based `start_line` → 1-based
3. Join:

   * `goid_crosswalk.file_path == scip_occ.rel_path`
   * `goid_crosswalk.start_line == scip_def_line_1b`
4. Join `goids.urn == goid_crosswalk.goid` to get `goid_h128`
5. Apply filters to avoid parameter symbols (at minimum: exclude `symbol.endswith(":")`)

## `native/graphs/scip_goid_crosswalk.py` (new)

```python
from __future__ import annotations

from datetime import UTC, datetime
import ibis
import ibis.expr.types as ir

from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.boundary_types import MaterializationMetadata
from codeintel.build.hamilton.materializers import DuckDBIbisTableSaver
from codeintel.build.hamilton.naming import materialize_node
from codeintel.build.hamilton.native.ibis_helpers import filter_tables_for_snapshot
from codeintel.build.hamilton.native.materialization_records import record_from_duckdb_materializations
from codeintel.build.hamilton.native.target_decorators import codeintel_target, TargetSpecDescriptor
from codeintel.build.hamilton.tagging import tag_compute, tag_helper
from codeintel.build.targets import TargetGraph
from codeintel.build.hamilton.run_records import TargetRunRecord
from hamilton.function_modifiers import SaveToObjectMetadataDecorator, source, value

SCIP_GOID_CROSSWALK_TARGET = "scip_goid_crosswalk"
SCIP_SYMBOL_GOIDS_TABLE_KEY = "core.scip_symbol_goids"


@tag_helper(domain="graphs", target=SCIP_GOID_CROSSWALK_TARGET)
def scip_goid_crosswalk__definition_occurrences(
    env: BuildEnv,
    q__core__scip_occurrences: ir.Table,
) -> ir.Table:
    """
    Keep only definition occurrences and normalize coordinates for joining.

    Important: Your core.scip_occurrences start_line appears to be 0-based.
    GOID rows use 1-based lineno. So: def_line_1b = start_line + 1.
    """
    (tables,) = (filter_tables_for_snapshot(env.snapshot, occ=q__core__scip_occurrences),)
    occ = tables["occ"]

    # roles bit 1 => definition
    is_def = occ.roles.bitwise_and(ibis.literal(1)) != 0

    # exclude parameter-like symbols (SemanticDB-style uses ":" for params)
    not_param = ~occ.symbol.endswith(":")

    return (
        occ.filter(is_def & not_param)
           .select(
               repo=occ.repo,
               commit=occ.commit,
               scip_symbol=occ.symbol,
               file_path=occ.rel_path,
               def_line_1b=occ.start_line + ibis.literal(1),
               def_col_0b=occ.start_col,
           )
    )


@SaveToObjectMetadataDecorator(
    [DuckDBIbisTableSaver],
    output_name_=materialize_node(SCIP_SYMBOL_GOIDS_TABLE_KEY),
    env=source("env"),
    graph=source("graph"),
    target_name=value(SCIP_GOID_CROSSWALK_TARGET),
    table_key=value(SCIP_SYMBOL_GOIDS_TABLE_KEY),
)
@tag_compute(domain="graphs", target=SCIP_GOID_CROSSWALK_TARGET)
def scip_symbol_goids(
    env: BuildEnv,
    q__core__goids: ir.Table,
    q__core__goid_crosswalk: ir.Table,
    scip_goid_crosswalk__definition_occurrences: ir.Table,
) -> ir.Table:
    """
    Join SCIP definition occurrences to GOIDs.

    This yields a direct mapping table that downstream serving/search can use:
      scip_symbol -> goid_h128 (+ file + line)
    """
    tables = filter_tables_for_snapshot(
        env.snapshot,
        goids=q__core__goids,
        cross=q__core__goid_crosswalk,
        defs=scip_goid_crosswalk__definition_occurrences,
    )
    goids = tables["goids"]
    cross = tables["cross"]
    defs = tables["defs"]

    joined = cross.left_join(
        defs,
        predicates=[
            cross.file_path == defs.file_path,
            cross.start_line == defs.def_line_1b,
        ],
    ).left_join(
        goids,
        predicates=[cross.goid == goids.urn],
    )

    now = ibis.literal(datetime.now(UTC))

    # “confidence” is currently a constant; you can enrich it by adding
    # additional join predicates (qualname matching, etc.)
    return joined.select(
        repo=cross.repo,
        commit=cross.commit,
        scip_symbol=defs.scip_symbol,
        goid_urn=cross.goid,
        goid_h128=goids.goid_h128,
        file_path=cross.file_path,
        def_line_1b=cross.start_line,
        confidence=ibis.literal("exact_line_match"),
        updated_at=now,
    ).filter(~joined.scip_symbol.isnull())


@tag_helper(domain="graphs", target=SCIP_GOID_CROSSWALK_TARGET)
def scip_goid_crosswalk__table_materializations(
    m__core__scip_symbol_goids: MaterializationMetadata,
) -> dict[str, MaterializationMetadata]:
    return {SCIP_SYMBOL_GOIDS_TABLE_KEY: m__core__scip_symbol_goids}


@codeintel_target(domain="graphs", target=SCIP_GOID_CROSSWALK_TARGET, spec=TargetSpecDescriptor())
def t__scip_goid_crosswalk(
    env: BuildEnv,
    graph: TargetGraph,
    t__scip: TargetRunRecord,    # explicit upstream check (optional if you rely on q__*)
    t__goids: TargetRunRecord,   # explicit upstream check (optional if you rely on q__*)
    scip_goid_crosswalk__table_materializations: dict[str, MaterializationMetadata],
) -> TargetRunRecord:
    """
    Create a proper run record from saver metadata.
    """
    # You can add explicit failure gating for upstream targets here.
    return record_from_duckdb_materializations(
        env=env,
        graph=graph,
        target_name=SCIP_GOID_CROSSWALK_TARGET,
        materializations=scip_goid_crosswalk__table_materializations,
    )
```

### Why `core.scip_symbol_goids` is better than mutating `core.goid_crosswalk`

* You avoid competing ownership of `core.goid_crosswalk` (currently produced by `t__goids`).
* You get a normalized “join table” that multiple downstream consumers can use (search index, symbol uses, UI links, etc.).
* You can iteratively improve matching confidence without breaking the base GOID extraction contract.

---

## Downstream integration points (what to update after this sub-DAG exists)

### 1) Search / serving (`storage/serving/search_index.py`)

You currently insert `ref_goid_h128 = NULL` for `scip_symbols`.

**Best-in-class:** join `core.scip_symbols` with `core.scip_symbol_goids` on `(repo, commit, symbol)` and set `ref_goid_h128` from the join.

### 2) Symbol uses edges (`t__symbol_uses__extract`)

If you want definition-use edges annotated with GOIDs deterministically:

* map “definition symbol” → goid via `core.scip_symbol_goids`
* (optional) map “use location” → enclosing goid via CST/AST spans (more involved)

### 3) Dependency coverage enforcement

Use:

* `core.scip_external_symbol_uses` (from SCIP)
* `core.dependency_registry` (from dependency_registry)
  to report:
* external projects referenced but not indexed
* version mismatches (referenced version vs indexed version)

---

## Complete list of tables / artifacts produced by this sub-DAG design

### Existing (from `t__scip`)

* Artifact: `build/scip/index.scip` (tracked as `scip_index`)
* Artifact: `build/scip/index.json` (tracked as `scip_json`)
* Table: `core.scip_symbols`
* Table: `core.scip_occurrences`

### New (dependency registry hooks)

* Table: `core.dependency_registry`
* Artifact: `build/dependency_registry_manifest.json` (tracked)
* Optional artifact directory (not contract-tracked by default):

  * `build/scip/deps/<package>/<version>/index.scip`
  * `build/scip/deps/<package>/<version>/index.json`

### New (crosswalk enrichment)

* Table: `core.scip_symbol_goids`
* Optional: `core.goid_crosswalk_enriched` (if you also want a denormalized materialization)

### New (dependency “hook” from SCIP; strongly recommended)

* Table: `core.scip_external_symbol_uses`

---

## Notes on “best-in-class” correctness edges (things you’ll want to implement alongside this)

1. **Fix range decoding correctness** in your SCIP JSON parsing path (`parse_scip_occurrences` currently ignores `endLine`).
2. Decide whether `core.scip_occurrences.start_line` is canonical **0-based or 1-based**; GOID extraction is clearly 1-based. If you keep SCIP 0-based (fine), always normalize at join boundaries (as shown).
3. Add a real SCIP symbol parser (instead of string splits) for:

   * detecting parameter symbols (avoid false GOID attachments)
   * extracting external project name/version robustly

---

If you want, I can follow up by writing the **exact `TableSchema` additions** for `output_registry.py` (`core.dependency_registry`, `core.scip_symbol_goids`, `core.scip_external_symbol_uses`) and the minimal patches needed in:

* `codeintel/ingestion/engine/scip.py` (thread `--project-version`, pyright config pass-through)
* `storage/serving/search_index.py` (populate `ref_goid_h128` for SCIP symbols)

…but the above is the full sub-DAG design with explicit node signatures and clear plug points into your current targets.
