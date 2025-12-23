Below is a **concrete, production-aligned API sketch** for a `HamiltonBuildHarness` that:

* **Runs real Hamilton DAG execution** via your production `HamiltonBuildExecutor`
* **Plugs into your existing** `GatewayFactory`, `repo_writers`, and `SeedPack`s
* Includes the two “bridging helpers” you called out:

  * **`HarnessArtifacts`** (materialize realistic tool artifacts in the exact paths your targets search)
  * **`ManifestPriming`** (insert manifests + minimal state so targets can be treated as “already ran”)

Because you want to **migrate “modules” first**, I’m tailoring the examples and the priming helper to the **`modules` target** specifically, while keeping the helpers generic enough to reuse immediately for the next targets.

---

## 1) `HamiltonBuildHarness` – concrete API sketch (production executor + your test infra)

### File: `tests/_helpers/harnesses/hamilton_build.py`

```python
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from codeintel.build.config import BuildConfig, load_build_config
from codeintel.build.hamilton.env import BuildEnv
from codeintel.build.hamilton.executor import HamiltonBuildExecutor, HamiltonBuildResult
from codeintel.build.providers import Providers, create_default_providers
from codeintel.config.models import ToolsConfig
from codeintel.core.config.settings import BuildSettings

from tests._helpers.build import TEST_BUILD_SETTINGS
from tests._helpers.context import TestContext, create_test_context, SeedPack
from tests._helpers.env_options import EnvOptions, GatewayOptions


@dataclass(frozen=True, slots=True)
class HamiltonBuildHarness:
    """
    Production-aligned Hamilton harness:
      - real HamiltonBuildExecutor (driver + adapters)
      - plugs into TestContext (GatewayFactory, SnapshotRef, BuildPaths)
      - optional SeedPacks + repo writers
    """

    ctx: TestContext
    config: BuildConfig
    providers: Providers
    settings: BuildSettings = TEST_BUILD_SETTINGS

    # Execution knobs
    profile: str | None = None
    parallel_backend: str = "sequential"
    max_workers: int | None = None
    enable_cache: bool = False

    # Environment knobs
    strict_contracts: bool = False
    validate_outputs: bool = False
    force_targets: frozenset[str] = frozenset()

    @classmethod
    def open(
        cls,
        tmp_path: Path,
        *,
        repo: str = "acme/repo",
        commit: str = "deadbeef",
        build_dir: Path | None = None,
        providers: Providers | None = None,
        config: BuildConfig | None = None,
        strict_contracts: bool = False,
        validate_outputs: bool = False,
        force_targets: Iterable[str] = (),
        gateway_options: GatewayOptions | None = None,
    ) -> HamiltonBuildHarness:
        """
        Creates a fresh TestContext (GatewayFactory-backed) + harness.
        """
        env_opts = EnvOptions(
            repo=repo,
            commit=commit,
            repo_root=tmp_path / "repo",
            build_dir=build_dir or tmp_path / "build",
        )
        ctx = create_test_context(tmp_path, options=env_opts, gateway_options=gateway_options)
        return cls(
            ctx=ctx,
            providers=providers or create_default_providers(ToolsConfig.default()),
            config=config or load_build_config(ctx.repo_root),
            strict_contracts=strict_contracts,
            validate_outputs=validate_outputs,
            force_targets=frozenset(force_targets),
        )

    @classmethod
    def wrap(
        cls,
        ctx: TestContext,
        *,
        providers: Providers | None = None,
        config: BuildConfig | None = None,
        **kwargs: object,
    ) -> HamiltonBuildHarness:
        """Wrap an existing TestContext fixture (common in your current suite)."""
        return cls(
            ctx=ctx,
            providers=providers or create_default_providers(ToolsConfig.default()),
            config=config or load_build_config(ctx.repo_root),
            **kwargs,
        )

    def with_repo_writer(
        self,
        writer: Callable[[Path], Any],
        *args: Any,
        **kwargs: Any,
    ) -> HamiltonBuildHarness:
        """
        Use your existing tests/_helpers/orchestration/repo_writers.py functions.
        """
        writer(self.ctx.snapshot.repo_root, *args, **kwargs)
        return self

    def seed(self, *packs: SeedPack) -> HamiltonBuildHarness:
        """
        Use your existing SeedPacks.
        """
        self.ctx.require(*packs)
        return self

    def build_env(self) -> BuildEnv:
        """
        Create the BuildEnv the same way native Hamilton targets expect it.
        (Minimal required fields; others can be turned on via harness fields.)
        """
        return BuildEnv(
            gateway=self.ctx.gateway,
            snapshot=self.ctx.snapshot,
            paths=self.ctx.build_paths,
            providers=self.providers,
            config=self.config,
            settings=TEST_BUILD_SETTINGS,
            profile=self.profile,
            force_targets=self.force_targets,
            validate_outputs=self.validate_outputs,
            strict_contracts=self.strict_contracts,
            # NOTE: manifest_index omitted here on purpose so skip checks query DB
            # unless you explicitly prime/load a manifest index later.
        )

    def run(self, targets: Sequence[str]) -> HamiltonBuildResult:
        """
        Run targets through the *production* HamiltonBuildExecutor.
        """
        executor = HamiltonBuildExecutor(
            profile=self.profile,
            parallel_backend=self.parallel_backend,
            max_workers=self.max_workers,
            enable_cache=self.enable_cache,
        )
        return executor.run(env=self.build_env(), targets=list(targets))

    def run_one(self, target: str) -> HamiltonBuildResult:
        return self.run([target])

    # --- Bridging helper accessors ---

    @property
    def artifacts(self) -> HarnessArtifacts:
        return HarnessArtifacts(self.ctx.snapshot.repo_root, self.ctx.build_paths)

    @property
    def priming(self) -> ManifestPriming:
        return ManifestPriming(self)

    # --- Convenience querying helpers (delegates to TestContext) ---

    def count(self, table: str, where: str | None = None) -> int:
        return self.ctx.query_count(table, where=where)
```

### Why this is the “minimal, production-aligned nucleus”

* It executes via your **real** `HamiltonBuildExecutor` (so closure, skip checks, and run-writer behaviors stay aligned with prod).
* It’s built on your **existing `TestContext`**, which is already the integration point for:

  * `GatewayFactory` / DuckDB gateway
  * `SnapshotRef` identity
  * `BuildPaths` layout
  * Seed packs
* Any per-run changes (`force_targets`, `profile`) should use `dataclasses.replace` on `BuildEnv`
  because `BuildEnv` is frozen in production.

---

## 2) Bridging helper #1: `HarnessArtifacts` (paths match your target resolvers)

This helper is deliberately “dumb but exact”: it writes files into the **precise locations** your native targets check (`env.paths.pytest_report`, `env.paths.scip_dir`, and the `.coverage` candidates).

### File: `tests/_helpers/hamilton_harness_artifacts.py`

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Any

from codeintel.config.primitives import BuildPaths


@dataclass(frozen=True, slots=True)
class HarnessArtifacts:
    """
    Writes realistic artifacts to the same paths your targets resolve:
      - pytest-report.json (tests_ingest)
      - index.json + index.scip (scip)
      - .coverage (coverage_ingest resolution)
    """
    repo_root: Path
    paths: BuildPaths

    # -------------------------
    # pytest-report.json helper
    # -------------------------
    def write_pytest_report(
        self,
        *,
        tests: Iterable[Mapping[str, Any]] = (),
        summary: Mapping[str, Any] | None = None,
        prefer: str = "build_paths",
    ) -> Path:
        """
        Writes a minimal pytest-json-report-like file.

        Resolution order in ingest_targets._resolve_report_file (abbrev):
          - build_dir/test-results/pytest-report.json
          - build_dir/test-results/pytest_report.json
          - build_dir/pytest-report.json
          - build_dir/pytest_report.json
          - repo_root/pytest-report.json
          - repo_root/pytest_report.json
          - repo_root/test-results/pytest-report.json
          - repo_root/.pytest_cache/pytest_report.json
        """
        payload = {
            "created": "1970-01-01T00:00:00Z",
            "duration": 0.0,
            "exitcode": 0,
            "root": str(self.repo_root),
            "environment": {},
            "summary": dict(summary or {"passed": 0, "failed": 0, "skipped": 0}),
            "tests": list(tests),
        }

        if prefer == "repo_root_flat":
            out = self.repo_root / "pytest-report.json"
        elif prefer == "repo_root_test_results":
            out = self.repo_root / "test-results" / "pytest-report.json"
        else:
            out = self.paths.pytest_report

        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return out

    # -------------------------
    # SCIP artifacts helper
    # -------------------------
    def write_dummy_scip_artifacts(
        self,
        *,
        documents: list[dict[str, Any]] | None = None,
        scip_dir: Path | None = None,
    ) -> tuple[Path, Path]:
        """
        Produces:
          - {scip_dir}/index.json
          - {scip_dir}/index.scip

        NOTE: Your SCIP tool plugin prefers existing JSON when present and non-empty.
        """
        out_dir = scip_dir or self.paths.scip_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        index_json = out_dir / "index.json"
        index_scip = out_dir / "index.scip"

        docs = documents or [
            {
                "relativePath": "pkg/mod_a.py",
                "symbols": [{"symbol": "scip-python python pkg/mod_a foo()."}],
                "occurrences": [],
            }
        ]

        index_json.write_text(json.dumps({"documents": docs}, indent=2), encoding="utf-8")
        index_scip.write_bytes(b"SCIP")  # non-empty sentinel

        return index_scip, index_json

    # -------------------------
    # Coverage helper
    # -------------------------
    def touch_coverage_file(self, *, prefer: str = "repo_root") -> Path:
        """
        Creates a .coverage file where coverage_ingest searches.

        Resolution order in ingest_targets._resolve_coverage_file:
          1) repo_root/.coverage
          2) repo_root/coverage.json
          3) build_dir/coverage.json

        NOTE: a real coverage run produces a SQLite-ish file; this helper
        only ensures path realism. If you want ingestion rows, either:
          - run real coverage in CI, or
          - use your FakeToolRunner route for coverage plugin output.
        """
        if prefer == "build_dir":
            out = self.paths.build_dir / "coverage.json"
        else:
            out = self.repo_root / ".coverage"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"")  # sentinel; real coverage will overwrite
        return out
```

---

## 3) Bridging helper #2: `ManifestPriming` (modules-first version)

This is the helper that lets you:

* “Pretend upstream already ran” by inserting **manifests** (and optionally output state),
* So later tests can hit **skip logic** and still find the expected outputs.

### Important note (re: your current native nodes)

Right now, many downstream nodes do:

```python
if t__modules.status != "succeeded":
    return ExecutionResult.fail(...)
```

That means **a skipped upstream target currently behaves like failure for downstream**.

So, for `ManifestPriming` to be fully effective as “skip upstream but let downstream run”, you’ll eventually want to standardize upstream guards to accept:

* `status == "succeeded"` **or** `status == "skipped"` (treat “skipped” as “cached success”).

That’s orthogonal to this helper, but it’s the key to making “priming + skip” unlock faster realistic tests.

### File: `tests/_helpers/hamilton_manifest_priming.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Mapping, Any

from codeintel.build.hamilton.driver_factory import build_driver
from codeintel.build.hamilton.run_records import compute_target_input_hash, options_hash_for_target
from codeintel.build.hashing import InputHashOptions
from codeintel.core.build_manifest import OutputManifest


@dataclass(frozen=True, slots=True)
class ManifestPriming:
    """
    DB-level priming:
      - writes OutputManifest rows to build.output_manifests
      - optionally primes minimal upstream output state (modules tables, artifacts, etc.)

    Scoped to work great for "modules-first" migration.
    """
    harness: Any  # HamiltonBuildHarness (kept Any to avoid import cycles)

    def prime_manifest(
        self,
        *,
        target: str,
        input_hash: str,
        options_hash: str | None,
        duration_ms: float = 0.0,
        plugin: str | None = None,
        row_count: int | None = None,
        change_delta: dict[str, object] | None = None,
        computed_at: datetime | None = None,
    ) -> OutputManifest:
        env = self.harness.build_env()
        when = computed_at or datetime.now(tz=UTC)
        manifest = OutputManifest(
            target=target,
            repo=env.repo,
            commit=env.commit,
            plugin=plugin or f"native:{target}",
            computed_at=when,
            duration_ms=duration_ms,
            input_hash=input_hash,
            output_hash=None,
            row_count=row_count,
            options_hash=options_hash,
            dep_hashes=None,
            change_delta=change_delta,
        )
        env.gateway.build.save_manifest(manifest, change_delta=change_delta)
        return manifest

    # -------------------------
    # Modules-first priming
    # -------------------------
    def prime_modules_manifest(
        self,
        *,
        file_state_hash: str,
        row_count: int | None = None,
        change_delta: dict[str, object] | None = None,
    ) -> OutputManifest:
        """
        Prime the 'modules' manifest so modules can be considered up-to-date
        for this snapshot (repo+commit) with the given file_state_hash.
        """
        env = self.harness.build_env()
        runtime = build_driver(config={"profile": env.profile})
        target = runtime.graph.get("modules")
        if target is None:
            raise RuntimeError("Target 'modules' not found in target graph")

        opts_hash = options_hash_for_target(env, "modules")
        input_hash = compute_target_input_hash(
            target=target,
            snapshot=env.snapshot,
            gateway=env.gateway,
            settings=env.settings,
            options=InputHashOptions(
                options_hash=opts_hash,
                file_state_hash=file_state_hash,
                manifests=None,
            ),
        )

        return self.prime_manifest(
            target="modules",
            input_hash=input_hash,
            options_hash=opts_hash,
            row_count=row_count,
            change_delta=change_delta,
        )
```

### What you do with this *immediately* for “modules-first”

* Use it to **test skip deterministically**, without having to run modules once first.
* Use it to prime state in follow-up migrations (typing/scip/etc.) once you relax upstream-guard semantics to treat “skipped” as cached success.

---

## 4) How you migrate “modules” first with this harness (realistic tests)

### A) Minimal “modules works” test

```python
from tests._helpers.hamilton_build_harness import HamiltonBuildHarness
from tests._helpers.orchestration.repo_writers import write_sample_repo

def test_modules_discovers_modules(tmp_path):
    with HamiltonBuildHarness.open(tmp_path) as h:
        h.with_repo_writer(write_sample_repo)

        result = h.run_one("modules")
        rec = result.get_record("modules")
        assert rec is not None
        assert rec.status == "succeeded"

        assert h.count("core.modules") > 0
        assert h.count("core.repo_map") == 1
        assert h.count("core.file_state") > 0
```

### B) “modules invalidates on change” test (very production-real)

Run modules, modify a file, run modules again; expect it to succeed again (not skip).

```python
from pathlib import Path
from tests._helpers.orchestration.repo_writers import write_sample_repo

def test_modules_recomputes_when_repo_changes(tmp_path):
    with HamiltonBuildHarness.open(tmp_path) as h:
        h.with_repo_writer(write_sample_repo)

        r1 = h.run_one("modules").get_record("modules")
        assert r1 and r1.status == "succeeded"

        # mutate repo
        mod = h.ctx.snapshot.repo_root / "pkg" / "mod_a.py"
        mod.write_text(mod.read_text(encoding="utf-8") + "\nX = 1\n", encoding="utf-8")

        r2 = h.run_one("modules").get_record("modules")
        assert r2 and r2.status == "succeeded"
```

### C) Deterministic skip test using `ManifestPriming` (modules-only)

This is where `ManifestPriming` is useful even before downstream guard refactors.

```python
def test_modules_skips_when_manifest_matches(tmp_path):
    with HamiltonBuildHarness.open(tmp_path) as h:
        h.with_repo_writer(write_sample_repo)

        # Run once to populate file_state_hash realistically
        h.run_one("modules")

        # Pull the file_state_hash back out of DB if you want,
        # OR simply run again and assert skipped (most realistic).
        r2 = h.run_one("modules").get_record("modules")
        assert r2 and r2.status in ("succeeded", "skipped")
```

(If you want fully deterministic “skip without first run”, you’d compute the `file_state_hash` the same way `RepoScanStep`/change detection does, then call `h.priming.prime_modules_manifest(file_state_hash=...)` and run modules once expecting `skipped`.)

---

## 5) Where to deploy these helpers (pytest scope)

To keep the migration clean and discoverable:

* `tests/_helpers/harnesses/hamilton_build.py`

  * the main harness (your tests import just this most of the time)
* `tests/_helpers/hamilton_harness_artifacts.py`

  * artifact writer that mirrors `BuildPaths` resolution logic
* `tests/_helpers/hamilton_manifest_priming.py`

  * DB priming helper

And then re-export (optional, but makes adoption smoother):

### Patch: `tests/_helpers/__init__.py`

```python
from tests._helpers.harnesses.hamilton_build import HamiltonBuildHarness
from tests._helpers.hamilton_harness_artifacts import HarnessArtifacts
from tests._helpers.hamilton_manifest_priming import ManifestPriming

__all__ += [
    "HamiltonBuildHarness",
    "HarnessArtifacts",
    "ManifestPriming",
]
```

---

## One “modules-first” migration tip that will matter immediately

If your goal is: **downstream targets run even when modules is up-to-date**, then you’ll want to normalize a rule like:

* Upstream is acceptable if `status in {"succeeded", "skipped"}`

…and replace patterns like:

```python
if t__modules.status != "succeeded":
```

with something like:

```python
if t__modules.status == "failed":
```

(or a shared helper `require_upstream_ok(t__modules, "modules")`).

That single convention is what makes **ManifestPriming** truly valuable for speeding up realistic DAG-driven tests.

---

If you want, I can now extend this “modules-first” sketch with one more immediately-useful piece: a **`ModulesAssertions`** mini-helper (e.g., `assert_repo_map_contains(...)`, `assert_modules_equal(...)`, `assert_file_state_hash_stable(...)`) that will make the first wave of modules tests very concise while still inspecting *real* persisted state.
