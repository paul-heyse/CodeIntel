
Below is a concrete “ModulesAssertions” mini-helper you can drop into your suite to make the first wave of **modules** tests extremely concise while still asserting against **real persisted state**:

* `core.modules` rows (module ↔ path)
* `core.repo_map.modules` JSON (module → path)
* `core.file_state` content hashes (path → blake2b via `HashChangeDetectionAdapter`)
* `build.output_manifests.change_delta["state_hash"]` (the stable, content-based state hash the pipeline uses for caching/skip semantics; this helper asserts it is present)

It’s designed so tests can say “here are the paths I expect exist” and the helper derives module names using the same convention as production (`path.replace("/", ".").removesuffix(".py")`).

---

## 1) New helper file: `tests/_helpers/assertions/modules.py`

```diff
*** /dev/null
--- b/tests/_helpers/assertions/modules.py
@@
+"""Assertions for the modules target and module inventory tables.
+
+This helper focuses on *persisted* state created by the modules target:
+  - core.modules
+  - core.repo_map
+  - core.file_state
+  - build.output_manifests(change_delta.state_hash)
+
+Primary goal:
+Make modules-first tests concise while still inspecting the same DB artifacts
+and hashing semantics production uses.
+"""
+
+from __future__ import annotations
+
+import json
+from dataclasses import dataclass
+from pathlib import Path
+from typing import TYPE_CHECKING
+
+from codeintel.config.primitives import SnapshotRef
+from codeintel.core.hashing.short import sha256_short
+from codeintel.core.paths import normalize_path
+from codeintel.ingestion.adapters.hash_change_detection import HashChangeDetectionAdapter
+from tests._helpers.assertions.common import format_assertion_message
+
+if TYPE_CHECKING:
+    from collections.abc import Iterable, Mapping, Sequence
+    from codeintel.storage.gateway import StorageGateway
+
+
+type ModuleMapSpec = Mapping[str, str] | Sequence[str] | Sequence[tuple[str, str]]
+
+
+def _derive_module_name_from_path(rel_path: str) -> str:
+    """Derive module name using the production naming convention."""
+    normalized = normalize_path(rel_path)
+    return normalized.replace("/", ".").removesuffix(".py")
+
+
+def _coerce_expected_module_map(expected: ModuleMapSpec) -> dict[str, str]:
+    """Coerce supported expected specs into {module -> normalized_path}."""
+    if isinstance(expected, Mapping):
+        return {str(k): normalize_path(str(v)) for k, v in expected.items()}
+
+    rows = list(expected)
+    if not rows:
+        return {}
+
+    first = rows[0]
+    if isinstance(first, str):
+        # Sequence[str] of rel paths
+        coerced: dict[str, str] = {}
+        for path in rows:  # type: ignore[assignment]
+            norm = normalize_path(str(path))
+            coerced[_derive_module_name_from_path(norm)] = norm
+        return coerced
+
+    if isinstance(first, tuple) and len(first) == 2:
+        # Sequence[(module, path)]
+        coerced = {}
+        for mod, path in rows:  # type: ignore[misc]
+            coerced[str(mod)] = normalize_path(str(path))
+        return coerced
+
+    msg = (
+        "Unsupported expected module spec. Use one of:\n"
+        "  - Mapping[module, path]\n"
+        "  - Sequence[path]\n"
+        "  - Sequence[(module, path)]\n"
+    )
+    raise TypeError(msg)
+
+
+def load_modules_module_map(
+    gateway: StorageGateway,
+    snapshot: SnapshotRef,
+) -> dict[str, str]:
+    """Load {module -> path} from core.modules for the snapshot."""
+    rows = gateway.con.execute(
+        """
+        SELECT module, path
+        FROM core.modules
+        WHERE repo = ? AND commit = ?
+        ORDER BY module
+        """,
+        [snapshot.repo, snapshot.commit],
+    ).fetchall()
+
+    module_to_path: dict[str, str] = {}
+    path_to_module: dict[str, str] = {}
+
+    for module, path in rows:
+        mod = str(module)
+        rel = normalize_path(str(path))
+
+        existing_path = module_to_path.get(mod)
+        if existing_path is not None and existing_path != rel:
+            raise AssertionError(
+                f"core.modules has duplicate module with different paths: "
+                f"module={mod!r} paths={[existing_path, rel]!r}"
+            )
+        module_to_path[mod] = rel
+
+        existing_mod = path_to_module.get(rel)
+        if existing_mod is not None and existing_mod != mod:
+            raise AssertionError(
+                f"core.modules has duplicate path with different modules: "
+                f"path={rel!r} modules={[existing_mod, mod]!r}"
+            )
+        path_to_module[rel] = mod
+
+    return module_to_path
+
+
+def _coerce_repo_map_modules_cell(value: object) -> dict[str, str]:
+    """Decode the repo_map.modules cell to a Python dict."""
+    if value is None:
+        return {}
+    if isinstance(value, dict):
+        return {str(k): normalize_path(str(v)) for k, v in value.items()}
+    if isinstance(value, (bytes, bytearray)):
+        value = value.decode("utf-8", errors="replace")
+    if isinstance(value, str):
+        raw = value.strip()
+        if not raw:
+            return {}
+        parsed = json.loads(raw)
+        if isinstance(parsed, dict):
+            return {str(k): normalize_path(str(v)) for k, v in parsed.items()}
+        raise AssertionError(f"core.repo_map.modules JSON was not an object: {type(parsed)}")
+    raise AssertionError(f"Unsupported core.repo_map.modules cell type: {type(value)}")
+
+
+def load_repo_map_modules(
+    gateway: StorageGateway,
+    snapshot: SnapshotRef,
+) -> dict[str, str]:
+    """Load {module -> path} from core.repo_map.modules for the snapshot."""
+    row = gateway.con.execute(
+        """
+        SELECT modules
+        FROM core.repo_map
+        WHERE repo = ? AND commit = ?
+        LIMIT 1
+        """,
+        [snapshot.repo, snapshot.commit],
+    ).fetchone()
+    if row is None:
+        raise AssertionError(f"core.repo_map row missing for {snapshot.repo}@{snapshot.commit}")
+    return _coerce_repo_map_modules_cell(row[0])
+
+
+def assert_modules_equal(
+    gateway: StorageGateway,
+    snapshot: SnapshotRef,
+    expected: ModuleMapSpec,
+    *,
+    allow_extra: bool = False,
+    message_prefix: str = "",
+) -> None:
+    """Assert core.modules matches expected module inventory."""
+    expected_map = _coerce_expected_module_map(expected)
+    actual_map = load_modules_module_map(gateway, snapshot)
+
+    if allow_extra:
+        missing = {k: v for k, v in expected_map.items() if actual_map.get(k) != v}
+        if missing:
+            raise AssertionError(
+                format_assertion_message(
+                    message_prefix,
+                    f"core.modules missing/mismatched expected entries: {missing}",
+                )
+            )
+        return
+
+    if actual_map != expected_map:
+        expected_keys = set(expected_map)
+        actual_keys = set(actual_map)
+        only_expected = sorted(expected_keys - actual_keys)
+        only_actual = sorted(actual_keys - expected_keys)
+        mismatched = sorted(
+            k for k in (expected_keys & actual_keys) if expected_map[k] != actual_map[k]
+        )
+        raise AssertionError(
+            format_assertion_message(
+                message_prefix,
+                "core.modules mismatch.\n"
+                f"  only_expected={only_expected}\n"
+                f"  only_actual={only_actual}\n"
+                f"  mismatched_paths={[(k, expected_map[k], actual_map[k]) for k in mismatched]}\n",
+            )
+        )
+
+
+def assert_repo_map_contains(
+    gateway: StorageGateway,
+    snapshot: SnapshotRef,
+    expected: ModuleMapSpec,
+    *,
+    strict: bool = False,
+    message_prefix: str = "",
+) -> None:
+    """Assert core.repo_map.modules contains expected module entries."""
+    expected_map = _coerce_expected_module_map(expected)
+    actual_map = load_repo_map_modules(gateway, snapshot)
+
+    if strict:
+        if actual_map != expected_map:
+            raise AssertionError(
+                format_assertion_message(
+                    message_prefix,
+                    f"core.repo_map.modules mismatch.\nexpected={expected_map}\nactual={actual_map}",
+                )
+            )
+        return
+
+    missing_or_mismatched = {
+        mod: path for mod, path in expected_map.items() if actual_map.get(mod) != path
+    }
+    if missing_or_mismatched:
+        raise AssertionError(
+            format_assertion_message(
+                message_prefix,
+                f"core.repo_map.modules missing/mismatched entries: {missing_or_mismatched}",
+            )
+        )
+
+
+def assert_repo_map_consistent_with_modules(
+    gateway: StorageGateway,
+    snapshot: SnapshotRef,
+    *,
+    message_prefix: str = "",
+) -> None:
+    """Assert repo_map.modules and core.modules describe the same inventory."""
+    modules_map = load_modules_module_map(gateway, snapshot)
+    repo_map = load_repo_map_modules(gateway, snapshot)
+    if modules_map != repo_map:
+        only_modules = sorted(set(modules_map) - set(repo_map))
+        only_repo_map = sorted(set(repo_map) - set(modules_map))
+        mismatched = sorted(
+            k for k in (set(modules_map) & set(repo_map)) if modules_map[k] != repo_map[k]
+        )
+        raise AssertionError(
+            format_assertion_message(
+                message_prefix,
+                "repo_map/modules inconsistency.\n"
+                f"  only_in_core.modules={only_modules}\n"
+                f"  only_in_core.repo_map={only_repo_map}\n"
+                f"  mismatched_paths={[(k, modules_map[k], repo_map[k]) for k in mismatched]}",
+            )
+        )
+
+
+def _compute_state_hash_from_content_hashes(path_to_hash: Mapping[str, str]) -> str:
+    """Compute stable state_hash the same way production does."""
+    payload = "|".join(
+        f"{rel_path}:{content_hash}" for rel_path, content_hash in sorted(path_to_hash.items())
+    )
+    if payload:
+        payload = f"{payload}|"
+    return sha256_short(payload, length=16, used_for_security=False)
+
+
+def compute_file_state_hash_from_table(
+    gateway: StorageGateway,
+    snapshot: SnapshotRef,
+    *,
+    language: str = "python",
+) -> str:
+    """Compute state hash using persisted core.file_state content_hash values."""
+    rows = gateway.con.execute(
+        """
+        SELECT rel_path, content_hash
+        FROM core.file_state
+        WHERE repo = ? AND commit = ? AND language = ?
+        """,
+        [snapshot.repo, snapshot.commit, language],
+    ).fetchall()
+    state = {normalize_path(str(p)): str(h) for p, h in rows}
+    return _compute_state_hash_from_content_hashes(state)
+
+
+def compute_file_state_hash_from_disk(
+    gateway: StorageGateway,
+    snapshot: SnapshotRef,
+) -> str:
+    """Compute state hash by hashing the on-disk files referenced by core.modules."""
+    modules_map = load_modules_module_map(gateway, snapshot)
+    path_to_hash: dict[str, str] = {}
+    for rel_path in modules_map.values():
+        abs_path = snapshot.repo_root / rel_path
+        if not abs_path.is_file():
+            raise AssertionError(f"Module path missing on disk: {rel_path} ({abs_path})")
+        digest = HashChangeDetectionAdapter.compute_file_digest(abs_path)
+        if digest is None:
+            raise AssertionError(f"Module path unreadable on disk: {rel_path} ({abs_path})")
+        path_to_hash[normalize_path(rel_path)] = digest.content_hash
+    return _compute_state_hash_from_content_hashes(path_to_hash)
+
+
+def assert_file_state_hash_stable(
+    gateway: StorageGateway,
+    snapshot: SnapshotRef,
+    *,
+    target: str = "modules",
+    previous: str | None = None,
+    verify_table: bool = True,
+    verify_disk: bool = True,
+    language: str = "python",
+    message_prefix: str = "",
+) -> str:
+    """Assert the persisted state_hash is consistent and stable.
+
+    Checks:
+      1) build.output_manifests(target).change_delta.state_hash exists
+      2) (optional) matches hash derived from core.file_state
+      3) (optional) matches hash recomputed from disk for files in core.modules
+      4) (optional) equals `previous` if provided
+
+    Returns the state_hash for convenient chaining in tests.
+    """
+    manifest = gateway.build.load_manifest(target=target, repo=snapshot.repo, commit=snapshot.commit)
+    if manifest is None:
+        raise AssertionError(
+            format_assertion_message(
+                message_prefix,
+                f"Missing build manifest for target={target} snapshot={snapshot.repo}@{snapshot.commit}",
+            )
+        )
+    delta = manifest.change_delta or {}
+    state_hash = delta.get("state_hash")
+    if not isinstance(state_hash, str) or not state_hash:
+        raise AssertionError(
+            format_assertion_message(
+                message_prefix,
+                f"Manifest missing change_delta.state_hash for target={target}. change_delta={delta}",
+            )
+        )
+
+    if previous is not None and state_hash != previous:
+        raise AssertionError(
+            format_assertion_message(
+                message_prefix,
+                f"state_hash changed unexpectedly: previous={previous} current={state_hash}",
+            )
+        )
+
+    if verify_table:
+        table_hash = compute_file_state_hash_from_table(gateway, snapshot, language=language)
+        if table_hash != state_hash:
+            raise AssertionError(
+                format_assertion_message(
+                    message_prefix,
+                    f"state_hash != hash(core.file_state): manifest={state_hash} table={table_hash}",
+                )
+            )
+
+    if verify_disk:
+        disk_hash = compute_file_state_hash_from_disk(gateway, snapshot)
+        if disk_hash != state_hash:
+            raise AssertionError(
+                format_assertion_message(
+                    message_prefix,
+                    f"state_hash != hash(disk): manifest={state_hash} disk={disk_hash}",
+                )
+            )
+
+    return state_hash
+
+
+@dataclass(frozen=True)
+class ModulesAssertions:
+    """Fluent wrapper for common module inventory assertions."""
+
+    gateway: StorageGateway
+    snapshot: SnapshotRef
+
+    def modules_equal(
+        self,
+        expected: ModuleMapSpec,
+        *,
+        allow_extra: bool = False,
+        message_prefix: str = "",
+    ) -> ModulesAssertions:
+        assert_modules_equal(
+            self.gateway,
+            self.snapshot,
+            expected,
+            allow_extra=allow_extra,
+            message_prefix=message_prefix,
+        )
+        return self
+
+    def repo_map_contains(
+        self,
+        expected: ModuleMapSpec,
+        *,
+        strict: bool = False,
+        message_prefix: str = "",
+    ) -> ModulesAssertions:
+        assert_repo_map_contains(
+            self.gateway,
+            self.snapshot,
+            expected,
+            strict=strict,
+            message_prefix=message_prefix,
+        )
+        return self
+
+    def inventory_consistent(self, *, message_prefix: str = "") -> ModulesAssertions:
+        assert_repo_map_consistent_with_modules(
+            self.gateway,
+            self.snapshot,
+            message_prefix=message_prefix,
+        )
+        return self
+
+    def file_state_hash_stable(
+        self,
+        *,
+        target: str = "modules",
+        previous: str | None = None,
+        verify_table: bool = True,
+        verify_disk: bool = True,
+        language: str = "python",
+        message_prefix: str = "",
+    ) -> str:
+        return assert_file_state_hash_stable(
+            self.gateway,
+            self.snapshot,
+            target=target,
+            previous=previous,
+            verify_table=verify_table,
+            verify_disk=verify_disk,
+            language=language,
+            message_prefix=message_prefix,
+        )
+
+
+__all__ = [
+    "ModulesAssertions",
+    "ModuleMapSpec",
+    "assert_file_state_hash_stable",
+    "assert_modules_equal",
+    "assert_repo_map_contains",
+    "assert_repo_map_consistent_with_modules",
+    "compute_file_state_hash_from_disk",
+    "compute_file_state_hash_from_table",
+    "load_modules_module_map",
+    "load_repo_map_modules",
+]
```

---

## 2) Export it from `tests/_helpers/assertions/__init__.py`

```diff
--- a/tests/_helpers/assertions/__init__.py
+++ b/tests/_helpers/assertions/__init__.py
@@
from tests._helpers.assertions.table_assertions import (
    assert_columns_not_null,
    assert_table_has_rows,
)
+from tests._helpers.assertions.modules import (
+    ModulesAssertions,
+    assert_file_state_hash_stable,
+    assert_modules_equal,
+    assert_repo_map_contains,
+    assert_repo_map_consistent_with_modules,
+    compute_file_state_hash_from_disk,
+    compute_file_state_hash_from_table,
+    load_modules_module_map,
+    load_repo_map_modules,
+)
@@
 __all__ = [
@@
    "assert_columns_not_null",
    "assert_table_has_rows",
+    "ModulesAssertions",
+    "assert_file_state_hash_stable",
+    "assert_modules_equal",
+    "assert_repo_map_contains",
+    "assert_repo_map_consistent_with_modules",
+    "compute_file_state_hash_from_disk",
+    "compute_file_state_hash_from_table",
+    "load_modules_module_map",
+    "load_repo_map_modules",
 ]
```

---

## 3) How you’d use it in “modules-first” tests

### A. Minimal “inventory correctness” test

```python
from tests._helpers.assertions import ModulesAssertions

def test_modules_writes_expected_inventory(harness) -> None:
    # harness: whatever you’re using to run the real modules target (HamiltonBuildHarness, etc.)
    # After harness.run("modules"), the DB should have core.modules/core.repo_map/core.file_state + manifest.

    harness.run_targets(["modules"])

    ModulesAssertions(harness.ctx.gateway, harness.ctx.snapshot).modules_equal(
        ["src/pkg/a.py", "src/pkg/b.py"]
    ).repo_map_contains(
        ["src/pkg/a.py", "src/pkg/b.py"]
    ).inventory_consistent()
```

### B. “state_hash stability across timestamp-only changes”

```python
from tests._helpers.assertions import ModulesAssertions

def test_modules_state_hash_ignores_mtime_changes(harness) -> None:
    harness.run_targets(["modules"])

    a = ModulesAssertions(harness.ctx.gateway, harness.ctx.snapshot)
    h1 = a.file_state_hash_stable()

    # touch without changing contents
    path = harness.ctx.snapshot.repo_root / "src/pkg/a.py"
    path.write_text(path.read_text(), encoding="utf-8")  # same content, new mtime

    harness.run_targets(["modules"])  # run again
    a.file_state_hash_stable(previous=h1)  # asserts unchanged
```

This is a high-signal test because it directly validates your “content-hash-only” caching semantics at the persisted-manifest level.

---

## 4) Where to deploy it in pytest scope

* **File:** `tests/_helpers/assertions/modules.py`
* **Export:** `tests/_helpers/assertions/__init__.py`

That keeps it consistent with:

* existing assertion helpers (`graphs.py`, `coverage_assertions.py`, `table_assertions.py`)
* the “tests/_helpers” organization you already have
* incremental migration: older tests can ignore it; new Hamilton-driven tests can immediately adopt it.

---
