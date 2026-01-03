"""Assertions for the modules target and module inventory tables."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from codeintel.config.primitives import SnapshotRef
from codeintel.core.hashing.short import sha256_short
from codeintel.core.paths import normalize_path
from codeintel.ingestion.adapters.hash_change_detection import (
    HashChangeDetectionAdapter,
)
from tests._helpers.assertions.common import format_assertion_message

if TYPE_CHECKING:
    from codeintel.storage.gateway import StorageGateway


type ModuleMapSpec = Mapping[str, str] | Sequence[str] | Sequence[tuple[str, str]]

EXPECTED_MODULE_TUPLE_LEN = 2


@dataclass(frozen=True)
class FileStateHashOptions:
    """Options for validating file state hashes."""

    target: str = "modules"
    previous: str | None = None
    verify_table: bool = True
    verify_disk: bool = True
    language: str = "python"
    message_prefix: str = ""


def _derive_module_name_from_path(rel_path: str) -> str:
    """Derive module name using the production naming convention.

    Returns
    -------
    str
        Module name derived from the normalized path.
    """
    normalized = normalize_path(rel_path)
    return normalized.replace("/", ".").removesuffix(".py")


def _coerce_expected_module_map(expected: ModuleMapSpec) -> dict[str, str]:
    """Coerce supported expected specs into {module -> normalized_path}.

    Returns
    -------
    dict[str, str]
        Normalized module inventory map.

    Raises
    ------
    TypeError
        If the expected spec is not one of the supported shapes.
    """
    if isinstance(expected, Mapping):
        return {str(k): normalize_path(str(v)) for k, v in expected.items()}

    rows = list(expected)
    if not rows:
        return {}

    first = rows[0]
    if isinstance(first, str):
        coerced: dict[str, str] = {}
        for path in rows:
            norm = normalize_path(str(path))
            coerced[_derive_module_name_from_path(norm)] = norm
        return coerced

    if isinstance(first, tuple) and len(first) == EXPECTED_MODULE_TUPLE_LEN:
        coerced = {}
        for mod, path in rows:
            coerced[str(mod)] = normalize_path(str(path))
        return coerced

    message = (
        "Unsupported expected module spec. Use one of:\n"
        "  - Mapping[module, path]\n"
        "  - Sequence[path]\n"
        "  - Sequence[(module, path)]\n"
    )
    raise TypeError(message)


def load_modules_module_map(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> dict[str, str]:
    """Load {module -> path} from core.modules for the snapshot.

    Returns
    -------
    dict[str, str]
        Mapping of module name to normalized relative path.

    Raises
    ------
    AssertionError
        If the table contains duplicate module or path entries.
    """
    rows = gateway.con.execute(
        """
        SELECT module, path
        FROM core.modules
        WHERE repo = ? AND commit = ?
        ORDER BY module
        """,
        [snapshot.repo, snapshot.commit],
    ).fetchall()

    module_to_path: dict[str, str] = {}
    path_to_module: dict[str, str] = {}

    for module, path in rows:
        mod = str(module)
        rel = normalize_path(str(path))

        existing_path = module_to_path.get(mod)
        if existing_path is not None and existing_path != rel:
            message = (
                "core.modules has duplicate module with different paths: "
                f"module={mod!r} paths={[existing_path, rel]!r}"
            )
            raise AssertionError(message)
        module_to_path[mod] = rel

        existing_mod = path_to_module.get(rel)
        if existing_mod is not None and existing_mod != mod:
            message = (
                "core.modules has duplicate path with different modules: "
                f"path={rel!r} modules={[existing_mod, mod]!r}"
            )
            raise AssertionError(message)
        path_to_module[rel] = mod

    return module_to_path


def _coerce_repo_map_modules_cell(value: object) -> dict[str, str]:
    """Decode the repo_map.modules cell to a Python dict.

    Returns
    -------
    dict[str, str]
        Mapping of module names to normalized relative paths.

    Raises
    ------
    AssertionError
        If the stored value is not a mapping-like representation.
    """
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(k): normalize_path(str(v)) for k, v in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        pairs = [
            item
            for item in value
            if isinstance(item, (list, tuple)) and len(item) == EXPECTED_MODULE_TUPLE_LEN
        ]
        if len(pairs) == len(value):
            return {str(k): normalize_path(str(v)) for k, v in pairs}
    if isinstance(value, (bytes, bytearray)):
        value = value.decode("utf-8", errors="replace")
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return {}
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return {str(k): normalize_path(str(v)) for k, v in parsed.items()}
        message = f"core.repo_map.modules JSON was not an object: {type(parsed)}"
        raise AssertionError(message)
    message = f"Unsupported core.repo_map.modules cell type: {type(value)}"
    raise AssertionError(message)


def load_repo_map_modules(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> dict[str, str]:
    """Load {module -> path} from core.repo_map.modules for the snapshot.

    Returns
    -------
    dict[str, str]
        Mapping of module name to normalized relative path.

    Raises
    ------
    AssertionError
        If the repo_map row is missing for the snapshot.
    """
    row = gateway.con.execute(
        """
        SELECT modules
        FROM core.repo_map
        WHERE repo = ? AND commit = ?
        LIMIT 1
        """,
        [snapshot.repo, snapshot.commit],
    ).fetchone()
    if row is None:
        message = f"core.repo_map row missing for {snapshot.repo}@{snapshot.commit}"
        raise AssertionError(message)
    return _coerce_repo_map_modules_cell(row[0])


def assert_modules_equal(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    expected: ModuleMapSpec,
    *,
    allow_extra: bool = False,
    message_prefix: str = "",
) -> None:
    """Assert core.modules matches expected module inventory.

    Raises
    ------
    AssertionError
        If the module inventory differs from the expected entries.
    """
    expected_map = _coerce_expected_module_map(expected)
    actual_map = load_modules_module_map(gateway, snapshot)

    if allow_extra:
        missing = {k: v for k, v in expected_map.items() if actual_map.get(k) != v}
        if missing:
            message = f"core.modules missing/mismatched expected entries: {missing}"
            raise AssertionError(format_assertion_message(message_prefix, message))
        return

    if actual_map != expected_map:
        expected_keys = set(expected_map)
        actual_keys = set(actual_map)
        only_expected = sorted(expected_keys - actual_keys)
        only_actual = sorted(actual_keys - expected_keys)
        mismatched = sorted(
            k for k in (expected_keys & actual_keys) if expected_map[k] != actual_map[k]
        )
        message = (
            "core.modules mismatch.\n"
            f"  only_expected={only_expected}\n"
            f"  only_actual={only_actual}\n"
            f"  mismatched_paths={[(k, expected_map[k], actual_map[k]) for k in mismatched]}\n"
        )
        raise AssertionError(format_assertion_message(message_prefix, message))


def assert_repo_map_contains(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    expected: ModuleMapSpec,
    *,
    strict: bool = False,
    message_prefix: str = "",
) -> None:
    """Assert core.repo_map.modules contains expected module entries.

    Raises
    ------
    AssertionError
        If expected entries are missing or mismatched.
    """
    expected_map = _coerce_expected_module_map(expected)
    actual_map = load_repo_map_modules(gateway, snapshot)

    if strict:
        if actual_map != expected_map:
            message = (
                f"core.repo_map.modules mismatch.\nexpected={expected_map}\nactual={actual_map}"
            )
            raise AssertionError(format_assertion_message(message_prefix, message))
        return

    missing_or_mismatched = {
        mod: path for mod, path in expected_map.items() if actual_map.get(mod) != path
    }
    if missing_or_mismatched:
        message = f"core.repo_map.modules missing/mismatched entries: {missing_or_mismatched}"
        raise AssertionError(format_assertion_message(message_prefix, message))


def assert_repo_map_consistent_with_modules(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    message_prefix: str = "",
) -> None:
    """Assert repo_map.modules and core.modules describe the same inventory.

    Raises
    ------
    AssertionError
        If the inventories are inconsistent.
    """
    modules_map = load_modules_module_map(gateway, snapshot)
    repo_map = load_repo_map_modules(gateway, snapshot)
    if modules_map != repo_map:
        only_modules = sorted(set(modules_map) - set(repo_map))
        only_repo_map = sorted(set(repo_map) - set(modules_map))
        mismatched = sorted(
            k for k in (set(modules_map) & set(repo_map)) if modules_map[k] != repo_map[k]
        )
        message = (
            "repo_map/modules inconsistency.\n"
            f"  only_in_core.modules={only_modules}\n"
            f"  only_in_core.repo_map={only_repo_map}\n"
            f"  mismatched_paths={[(k, modules_map[k], repo_map[k]) for k in mismatched]}"
        )
        raise AssertionError(format_assertion_message(message_prefix, message))


def _compute_state_hash_from_content_hashes(path_to_hash: Mapping[str, str]) -> str:
    """Compute stable state_hash the same way production does.

    Returns
    -------
    str
        Stable hash computed from the supplied content hashes.
    """
    payload = "|".join(
        f"{rel_path}:{content_hash}" for rel_path, content_hash in sorted(path_to_hash.items())
    )
    if payload:
        payload = f"{payload}|"
    return sha256_short(payload, length=16, used_for_security=False)


def compute_file_state_hash_from_table(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    *,
    language: str = "python",
) -> str:
    """Compute state hash using persisted core.file_state content_hash values.

    Returns
    -------
    str
        Stable state hash from the file_state table.
    """
    rows = gateway.con.execute(
        """
        SELECT rel_path, content_hash
        FROM core.file_state
        WHERE repo = ? AND commit = ? AND language = ?
        """,
        [snapshot.repo, snapshot.commit, language],
    ).fetchall()
    state = {normalize_path(str(p)): str(h) for p, h in rows}
    return _compute_state_hash_from_content_hashes(state)


def compute_file_state_hash_from_disk(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
) -> str:
    """Compute state hash by hashing the on-disk files referenced by core.modules.

    Returns
    -------
    str
        Stable state hash from the on-disk module files.

    Raises
    ------
    AssertionError
        If referenced module files are missing or unreadable.
    """
    modules_map = load_modules_module_map(gateway, snapshot)
    path_to_hash: dict[str, str] = {}
    for rel_path in modules_map.values():
        abs_path = snapshot.repo_root / rel_path
        if not abs_path.is_file():
            message = f"Module path missing on disk: {rel_path} ({abs_path})"
            raise AssertionError(message)
        digest = HashChangeDetectionAdapter.compute_file_digest(abs_path)
        if digest is None:
            message = f"Module path unreadable on disk: {rel_path} ({abs_path})"
            raise AssertionError(message)
        path_to_hash[normalize_path(rel_path)] = digest.content_hash
    return _compute_state_hash_from_content_hashes(path_to_hash)


def assert_file_state_hash_stable(
    gateway: StorageGateway,
    snapshot: SnapshotRef,
    options: FileStateHashOptions | None = None,
) -> str:
    """Assert the persisted state_hash is consistent and stable.

    Returns
    -------
    str
        The validated state hash from the build manifest.

    Raises
    ------
    AssertionError
        If the manifest is missing or the hash values are inconsistent.
    """
    resolved = options or FileStateHashOptions()
    manifest = gateway.build.load_manifest(
        target=resolved.target,
        repo=snapshot.repo,
        commit=snapshot.commit,
    )
    if manifest is None:
        message = (
            f"Missing build manifest for target={resolved.target} "
            f"snapshot={snapshot.repo}@{snapshot.commit}"
        )
        raise AssertionError(format_assertion_message(resolved.message_prefix, message))
    delta = manifest.change_delta or {}
    state_hash = delta.get("state_hash")
    if not isinstance(state_hash, str) or not state_hash:
        message = (
            f"Manifest missing change_delta.state_hash for target={resolved.target}. "
            f"change_delta={delta}"
        )
        raise AssertionError(format_assertion_message(resolved.message_prefix, message))

    if resolved.previous is not None and state_hash != resolved.previous:
        message = (
            f"state_hash changed unexpectedly: previous={resolved.previous} current={state_hash}"
        )
        raise AssertionError(format_assertion_message(resolved.message_prefix, message))

    if resolved.verify_table:
        table_hash = compute_file_state_hash_from_table(
            gateway,
            snapshot,
            language=resolved.language,
        )
        if table_hash != state_hash:
            message = (
                f"state_hash != hash(core.file_state): manifest={state_hash} table={table_hash}"
            )
            raise AssertionError(format_assertion_message(resolved.message_prefix, message))

    if resolved.verify_disk:
        disk_hash = compute_file_state_hash_from_disk(gateway, snapshot)
        if disk_hash != state_hash:
            message = f"state_hash != hash(disk): manifest={state_hash} disk={disk_hash}"
            raise AssertionError(format_assertion_message(resolved.message_prefix, message))

    return state_hash


@dataclass(frozen=True)
class ModulesAssertions:
    """Fluent wrapper for common module inventory assertions."""

    gateway: StorageGateway
    snapshot: SnapshotRef

    def modules_equal(
        self,
        expected: ModuleMapSpec,
        *,
        allow_extra: bool = False,
        message_prefix: str = "",
    ) -> ModulesAssertions:
        assert_modules_equal(
            self.gateway,
            self.snapshot,
            expected,
            allow_extra=allow_extra,
            message_prefix=message_prefix,
        )
        return self

    def repo_map_contains(
        self,
        expected: ModuleMapSpec,
        *,
        strict: bool = False,
        message_prefix: str = "",
    ) -> ModulesAssertions:
        assert_repo_map_contains(
            self.gateway,
            self.snapshot,
            expected,
            strict=strict,
            message_prefix=message_prefix,
        )
        return self

    def inventory_consistent(self, *, message_prefix: str = "") -> ModulesAssertions:
        assert_repo_map_consistent_with_modules(
            self.gateway,
            self.snapshot,
            message_prefix=message_prefix,
        )
        return self

    def file_state_hash_stable(
        self,
        *,
        options: FileStateHashOptions | None = None,
    ) -> str:
        return assert_file_state_hash_stable(self.gateway, self.snapshot, options)


__all__ = [
    "FileStateHashOptions",
    "ModuleMapSpec",
    "ModulesAssertions",
    "assert_file_state_hash_stable",
    "assert_modules_equal",
    "assert_repo_map_consistent_with_modules",
    "assert_repo_map_contains",
    "compute_file_state_hash_from_disk",
    "compute_file_state_hash_from_table",
    "load_modules_module_map",
    "load_repo_map_modules",
]
