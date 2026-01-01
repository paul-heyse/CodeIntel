
Below is a **drop-in “next increment”** that turns your existing `normalize_for_contract()` into a real **contract-boundary normalizer**, with:

* **FieldMask canonicalization** (all `google.protobuf.FieldMask` anywhere in the tree)
* **Selective unknown handling** via **message full-name allowlists** (glob patterns)
* A single deterministic **contract JSON policy** (version-adaptive)
* A unified **`assert_contract_equivalent(a, b)`** comparator (JSON diff on failure)

I’m giving this as **repo-ready files/patches**. You can copy/paste as-is.

---

## 0) Update your targets config

### `tests/protobuf/contract_targets.py` (add this block)

```python
# ----------------------------
# Contract-boundary policy knobs
# ----------------------------

# Unknown-field policy is resolved per message full_name using fnmatch globs.
# Precedence: STRIP_UNKNOWN_FOR_MESSAGES > PRESERVE_UNKNOWN_FOR_MESSAGES > UNKNOWN_POLICY_DEFAULT
UNKNOWN_POLICY_DEFAULT: str = "strip"  # "strip" | "preserve"

# Examples:
#   ["my.pkg.*"]          (all messages in package)
#   ["my.pkg.Foo", "google.protobuf.Any"]
PRESERVE_UNKNOWN_FOR_MESSAGES: list[str] = []
STRIP_UNKNOWN_FOR_MESSAGES: list[str] = []

# Canonicalize FieldMask paths for stable diffs/cache keys.
CANONICALIZE_FIELDMASKS: bool = True

# Deterministic JSON emission policy for snapshots and contract equivalence.
# These values are applied via a version-adaptive wrapper around json_format.MessageToDict
# (older protobuf uses including_default_value_fields; newer uses always_print_fields_with_no_presence).
JSON_SNAPSHOT_POLICY: dict[str, object] = {
    "preserving_proto_field_name": True,
    "always_print_fields_with_no_presence": True,
    "use_integers_for_enums": False,
    "float_precision": None,  # or an int like 6/9/15 if you want to bound float noise
    "indent": 2,
}
```

(You can keep your existing `UNKNOWN_FIELD_POLICY` if you want; the loader below will fall back to it.)

---

## 1) Add a session fixture for options

### `tests/protobuf/conftest.py` (new)

```python
from __future__ import annotations

import pytest

from .contract_config import ContractConfig
from .contract_harness import load_contract_options
from . import contract_targets as targets


@pytest.fixture(scope="session")
def pb_contract_options():
    cfg = ContractConfig.from_env()
    return load_contract_options(targets, cfg)
```

Now every test can just accept `pb_contract_options`.

---

## 2) Extend the harness: options, normalization, deterministic JSON, comparator

### `tests/protobuf/contract_harness.py` (add/replace sections)

#### 2.1 Add imports (near top)

```python
import difflib
import fnmatch
from dataclasses import dataclass
from typing import Callable, Literal
```

#### 2.2 Add option dataclasses + loader (place near other config helpers)

```python
UnknownPolicy = Literal["preserve", "strip"]


@dataclass(frozen=True)
class ContractJsonPolicy:
    preserving_proto_field_name: bool = True
    always_print_fields_with_no_presence: bool = True
    use_integers_for_enums: bool = False
    float_precision: int | None = None
    indent: int = 2


@dataclass(frozen=True)
class ContractBoundaryOptions:
    unknown_default: UnknownPolicy = "strip"
    preserve_unknown_for: tuple[str, ...] = ()
    strip_unknown_for: tuple[str, ...] = ()
    canonicalize_field_masks: bool = True
    json_policy: ContractJsonPolicy = ContractJsonPolicy()


def load_contract_options(targets_module, cfg) -> ContractBoundaryOptions:
    """
    Merge ContractConfig (env-driven) + contract_targets.py (repo-driven).
    Precedence:
      - unknown_default: contract_targets.UNKNOWN_POLICY_DEFAULT
          else contract_targets.UNKNOWN_FIELD_POLICY
          else cfg.unknown_field_policy
      - patterns: contract_targets lists (default empty)
      - json policy / fieldmask toggle: contract_targets (defaulted)
    """
    # Unknown default
    unknown_default = getattr(targets_module, "UNKNOWN_POLICY_DEFAULT", None)
    if unknown_default is None:
        unknown_default = getattr(targets_module, "UNKNOWN_FIELD_POLICY", None)
    if unknown_default is None:
        unknown_default = getattr(cfg, "unknown_field_policy", "strip")

    unknown_default = str(unknown_default).strip().lower()
    if unknown_default not in {"strip", "preserve"}:
        raise ValueError(f"Unknown policy default must be 'strip'|'preserve', got {unknown_default!r}")

    preserve_patterns = tuple(getattr(targets_module, "PRESERVE_UNKNOWN_FOR_MESSAGES", []) or [])
    strip_patterns = tuple(getattr(targets_module, "STRIP_UNKNOWN_FOR_MESSAGES", []) or [])

    canonicalize_fm = bool(getattr(targets_module, "CANONICALIZE_FIELDMASKS", True))

    jp = getattr(targets_module, "JSON_SNAPSHOT_POLICY", {}) or {}
    json_policy = ContractJsonPolicy(
        preserving_proto_field_name=bool(jp.get("preserving_proto_field_name", True)),
        always_print_fields_with_no_presence=bool(jp.get("always_print_fields_with_no_presence", True)),
        use_integers_for_enums=bool(jp.get("use_integers_for_enums", False)),
        float_precision=jp.get("float_precision", None),
        indent=int(jp.get("indent", 2)),
    )

    return ContractBoundaryOptions(
        unknown_default=unknown_default,  # type: ignore[arg-type]
        preserve_unknown_for=preserve_patterns,
        strip_unknown_for=strip_patterns,
        canonicalize_field_masks=canonicalize_fm,
        json_policy=json_policy,
    )
```

#### 2.3 Add deterministic “contract JSON” wrapper (keep your existing canonical_json if you like)

```python
def contract_json(message: Message, opts: ContractBoundaryOptions) -> str:
    """
    Deterministic JSON emission under the configured contract policy.
    Uses your existing _message_to_dict() version-adaptive wrapper.
    """
    d = _message_to_dict(
        message,
        preserving_proto_field_name=opts.json_policy.preserving_proto_field_name,
        always_print_fields_with_no_presence=opts.json_policy.always_print_fields_with_no_presence,
        use_integers_for_enums=opts.json_policy.use_integers_for_enums,
        float_precision=opts.json_policy.float_precision,
    )
    return json.dumps(d, sort_keys=True, indent=opts.json_policy.indent, ensure_ascii=True)
```

#### 2.4 Replace your current `normalize_for_contract()` with a full boundary normalizer

Add these helpers (they avoid `UnknownFields()` entirely and work under upb):

```python
def clone_message(msg: Message) -> Message:
    c = msg.__class__()  # type: ignore[call-arg]
    c.CopyFrom(msg)
    return c


def resolve_unknown_policy(full_name: str, opts: ContractBoundaryOptions) -> UnknownPolicy:
    # Precedence: explicit strip patterns > preserve patterns > default
    for pat in opts.strip_unknown_for:
        if fnmatch.fnmatchcase(full_name, pat):
            return "strip"
    for pat in opts.preserve_unknown_for:
        if fnmatch.fnmatchcase(full_name, pat):
            return "preserve"
    return opts.unknown_default


def normalize_for_contract_inplace(msg: Message, opts: ContractBoundaryOptions) -> Message:
    """
    In-place normalization at the contract boundary:
      1) Apply selective unknown-field stripping/preservation based on message full_name allowlists
      2) Canonicalize FieldMask paths everywhere (optional)
    """
    resolver = lambda name: resolve_unknown_policy(name, opts)
    _apply_unknown_policy(msg, resolver)
    if opts.canonicalize_field_masks:
        _canonicalize_field_masks(msg)
    return msg


def normalize_for_contract(msg: Message, opts: ContractBoundaryOptions) -> Message:
    """
    Safe normalizer that returns a normalized clone (does not mutate caller-owned message).
    """
    c = clone_message(msg)
    return normalize_for_contract_inplace(c, opts)
```

Now add the internal traversal + selective unknown algorithm:

```python
def _iter_child_messages(msg: Message) -> Iterator[Message]:
    # Uses FieldDescriptor.is_repeated to avoid label deprecation warnings in protobuf 6.x
    for fd, value in msg.ListFields():
        if fd.cpp_type != FieldDescriptor.CPPTYPE_MESSAGE:
            continue

        if fd.is_repeated:
            # map fields are represented as repeated entry messages; the runtime exposes them as dict-like
            if fd.message_type and fd.message_type.GetOptions().map_entry:
                for _, v in value.items():
                    if isinstance(v, Message):
                        yield v
            else:
                for v in value:
                    yield v
        else:
            yield value


def _collect_topmost_preserve_descendants(root: Message, resolver: Callable[[str], UnknownPolicy]) -> list[Message]:
    """
    Collect preserve-policy nodes in the subtree, but STOP descending once we hit a preserve node.
    This avoids restoring nested preserve nodes whose references could be invalidated by restoring an ancestor.
    """
    out: list[Message] = []

    def walk(node: Message) -> None:
        if resolver(node.DESCRIPTOR.full_name) == "preserve":
            out.append(node)
            return
        for ch in _iter_child_messages(node):
            walk(ch)

    for ch in _iter_child_messages(root):
        walk(ch)
    return out


def _apply_unknown_policy(root: Message, resolver: Callable[[str], UnknownPolicy]) -> None:
    """
    Selective unknown-field stripping/preservation without UnknownFields() access:

    If a node's policy is 'strip':
      - snapshot bytes for all TOPMOST preserve descendants
      - call node.DiscardUnknownFields() (recursive strip)
      - restore each preserve descendant by ParseFromString(saved_bytes)
      - recurse to handle strip nodes inside restored preserve subtrees
    """
    def rec(node: Message) -> None:
        if resolver(node.DESCRIPTOR.full_name) == "strip":
            preserve_nodes = _collect_topmost_preserve_descendants(node, resolver)
            saved = [(p, p.SerializeToString()) for p in preserve_nodes]
            node.DiscardUnknownFields()
            for p, b in saved:
                p.ParseFromString(b)

        for ch in _iter_child_messages(node):
            rec(ch)

    rec(root)
```

And FieldMask canonicalization across the tree (preserves unknown bytes because it only mutates `paths`):

```python
def _canonicalize_field_masks(root: Message) -> None:
    from google.protobuf import field_mask_pb2

    fm_full_name = field_mask_pb2.FieldMask.DESCRIPTOR.full_name

    def rec(node: Message) -> None:
        if node.DESCRIPTOR.full_name == fm_full_name:
            tmp = field_mask_pb2.FieldMask()
            tmp.CanonicalFormFromMask(node)  # writes canonical paths into tmp; returns None

            paths = getattr(node, "paths", None)
            if paths is not None:
                del paths[:]          # mutate only the repeated field
                paths.extend(tmp.paths)

        for ch in _iter_child_messages(node):
            rec(ch)

    rec(root)
```

---

## 3) Add the single comparator: `assert_contract_equivalent(a, b)`

Add this to `contract_harness.py`:

```python
def assert_contract_equivalent(
    expected: Message,
    actual: Message,
    opts: ContractBoundaryOptions,
    *,
    context: str = "",
) -> None:
    """
    Contract equivalence:
      - descriptor must match
      - both sides are normalized under contract-boundary policy
      - compared via deterministic contract JSON
    On mismatch: raises with a unified diff.
    """
    if expected.DESCRIPTOR.full_name != actual.DESCRIPTOR.full_name:
        raise AssertionError(
            f"Descriptor mismatch: {expected.DESCRIPTOR.full_name} vs {actual.DESCRIPTOR.full_name}"
            + (f" ({context})" if context else "")
        )

    ne = normalize_for_contract(expected, opts)
    na = normalize_for_contract(actual, opts)

    je = contract_json(ne, opts)
    ja = contract_json(na, opts)

    if je != ja:
        diff = "\n".join(
            difflib.unified_diff(
                je.splitlines(),
                ja.splitlines(),
                fromfile="expected",
                tofile="actual",
                lineterm="",
            )
        )
        raise AssertionError(
            "Contract equivalence failed"
            + (f" ({context})" if context else "")
            + ":\n"
            + diff[:20000]  # cap giant diffs
        )
```

---

## 4) Update the unknown-policy test to use per-type allowlists

### `tests/protobuf/test_contract_presence_unknowns.py` (update only the unknown test)

Replace the body of `test_unknown_field_policy_roundtrip()` with this:

```python
def test_unknown_field_policy_roundtrip(pb_contract_options) -> None:
    mods = load_pb2_modules(targets.PROTO_MODULE_IMPORTS, targets.PROTO_PACKAGE_ROOTS)
    if not mods:
        pytest.skip("No protobuf modules configured in contract_targets.py")

    classes = _iter_message_classes(mods)

    import os
    cap = os.getenv("PROTOBUF_CONTRACT_MAX_MESSAGES")
    if cap:
        classes = classes[: int(cap)]

    for cls in classes:
        policy = resolve_unknown_policy(cls.DESCRIPTOR.full_name, pb_contract_options)
        assert_unknown_field_policy_roundtrip(cls, policy)  # existing probe test still works
```

This keeps your existing wire-probe roundtrip assertion but now drives it with the **same resolver** your normalizer uses.

---

## 5) How you use this everywhere (the point)

### Golden snapshot emission (deterministic)

```python
snap = contract_json(normalize_for_contract(msg, pb_contract_options), pb_contract_options)
assert_golden_text(path, snap, update=cfg.update_goldens)
```

### Comparing expected vs actual in any test

```python
assert_contract_equivalent(expected_msg, actual_msg, pb_contract_options, context="my feature X")
```

### Normalizing before serving / caching

```python
normalized = normalize_for_contract(msg, pb_contract_options)
cache_key = sha256(contract_json(normalized, pb_contract_options).encode("utf-8")).hexdigest()
```

---

### Why this design holds up

* **FieldMask canonicalization** prevents spurious diffs from path ordering / redundant covered paths.
* **Selective unknowns** lets you preserve forward-compat payload where you intend (e.g., `Any`, specific messages) while still stripping unknowns elsewhere to stabilize snapshots and cache keys.
* **Contract JSON policy** gives a single “source of truth” for diffs, golden tests, and equivalence.
* **Comparator** ensures every test asserts the *contract*, not incidental encoding.

If you want one more increment after this: I can add a small `contract_boundary.py` “public façade” module (thin wrappers + type exports) so production code can import the same normalizer without reaching into `tests/`, and so your test suite uses a single stable import path.
