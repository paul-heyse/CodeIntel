# Quality Report Speed Improvement Plan

This document defines implementation scopes to speed up the quality report pipeline
by reducing IO, minimizing redundant AST parsing, and parallelizing safe steps.
Each scope includes code patterns, target files, and a checklist.

---

## Scope 1: rpygrep candidate prefilter for string-based guardrails

Status: Partial (completed for `tools/pylist_guardrail.py`; `tools/guardrails.py` pending)

### Goal
Replace full-file Python scans that only need literal substring detection with
ripgrep-backed candidate selection. This reduces full file reads and Python-level
looping.

### Code pattern
```python
from pathlib import Path
from rpygrep import RipGrepSearch


def find_candidate_files(root: Path, pattern: str) -> set[Path]:
    return {
        result.path
        for result in (
            RipGrepSearch(working_directory=root)
            .patterns_are_not_regex()
            .add_pattern(pattern)
            .include_types(["py"])
            .add_extra_options(["--no-config", "--no-mmap"])
        ).run()
    }
```

### Target files
- tools/pylist_guardrail.py
- tools/guardrails.py

### Checklist
- Completed: `tools/pylist_guardrail.py` uses rpygrep prefiltering.
- Replace full-tree `rglob("*.py")` loops for literal patterns with rpygrep.
- Preserve existing allowlist/include-prefix logic by applying it to matched paths.
- Keep stdout/stderr formatting identical to current outputs.
- Add unit-level tests for candidate filtering if existing test harnesses exist.

---

## Scope 2: rpygrep prefilter + AST second pass for AST-based lints

Status: Completed

### Goal
Use ripgrep to narrow the file set before AST parsing, then run AST checks on
only those files. This reduces parsing time and total file IO.

### Code pattern
```python
from pathlib import Path
from rpygrep import RipGrepSearch


def filter_files(root: Path, patterns: list[str]) -> set[Path]:
    search = RipGrepSearch(working_directory=root).patterns_are_not_regex()
    for pattern in patterns:
        search = search.add_pattern(pattern)
    search = search.include_types(["py"]).add_extra_options(["--no-config"])
    return {result.path for result in search.run()}


# Example: prefilter for `to_table(` calls
candidate_paths = filter_files(repo_root, ["to_table("])
for path in candidate_paths:
    lint_file(path)
```

### Target files
- tools/lint_no_raw_pyarrow_compute_in_nodes.py
- tools/lint_no_materialize_in_nodes.py
- tools/lint_analytics_rowset_guardrails.py
- tools/lint_analytics_finalize_writes.py
- tools/lint_analytics_iter_rows.py

### Checklist
- Completed: rpygrep prefiltering is wired into all target lints.
- Identify the minimal literal patterns that safely capture candidate files.
- Gate AST parsing on the candidate path set.
- Ensure allowlists are applied after candidate filtering.
- Preserve current exit codes and error message formatting.

---

## Scope 3: ast-grep-py structural queries to replace AST walkers

Status: Not started

### Goal
Replace hand-written `ast.walk` logic with structural queries, while preserving
existing rule semantics. This reduces custom traversal code and makes rules
self-describing.

### Code pattern
```python
from ast_grep_py import SgRoot


def find_to_table_calls(source: str) -> list[int]:
    root = SgRoot(source, "python").root()
    matches = root.find_all(pattern="$OBJ.to_table($$$ARGS)")
    return [match.range().start.line + 1 for match in matches]
```

### Target files
- tools/lint_no_materialize_in_nodes.py
- tools/lint_no_raw_pyarrow_compute_in_nodes.py
- tools/lint_analytics_iter_rows.py
- tools/lint_analytics_finalize_writes.py
- tools/lint_analytics_rowset_guardrails.py

### Checklist
- Translate each AST predicate into a structural `pattern` or `rule` config.
- Preserve all allowlist logic and path scoping.
- Use `range().start.line + 1` for 1-based line reporting.
- Validate on a sample of known-positive and known-negative cases.

---

## Scope 4: Consolidate overlapping AST lints into shared passes

Status: Not started

### Goal
Avoid multiple parses of the same file set. Combine analytics lints and
build/ingestion lints into single-pass collectors.

### Code pattern
```python
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LintFinding:
    path: Path
    line: int
    message: str


def scan_analytics_module(path: Path, source: str) -> list[LintFinding]:
    findings: list[LintFinding] = []
    # Run multiple checks over a single parsed tree or SgRoot.
    return findings
```

### Target files
- tools/lint_analytics_rowset_guardrails.py
- tools/lint_analytics_finalize_writes.py
- tools/lint_analytics_iter_rows.py
- tools/lint_no_raw_pyarrow_compute_in_nodes.py
- tools/lint_no_materialize_in_nodes.py

### Checklist
- Define a shared scanner module for analytics checks.
- Define a shared scanner module for build/ingestion checks.
- Keep output format identical (one combined stderr stream).
- Ensure rule-level exit codes still reflect any violation.

---

## Scope 5: guardrails.py regex scan via rpygrep + allowlist filter

Status: Not started

### Goal
Replace Python-side regex scans with ripgrep regex search, then apply the
existing allowlist/include-prefix rules on matched paths.

### Code pattern
```python
from pathlib import Path
from rpygrep import RipGrepSearch


def rg_regex_matches(root: Path, pattern: str) -> set[Path]:
    return {
        result.path
        for result in (
            RipGrepSearch(working_directory=root)
            .add_pattern(pattern)
            .include_types(["py"])
            .add_extra_options(["--no-config"])
        ).run()
    }
```

### Target files
- tools/guardrails.py

### Checklist
- Convert each `Guardrail.pattern` to an equivalent ripgrep regex string.
- Apply `include_prefixes` and `allow_prefixes` after candidate paths are found.
- Preserve line-number reporting for matches (use result match offsets).
- Confirm no behavior change for rules that use multiline regexes.

---

## Scope 6: Parallelize safe quality report steps after ruff

Status: Not started

### Goal
Run non-mutating checks concurrently once ruff finishes, reducing total wall time.

### Code pattern
```python
async def run_suite(commands: list[CommandSpec], repo_root: Path) -> list[CommandResult]:
    ruff = next(cmd for cmd in commands if cmd.name == "ruff_check")
    other = [cmd for cmd in commands if cmd.name != "ruff_check"]
    results: list[CommandResult] = [await run_command(ruff, repo_root)]
    results.extend(await asyncio.gather(*(run_command(cmd, repo_root) for cmd in other)))
    return results
```

### Target files
- tools/quality_report.py

### Checklist
- Keep ruff as the first step to avoid concurrent file mutation.
- Execute all remaining commands with `asyncio.gather`.
- Preserve report ordering by storing results in a deterministic order.
- Confirm stdout/stderr output remains unchanged.

---

## Scope 7: Shared file listing utility for lints

Status: Completed

### Goal
Avoid repeated `rglob("*.py")` traversals by providing a shared file enumerator
that caches results per directory group.

### Code pattern
```python
from functools import lru_cache
from pathlib import Path
from typing import Sequence


def list_python_files(root: Path, rel_roots: Sequence[str]) -> tuple[Path, ...]:
    return _list_python_files_cached(root, tuple(rel_roots))


@lru_cache(maxsize=32)
def _list_python_files_cached(root: Path, rel_roots: tuple[str, ...]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for rel_root in rel_roots:
        base = root / rel_root
        if not base.exists():
            continue
        for path in base.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            paths.append(path)
    return tuple(paths)
```

### Target files
- tools/lint_file_utils.py
- tools/lint_no_raw_pyarrow_compute_in_nodes.py
- tools/lint_no_materialize_in_nodes.py
- tools/lint_analytics_rowset_guardrails.py
- tools/lint_analytics_finalize_writes.py
- tools/lint_analytics_iter_rows.py
- tools/pylist_guardrail.py

### Checklist
- Completed: helper added in `tools/lint_file_utils.py` and consumed by target lints.
- Introduce a shared helper module for file listing.
- Replace ad-hoc rglob loops with the shared helper.
- Keep sorting stable if output depends on ordering.

---

## Scope 8: Schema diff fast-path for local runs

Status: Completed in `tools/quality_report.py` (auto-adds `--actual` when present)

### Goal
Avoid runtime schema compilation when a precomputed manifest is available,
allowing a faster local quality report.

### Code pattern
```python
# CLI usage pattern
uv run python -m tools.schema_diff \
  --expected build/serving/artifacts/schema_manifest.json \
  --actual build/serving/artifacts/schema_manifest.current.json
```

### Target files
- tools/schema_diff.py
- tools/quality_report.py

### Checklist
- Completed: `tools/quality_report.py` passes `--actual` when a current manifest exists.
- Add or document a fast-path in `tools/quality_report.py` to pass `--actual` when present.
- Keep defaults unchanged to avoid CI behavior changes.
- Document the local speed-up in the quality report README or docs.

---

## Scope 9: Verification and regression checks

Status: Deferred (per request to validate after other changes settle)

### Goal
Ensure all speed changes preserve exact behavior and error reporting.

### Code pattern
```bash
uv run python -m tools.quality_report --output build/quality-results/quality_report.json
```

### Target files
- tools/quality_report.py
- tools/* (lint/guardrail scripts)

### Checklist
- Run quality report before and after changes to compare outputs.
- Validate sample violations for each lint rule still report correct lines.
- Confirm no change in exit codes or error messages.
- Capture timing data for a before/after comparison.

---

## Recommended next action sets for remaining scope

### Action Set 2: Structural lint refactors (Scopes 3 and 4)
1. Replace AST walkers with ast-grep-py structural matches in the lint scripts for
   `tools/lint_no_raw_pyarrow_compute_in_nodes.py`, `tools/lint_no_materialize_in_nodes.py`,
   `tools/lint_analytics_iter_rows.py`, `tools/lint_analytics_finalize_writes.py`, and
   `tools/lint_analytics_rowset_guardrails.py`.
2. Consolidate overlapping analytics checks into a single scan module, and do the same for
   build/ingestion checks, so each file is parsed only once per run.
3. Preserve allowlist logic, 1-based line reporting, and exact stderr formatting.

### Action Set 3: Pipeline-level optimizations and validation (Scopes 5, 6, 9)
1. Move `tools/guardrails.py` regex scanning to rpygrep regex searches and reapply allowlists
   on the resulting match set.
2. Parallelize post-ruff quality checks in `tools/quality_report.py` while keeping ruff first
   and preserving deterministic report ordering.
3. Perform deferred validation: run the full quality report and spot-check known violations
   to confirm unchanged behavior and error messages.
