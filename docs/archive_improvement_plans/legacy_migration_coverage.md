Perfect, let’s go full “rip the bandaid off.”

Below is a revised plan that **assumes we are free to break tests and internal APIs** and that we want **zero legacy/compat shims** around coverage, with **CoverageReport as the only go‑forward domain type at the ToolService layer**. 

I’ll:

* Define the target architecture in one sentence
* Then give concrete, patch‑style steps (with code sketches)
* Then list everything you / tests need to update

---

## Target: single, clean coverage path

**Go‑forward architecture:**

> `coverage.py` plugin → `CoverageReport` → `ToolService.run_coverage_report(...) -> CoverageReport` → `ToolRunnerAdapter.run_coverage(...) -> CoverageResult` → `CoverageIngestStep`.

What goes away:

* `CoverageFileReport` (legacy dataclass)
* Any conversion CoverageReport → CoverageFileReport
* `ToolService.run_coverage_json` and all mentions of “legacy” coverage in docstrings

Nothing else in the ingestion / analytics graph has to change.

---

## Patch 1 – Make `CoverageReport` the canonical ToolService API

### 1.1 Remove `CoverageFileReport` entirely

**File:** `ingestion/tool_service.py`

Delete the whole dataclass:

```python
@dataclass(frozen=True)
class CoverageFileReport:
    """Normalized coverage summary for a single file.

    This dataclass provides backward compatibility with existing code
    that expects this interface from ToolService.
    """
    rel_path: str
    executed_lines: set[int]
    missing_lines: set[int]

    @classmethod
    def from_summary(cls, summary: CoverageFileSummary) -> CoverageFileReport:
        """Build a CoverageFileReport from a CoverageFileSummary."""
        return cls(
            rel_path=summary.rel_path,
            executed_lines=set(summary.executed_lines),
            missing_lines=set(summary.missing_lines),
        )
```

Then:

* Remove the now‑unused import of `CoverageFileSummary` at the top of the file (it’s only needed for that dataclass).
* Remove any references to `CoverageFileReport` in type hints / docstrings in this file (we’ll touch one in the next step).

### 1.2 Replace `run_coverage_json` with `run_coverage_report` (no shim, no legacy semantics)

Still in `ToolService`:

1. **Delete** the existing `run_coverage_json` method (the whole block around lines 284–341).
2. **Add** the new canonical method:

```python
from codeintel.ingestion.tools.results import CoverageReport

class ToolService:
    ...

    async def run_coverage_report(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> CoverageReport:
        """Run coverage JSON export and return a CoverageReport.

        Parameters
        ----------
        repo_root
            Repository root directory.
        coverage_file
            Optional explicit coverage data file path.
        output_path
            Optional path for JSON output; defaults to a cache location.

        Returns
        -------
        CoverageReport
            Parsed coverage data for all files. On failure or when the
            coverage tool is missing, returns CoverageReport.empty().
        """
        data_file = coverage_file or self.tools_config.coverage_file
        target_output = output_path or (self.runner.cache_dir / "coverage.json")

        plugin_result = await self.run_plugin(
            "coverage",
            repo_root=repo_root,
            coverage_file=data_file,
            output_path=target_output,
        )

        # Clean up the JSON artifact regardless of status
        json_path = plugin_result.artifacts.get("coverage_json", target_output)
        await to_thread.run_sync(_unlink_missing, json_path)

        # Missing binary -> warn, but don't crash the ingest pipeline
        if plugin_result.status is ToolStatus.NOT_FOUND:
            log.warning("coverage binary not found; skipping coverage ingestion")
            return CoverageReport.empty()

        # Non‑zero exit -> warn, but behave like "no coverage data"
        if plugin_result.status is not ToolStatus.OK:
            log.warning(
                "coverage CLI failed or returned non-zero exit; status=%s error=%r",
                plugin_result.status,
                plugin_result.error,
            )
            return CoverageReport.empty()

        parsed = plugin_result.parsed
        if isinstance(parsed, CoverageReport):
            return parsed

        log.warning(
            "coverage plugin returned unexpected parsed payload type: %r",
            type(parsed),
        )
        return CoverageReport.empty()
```

Notes:

* No legacy list type, no compatibility path.
* Behaviour matches current ingest semantics:

  * coverage not installed / CLI failure → “no coverage” (but not a hard failure).
  * Only true runtime exceptions are treated as fatal (in the adapter), as today.

(If you *want* to make coverage failures hard errors in the future, you’d change this method to raise instead of returning `CoverageReport.empty()`, but that’s a separate behaviour decision, not required for killing legacy code.)

---

## Patch 2 – Make the adapter consume `CoverageReport` directly

**File:** `ingestion/adapters/tool_runner.py`

Currently:

```python
reports = await self._service.run_coverage_json(
    repo_root,
    coverage_file=coverage_file,
    output_path=output_path,
)
...
files = [
    CoverageFileData(
        rel_path=report.rel_path,
        executed_lines=frozenset(report.executed_lines),
        missing_lines=frozenset(report.missing_lines),
    )
    for report in reports
]
return CoverageResult(status=ToolStatus.OK, files=files, duration_s=duration)
```

Replace `run_coverage` with a version that takes the **new** canonical `CoverageReport`:

```python
from codeintel.ingestion.tools.results import CoverageReport, CoverageFileSummary
from codeintel.ingestion.ports.tools import CoverageFileData, CoverageResult, ToolStatus

class ToolRunnerAdapter(IngestToolPort):
    ...

    async def run_coverage(
        self,
        repo_root: Path,
        *,
        coverage_file: Path | None = None,
        output_path: Path | None = None,
    ) -> CoverageResult:
        """Run coverage tool to export coverage data."""
        start = time.perf_counter()
        try:
            report = await self._service.run_coverage_report(
                repo_root,
                coverage_file=coverage_file,
                output_path=output_path,
            )
            duration = time.perf_counter() - start

            # On failures or missing binary, ToolService already returns
            # CoverageReport.empty(), so treat this as "no coverage data".
            files = [
                CoverageFileData(
                    rel_path=summary.rel_path,
                    executed_lines=summary.executed_lines,
                    missing_lines=summary.missing_lines,
                    # excluded_lines remains default (empty) for now
                )
                for summary in report.files
            ]

            return CoverageResult(
                status=ToolStatus.OK,
                files=files,
                duration_s=duration,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            duration = time.perf_counter() - start
            log.warning("coverage execution failed: %s", exc)
            return CoverageResult(
                status=ToolStatus.FAILED,
                error=str(exc),
                duration_s=duration,
            )
```

Key points:

* We no longer mention `CoverageFileReport` **at all**.
* Adapter only ever sees the canonical `CoverageReport` from the plugin, via the new ToolService API.
* `CoverageIngestStep` continues to consume `CoverageResult` and check `result.error` exactly as before — no change needed.

---

## Patch 3 – Clean up any lingering references

Now that the code path is `CoverageReport`‑only, make sure nothing references the old type or method.

### 3.1 Remove all `CoverageFileReport` references

We already saw via search that `CoverageFileReport` only appears in `tool_service.py`. After you delete the dataclass and the old method:

* Run a quick search for `"CoverageFileReport"` in the repo to confirm it’s gone.
* If you run mypy/pyright, they’ll also flag any dangling references.

### 3.2 Remove `run_coverage_json` references

We know the only call site is `ToolRunnerAdapter.run_coverage`. Once you’ve updated that to `run_coverage_report`:

* Delete the `run_coverage_json` method from `ToolService`.
* Run a search for `"run_coverage_json("` and ensure no hits.

### 3.3 Update docstrings and comments

* In `tool_service.py`:

  * Remove/adjust the comment `# Convert parsed CoverageReport to legacy CoverageFileReport list`.
  * Update any docstrings that mention “legacy coverage interface” to simply talk about `CoverageReport`.

After this, there is **no compatibility / legacy layer** left in the coverage path.

---

## Patch 4 – Tests and callers to update

You said tests will be updated, and there are no external consumers. Here’s what needs to change test‑wise (if such tests exist):

1. **Tests that call `ToolService.run_coverage_json`**

   * Change to call `run_coverage_report`.
   * Update expectations from `list[CoverageFileReport]` to a single `CoverageReport`:

     * Instead of `len(reports)` and `reports[0].rel_path`, use:

       * `report.files`, e.g. `len(report.files)` and `report.files[0].rel_path`.
   * If tests asserted exact type `CoverageFileReport`, delete those assertions or migrate to `CoverageFileSummary`.

2. **Tests that make assertions about “legacy” semantics**

   * Remove any references to `CoverageFileReport` in test fixtures.
   * If you had tests specifically for “legacy compatibility” behaviour, they are no longer relevant and can be deleted or rewritten to focus on the CoverageReport flow.

3. **Tests for `ToolRunnerAdapter.run_coverage`**

   * Might need to adapt mocks:

     * Instead of mocking `run_coverage_json` to return a list of simple objects, mock `run_coverage_report` to return a `CoverageReport` with `files=[CoverageFileSummary(...)]`.

4. **Coverage ingest step tests**

   * These **should not change** as long as they only assert on `CoverageResult`:

     * `status`, `error`, and `files` stay the same.

---

## Quick checklist

If you want to sanity‑check that you’ve fully migrated:

* [ ] `CoverageFileReport` dataclass removed.
* [ ] No string `"CoverageFileReport"` anywhere in the codebase.
* [ ] `ToolService` exposes **only** `run_coverage_report` for coverage.
* [ ] No string `"run_coverage_json("` in the codebase.
* [ ] `ToolRunnerAdapter.run_coverage` calls `run_coverage_report`, consumes `CoverageReport`, and returns `CoverageResult`.
* [ ] `CoverageIngestStep` unchanged and still compiles.
* [ ] Tests updated to use `CoverageReport` at the ToolService layer.

Once this is done, the coverage path is fully aligned with your go‑forward architecture, with **zero legacy coverage types and no compatibility shim** in the ingestion pipeline.

If you’d like, the next cluster we can do in this same “no compatibility at all” style is the **CLI argparse compatibility layer** or the **SCIP config legacy path** — both are similarly self‑contained for an aggressive cleanup.
