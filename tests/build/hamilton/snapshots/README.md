# CLI Golden Snapshots

This directory contains golden snapshot tests for the CodeIntel CLI build commands.

## Quick Start

### Run Snapshot Tests

```bash
# Run all CLI snapshot tests
pytest -m cli_snapshot

# Run with verbose output to see case names
pytest -m cli_snapshot -v
```

### Update Snapshots

When CLI output changes intentionally, update the golden files:

```bash
pytest -m cli_snapshot --update-cli-snapshots
```

### Filter Tests

```bash
# Run only PR-14 tests
pytest -m cli_snapshot --cli-snapshot-tags pr14

# Run only graph-related tests
pytest -m cli_snapshot --cli-snapshot-tags graph

# Run tests matching a pattern
pytest -m cli_snapshot --cli-snapshot-pattern "pr14_*"

# Combine filters
pytest -m cli_snapshot --cli-snapshot-tags pr14,graph --cli-snapshot-pattern "*help*"
```

### List Available Cases

```bash
pytest -m cli_snapshot --list-cli-snapshots
```

## Adding New Snapshots

### 1. Add Case to Manifest

Edit `manifest.yaml` and add a new case:

```yaml
cases:
  - name: "my_new_case"
    tags: ["pr15", "explain", "json", "tiny"]
    args: ["build", "explain", "function_metrics", "--format", "json"]
    snapshot: "my_new_case.json"
```

### 2. Generate Snapshot File

```bash
pytest -m cli_snapshot -k my_new_case --update-cli-snapshots
```

This creates `my_new_case.json` with the normalized CLI output.

### 3. Verify

```bash
pytest -m cli_snapshot -k my_new_case
```

## Manifest Schema

### Case Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | string | required | Unique case identifier |
| `args` | list[str] | required | CLI arguments (without program name) |
| `tags` | list[str] | `[]` | Tags for filtering |
| `kind` | "json" \| "text" | "json" | Output format |
| `output` | "stdout" \| "stderr" \| "both" | "stdout" | Output stream |
| `exit_code` | int | 0 | Expected exit code |
| `snapshot` | string | `{name}.json` or `{name}.txt` | Snapshot filename |
| `env` | dict[str, str] | `{}` | Environment overrides |
| `strip_keys` | list[str] | `[]` | Additional JSON keys to remove |
| `replace` | list[{pattern, repl}] | `[]` | Regex replacements for text |

### Tag Taxonomy

Use consistent tags for easy filtering:

- **PR tags**: `pr08`, `pr09`, `pr10`, `pr11`, `pr12`, `pr13`, `pr14`, `pr15`
- **Command tags**: `graph`, `plan`, `explain`, `history`, `status`, `run`
- **Format tags**: `json`, `dot`, `mermaid`, `text`
- **Scope tags**: `tiny`, `integration`
- **Mode tags**: `generated`, `phase0`

## Normalization

### JSON Snapshots

Dynamic fields are automatically stripped:

- `run_id`
- `duration_ms`, `duration_seconds`, `total_duration_ms`
- `started_at`, `completed_at`, `recorded_at`, `computed_at`
- `timestamp`, `now`

Add custom keys via `strip_keys` in the case definition.

### Text Snapshots

Text output is normalized:

1. Line endings unified to `\n`
2. Trailing whitespace removed per line
3. Leading/trailing blank lines stripped
4. Optional regex replacements applied

Use `replace` for dynamic content like paths or IDs:

```yaml
replace:
  - pattern: "/tmp/[^\\s]+"
    repl: "<TMP>"
```

## Fail-Fast Mode

Stop on first failure for faster debugging:

```bash
pytest -m cli_snapshot --cli-snapshot-fail-fast
```

## Files

- `manifest.yaml` - Test case definitions (YAML format)
- `_snapshot.py` - Normalization and assertion helpers
- `_manifest.py` - Typed manifest loader
- `_runner.py` - CLI execution and comparison
- `*.json` - JSON golden snapshots
- `*.txt` - Text golden snapshots (DOT, Mermaid, help text)

