# Python Wiring Pack Executor

This artifact contains a **wiring pack executor** that:
1) Uses **rpygrep** (required) to discover candidate files/spans
2) Uses **ast-grep-py** to confirm and extract structured captures
3) Uses **LibCST** to enrich output with enclosing def + qualified name and byte spans

It outputs a deterministic JSON wiring-edge list:
- entry_kind / entry_key
- hook_span (byte + line/col)
- target (enclosing def qname / def span; plus optional resolved handler target)
- provenance (pack_id, rule_id, rpygrep pattern ids)
- evidence snippets

## Install

```bash
pip install rpygrep ast-grep-py libcst pyyaml
# plus ripgrep binary `rg` in PATH
```

## Run (single pack)

```bash
python -m wiring_executor.cli run \
  --repo /path/to/repo \
  --pack /path/to/packs/wiring_packs/python/fastapi_routes.json \
  --pack-root /path/to/packs \
  --out wiring.json
```

## Run (all packs)

```bash
python -m wiring_executor.cli run-all \
  --repo /path/to/repo \
  --packs /path/to/packs/wiring_packs/python/*.json \
  --pack-root /path/to/packs \
  --out wiring_all.json
```
