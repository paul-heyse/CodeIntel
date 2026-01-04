#!/usr/bin/env bash
set -euo pipefail

pattern="import polars|polars\\.LazyFrame|pl\\.LazyFrame"
if command -v rg >/dev/null 2>&1; then
  search_cmd=(rg -n "$pattern" src/codeintel/build/hamilton/native/graphs)
else
  search_cmd=(grep -R -n -E "$pattern" src/codeintel/build/hamilton/native/graphs)
fi
if "${search_cmd[@]}"; then
  echo "Arrow-first guard failed: Polars usage detected in graph compute modules." >&2
  exit 1
fi
