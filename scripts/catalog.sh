#!/usr/bin/env bash
set -euo pipefail

# Convenience wrapper for generating the dataset catalog locally.

REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CATALOG_DIR="${CATALOG_DIR:-${REPO_ROOT}/build/catalog}"
SAMPLE_ROWS="${SAMPLE_ROWS:-3}"

codeintel datasets catalog \
  --repo-root "${REPO_ROOT}" \
  --repo "${CODEINTEL_REPO:-demo/repo}" \
  --commit "${CODEINTEL_COMMIT:-deadbeef}" \
  --db-path "${CODEINTEL_DB_PATH:-build/db/db.duckdb}" \
  --build-dir "${REPO_ROOT}/build" \
  --document-output-dir "${REPO_ROOT}/Document Output" \
  --sample-rows "${SAMPLE_ROWS}" \
  --output-dir "${CATALOG_DIR}"
