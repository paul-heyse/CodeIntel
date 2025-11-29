#!/usr/bin/env bash
set -euo pipefail

# Generate dataset catalog and contract diffs for CI artifacts.
REPO_ROOT="${REPO_ROOT:-$(pwd)}"
CATALOG_DIR="${CATALOG_DIR:-${REPO_ROOT}/build/catalog}"
SUMMARY_FILE="${SUMMARY_FILE:-${CATALOG_DIR}/contract_docs_summary.txt}"
mkdir -p "${CATALOG_DIR}"

DB_PATH="${CODEINTEL_DB_PATH:-build/db/db.duckdb}"
if [ ! -f "${DB_PATH}" ]; then
  echo "Database not found at ${DB_PATH}; skipping catalog/diff generation."
  echo "dataset_diff_exit_code=0" > "${SUMMARY_FILE}"
  echo "diff_skipped=true" >> "${SUMMARY_FILE}"
  exit 0
fi

echo "Generating dataset catalog into ${CATALOG_DIR}"
codeintel datasets catalog --repo-root "${REPO_ROOT}" --repo "${CODEINTEL_REPO:-demo/repo}" --commit "${CODEINTEL_COMMIT:-deadbeef}" --db-path "${CODEINTEL_DB_PATH:-build/db/db.duckdb}" --build-dir "${REPO_ROOT}/build" --document-output-dir "${REPO_ROOT}/Document Output" --sample-rows 0 --output-dir "${CATALOG_DIR}"

echo "Writing dataset specs snapshot"
codeintel datasets snapshot --repo-root "${REPO_ROOT}" --repo "${CODEINTEL_REPO:-demo/repo}" --commit "${CODEINTEL_COMMIT:-deadbeef}" --db-path "${CODEINTEL_DB_PATH:-build/db/db.duckdb}" --build-dir "${REPO_ROOT}/build" --document-output-dir "${REPO_ROOT}/Document Output" --output "${CATALOG_DIR}/dataset_specs.json"

DIFF_EXIT=0
if git rev-parse --verify --quiet "${CI_BASE_REF:-}" >/dev/null 2>&1; then
  echo "Running dataset diff against ${CI_BASE_REF}"
  set +e
  codeintel datasets diff --repo-root "${REPO_ROOT}" --repo "${CODEINTEL_REPO:-demo/repo}" --commit "${CODEINTEL_COMMIT:-deadbeef}" --db-path "${CODEINTEL_DB_PATH:-build/db/db.duckdb}" --build-dir "${REPO_ROOT}/build" --document-output-dir "${REPO_ROOT}/Document Output" --baseline "${CATALOG_DIR}/dataset_specs.json" --against-ref "${CI_BASE_REF}" --baseline-path build/dataset_specs.json
  DIFF_EXIT=$?
  set -e
  echo "dataset_diff_exit_code=${DIFF_EXIT}" > "${SUMMARY_FILE}"
  if [ "${DIFF_EXIT}" -eq 0 ]; then
    echo "No dataset diffs detected."
  elif [ "${DIFF_EXIT}" -eq 1 ]; then
    echo "Dataset diffs detected (exit 1)."
  else
    echo "Dataset diff failed with exit ${DIFF_EXIT}."
  fi
else
  echo "CI_BASE_REF not set; skipping diff against ref."
  echo "dataset_diff_exit_code=0" > "${SUMMARY_FILE}"
  echo "diff_skipped=true" >> "${SUMMARY_FILE}"
fi

echo "Catalog artifacts ready under ${CATALOG_DIR}"
exit "${DIFF_EXIT}"
