#!/usr/bin/env bash
# Schema drift detection CI gate.
#
# This script compares the current schema manifest against an expected baseline,
# failing if breaking changes are detected.
#
# Usage:
#   scripts/ci/schema_drift.sh [MANIFEST_PATH]
#
# Arguments:
#   MANIFEST_PATH  Path to expected manifest (default: tests/build/hamilton/snapshots/pr63_schema_manifest_native.json)
#
# Environment:
#   FAIL_ON_ANY    If set to "true", fail on any drift, not just breaking changes
#
set -euo pipefail

MANIFEST_PATH="${1:-tests/build/hamilton/snapshots/pr63_schema_manifest_native.json}"
FAIL_ON_ANY="${FAIL_ON_ANY:-false}"

echo "=== Schema Drift Detection ==="
echo "Expected manifest: ${MANIFEST_PATH}"

# Build diff command arguments
DIFF_ARGS=(
    build schema diff
    --expected "${MANIFEST_PATH}"
    --infer-native
    --stable
    --fail-on-breaking
)

if [ "${FAIL_ON_ANY}" = "true" ]; then
    DIFF_ARGS+=(--fail-on-any)
    echo "Mode: Fail on ANY drift"
else
    echo "Mode: Fail on BREAKING changes only"
fi

echo ""
echo "Running: codeintel ${DIFF_ARGS[*]}"
echo ""

# Run the diff command
# Exit code 0 = no drift or non-breaking drift (when fail-on-any is false)
# Exit code non-zero = drift detected (breaking, or any when fail-on-any is true)
if codeintel "${DIFF_ARGS[@]}"; then
    echo ""
    echo "=== Schema drift check PASSED ==="
    exit 0
else
    EXIT_CODE=$?
    echo ""
    echo "=== Schema drift check FAILED ==="
    echo ""
    echo "To update the expected manifest, run:"
    echo "  codeintel build schema migrate --expected ${MANIFEST_PATH} --infer-native --stable --no-dry-run"
    echo ""
    exit "${EXIT_CODE}"
fi
