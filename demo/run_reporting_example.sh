#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS="${SCRIPT_DIR}/reporting_example_results.jsonl"
OUT_DIR="${TMPDIR:-/tmp}/versa-reporting-example"

mkdir -p "${OUT_DIR}"

versa-visualize "${RESULTS}" \
  --out "${OUT_DIR}/report.html" \
  --csv "${OUT_DIR}/report.csv" \
  --markdown "${OUT_DIR}/report.md" \
  --group-by model

echo "Wrote example reports to ${OUT_DIR}"
echo "Open ${OUT_DIR}/report.html to inspect the visualization report."
