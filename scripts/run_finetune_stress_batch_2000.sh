#!/usr/bin/env bash
set -euo pipefail

# 2000-row stress batch for local TabPFN fine-tuning limits.
# Each trial is followed by a reload validation check.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

run_trial() {
  local label="$1"
  local device="$2"
  local rows="$3"
  local context="$4"
  local steps="$5"

  echo ""
  echo "===== ${label} ====="
  echo "device=${device} rows=${rows} context=${context} steps=${steps}"

  (cd "$REPO_ROOT" && python scripts/run_small_finetune_classifier_trial.py \
    --device "$device" \
    --rows "$rows" \
    --context-samples "$context" \
    --max-finetune-steps "$steps")

  (cd "$REPO_ROOT" && python scripts/check_saved_finetune_classifier_model.py --device cpu)
}

echo "Running TabPFN 2000-row stress batch..."
echo "Repo root: $REPO_ROOT"

run_trial "S1" cpu 2000 64 1
run_trial "S2" cpu 2000 128 1
run_trial "S3" cpu 2000 128 3
run_trial "S4" mps 2000 64 1

echo ""
echo "Stress batch complete. Review:"
echo "- outputs/current/tables/tabpfn_finetune_trial_results.csv"
echo "- outputs/current/tables/tabpfn_finetune_reload_checks.csv"
echo "- outputs/current/models/"
