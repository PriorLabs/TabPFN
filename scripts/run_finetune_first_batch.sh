#!/usr/bin/env bash
set -euo pipefail

# Runs the first limit-finding batch from docs/analyses/tabpfn_finetune_limit_test_plan.md.
# For each trial:
# 1) run fine-tune trial
# 2) run saved-model reload check

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

  # Validate persistence using the latest saved artifact from the trial log.
  (cd "$REPO_ROOT" && python scripts/check_saved_finetune_classifier_model.py --device cpu)
}

echo "Running TabPFN fine-tune first-batch test matrix..."
echo "Repo root: $REPO_ROOT"

# First batch from tabpfn_finetune_limit_test_plan.md
run_trial "A2" cpu 500 64 1
run_trial "A3" cpu 1000 64 1
run_trial "B2" cpu 1000 128 1
run_trial "C2" cpu 1000 128 3
run_trial "D1" mps 1000 64 1

echo ""
echo "Batch complete. Review:"
echo "- outputs/current/tables/tabpfn_finetune_trial_results.csv"
echo "- outputs/current/tables/tabpfn_finetune_reload_checks.csv"
echo "- outputs/current/models/"
