#!/usr/bin/env bash
set -euo pipefail

# 3000-row crossover confirmation batch: CPU vs MPS with matched configs.

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

echo "Running TabPFN crossover batch at 3000 rows..."
echo "Repo root: $REPO_ROOT"

# Matched CPU/MPS comparisons
run_trial "X1-cpu" cpu 3000 64 1
run_trial "X1-mps" mps 3000 64 1
run_trial "X2-cpu" cpu 3000 128 1
run_trial "X2-mps" mps 3000 128 1

echo ""
echo "Crossover batch complete. Review:"
echo "- outputs/current/tables/tabpfn_finetune_trial_results.csv"
echo "- outputs/current/tables/tabpfn_finetune_reload_checks.csv"
echo "- outputs/current/models/"
