# Small TabPFN Classifier Fine-Tuning Methodology

Scope note: this methodology is classifier-only and uses `TabPFNClassifier` for all fine-tuning and evaluation steps described here.

## Objective

Run a small, low-risk fine-tuning exercise on a classifier to answer three questions:

1. Can the local machine execute fine-tuning reliably?
2. Does a single fine-tuning step or one short epoch complete within acceptable time and memory?
3. Is the setup stable enough to scale to a larger pilot run?

## Recommended Method

Use a staged approach rather than jumping directly to a large run.

### Stage 1: Environment Validation

1. Use the local upstream TabPFN source tree when fine-tuning APIs are required.
2. Confirm that the dataset has a valid classification target.
3. Prefer `cpu` as the first device on this Apple Silicon machine for small fine-tuning trials, because the local smoke test showed better performance on `cpu` than `mps`.

### Stage 2: Tiny Smoke Test

1. Select a binary classification dataset with a known target.
2. Sample a very small subset with stratification.
3. Run:
   - initial evaluation before fine-tuning
   - one fine-tuning step
   - post-step evaluation
4. Capture:
   - wall time
   - peak memory
   - ROC AUC
   - log loss
5. Append each run to a structured CSV log so trials can be compared over time.
6. Save the fitted model artifact so successful trials can be reloaded and compared later.

Recommended defaults:

- rows: `300`
- device: `cpu`
- epochs: `1`
- max fine-tune steps: `1`
- context samples: `64`
- random seed: `42`

### Stage 3: Stability Check

Treat the run as successful if:

1. the script completes without import or device errors
2. one fine-tuning step executes successfully
3. runtime and memory remain comfortably within local limits
4. evaluation runs both before and after the fine-tuning step

### Stage 4: Scale-Up Path

If the smoke test succeeds:

1. increase rows from `300` to `1000`
2. increase context samples from `64` to `128`
3. keep epochs at `1`
4. benchmark `cpu` again before retrying `mps`
5. only then move to multi-epoch trials

## Test Case

### Dataset

- Source: `data/raw/coil2000.csv`
- Target: `CARAVAN`
- Task: binary classification

### Run Definition

- subset size: `300`
- split: stratified train/test, `70/30`
- device: `cpu`
- estimators: `2`
- fine-tune steps: `1`
- context samples: `64`
- metrics: ROC AUC and log loss

### Expected Outcome

This test is not intended to prove model improvement. It is intended to validate the fine-tuning path, environment configuration, and local execution characteristics.

Success means the workflow is technically viable and safe to scale.

## Result Recording

Record each run to:

- `outputs/current/tables/tabpfn_finetune_trial_results.csv`

Store each fine-tuned model artifact at:

- `outputs/current/models/<timestamp>_tabpfn_finetune_<target>_<device>_<rows>.tabpfn_fit`

Each row should include at least:

- timestamp
- dataset and target
- device
- rows, context samples, and estimator count
- saved model path
- fine-tune steps executed
- initial and post-step metrics
- wall time
- memory usage

This keeps smoke tests, device comparisons, and later scale-up trials in one consistent experiment log.

## Reload Validation

After saving a `.tabpfn_fit` artifact, run a reload check to confirm the model can be restored and used for inference.

Recommended script:

- `scripts/check_saved_finetune_classifier_model.py`

Default behavior:

1. read the latest saved-model row from `outputs/current/tables/tabpfn_finetune_trial_results.csv`
2. reconstruct the same sampled dataset and holdout split from the logged config
3. reload the `.tabpfn_fit` artifact on `cpu`
4. evaluate ROC AUC and log loss on the reconstructed holdout set
5. append the check result to `outputs/current/tables/tabpfn_finetune_reload_checks.csv`

This gives us a minimal persistence guarantee: not only did we save a fine-tuned model, we proved that it can be reloaded and scored later.
