# Multi-Dataset GLM vs TabPFN Summary

Date: 2026-03-29

## Objective
Evaluate whether TabPFN (client backend) outperforms a logistic regression baseline (GLM) across multiple insurance datasets, not only on EU Direct Lapse.

## Datasets

| Dataset | Rows | Target | Positive Rate |
|---|---:|---|---:|
| EU Direct Lapse | 23,060 | `lapse` | 12.81% |
| COIL 2000 (NL) | 9,822 | `CARAVAN` | 5.97% |
| Aus. Vehicle (AU) | 67,856 | `ClaimOcc` | 6.81% |
| freMTPL2 Binary (FR) | 50,000 | `ClaimIndicator` | 5.02% |

## Experimental Setup

- Split: stratified 80/20 train-test.
- Reproducibility seed: `RANDOM_SEED = 42`.
- Train cap: `GLOBAL_MAX_TRAIN = 10,000` for fair comparison across models.
- Metrics: ROC AUC and PR AUC.
- GLM: `LogisticRegression(max_iter=1000, class_weight='balanced')`.
- TabPFN: `tabpfn_client.TabPFNClassifier(random_state=42)`.
- TabPFN inference guard: batched prediction with `MAX_PRED_BATCH = 10,000` due to client API per-call row limits.

## Head-to-Head Results (Observed)

Source: `data/processed/glm_vs_tabpfn_head_to_head.csv`

| Dataset | GLM ROC | TabPFN ROC | Delta ROC (TabPFN-GLM) | GLM PR | TabPFN PR | Delta PR (TabPFN-GLM) |
|---|---:|---:|---:|---:|---:|---:|
| EU Direct Lapse | 0.5943 | 0.5863 | -0.0080 | 0.1688 | 0.1719 | +0.0031 |
| COIL 2000 (NL) | 0.6956 | 0.7178 | +0.0222 | 0.1332 | 0.1286 | -0.0047 |
| Aus. Vehicle (AU) | 0.6587 | 0.6591 | +0.0004 | 0.1063 | 0.1059 | -0.0004 |
| freMTPL2 Binary (FR) | 0.5981 | 0.6131 | +0.0151 | 0.0730 | 0.0808 | +0.0078 |

## Interpretation

- ROC AUC: TabPFN wins on 3/4 datasets.
- PR AUC: TabPFN wins on 2/4 datasets.
- EU Direct Lapse is behaviorally different from the other datasets:
  - GLM is stronger on ROC there, while TabPFN only slightly improves PR.
- Conclusion: performance is dataset-dependent; no universal winner across all metrics and datasets.

## Repeatability Protocol

### 1. Environment

Preferred kernel: `.venv312` (Python 3.12.x).

```bash
cd "/Users/Scott/Documents/Data Science/ADSWP/TabPFN-work-scott"
source .venv312/bin/activate
python -V
```

Expected key packages:

```bash
pip show tabpfn tabpfn-client scikit-learn xgboost catboost | grep -E "^(Name|Version)"
```

Note for macOS + XGBoost: OpenMP runtime is required.

```bash
brew install libomp
```

### 2. Data availability

```bash
ls data/raw/
```

Expected files:
- `eudirectlapse.csv`
- `coil2000.csv`
- `ausprivauto0405.csv`
- `freMTPL2freq_binary.csv`

If missing, regenerate via:

```bash
python scripts/download_datasets.py
```

### 3. Run analysis

Option A (Notebook):
- Open `notebooks/baseline_experiments/07_multi_dataset_benchmark.ipynb`.
- Select the `.venv312` kernel.
- Run all cells from top to bottom.

Option B (Scripted head-to-head reproducibility run):

```bash
python /tmp/glm_vs_tabpfn_head_to_head.py
```

### 4. Expected artifacts

- `data/processed/multi_dataset_benchmark_results.csv`
- `data/processed/glm_vs_tabpfn_head_to_head.csv`
- `data/processed/multi_dataset_roc_comparison.png`

### 5. Validation checks

- Confirm all 4 datasets appear in output CSVs.
- Confirm train cap was applied (`train_rows <= 10000`).
- Confirm TabPFN batching is active for test sets > 10,000 rows.
- Expect small run-to-run drift only if external service behavior changes; local split/model seed is fixed.

## Risks / Caveats

- `tabpfn_client` requires authenticated API access and accepted license terms.
- Cloud backend behavior and latency can vary over time.
- Metric ranking can differ between ROC and PR on imbalanced datasets.

## Recommended Reporting Language

"Across four insurance datasets, TabPFN generally improved ranking discrimination (ROC AUC) versus GLM, but gains were not uniform across metrics or datasets. EU Direct Lapse remained a notable exception where GLM retained a ROC advantage, indicating dataset-specific behavior rather than a universal model winner."
