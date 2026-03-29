# Appendix A: Reproducibility Package

## Scope

This appendix documents how to reproduce the first-round results reported in:
- There's Life in the Old GLM Yet!

It is intended to make replication practical for readers by providing exact file paths, execution order, and verification checks.

---

## A1. Environment

Minimum requirements:
- Python 3.8+
- scikit-learn >= 1.0
- pandas
- numpy
- xgboost
- catboost
- tabpfn-client (for API-backed TabPFN workflow) or equivalent local TabPFN setup

Notes:
- Some TabPFN calls depend on external API availability and credentials.
- On macOS, XGBoost may require OpenMP runtime (`libomp`).

---

## A2. Data Inputs

Expected raw datasets in `data/raw/`:
- `eudirectlapse.csv`
- `coil2000.csv`
- `ausprivauto0405.csv`
- `freMTPL2freq_binary.csv`
- `freMTPL2freq.csv`

If one or more are missing, regenerate via:

```bash
python scripts/download_datasets.py
```

---

## A3. Primary Notebooks (Execution Order)

Run these notebooks in order:
1. `notebooks/baseline_experiments/02_tabpfn_vs_glm_lapse.ipynb`
2. `notebooks/baseline_experiments/04_probability_calibration.ipynb`
3. `notebooks/baseline_experiments/07_multi_dataset_benchmark.ipynb`
4. `notebooks/baseline_experiments/08_multi_dataset_regression_benchmark.ipynb`

Rationale:
- 02 reproduces core EU lapse benchmark.
- 04 reproduces calibration uplift findings.
- 07 reproduces multi-dataset classification results.
- 08 reproduces multi-dataset regression baseline results.

---

## A4. Fixed Seeds and Determinism

Use the same seeds as the paper workflow:
- Train/test split seed: `45`
- Additional stochastic components used in extended analysis: `943321`

Determinism caveat:
- Small run-to-run drift can occur for API-backed TabPFN inference due to backend/service changes.
- Local ML baselines should be highly stable given fixed seeds.

---

## A5. Expected Output Artifacts

After a successful run, verify these processed outputs exist in `data/processed/`:
- `glm_vs_tabpfn_head_to_head.csv`
- `multi_dataset_benchmark_results.csv`
- `multi_dataset_regression_benchmark_results.csv`
- `multi_dataset_roc_comparison.png`
- `multi_dataset_regression_rmse_comparison.png`

---

## A6. Validation Checklist

Minimum checks before claiming replication:
1. EU Direct lapse GLM vs TabPFN metrics are recreated from notebook 02.
2. Calibration analysis in notebook 04 shows Brier improvement after isotonic calibration.
3. Multi-dataset classification table (notebook 07) is regenerated and written to `data/processed/`.
4. Multi-dataset regression benchmark table (notebook 08) is regenerated and written to `data/processed/`.
5. Seeds and split logic match the values listed in this appendix.

---

## A7. Reporting Guidance

When reporting reproduced results, separate:
- Fully reproducible outputs generated from current notebooks and CSV artifacts.
- Earlier observed values that require explicit re-run confirmation in the current environment.

This distinction prevents mixing historical observations with currently reproducible evidence.
