# Follow-Up to Initial TabPFN Insurance Study (Short Journal Format)

## Abstract

Our initial study on EU Direct Lapse showed that TabPFN did not dominate logistic regression (GLM) on raw discrimination, but improved probability quality after calibration. This follow-up evaluates whether that pattern generalizes across additional insurance classification datasets and extends the analysis to regression tasks. Classification testing covered four datasets (EU Direct Lapse, COIL 2000, Aus. Vehicle, freMTPL2 Binary) with ROC AUC and PR AUC as primary metrics. Regression benchmarking covered three datasets (freMTPL2 Frequency, EU Direct Premium, AUS Auto Vehicle Value) using MAE, RMSE, and R2 with LinearRegression, RandomForestRegressor, and CatBoostRegressor baselines.

In head-to-head GLM vs TabPFN classification comparisons, TabPFN improved ROC AUC on 3/4 datasets and PR AUC on 2/4 datasets; EU Direct Lapse remained an exception where GLM retained ROC advantage. Regression results were target-dependent: CatBoost was strongest on freMTPL2 Frequency and AUS Auto Vehicle Value, while RandomForest performed best on EU Direct Premium in the current reproducible artifact. Prior observed TabPFN regressor results (earlier run state) indicated strong premium-task performance but mixed results on frequency metrics. Overall, findings support a dataset- and KPI-specific model selection strategy rather than a single default winner.

## Methods

### Study Design

This follow-up extends the first-round single-dataset analysis in two directions:
1. Multi-dataset insurance classification benchmarking (TabPFN vs GLM).
2. Insurance-relevant regression benchmarking with classical baselines, plus prior observed TabPFN regressor evidence explicitly labeled as needing rerun confirmation in current state.

### Classification Data and Metrics

Datasets:
- EU Direct Lapse (n=23,060)
- COIL 2000 (NL) (n=9,822)
- Aus. Vehicle (AU) (n=67,856)
- freMTPL2 Binary (FR) (n=50,000)

Protocol:
- Stratified train/test split.
- Fixed random seed (`RANDOM_SEED=42`) in multi-dataset benchmark workflow.
- Train cap for fairness (`GLOBAL_MAX_TRAIN=10,000`).

Models:
- LogisticRegression (GLM, class-weighted)
- TabPFNClassifier (client backend)

Metrics:
- ROC AUC
- PR AUC

### Regression Data and Metrics

Datasets:
- freMTPL2 Frequency (FR), target `ClaimNb`
- EU Direct Premium (pure), target `prem_pure`
- AUS Auto Vehicle Value, target `VehValue`

Models:
- LinearRegression
- RandomForestRegressor
- CatBoostRegressor

Metrics:
- MAE
- RMSE
- R2

### Reproducibility Layer

Current reproducible artifacts:
- `data/processed/glm_vs_tabpfn_head_to_head.csv`
- `data/processed/multi_dataset_benchmark_results.csv`
- `data/processed/multi_dataset_regression_benchmark_results.csv`

Additional regression evidence for TabPFN from earlier interrupted runs is treated as prior observed evidence and not merged into the reproducible baseline layer without rerun confirmation.

## Results

### Classification: GLM vs TabPFN Head-to-Head

From `glm_vs_tabpfn_head_to_head.csv`:

| Dataset | GLM ROC | TabPFN ROC | Delta ROC | GLM PR | TabPFN PR | Delta PR |
|---|---:|---:|---:|---:|---:|---:|
| EU Direct Lapse | 0.5943 | 0.5863 | -0.0080 | 0.1688 | 0.1719 | +0.0031 |
| COIL 2000 (NL) | 0.6956 | 0.7178 | +0.0222 | 0.1332 | 0.1286 | -0.0047 |
| Aus. Vehicle (AU) | 0.6587 | 0.6591 | +0.0004 | 0.1063 | 0.1059 | -0.0004 |
| freMTPL2 Binary (FR) | 0.5981 | 0.6131 | +0.0151 | 0.0730 | 0.0808 | +0.0078 |

Summary:
- ROC AUC wins for TabPFN: 3/4 datasets.
- PR AUC wins for TabPFN: 2/4 datasets.
- EU Direct Lapse remains a dataset-level exception (GLM ROC advantage).

### Regression: Baseline Benchmark (Current Artifact)

From `multi_dataset_regression_benchmark_results.csv`:

- freMTPL2 Frequency (FR): CatBoost best on MAE, RMSE, R2.
- EU Direct Premium (pure): RandomForest best on MAE, RMSE, R2.
- AUS Auto Vehicle Value: CatBoost best on MAE, RMSE, R2.

Representative values:
- freMTPL2 Frequency, CatBoost: MAE 0.0763, RMSE 0.2169, R2 0.1716
- EU Direct Premium, RandomForest: MAE 12.2848, RMSE 19.7943, R2 0.9899
- AUS Auto Vehicle Value, CatBoost: MAE 0.4325, RMSE 0.7503, R2 0.6290

### Prior Observed TabPFN Regression Evidence (Earlier Run)

Earlier observed (to be revalidated in current run state):
- freMTPL2 Frequency: MAE 0.0489; RMSE 0.2210; R2 0.1399
- EU Direct Premium: MAE 10.8203; RMSE 18.3153; R2 0.9913

Pattern suggested by these earlier observations:
- Potential strength on premium-style continuous targets.
- Mixed behavior on frequency targets where CatBoost remains strong on RMSE/R2.

## Discussion

The follow-up confirms and broadens the main practical conclusion from round 1: there is no universal model winner for insurance tabular tasks. Classification performance is metric-dependent, with TabPFN showing more consistent relative gains in ROC than PR, and a notable exception on EU Direct Lapse. Regression behavior is target-dependent; count/noisy outcomes favor strong boosting baselines, while premium-style targets may provide conditions where TabPFN can be highly competitive.

For actuarial deployment, a KPI-first strategy is warranted:
- If ranking discrimination is primary (ROC-focused classification), TabPFN should be included as a lead candidate.
- If precision-recall behavior drives decisions (imbalanced classification), direct PR-side comparison is mandatory.
- For regression, model choice should be conditioned on target type (count/noisy vs premium/value continuous), not brand preference.

A second implication is methodological: evidence layers must remain explicit. Current reproducible regression artifacts and earlier observed TabPFN regression values should not be blended without rerun confirmation. This distinction is essential for scientific transparency and production governance.

Overall, the extended testing supports a model portfolio view: GLM and tree baselines remain strong in structured insurance settings, while TabPFN is a high-value challenger whose benefit is context-specific and can be substantial in the right regime.
