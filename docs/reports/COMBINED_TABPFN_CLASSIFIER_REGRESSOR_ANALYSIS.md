# Combined Analysis: TabPFN Classifier and Regressor vs Existing Models

## Objective
Provide one consolidated view of findings from:
1. Classification benchmarking (TabPFN vs GLM)
2. Regression benchmarking (classical baselines on expanded datasets, plus prior TabPFN regression evidence)

This report distinguishes between:
- **Current reproducible artifacts on disk**
- **Prior TabPFN regression observations captured earlier in the workflow**

---

## Data Sources

### Current artifacts (reproducible now)
- `data/processed/glm_vs_tabpfn_head_to_head.csv`
- `data/processed/multi_dataset_regression_benchmark_results.csv`

### Additional prior observations (earlier run)
A previous regression run included TabPFNRegressor on 2 datasets before API-limit interruptions. Those values are included below as prior observed evidence and should be re-run for strict reproducibility in the current notebook state.

---

## 1) Classification Findings (TabPFN vs GLM)

Datasets evaluated:
- EU Direct Lapse
- COIL 2000 (NL)
- Aus. Vehicle (AU)
- freMTPL2 Binary (FR)

### Head-to-head summary
- **ROC AUC wins (TabPFN over GLM): 3/4 datasets**
- **PR AUC wins (TabPFN over GLM): 2/4 datasets**

### Per-dataset deltas (TabPFN - GLM)
- **EU Direct Lapse:** ROC -0.0080, PR +0.0031
- **COIL 2000:** ROC +0.0222, PR -0.0047
- **Aus. Vehicle:** ROC +0.0004, PR -0.0004
- **freMTPL2 Binary:** ROC +0.0151, PR +0.0078

### Interpretation
- TabPFN is generally stronger on ranking quality (ROC) across these binary tasks.
- PR behavior is mixed, indicating sensitivity to class imbalance and score calibration by dataset.
- The strongest overall classification lift appears on freMTPL2 Binary (both ROC and PR improved).

---

## 2) Regression Findings (Current 3-dataset baseline run)

Datasets currently executed in the latest artifact:
- freMTPL2 Frequency (FR), target ClaimNb
- EU Direct Premium (pure), target prem_pure
- AUS Auto Vehicle Value, target VehValue

Models in latest artifact:
- LinearRegression
- RandomForestRegressor
- CatBoostRegressor

### Best model by dataset (current artifact)
- **freMTPL2 Frequency:** CatBoost (best MAE, RMSE, R2)
- **EU Direct Premium:** RandomForest (best MAE, RMSE, R2)
- **AUS Auto Vehicle Value:** CatBoost (best MAE, RMSE, R2)

### Interpretation
- For noisy/count-like targets, boosted trees (CatBoost) are strongest among classical baselines.
- For premium prediction with already high explainability, RandomForest slightly edges other baselines.
- For nonlinear continuous value prediction (vehicle value), CatBoost shows clear superiority.

---

## 3) Prior TabPFN Regressor Evidence (2 datasets, earlier run)

Earlier observed values (to be revalidated in current run state):

### freMTPL2 Frequency (FR)
- TabPFN MAE 0.0489 (best)
- TabPFN RMSE 0.2210 (close to CatBoost 0.2169)
- TabPFN R2 0.1399 (below CatBoost 0.1716)

### EU Direct Premium (pure)
- TabPFN MAE 10.8203 (best)
- TabPFN RMSE 18.3153 (best)
- TabPFN R2 0.9913 (best)

### Interpretation
- On premium regression, TabPFN showed consistent gains over classical baselines.
- On claim frequency, TabPFN improved MAE but did not beat CatBoost on RMSE/R2.
- Net pattern: TabPFN can provide substantial value, but not uniformly across every regression metric.

---

## 4) Combined Cross-Task Conclusions

1. **TabPFN is broadly competitive and often strong, but not universally dominant.**
2. **Classification:** Most robust gains appear in ROC across diverse datasets; PR gains are dataset-specific.
3. **Regression:** TabPFN appears especially promising for premium/continuous targets; boosted trees remain hard to beat on noisy count targets.
4. **Best baseline competitor:** CatBoost is the most consistent classical challenger overall.
5. **Model selection should stay dataset- and metric-specific** rather than adopting a single default winner.

---

## 5) Practical Recommendations

1. Keep TabPFN as a primary candidate for:
   - Binary classification where ranking quality (ROC) is critical
   - Continuous premium-style regression targets
2. Keep CatBoost as a primary classical fallback/challenger for:
   - Count-frequency style regression targets
   - Cases requiring fast inference with strong baseline reliability
3. Re-run full regression with TabPFN in the current notebook state once limits/environment are clear, then finalize production selection from a single fully reproducible artifact.

---

## 6) Reproducibility Note

This combined report is intentionally transparent about evidence layers:
- Classification conclusions come directly from current CSV artifact.
- Regression baseline conclusions come directly from current CSV artifact.
- TabPFN regression conclusions are from earlier observed runs and should be re-executed in the latest notebook flow before final sign-off.

---

## 7) Circumstances Matrix: When TabPFN Performs Well / Not Well

### TabPFN tends to perform well when
1. **The task is binary classification and ranking quality matters most (ROC AUC).**
   - Evidence: TabPFN ROC wins on 3/4 classification datasets.
2. **The regression target is continuous with strong structured signal (premium-style).**
   - Evidence: Earlier EU premium run showed best MAE, RMSE, and R2 for TabPFN.
3. **Feature interactions are likely nonlinear and not fully captured by linear baselines.**
   - Evidence: Gains over GLM in most ROC comparisons suggest richer pattern capture.

### TabPFN tends to perform less well when
1. **Count/noisy regression targets are evaluated using variance-sensitive metrics (RMSE, R2).**
   - Evidence: On freMTPL2 Frequency, TabPFN MAE was best but CatBoost was better on RMSE/R2.
2. **PR AUC is the primary metric in highly imbalanced classification.**
   - Evidence: PR wins were mixed (2/4), even where ROC improved.
3. **A strong tree baseline is already near the practical ceiling.**
   - Evidence: RandomForest/CatBoost remain strongest classical competitors across regression datasets.

---

## 8) Practical Decision Rule (Current Evidence)

Use this as a default selection heuristic before full retuning:

1. **Classification:**
   - If KPI is ROC AUC -> start with TabPFN, then compare CatBoost.
   - If KPI is PR AUC -> run TabPFN and CatBoost side by side; pick by PR directly.
2. **Regression:**
   - Continuous premium/value target -> prioritize TabPFN and CatBoost.
   - Count-frequency target -> prioritize CatBoost first; include TabPFN as challenger for MAE-focused use cases.
3. **Production fallback:**
   - If TabPFN gain over best baseline is small (<1-2% relative on primary KPI), prefer the faster/simpler baseline.

---

## 9) What to Measure Next to Confirm These Circumstances

To make the "when it works" map rigorous, the next run should log the following per dataset:

1. Target type: binary / count / continuous
2. Imbalance and sparsity: positive rate, zero-inflation, missingness
3. Signal profile: baseline linear R2 (or separability proxy)
4. Metric split: ROC vs PR for classification; MAE vs RMSE/R2 for regression
5. Relative lift of TabPFN over best classical baseline

This will convert current qualitative patterns into a quantitative operating policy for model selection.
