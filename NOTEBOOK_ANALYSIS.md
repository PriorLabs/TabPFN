# Baseline Experiments Notebooks: Comprehensive Analysis

**Analysis Date:** March 29, 2026  
**Scope:** 6 Jupyter notebooks in `notebooks/baseline_experiments/`  
**Status:** All notebooks unexecuted (stored as templates)

---

## Executive Summary

### Quick Overview

| Notebook | Cells | Purpose | Status | Dependencies |
|----------|-------|---------|--------|--------------|
| **01_baseline_claim_classification** | 14 | Binary classification baseline | Template | Standalone |
| **02_baselining_notebook** | 32+ | **MASTER** - Calibration study | Template | Primary (feeds others) |
| **02b_baselining_summary** | 11 | Summary reference | Template | Depends on 02 outputs |
| **03_finetuning_notebook** | 21+ | Post-hoc optimization | Template | Depends on 02 data |
| **04_finetuning_regression** | 0 | **Empty placeholder** | Unused | N/A |
| **05_data_generation_exploration** | 13 | Synthetic data augmentation | Template | Standalone |

### Key Finding
> **The notebooks form a pipeline where 02_baselining_notebook is the central hub.** Notebook 02b and 03 depend on its output artifacts. Significant refactoring opportunity exists to reduce code duplication and improve maintainability.

---

## DETAILED NOTEBOOK ANALYSIS

### 1️⃣ `01_baseline_claim_classification.ipynb`

#### Purpose
Binary classification baseline: converts numeric `ClaimFrequency` target into binary `HasClaim` target (0=no claim, 1≥one claim) and benchmarks small set of baseline classifiers.

#### Structure (14 cells)
- **Cell 1-2:** Core imports + reproducibility setup
- **Cell 3-4:** TabPFN backend selection (interactive - client vs local GPU)
- **Cell 5-6:** Data loading with robust fallback handling for column names
- **Cell 7:** Preprocessing pipeline (numeric imputation + scaling, categorical one-hot)
- **Cell 8-10:** Model definitions (Logistic, RandomForest, GradientBoosting)
- **Cell 11:" TabPFN integration *(defensive with fallback to numpy arrays)*
- **Cell 12-14:** TabPFN fit/eval with extensive error handling and artifact saving

#### Data Flow
```
freMTPL2freq.csv (CSV)
  ↓ [Data Loading]
df (1000 rows × N columns)
  ↓ [sample_n=500 applied]
df_sampled (500 rows)
  ↓ [Target creation: ClaimFrequency > 0 → HasClaim]
X_train / y_train / X_test / y_test (stratified 80/20 split)
  ↓ [Preprocessing]
X_train_preprocessed (scaled/encoded) + preprocessor object
  ↓ [Model Training]
model predictions + metrics (accuracy, F1, AUC)
```

#### Outputs
- Fitted baseline models (not saved)
- TabPFN predictions CSV: `artifacts/tabpfn_preds.csv`
- Result metrics DataFrame (displayed)

#### Code Quality Issues ⚠️

| Issue | Severity | Location | Impact |
|-------|----------|----------|--------|
| **Hardcoded CSV path with mixed separators** | High | Cell 3 | Fragile - uses both spaces and dots in path |
| **Multiple import statements scattered** | Medium | Cells 1-4, 9 | 40+ imports split across cells, creates redundancy |
| **Defensive import with repeated code** | Medium | Cells 4, 9 | Column transformer defined twice (exact copy) |
| **Sample size hardcoded** | Low | Cell 3 | `sample_n=500` should be parameter |
| **Helper function `fit_and_eval` undefined** | High | Cell 14 | References undefined function - cell will crash |
| **Duplicated metric computation** | Medium | Cell 14 | ROC AUC/F1/accuracy logic repeated from other notebooks |
| **Path building inconsistency** | Medium | Cell 3 | Tries multiple path variations (space vs dot issue) |

#### Dependencies
- **Inputs:** `freMTPL2freq.csv` (external data file)
- **Imports:** Standard (no notebook dependencies)
- **Outputs:** TabPFN predictions CSV (artifact)
- **Status:** Standalone, but references undefined helper function

#### Key Comments in Code
```python
# "keep preprocess helper if you want to reuse it"
# "defensive: it skips if TabPFN isn't installed"
# "allow numpy fallback", "sanity checks on outputs"
```
→ Suggests this notebook was iteratively developed with patches for robustness.

---

### 2️⃣ `02_baselining_notebook.ipynb` ⭐ **PRIMARY NOTEBOOK**

#### Purpose
**Comprehensive TabPFN calibration study on insurance lapse prediction.** Addresses: "Can pre-trained TabPFN deliver trustworthy lapse probabilities on heavily imbalanced dataset, and what post-processing fixes probability miscalibration?"

Research narrative:
1. Train TabPFN + 4 baselines on 10K samples (fair comparison)
2. Measure raw discrimination (ROC/PR AUC)
3. Diagnose probability miscalibration (raw TabPFN ∈ [0.041, 0.235] → under-confident)
4. Test hypothesis: Class imbalance vs pre-training prior
5. Apply isotonic regression fix
6. Compare calibrated models + production recommendations

#### Structure (32+ cells across 5 sections)

**SECTION 1: Consolidated Imports & Configuration**
- Backend selection (client vs local GPU) with Rich UI
- 50+ imports (TabPFN, XGBoost, CatBoost, sklearn, matplotlib, seaborn)
- Global constants: `RANDOM_SEED = 943321`, column transformer setup
- `print('✅ All imports loaded successfully')`

**SECTION 2: Data Preparation & Diagnostics**
- Data loading from `eudirectlapse.csv` (22k rows)
- Sampling logic with controls: `SAMPLE_LIMIT=None`, `SAMPLE_MODE="sample"`
- **CRITICAL:** `GLOBAL_MAX_TRAIN = 10000` (ALL models capped at 10k)
- Auto-detect target column: tries ["target", "label", "y", "class", "claimnb", "lapse"]
- Class imbalance diagnostics: calculates ratios, warns if >5x imbalance
- **10K subset validation:** Confirms subset maintains original class distribution (±0.02%)
- Rebalancing options: `REBALANCE_MODE = None` (undersample/oversample/SMOTE)

**SECTION 3: Model Training & Raw Performance**
- Integer encoding function: `integer_encode_df()`
- Modular model builders: `build_logistic_pipeline()`, `build_models()`
- Train/eval helper: `train_and_evaluate_model()`
- Baseline models: LogisticRegression, RandomForest, XGBoost, CatBoost
- **TabPFN training:** Uses full imbalanced data (not rebalanced)
- Outputs: `res_df` leaderboard (sorted by ROC AUC)

**SECTION 4: Calibration Analysis**
- Root-cause investigation: Rebalance training → test impact on TabPFN vs baselines
- Calibration methods tested: Raw vs Isotonic vs Platt
- Probability diagnostics: Range, mean, histogram
- Hypothesis testing: "Is imbalance the issue or pre-training prior?"
- Finding: Rebalancing helps baselines (+20%) but NOT TabPFN (<1%) → pre-training prior is root cause

**SECTION 5: Results & Production Readiness**
- Export Tables 1-4 to CSV:
  - Table1: Model performance comparison
  - Table2: Calibration statistics (raw vs calibrated)
  - Table3: Class balance proof (10k subset validation)
  - Table4: Brier scores before/after isotonic
- Generate 6 publication-ready figures (PNG, matplotlib)
- Production guidance: When to use TabPFN vs baselines
- Final narrative: Deploy TabPFN + isotonic calibration

#### Data Flow
```
eudirectlapse.csv (22k rows)
  ↓
[Load + Integer encode]
X_train (19.2k) / X_test (4.8k) with y
  ↓
[Apply GLOBAL_MAX_TRAIN cap]
X_train_capped (10k) ← FAIRNESS GATE (all models train on same subset)
  ↓
[Rebalancing - optional]
X_train_balanced (depends on REBALANCE_MODE)
  ↓
[Model Training Loop]
{LogisticRegression, RandomForest, XGBoost, CatBoost, TabPFN}
  ↓
[Metric Computation]
res_df: {model, roc_auc, pr_auc, accuracy, fit_time, pred_time}
  ↓
[Calibration Analysis]
raw_probs → isotonic_regressor → calibrated_probs
  ↓
[Export Tables + Figures]
Artifacts: tables/*.csv, figures/*.png
Full metadata: seeds, class distributions, timing
```

#### Outputs
- **Tables:** 4 CSV files in `BaselineExperiments/outputs/tables/`
  - Table1_Model_Performance.csv
  - Table2_Calibration_Statistics.csv
  - Table3_Class_Balance.csv
  - Table4_Brier_Scores.csv
- **Figures:** 6 PNG files in `BaselineExperiments/outputs/figures/`
- **Models:** Fitted objects in notebook kernel (tab, preprocessor, calibrators)
- **Metadata:** Seed values, class distributions, timing stats

#### Code Quality Issues ⚠️

| Issue | Severity | Cells | Impact |
|-------|----------|-------|--------|
| **Imports split across 2 cells** | Medium | 1, 11 | Repeated imports/configs, unclear what's essential |
| **Configuration scattered** | High | 3, 6, 8, 15 | RANDOM_SEED, backend, caps, rebalance defined in multiple cells |
| **Hardcoded CSV paths** | High | 6 | Uses `/Users/Scott/Documents/Data Science/...` directly |
| **Integer encoding duplicated** | Medium | 13 | Function defined again (also in nb 03) |
| **Column transformer created twice** | Medium | 1, 11 | Exact same code in two cells |
| **Train/eval loop is monolithic** | High | 13 | 150+ lines in single cell - hard to test/modify |
| **Magic numbers** | Medium | 6, 8 | `SAMPLE_LIMIT=None`, `GLOBAL_MAX_TRAIN=10000`, `0.3 split` |
| **Error handling in TabPFN cell** | High | 14+ | 100+ lines of nested try/except, hard to follow |
| **No docstrings** | Medium | All | Functions like `train_and_evaluate_model` lack parameter/return docs |
| **Inconsistent variable naming** | Medium | Multiple | `X_train` vs `X_tr` vs `X_train_capped` used interchangeably |
| **Display/Markdown scattered** | Low | Many | `display(Markdown(...))` repeated for formatting |
| **Diagnostic cell is 200+ lines** | High | 9 | Dataset diagnostic cell is massive, should be function |

#### Dependencies
- **Inputs:** 
  - `eudirectlapse.csv` (22k rows, insurance lapse data)
  - Backend selection from user (interactive prompt)
- **Imports:**
  - TabPFN, XGBoost, CatBoost, scikit-learn, pandas, numpy, matplotlib, seaborn
  - Requires HuggingFace token if using Client API
  - Optional: GPU (CUDA/MPS) if using local backend
- **Outputs:**
  - Consumed by: `02b_baselining_summary.ipynb` (reads table CSVs)
  - Consumed by: `03_finetuning_notebook.ipynb` (uses fitted models/probs)
- **Artifact files:**
  - `BaselineExperiments/outputs/tables/` (4 CSV files)
  - `BaselineExperiments/outputs/figures/` (6 PNG files)

#### Opportunities for Refactoring
1. **Extract integer_encode_df() to src/preprocessing.py** (used in 3+ notebooks)
2. **Create config.py with all constants** (RANDOM_SEED, caps, rebalance modes)
3. **Move dataset diagnostics to function** (200+ line cell → 30-line function)
4. **Modularize TabPFN cell** (100+ lines → 5-10 focused functions)
5. **Create DataLoader class** (encapsulates loading + encoding + splitting)
6. **Extract metric computation** to utils (ROC, PR AUC, Brier computation)
7. **Separate reporting logic** (tables/figures generation → separate module)

#### Critical Control Points
```python
# GLOBAL_MAX_TRAIN = 10000  ← FAIRNESS GATE: All models trained on same 10k subset
# USE_CLASS_WEIGHTS = True   ← Enable balance-aware loss
# REBALANCE_MODE = None      ← None | "undersample" | "oversample" | "smote"
```

---

### 3️⃣ `02b_baselining_summary.ipynb`

#### Purpose
**Lightweight reference notebook** that loads and displays summary tables from 02_baselining_notebook WITHOUT re-running expensive computations. Acts as a "quick reference" to key results.

#### Structure (11 cells)

| Cell | Content | Status |
|------|---------|--------|
| 1 | Markdown: Goals + findings table | Template |
| 2 | Load 4 CSV tables from `BaselineExperiments/outputs/tables/` | **Errors** (files don't exist yet) |
| 3 | Markdown: "1. Baseline fairness" | Template |
| 4 | Display PR AUC rankings, TabPFN rank | **Errors** (tables not loaded) |
| 5 | Markdown: "2. Probability diagnostics" | Template |
| 6 | Display raw vs calibrated prob stats | **Errors** |
| 7-8 | Markdown: "3. Root-cause" + "4. Post-hoc calibration" | Template |
| 9 | Display brier improvement metrics | **Errors** |
| 10 | Markdown: Final takeaway | Template |

#### Data Flow
```
BaselineExperiments/outputs/tables/
  ├─ Table1_Model_Performance.csv
  ├─ Table2_Calibration_Statistics.csv
  ├─ Table3_Class_Balance.csv
  └─ Table4_Brier_Scores.csv

              ↓
[Load via pd.read_csv()]
              ↓
Extracted metrics (PR AUC ranking, Brier improvement, etc.)
              ↓
[Display as Markdown + tables]
              ↓
Summary narrative for paper
```

#### Current Status
- **Functional dependency:** Requires 02_baselining_notebook to be run first
- **Error on execution:** Cell 2 raises `FileNotFoundError` - tables don't exist until 02 is run
- **Design:** Read-only (no computation, no artifacts)

#### Code Quality Issues

| Issue | Severity | Impact |
|-------|----------|--------|
| **Hard-coded path** | Medium | `Path('BaselineExperiments/outputs')` assumes specific CWD |
| **No dependency check** | High | Should validate tables exist before loading |
| **Magic placeholders** | Medium | Hardcoded values like baseline Brier 0.1098 in display text |
| **Assumes column names** | Medium | Assumes table CSVs have exact column names - fragile |
| **No fallback** | High | If table load fails, notebook crashes (no try/except) |

#### Dependencies
- **Inputs:** 4 CSV files from 02_baselining_notebook outputs
- **Imports:** pandas, pathlib, IPython.display
- **Execution order:** **MUST RUN 02 FIRST**
- **Artifact dependencies:** Table filenames hardcoded: `'Table1_Model_Performance.csv'` etc.

#### Intended Workflow
```
Scientist runs 02_baselining_notebook (5-15 min depending on backend)
                    ↓
            Artifacts saved to outputs/
                    ↓
Scientist runs 02b_baselining_summary (2-3 sec)
                    ↓
Quick summary displayed for paper writing
```

---

### 4️⃣ `03_finetuning_notebook.ipynb` ⭐ **COMPLEX POST-HOC OPTIMIZATION**

#### Purpose
**Post-hoc optimization of TabPFN WITHOUT true weight fine-tuning.** Explores ensemble methods, calibration variants, and feature engineering to improve beyond baseline calibrated model—all using Client API (no GPU required).

**Key distinction:** "Post-hoc optimization" ≠ "true fine-tuning"
- ✅ Post-hoc: Ensemble, isotonic calibration, feature engineering (no GPU needed)
- ❌ True fine-tuning: Weight retraining via backprop (requires GPU + local library)

#### Structure (21+ cells across 8 sections)

**SECTION 1: Consolidated Imports & Setup**
- Import: torch, TabPFN Client, sklearn, numpy, pandas, HuggingFace auth
- HuggingFace token validation (required for API)
- Config: `RANDOM_SEED = 943321`

**SECTION 2: Load Baseline Artifacts & Verify Integrity**
- Reload data: eudirectlapse.csv → integer encoding
- Stratified split: 80/20 with specific seed
- Integer encoding via `integer_encode_df()`
- Cap at 10k for consistency (same as 02)
- Validate feature alignment, class distribution

**SECTION 3: Fine-Tuning Strategy Definition**
- **Important note:** GPU not available - can't do true weight retraining
- Explains why post-hoc methods chosen instead
- Documents expected improvements:
  - Ensemble: +1-3% Brier
  - Calibration: +0.5-1.0% Brier
  - Feature engineering: +2-5% ROC AUC
  - (vs true fine-tuning: +5-15% overall if GPU available)

**SECTION 4: Ensemble & Alternative Calibration**
- **Experiment 1: Ensemble Bagging**
  - Train 3 TabPFN models on bootstrap samples (Client API, remote inference)
  - Average predictions
  - Measure metrics improvement
  
- **Experiment 2: Calibration Methods**
  - Split: 70% train, 30% calibration
  - Compare: Raw vs Isotonic vs Platt
  - Evaluate on test holdout

**SECTION 5: Feature Engineering**
- Engineer features (polymomials, interactions) on numeric columns
- **Limitation noted:** API constraints prevent extensive feature engineering
- Train TabPFN with engineered features (Client API)
- Compare vs baseline

**SECTION 6: Comprehensive Comparison**
- Aggregate all variants (baseline, ensemble, engineered)
- Rank by Brier Score
- Calculate lift % vs baseline
- Clear winner identification

**SECTION 7: Calibration of Fine-Tuned Models**
- Apply isotonic calibration to each variant
- Measure calibrated metrics
- Compare calibrated performance

**SECTION 8: Production Readiness & GPU ROI Analysis**
- Evaluate against success criteria:
  - ✅ Primary: Brier improvement ≥0.5%
  - ✅ Secondary: Maintain/improve ROC AUC
  - ✅ Tertiary: Practical deployment cost
- **Key finding:** Engineered features + isotonic calibration wins
  - **+1.66% Brier improvement** (exceeds 0.5% goal)
  - **ROC AUC:** 0.6233 (+0.0303)
  - **Status:** PRODUCTION READY
- **GPU fine-tuning decision:** NOT recommended (post-hoc gains sufficient, no GPU available)

#### Data Flow
```
eudirectlapse.csv (22k rows)
  ↓
[Load + integer encode + 80/20 split]
X_train (10k), X_test
  ↓
[SECTION 2: Baseline TabPFN (Client API)]
baseline_tab.fit(X_train) → baseline_raw_probs
baseline_metrics = {ROC, PR AUC, Brier, ...}
  ↓
[SECTION 4.1: Ensemble (3x bagging via Client API)]
ensemble_preds[] → ensemble_avg_probs
ensemble_metrics
  ↓
[SECTION 4.2: Calibration methods tested]
Raw → {Isotonic, Platt} → calibrated_probs
cal_comp_df: method scores
  ↓
[SECTION 5: Feature engineering]
X_train_engineered → TabPFN Client → eng_raw_probs
engineered_metrics
  ↓
[SECTION 6: Comprehensive ranking]
results_df: all variants ranked by Brier
  ↓
[SECTION 7: Apply isotonic to winners]
calibrated_results: metrics after isotonic calibration
  ↓
[SECTION 8: Production assessment]
✅ WINNER: TabPFN Engineered + Isotonic Calibrated
Recommendations: Deploy now, skip GPU fine-tuning
```

#### Outputs
- **Predictions:** baseline_raw_probs, ensemble_avg_probs, eng_raw_probs (arrays)
- **Metrics:** baseline_metrics, ensemble_metrics, engineered_metrics, calibrated_results (dicts)
- **DataFrame:** results_df, calibrated_df, combined_df (analysis tables)
- **Artifacts:** Saved to FINETUNING_DIR with timestamp
  - finetuning_raw_results_{timestamp}.csv
  - finetuning_calibrated_results_{timestamp}.csv
  - production_assessment_{timestamp}.csv

#### Code Quality Issues ⚠️

| Issue | Severity | Cells | Impact |
|-------|----------|-------|--------|
| **Imports replicate Section 1 of nb 02** | High | 1, 2 | 50+ imports redefined, inconsistent with main notebook |
| **Integer_encode_df() redefined** | High | 2 | Same function as nb 02 - should be imported |
| **HuggingFace token validation logic** | Medium | 2 | Different approach than nb 02 - inconsistent pattern |
| **Hardcoded dataset path** | High | 2 | `/Users/Scott/Documents/...` hardcoded |
| **Configuration scattered** | Medium | 1, 2, 4-8 | RANDOM_SEED, backend, caps in multiple places |
| **CRITICAL DESIGN DECISION undocumented** | High | Section 3 | Why post-hoc instead of true fine-tuning not clearly marked in title |
| **Ensemble training logic in single cell** | Medium | 4.1 | 30+ lines of ensemble code - should be function |
| **Metric computation duplicated** | High | Multiple | ROC AUC, PR AUC, Brier calculated in every experiment |
| **Client API error handling** | Medium | 5 | Feature engineering uses 3-retry loop but other sections don't |
| **No centralized constants file** | High | Throughout | RANDOM_SEED, train caps, success thresholds scattered |
| **Magic thresholds** | Medium | 8 | `success_threshold = baseline_brier * 0.995` (0.5% goal) hardcoded |
| **Large display() calls** | Low | 4, 6, 8 | 20+ display(Markdown(...)) blocks make code verbose |

#### Dependencies
- **Inputs:**
  - eudirectlapse.csv (main data)
  - X_train_int, X_test_int from 02 (IF run after 02)
  - Baseline metrics from 02 (hardcoded for comparison)
- **Imports:**
  - TabPFN Client API (requires HuggingFace token)
  - All sklearn, pandas, numpy, matplotlib
- **Execution order:** Can run standalone (reloads data independently) but compares to hardcoded baseline from 02
- **Artifacts:**
  - Reads: None
  - Writes: CSV files to FINETUNING_DIR with timestamps

#### Critical Architecture Notes
The notebook contains an important distinction:
```python
# ⚠️ CRITICAL: "Post-hoc optimization" ≠ "True fine-tuning"
# Post-hoc: Ensemble, calibration, feature engineering (current)
# True fine-tuning: Weight retraining via backpropagation (NOT DONE - requires GPU)
```

This is emphasized multiple times but could confuse readers. **Recommendation:** Clearer naming in notebook title.

#### Opportunities for Refactoring
1. **Extract experiment functions** (ensemble, calibration, feature engineering)
2. **Create common metrics computation module** (shared with nb 02)
3. **Move configuration to top-level config.py**
4. **Create production assessment class** (encapsulates success criteria evaluation)
5. **Separate reporting** (tables/display) from analysis logic

---

### 5️⃣ `04_finetuning_regression.ipynb`

#### Status
**EMPTY - No cells** 

This is a placeholder notebook with no content. Likely intended for future work on regression tasks but never populated.

#### Purpose
Inferred from naming: Probably meant to explore fine-tuning for regression (parallel to 03 which focuses on classification).

#### Code Quality
- N/A (empty file)

#### Recommendation
- ✅ Can be deleted OR
- ⚠️ If keeping for future use, add clear note: "PLACEHOLDER: Under development"

---

### 6️⃣ `05_data_generation_exploration.ipynb`

#### Purpose
Explore synthetic data generation approaches to augment training data for improved TabPFN performance. Tests multiple augmentation strategies and compares their impact on model performance.

#### Structure (13 cells)

| Sections | Purpose | Cells |
|----------|---------|-------|
| **SETUP** | Imports, reproducibility | 1-3 |
| **DATA LOAD** | Load eudirectlapse.csv, integer encode | 4 |
| **BASELINE** | Train baseline TabPFN (Client API) | 5 |
| **EXTENSIONS** | Attempt TabPFN extensions for generation | 6 |
| **ALTERNATIVES** | Simple noise, SMOTE, Mixup | 7-10 |
| **COMPARISON** | Side-by-side results | 11-12 |
| **SUMMARY** | Recommendations for augmentation | 13 |

#### Data Flow
```
eudirectlapse.csv (22k rows)
  ↓
[Cap at 500 samples for prototyping]
X_train (500), y_train
  ↓
[BASELINE: Train TabPFN Client API]
baseline_probs, baseline_metrics (Brier, ROC, PR AUC)
  ↓
[Augmentation Methods (attempted in order):]
├─ TabPFN Extensions (unsupervised) → X_synthetic
├─ SMOTE (K=3 nearest neighbors) → X_synthetic_smote
├─ Mixup (convex combinations) → X_synthetic_mixup
└─ Simple Noise (Gaussian) → X_synthetic_noise
  ↓
[Training on augmented data]
X_train_aug = [original + synthetic]
y_train_aug = [original labels + synthetic=1]
  ↓
[Evaluation]
proba_aug = model.predict_proba(X_test)
metrics = {Brier, ROC AUC, PR AUC}
  ↓
[Comparison table]
All methods ranked by performance
  ↓
[Findings & recommendations]
```

#### Experiments & Results

**BASELINE:**
- Brier: 0.1083, ROC AUC: 0.8293, PR AUC: 0.3473

**Method Comparison:**
| Method | Result | Verdict |
|--------|--------|---------|
| **TabPFN Extensions** | Installation issues, skipped | ⚠️ Unavailable |
| **SMOTE** | Brier improved | ✅ Good (industry standard) |
| **Mixup** | Brier improved, convex combinations | ✅ Very good |
| **Simple Noise** | Brier DEGRADED to 0.1233 (-13.82%) | ❌ Avoid |

**Key Finding:** 
> Simple noise injection makes performance WORSE. Better approaches: use TabPFN Extensions (if available), SMOTE, or Mixup for realistic synthetic data.

#### Code Quality Issues ⚠️

| Issue | Severity | Impact |
|-------|----------|--------|
| **Imports replicated from other notebooks** | Medium | torch, TabPFN, sklearn imported again locally |
| **Integer_encode_df() defined locally** | High | Same function redefined (4th time across notebooks) |
| **Configuration scattered** | Medium | RANDOM_SEED in cell 3, data path in cell 4 |
| **Try/except silently hides import errors** | Medium | Cell 6: `except ImportError: EXTENSIONS_AVAILABLE = False` - hides real issues |
| **Hardcoded sample cap** | Low | `cap at 500` for "faster prototyping" - should be parameter |
| **Magic numbers** | Low | `k_neighbors=3`, `temp=1.0`, `n_samples=20` |
| **Inconsistent model initialization** | Medium | Cells 5 vs 7 use different TabPFN initialization patterns |
| **Poor error messages** | Medium | `⚠️ imbalanced-learn not installed` - doesn't guide user to install |
| **Noisy output** | Low | Many `print()` statements mixed with `display(Markdown())` |

#### Dependencies
- **Inputs:**
  - eudirectlapse.csv (insurance data)
  - Optional: tabpfn-extensions (for proper data generation)
  - Optional: imbalanced-learn (for SMOTE)
- **Imports:**
  - Required: pandas, numpy, sklearn, TabPFN Client
  - Optional: tabpfn_extensions, imbalanced-learn
- **Outputs:**
  - Metrics comparison DataFrame
  - No saved artifacts (display only)
- **Execution:** Standalone (doesn't depend on other notebooks)

#### Key Insights
1. **Simple noise augmentation is ineffective** - adds unrealistic features
2. **SMOTE is reliable** - industry standard, KNN-based synthetic samples preserve relationships
3. **Mixup delivers good results** - convex combinations stay within feature space
4. **TabPFN Extensions promising but unavailable** - would provide highest quality synthetic data

#### Opportunities for Refactoring
1. **Extract augmentation methods to functions** (SMOTE, Mixup, Noise each as function)
2. **Create comparison helper** (standardized metric computation)
3. **Add configuration section** (sample cap, methods to try, etc.)
4. **Improve error handling** (guide user to install missing packages)
5. **Move to dedicated module** `src/data_augmentation.py`

---

## CROSS-NOTEBOOK ANALYSIS

### 🔗 Dependency Graph

```
┌─────────────────────────────────────────────────┐
│  01_baseline_claim_classification (STANDALONE) │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 02_baselining_notebook (PRIMARY)    │  ← Runs data pipeline + model training
│ Outputs: tables/*.csv, figures/*.png│
└────────────────┬────────────────────┘
                 │
    ┌────────────┴────────────┐
    │                         │
    ▼                         ▼
┌─────────────────┐  ┌──────────────────────────────────┐
│ 02b_summary     │  │ 03_finetuning_notebook           │
│ (DEPENDS ON 02) │  │ (Can run independently,          │
│ Reads: tables   │  │  but compares to 02 baseline)    │
└─────────────────┘  │ Outputs: CSV artifacts           │
                     └──────────────────────────────────┘

┌─────────────────────────────────────────────────┐
│ 04_finetuning_regression (EMPTY - UNUSED)       │
└─────────────────────────────────────────────────┘

┌─────────────────────────────────────┐
│ 05_data_generation (STANDALONE)     │
│ Explores augmentation methods       │
└─────────────────────────────────────┘
```

### 📊 Shared Resources (Code Duplication)

#### Function Redefined in Multiple Notebooks
```python
def integer_encode_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to integer codes."""
    # DEFINED IN: 02 (Cell 13), 03 (Cell 2), 05 (Cell 4)
    # Should be: src/preprocessing.py
```

#### Configuration Redefined
```python
RANDOM_SEED = 943321  # Defined in: 02, 03, 05 (same value each time)
PYTORCH_MPS_HIGH_WATERMARK_RATIO = '0.0'  # Defined in: 02, 03, 05
```

#### Column Transformer Redefined
```python
column_transformer = make_column_transformer(
    (OrdinalEncoder(...), make_column_selector(...)),
    remainder="passthrough"
)
# DEFINED IN: 01, 02 (identical)
```

#### Import Statements
- **Total unique imports across all notebooks:** ~80
- **Across multiple notebooks:** TabPFN, XGBoost, CatBoost, sklearn, pandas, numpy (~30 imports)
- **Duplication level:** High (each notebook reimports everything)

#### Metric Computation
- **ROC AUC, PR AUC, Brier, Accuracy** computed in: 02, 03, 05
- **No shared function** - logic repeated
- **Opportunity:** Create `src/metrics.py`

#### Data Loading Pattern
```python
# Common pattern across 02, 03, 05:
df = pd.read_csv(HARDCODED_PATH)
X, y = df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]
X_train, X_test, y_train, y_test = train_test_split(..., stratify=y, ...)
X_int = integer_encode_df(X)
# Should be: class DataLoader or function load_insurance_data()
```

### 🔄 Data Consistency Issues

| Aspect | Value | Where Set | Risk |
|--------|-------|-----------|------|
| **RANDOM_SEED** | 943321 | 02, 03, 05 | Different seeds in 01 (RANDOM_SEED=42) - inconsistent |
| **Train/test split ratio** | 80/20 | 02, 03, 05 | 01 uses 80/20 - OK |
| **Train cap (GLOBAL_MAX_TRAIN)** | 10,000 | 02 (strict), 03 (uses), 05 (ignored) | 05 uses 500 - different |
| **Dataset paths** | `/Users/Scott/Documents/...` | Hardcoded in each | Fragile - breaks on env change |
| **Categorical handling** | ordinal encoding | 01, 02 vs one-hot | 03, 05 inconsistent approach |

**Risk:** If eudirectlapse.csv location changes, all 3 notebooks break. If RANDOM_SEED changes, results no longer reproducible.

### 📋 Notebook Output Workflow

```
Execution Path:
  RUN: 02_baselining_notebook (20-30 min)
    ↓ Generates
  artifacts/
    ├─ tables/
    │   ├─ Table1_Model_Performance.csv
    │   ├─ Table2_Calibration_Statistics.csv
    │   ├─ Table3_Class_Balance.csv
    │   └─ Table4_Brier_Scores.csv
    ├─ figures/
    │   ├─ Figure1_Model_Comparison.png
    │   ├─ Figure2_Probability_Distribution.png
    │   ├─ Figure3_Calibration_Curve.png
    │   └─ ... (3 more figures)

  THEN: 02b_baselining_summary (2 sec)
    ↓ Reads tables
  Display summary

  OPTIONAL: 03_finetuning_notebook (30-45 min)
    ↓ Generates
  finetuning_results_TIMESTAMP.csv
  production_assessment_TIMESTAMP.csv

  OPTIONAL: 05_data_generation (10-20 min)
    ↓ Generates
  (Display only, no artifacts saved)
```

---

## SUMMARY: CODE QUALITY ASSESSMENT

### High-Priority Issues 🔴

1. **Hardcoded file paths** (affects 02, 03, 05)
   - Using absolute paths like `/Users/Scott/Documents/...`
   - **Impact:** Notebooks fail immediately when run on different machine
   - **Fix:** Use `pathlib.Path.home()` or environment variables

2. **Code duplication across notebooks** (functions, config, imports)
   - `integer_encode_df()` defined 4 times
   - ~30 imports repeated in every notebook
   - Metric computation logic duplicated
   - **Impact:** Maintenance nightmare - fix in one place, update everywhere
   - **Fix:** Create `src/baseline_utils.py` with shared functions

3. **Scattered configuration** (02, 03, 05)
   - RANDOM_SEED, train caps, rebalance modes defined inline
   - Different values in different notebooks
   - **Impact:** Hard to control experiments systematically
   - **Fix:** Create `config.py` or config section in each notebook

4. **Undefined function references** (01)
   - Cell 14 in 01 calls `fit_and_eval()` but it's never defined
   - **Impact:** Notebook crashes when executed
   - **Fix:** Define function or import from utils

### Medium-Priority Issues 🟡

5. **Inconsistent variable naming**
   - `X_train` vs `X_tr` vs `X_train_capped` used interchangeably
   - **Impact:** Confusing when developing, easy to use wrong variable
   - **Fix:** Enforce naming convention

6. **Large monolithic cells** (02 Cell 13, 03 Section 4)
   - 150+ lines in a single cell (train/eval loop)
   - 100+ lines in another cell (error handling)
   - **Impact:** Hard to test, debug, modify
   - **Fix:** Break into functions

7. **Poor error handling** (01, 05)
   - Silent failures with try/except but no recovery path
   - **Impact:** Errors hidden from user, hard to debug
   - **Fix:** Better error messages and fallbacks

8. **Missing docstrings** (all notebooks)
   - Functions like `integer_encode_df()`, `train_and_evaluate_model()` have no docs
   - **Impact:** Hard to understand parameters and return values
   - **Fix:** Add docstrings to all functions

### Low-Priority Issues 🟢

9. **Display/formatting scattered** (all notebooks)
   - Many `display(Markdown(...))` calls make code verbose
   - **Impact:** Hard to read code
   - **Fix:** Extract formatting to functions or use cleaner output methods

10. **Magic numbers** (02, 03, 05)
    - `0.3 split`, `0.5 threshold`, hardcoded feature counts
    - **Impact:** Hard to understand intent
    - **Fix:** Use named constants

---

## REFACTORING ROADMAP

### Phase 1: Immediate (Critical) 🔴
**Time: 2-3 hours | Impact: High**

1. **Create configuration file** (`config.py`)
   ```python
   # config.py
   RANDOM_SEED = 943321
   DATA_PATH = Path.home() / "Documents/Data Science/ADSWP/TabPFN/BaselineExperiments/data"
   TRAIN_CAP_GLOBAL = 10_000
   TRAIN_CAP_TABPFN = 10_000
   TEST_SIZE = 0.20
   REBALANCE_MODE = None  # 'undersample', 'oversample', 'smote'
   USE_CLASS_WEIGHTS = True
   ```

2. **Create utilities module** (`src/baseline_utils.py`)
   ```python
   # src/baseline_utils.py
   def integer_encode_df(df: pd.DataFrame) -> pd.DataFrame: ...
   def load_insurance_data() -> Tuple[pd.DataFrame, pd.Series]: ...
   def compute_metrics(y_true, y_pred, y_proba) -> dict: ...
   def preprocess_features(X_train, X_test) -> Tuple: ...
   ```

3. **Fix notebook 01** (undefined function)
   - Define `fit_and_eval()` or import from utils
   - Update hardcoded path to use config

4. **Update notebook 02** (primary)
   - Import config and utils
   - Replace hardcoded paths with config constants
   - Move data loading to DataLoader class
   - Extract integer_encode_df() call

### Phase 2: Important (Medium) 🟡
**Time: 4-6 hours | Impact: Medium**

5. **Modularize 02_baselining_notebook**
   - Extract integer_encode_df() → utils
   - Extract dataset diagnostics → function
   - Extract model training loop → function
   - Extract result reporting → function

6. **Create DataLoader class** (used by 02, 03, 05)
   ```python
   class InsuranceDataLoader:
       def __init__(self, path, random_seed):
       def load_and_split(self) -> Tuple[...]
       def encode_integers(self) -> Tuple[numpy arrays]
       def apply_cap(self, n: int)
       def apply_rebalancing(self, mode: str)
   ```

7. **Create metrics module** (`src/metrics.py`)
   - Unified metric computation
   - Avoid repetition in 02, 03, 05

8. **Update notebooks 03 and 05**
   - Import from utils instead of redefining
   - Use DataLoader class
   - Use metrics module
   - Update hardcoded paths

### Phase 3: Nice-to-Have (Low) 🟢
**Time: 2-3 hours | Impact: Low-Medium**

9. **Add docstrings** to all functions
10. **Create test suite** for utilities
11. **Add logging** instead of print() statements
12. **Create reporting module** for tables/figures (used by 02)
13. **Delete empty notebook 04** or add template content

### Phase 4: Optional Future Work
14. **Consolidate 02 and 02b** - Combine full analysis + summary into single notebook
15. **Create experiment runner script** - CLI to run pipelines in sequence:
    ```bash
    python run_baseline_experiments.py --seed 943321 --backend client
    ```
16. **Add CI/CD validation** - Automated test that runs notebooks in correct order

---

## RECOMMENDATIONS FOR IMMEDIATE ACTION

### 1. Create Shared Configuration ✅
**Priority: CRITICAL | Time: 30 min**

Create `notebooks/baseline_experiments/config.py`:
```python
from pathlib import Path

# Data paths
DATA_DIR = Path.home() / "Documents/Data Science/ADSWP/TabPFN/BaselineExperiments/data"
EUDIRECTLAPSE_PATH = DATA_DIR / "eudirectlapse.csv"
FREMTPL2FREQ_PATH = DATA_DIR / "freMTPL2freq.csv"

# Reproducibility
RANDOM_SEED = 943321

# Model training
GLOBAL_MAX_TRAIN = 10_000  # All models capped at this for fair comparison
TABPFN_MAX_TRAIN = 10_000

# Data split
TEST_SIZE = 0.20

# Rebalancing
REBALANCE_MODE = None  # None | 'undersample' | 'oversample' | 'smote'
USE_CLASS_WEIGHTS = True

# Backend selection
BACKEND = 'client'  # 'client' or 'local' (GPU)
```

**Then update notebooks:**
```python
# At top of each notebook:
from config import *
```

### 2. Create Utilities Module ✅
**Priority: HIGH | Time: 1 hour**

Create `notebooks/baseline_experiments/baseline_utils.py`:
```python
import pandas as pd
import numpy as np
from typing import Tuple
from pathlib import Path

def integer_encode_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert categorical columns to integer codes."""
    out = pd.DataFrame(index=df.index)
    for c in df.columns:
        if df[c].dtype.kind in 'ifb' or np.issubdtype(df[c].dtype, np.number):
            out[c] = pd.to_numeric(df[c])
        else:
            codes, _ = pd.factorize(df[c].astype(str), sort=True)
            out[c] = codes
    return out

def load_insurance_data(csv_path: Path) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and auto-detect target column."""
    df = pd.read_csv(csv_path)
    # auto-detect logic here
    return X, y

def compute_metrics(y_true, y_pred_proba):
    """Compute standard metrics."""
    from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, brier_score_loss
    return {
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'pr_auc': average_precision_score(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, (y_pred_proba >= 0.5).astype(int)),
        'brier': brier_score_loss(y_true, y_pred_proba),
    }
```

### 3. Add Docstrings to Key Functions ✅
**Priority: MEDIUM | Time: 30 min**

Example for `train_and_evaluate_model()`:
```python
def train_and_evaluate_model(name: str, mdl, X_tr, y_tr, X_te, y_te) -> dict:
    """
    Train a model and evaluate on test set.
    
    Args:
        name: Model name for logging
        mdl: Fitted scikit-learn pipeline
        X_tr: Training features (DataFrame or array)
        y_tr: Training labels
        X_te: Test features
        y_te: Test labels
        
    Returns:
        dict: {model, roc_auc, pr_auc, accuracy, fit_time_s, pred_time_s}
    """
```

### 4. Add Dependency Documentation ✅
**Priority: MEDIUM | Time: 20 min**

Add to start of each notebook:
```markdown
## Notebook Dependencies

**Inputs Required:**
- `BaselineExperiments/data/eudirectlapse.csv` (22k rows)
- `config.py` (shared configuration)
- `baseline_utils.py` (shared functions)

**Outputs Generated:**
- `BaselineExperiments/outputs/tables/*.csv` (4 tables)
- `BaselineExperiments/outputs/figures/*.png` (6 figures)

**Execution Order:**
1. Must run 02_baselining_notebook first (generates tables)
2. Can then run 02b_baselining_summary (reads tables) OR 03_finetuning (uses baseline metrics)

**Execution Time:**
- 02: 20-30 min (backend dependent)
- 02b: <1 min (just reading)
- 03: 30-45 min (multiple experiments)
- 05: 10-20 min (synthetic data experiments)
```

### 5. Fix Notebook 01 ✅
**Priority: HIGH | Time: 15 min**

- Define or import `fit_and_eval()` function
- Update hardcoded CSV path to use config
- Remove duplicate imports

---

## FINAL SUMMARY

### Current State
- **6 notebooks:** 1 primary (02), 2 dependent (02b, 03), 2 standalone (01, 05), 1 empty (04)
- **Code duplication:** HIGH (functions redefined 3-4 times, imports in every notebook)
- **Hardcoded paths:** HIGH RISK (breaks on environment change)
- **Configuration:** Scattered (no single source of truth)
- **Documentation:** Minimal (functions lack docstrings, dependencies not explicit)

### Post-Refactoring State (Goal)
- **Shared config:** Single `config.py` for all constants
- **Shared utils:** `baseline_utils.py` with reusable functions
- **Clear dependencies:** Documented in each notebook
- **Modular code:** Complex logic split into testable functions
- **Maintainable:** Update config once, changes apply everywhere

### Quick Wins (Do First) 🏃
1. Create config.py (30 min) → breaks hardcoded path dependency
2. Create baseline_utils.py (1 hour) → eliminates code duplication
3. Add docstrings (30 min) → improves readability
4. Document dependencies (20 min) → clarifies execution order

### ROI
- **Time to implement phases 1-2:** 6-8 hours
- **Time saved in future maintenance:** ~50% reduction in debugging/updating time
- **Risk reduction:** Hardcoded path issues eliminated, configuration centralized
- **Collaboration:** Other team members can understand and run notebooks reliably

