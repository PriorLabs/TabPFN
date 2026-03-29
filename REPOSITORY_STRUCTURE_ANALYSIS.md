# TabPFN-work-scott Repository Structure Analysis

**Generated:** March 29, 2026  
**Total Size:** 2.8 GB  
**Total Files:** 99  
**Total Directories:** 15

---

## Executive Summary

This repository contains a mix of **Jupyter notebooks, Python scripts, data files, outputs, and documentation** organized across two main project folders. The structure shows signs of **organic growth with experimental work**, including significant **redundancy, cache accumulation, and inconsistent naming patterns**.

### Key Findings:
- ✅ **10 Jupyter notebooks** (experimental work across multiple projects)
- ✅ **4 Python modules** (supporting scripts)
- ⚠️ **31 CSV outputs** (many duplicates from different runs)
- ⚠️ **19 cache/compiled files** (`.pyc`, `__pycache__` directories, `.pkl` files)
- ⚠️ **17 markdown documentation files** (some redundant)
- ⚠️ **Duplicate data files** in different locations
- ⚠️ **Spaces in directory names** (`ADSWP Project/`, `TabPFN ...`) - non-standard

---

## Directory Structure

```
TabPFN-work-scott/
│
├── [ROOT LEVEL] - 4 files
│   ├── ADDED_FILES_COMPARISON.md (3.1 KB)
│   ├── ADDED_FILES_COMPLETE_LIST.txt (19 KB)
│   ├── CLEANUP_COMPLETE.md (2.0 KB)
│   └── CLEANUP_PLAN.md (1.7 KB)
│
├── ADSWP Project/ [8 files, ~5.2 GB]
│   ├── eudirectlapse.csv (3.2 MB) ⚠️ DUPLICATE
│   ├── TabPFN Classifier on eudirectlapse.ipynb (119 KB)
│   ├── TabPFN_on_freMTPL.ipynb (1.3 MB)
│   ├── TabPFN_ausprivauto0405.R (7.8 KB)
│   ├── TabPFN_freMTPL.R (12 KB)
│   ├── tabpfn_embedding_workflow.ipynb (20 KB)
│   ├── usautoBI_fit.ipynb (31 KB)
│   │
│   └── baselining/ [7 compiled files in __pycache__]
│       ├── __pycache__/ [6 .pyc files] ⚠️ CACHE
│       ├── archive/tests/__pycache__/ [1 .pyc file] ⚠️ CACHE
│       └── [No source files visible - only compiled artifacts]
│
└── BaselineExperiments/ [60+ files, ~2.3 GB]
    ├── [ROOT LEVEL - 27 files]
    │   ├── 📊 Documentation Files [17 markdown/style files]
    │   │   ├── ARTICLE_REVISED_COMPLETE.md (22 KB)
    │   │   ├── STREAMLINING_SUMMARY.md (7.0 KB)
    │   │   ├── TECHNICAL_COMPANION.md (14 KB)
    │   │   ├── FINETUNING_SUMMARY.md (5.5 KB)
    │   │   ├── STATUS_REPORT_FINAL.md (3.8 KB)
    │   │   ├── UNIFIED_PAPER_FINAL.md (11 KB)
    │   │   ├── UNIFIED_PAPER_STRUCTURE.md (6.5 KB)
    │   │   ├── BEFORE_AFTER_COMPARISON.md (15 KB)
    │   │   ├── QUICK_REFERENCE_STREAMLINED.md (5.6 KB)
    │   │   ├── QUICK_REFERENCE.md [EMPTY - 0 bytes] ⚠️
    │   │   ├── SECURITY_INCIDENT_RESOLVED.md (8.0 KB)
    │   │   ├── INDEX_DOCUMENTATION.md (2.3 KB)
    │   │   ├── baselining_notebook_summary.md (8.8 KB)
    │   │   ├── class_imbalance_analysis_summary.md (varies)
    │   │   ├── UNIFIED_PAPER_STRUCTURE.md (6.5 KB)
    │   │   ├── The humble logistic regression model tak.sty (11 KB)
    │   │   └── There's Life in the Old GLM Yet!.sty (23 KB)
    │   │
    │   ├── 📓 Jupyter Notebooks [6 files]
    │   │   ├── baselining_notebook.ipynb (2.4 MB) ⭐ LARGE
    │   │   ├── baseline_claim_classification.ipynb (37 KB)
    │   │   ├── baselining_summary_notebook.ipynb (12 KB)
    │   │   ├── finetuning_notebook.ipynb (varies)
    │   │   ├── finetuning_regression_notebook.ipynb (varies)
    │   │   └── data_generation_exploration.ipynb (varies)
    │   │
    │   ├── 🐍 Python Modules [4 files]
    │   │   ├── model_training.py
    │   │   ├── data_loader.py
    │   │   ├── evaluation_metrics.py
    │   │   └── cleanup_outputs.py
    │   │
    │   ├── 📦 Cache Files [3 .pyc files]
    │   │   └── __pycache__/ ⚠️ CACHE DIRECTORY
    │   │
    │   └── model_output.csv (varies)
    │
    ├── data/ [2 CSV files - 33.2 MB total]
    │   ├── eudirectlapse.csv (3.2 MB) ⚠️ DUPLICATE
    │   └── freMTPL2freq.csv (30 MB)
    │
    ├── outputs/ [50+ files, 2.1 MB + large data files]
    │   ├── figures/ [6 PNG files, 2.1 MB] ✅ Well-organized
    │   │   ├── Figure1_Model_Performance_Comparison.png
    │   │   ├── Figure2_Calibration_Diagnosis.png
    │   │   ├── Figure3_PostHoc_Calibration.png
    │   │   ├── Figure4_Probability_Distribution_by_Class.png
    │   │   ├── Figure5_Imbalance_Hypothesis_Test.png
    │   │   └── Figure6_MultiMetric_Radar.png
    │   │
    │   ├── finetuning/ [9 CSV files, 36 KB]
    │   │   ├── finetuning_calibrated_results_20251127_234933.csv
    │   │   ├── finetuning_calibrated_results_20251128_002217.csv
    │   │   ├── finetuning_calibrated_results_20251128_002609.csv
    │   │   ├── finetuning_raw_results_*.csv [3 files] ⚠️ VERSIONED
    │   │   └── production_assessment_*.csv [3 files] ⚠️ VERSIONED
    │   │
    │   ├── shap/ [2 large data files, 64 KB]
    │   │   ├── tabpfn_shap_inputs.parquet
    │   │   └── tabpfn_shap_values.npy
    │   │
    │   ├── tables/ [4 CSV files, 16 KB] ✅ Well-organized
    │   │   ├── Table1_Model_Performance.csv
    │   │   ├── Table2_Calibration_Statistics.csv
    │   │   ├── Table3_Class_Balance.csv
    │   │   └── Table4_Brier_Scores.csv
    │   │
    │   ├── Model Comparison Outputs [20 CSV + PKL files]
    │   │   ├── model_comparison.csv (latest)
    │   │   └── model_comparison_YYYYMMDD_HHMMSS.(csv|pkl) [19 versioned pairs] ⚠️ HISTORICAL
    │   │
    │   ├── Probability Summary Files [3 CSV files]
    │   │   ├── proba_summary_20251026_004703.csv
    │   │   ├── proba_summary_20251026_195320.csv
    │   │   └── proba_summary_20251109_152800.csv
    │   │
    │   ├── consensus_importances.csv (4 KB)
    │   │
    │   └── catboost_info/ [Training metadata]
    │       ├── catboost_training.json
    │       ├── learn_error.tsv
    │       ├── time_left.tsv
    │       ├── learn/events.out.tfevents
    │       └── tmp/
    │
    └── __pycache__/ [3 .pyc files] ⚠️ CACHE DIRECTORY
```

---

## File Type Distribution

| Type | Count | Notes |
|------|-------|-------|
| **CSV** | 31 | Data files and outputs (mostly in `/outputs/`) |
| **Markdown** | 17 | Documentation files (can be consolidated) |
| **PyC (compiled)** | 10 | Cache files (should be removed) |
| **Jupyter Notebooks** | 10 | Experimental work across projects |
| **PKL (pickles)** | 9 | Model comparison artifacts (duplicative) |
| **PNG** | 6 | Figures (well-organized) |
| **Python** | 4 | Core modules |
| **TSV** | 2 | CatBoost metrics |
| **STY (LaTeX)** | 2 | Document templates (unusual) |
| **R Scripts** | 2 | Legacy analysis |
| **Other** | 1 | Parquet, JSON, npy, txt, tfevents |

---

## Naming Convention Analysis

### ✅ Good Patterns Found:
- **Figures:** `Figure1_Model_Performance_Comparison.png` - Clear, numbered, descriptive
- **Tables:** `Table1_Model_Performance.csv` - Consistent numbering
- **Notebooks:** Generally descriptive names (`baselining_notebook.ipynb`, `finetuning_notebook.ipynb`)

### ⚠️ Poor/Inconsistent Patterns:

1. **Spaces in Directory Names:**
   - `ADSWP Project/` - Non-standard, should be `adswp_project/` or `ADSWP_Project/`
   - `TabPFN Classifier on eudirectlapse.ipynb` - Inconsistent with other files

2. **Versioned Files (No Clear Strategy):**
   ```
   model_comparison_20251110_020316.csv
   model_comparison_20251110_020316.pkl
   model_comparison_20251110_020756.csv
   model_comparison_20251110_020756.pkl
   ... [19 total pairs] ❌ CONFUSING
   ```
   - No clear which is "latest"
   - Both CSV and PKL versions for same run
   - Timestamps suggest experimental iterations

3. **Finetuning Outputs:**
   ```
   finetuning_calibrated_results_20251127_234933.csv
   finetuning_calibrated_results_20251128_002217.csv
   finetuning_raw_results_20251127_234933.csv
   finetuning_raw_results_20251128_002217.csv
   production_assessment_20251127_234933.csv
   production_assessment_20251128_002217.csv
   ```
   - Multiple parallel runs with same timestamps
   - Inconsistent structure

4. **R Files (Deprecated?):**
   - `TabPFN_ausprivauto0405.R`
   - `TabPFN_freMTPL.R`
   - No corresponding Python versions or clear status

5. **LaTeX Style Files (Unusual):**
   - `The humble logistic regression model tak.sty` (11 KB)
   - `There's Life in the Old GLM Yet!.sty` (23 KB)
   - These belong in a papers/docs folder, not here

6. **Empty Files:**
   - `QUICK_REFERENCE.md` - 0 bytes (should be removed or filled)

---

## Redundancy & Duplication Analysis

### 🔴 **CRITICAL REDUNDANCY**

#### 1. **Duplicate Data Files**
```
ADSWP Project/eudirectlapse.csv (3.2 MB)
BaselineExperiments/data/eudirectlapse.csv (3.2 MB)
                                    ↑ DUPLICATE
```
**Impact:** 3.2 MB wasted storage + confusion about which is authoritative

#### 2. **Multiple Model Comparison Runs**
```
20 timestamped pairs of:
├── model_comparison_YYYYMMDD_HHMMSS.csv
└── model_comparison_YYYYMMDD_HHMMSS.pkl
```
**Impact:** Combined ~80 KB of duplicate comparison runs  
**Likely:** Experimental iterations; only latest needed for most analysis

#### 3. **Multiple Finetuning Runs**
```
3 sets of:
├── finetuning_calibrated_results_YYYYMMDD_HHMMSS.csv
├── finetuning_raw_results_YYYYMMDD_HHMMSS.csv
└── production_assessment_YYYYMMDD_HHMMSS.csv
```
**Likely:** Different parameter configurations or dataset splits

#### 4. **Probability Summary Outputs**
```
proba_summary_20251026_004703.csv
proba_summary_20251026_195320.csv (same date, different times)
proba_summary_20251109_152800.csv (later)
```
**Question:** Are all three versions still needed?

### 🟡 **MODERATE REDUNDANCY**

#### 5. **Documentation File Duplication**
```
QUICK_REFERENCE.md (0 bytes)
QUICK_REFERENCE_STREAMLINED.md (5.6 KB)
                    ↑ Streamlined version suggests first was superseded
```

```
UNIFIED_PAPER_FINAL.md (11 KB)
UNIFIED_PAPER_STRUCTURE.md (6.5 KB)
                    ↑ Structure doc should precede final
```

```
baselining_notebook_summary.md
baselining_summary_notebook.ipynb
                    ↑ Both covering same content?
```

#### 6. **Notebook Summary Variants**
```
BEFORE_AFTER_COMPARISON.md
ARTICLE_REVISED_COMPLETE.md
STREAMLINING_SUMMARY.md
STATUS_REPORT_FINAL.md
```
**Likely:** Progress documentation that should be archived

#### 7. **Cache Accumulation** ⚠️
- 10 `.pyc` files in `__pycache__/` directories
- 9 `.pkl` model comparison artifacts
- These are automatically generated and should not be versioned

---

## Organization Issues by Severity

### 🔴 **HIGH SEVERITY**

1. **Space in Directory Name** - `ADSWP Project/`
   - Breaks command-line scripts, requires quoting
   - **Fix:** Rename to `adswp_project/`

2. **Duplicate Data Files** - `eudirectlapse.csv`
   - **Fix:** Keep single source in `BaselineExperiments/data/`, remove from `ADSWP Project/`

3. **No .gitignore Strategy** (Implied)
   - `__pycache__/` and `.pyc` files in version control
   - `.pkl` pickle files in version control (non-portable)
   - **Fix:** Add proper `.gitignore`

### 🟡 **MEDIUM SEVERITY**

4. **Unclear Model Version Strategy**
   - 20 dated model_comparison files with no clear "current" version
   - **Fix:** Archive old runs, keep only latest + important baselines

5. **LaTeX Files in Root**
   - `.sty` files belong in dedicated documents/papers folder
   - **Fix:** Create `docs/papers/` directory

6. **R Scripts (Deprecated?)**
   - No corresponding Python versions visible
   - Missing context about status
   - **Fix:** Either port to Python or move to legacy/archive

7. **ADSWP Project Structure**
   - `baselining/` contains only `__pycache__/` directories
   - Source Python modules missing or moved
   - Archive directory buried in subdirectories
   - **Fix:** Clarify what this folder should contain

### 🟢 **LOW SEVERITY**

8. **Mixed Notebook Naming**
   - Some have spaces: `TabPFN Classifier on eudirectlapse.ipynb`
   - Inconsistent with others: `baselining_notebook.ipynb`
   - **Fix:** Standardize naming convention

9. **Empty Documentation**
   - `QUICK_REFERENCE.md` (0 bytes)
   - **Fix:** Remove or repurpose

10. **CatBoost Training Artifacts**
    - `catboost_info/` contains auto-generated training logs
    - Should likely be in .gitignore
    - **Fix:** Add to .gitignore if preserved

---

## Sizing Analysis

| Directory | Size | Contains |
|-----------|------|----------|
| **ADSWP Project/** | ~5.2 GB | Large notebooks (1.3 GB), CSV data (3.2 GB) |
| **BaselineExperiments/data/** | 33.2 MB | CSV files (freMTPL: 30 MB, eudirectlapse: 3.2 MB) |
| **BaselineExperiments/outputs/figures/** | 2.1 MB | 6 PNG files (well-sized) |
| **BaselineExperiments/outputs/finetuning/** | 36 KB | 9 CSV result files |
| **BaselineExperiments/outputs/tables/** | 16 KB | 4 summary tables |
| **BaselineExperiments/outputs/shap/** | 64 KB | Parquet + numpy arrays |
| **Root/Cache & Metadata** | Negligible | ~100 KB total |

### Storage Optimization Opportunities:
- Remove duplicate `eudirectlapse.csv`: **-3.2 MB**
- Archive old model_comparison runs: **-~50 KB**
- Remove `.pyc` and `__pycache__/`: **-~5 KB** (clean, not size)
- Archive old finetuning runs: **-~15 KB**
- Consolidate documentation: **-~5 KB**

**Potential Cleanup: ~70 KB of metadata, but keep data/outputs intact**

---

## Best Practices Violations

| Issue | Current | Recommended |
|-------|---------|-------------|
| **Directory naming** | Spaces: `ADSWP Project/` | `adswp_project/` or `ADSWP_Project/` |
| **Python cache** | Versioned in root | Should be in .gitignore |
| **Output versioning** | Manual timestamps, no clear latest | Centralized metadata or single "current" |
| **Data location** | Scattered (root + data/) | Single `data/` source of truth |
| **Documentation** | Mixed in root + scattered | `docs/` with subdirs: `analyses/`, `reports/`, `papers/` |
| **Notebook organization** | Root of projects | `notebooks/` subdirectory |
| **Source code** | Limited Python modules | `src/` or `modules/` directory |

---

## Recommended Reorganization Structure

```
TabPFN-work-scott/
│
├── README.md                           # Top-level overview
├── .gitignore                          # Include __pycache__, *.pyc, *.pkl, *.egg-info
│
├── data/
│   ├── raw/
│   │   ├── eudirectlapse.csv          # Single source of truth (3.2 MB)
│   │   └── freMTPL2freq.csv           # Single source of truth (30 MB)
│   └── processed/                      # Any processed versions
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── evaluation_metrics.py
│   ├── model_training.py
│   └── utils/
│
├── notebooks/
│   ├── adswp_project/
│   │   ├── 01_TabPFN_classifier_eudirectlapse.ipynb
│   │   ├── 02_TabPFN_freMTPL.ipynb
│   │   ├── 03_usautoBI_fit.ipynb
│   │   └── 04_tabpfn_embedding_workflow.ipynb
│   │
│   └── baseline_experiments/
│       ├── 01_baseline_claim_classification.ipynb
│       ├── 02_baselining_notebook.ipynb
│       ├── 03_finetuning_notebook.ipynb
│       ├── 04_finetuning_regression.ipynb
│       └── 05_data_generation_exploration.ipynb
│
├── outputs/
│   ├── current/                        # Latest results
│   │   ├── figures/                    # PNG exports
│   │   ├── tables/                     # Summary tables
│   │   ├── model_comparison.csv
│   │   └── model_metrics.json
│   │
│   ├── archive/                        # Versioned runs for reference
│   │   ├── finetuning_runs_20251127/
│   │   ├── finetuning_runs_20251128/
│   │   └── model_comparisons_historical/
│   │
│   ├── shap/                           # SHAP analysis outputs
│   └── catboost_info/
│
├── docs/
│   ├── papers/
│   │   ├── The_humble_logistic_regression_model.sty
│   │   ├── Theres_Life_in_the_Old_GLM_Yet.sty
│   │   └── papers_readme.md
│   │
│   ├── reports/
│   │   ├── ARTICLE_REVISED_COMPLETE.md
│   │   ├── TECHNICAL_COMPANION.md
│   │   ├── FINETUNING_SUMMARY.md
│   │   └── reports_index.md
│   │
│   ├── analyses/
│   │   ├── class_imbalance_analysis_summary.md
│   │   ├── baselining_notebook_summary.md
│   │   └── analyses_index.md
│   │
│   └── status/                         # Archived status docs
│       ├── CLEANUP_COMPLETE.md
│       ├── STATUS_REPORT_FINAL.md
│       └── SECURITY_INCIDENT_RESOLVED.md
│
├── legacy/                             # Deprecated/archived items
│   ├── adswp_project_scripts/
│   │   ├── TabPFN_ausprivauto0405.R
│   │   └── TabPFN_freMTPL.R
│   │
│   ├── archived_results/
│   │   ├── model_comparison_20251110_020316.csv
│   │   ├── proba_summary_20251026_004703.csv
│   │   └── [other old runs]
│   │
│   └── legacy_readme.md
│
└── .env.example                        # If environment variables are used
```

---

## Quick Statistics

| Metric | Value |
|--------|-------|
| **Total Repository Size** | 2.8 GB |
| **Total Files** | 99 |
| **Total Directories** | 15 |
| **Largest Single File** | baselining_notebook.ipynb (2.4 MB) |
| **Largest Directory** | ADSWP Project/ (5.2 GB) |
| **Duplicate Data** | eudirectlapse.csv (3.2 MB × 2) |
| **Cache/Compiled Files** | 19 items (10 .pyc + 9 .pkl) |
| **Documentation Files** | 17 markdown + 2 style files |
| **Experimental Outputs** | 31 CSV files versioned by timestamp |
| **Code Files** | 4 Python modules + 10 notebooks + 2 R scripts |

---

## Action Items Summary

### Immediate (Critical):
- [ ] Remove duplicate `ADSWP Project/eudirectlapse.csv`
- [ ] Rename `ADSWP Project/` to `adswp_project/` (remove space)
- [ ] Create `.gitignore` to exclude `__pycache__/`, `*.pyc`, `*.pkl`

### Short-term (High Priority):
- [ ] Archive old model comparison runs (keep latest only)
- [ ] Archive old finetuning experiment outputs
- [ ] Move `.sty` files to dedicated `docs/papers/` folder
- [ ] Clarify status of R scripts (port to Python or deprecate)
- [ ] Explain contents of `ADSWP Project/baselining/` (only has cache)

### Medium-term (Nice to Have):
- [ ] Consolidate documentation across multiple `.md` files
- [ ] Organize notebooks into `notebooks/adswp_project/` and `notebooks/baseline_experiments/`
- [ ] Move Python modules to `src/` directory
- [ ] Create proper outputs versioning strategy (current + archive)

### Long-term (Best Practices):
- [ ] Implement proper project template structure
- [ ] Add automated testing for reproducibility
- [ ] Document data lineage and transformations
- [ ] Create a proper research project README
