# Files Added to Forked Repository (TabPFN-work-scott)

## Summary
**Total Files Added: 321**

These files exist in your working fork (`TabPFN-work-scott`) but **not** in the upstream repository (`TabPFN-upstream`).

---

## Major Directories Added

### 1. **ADSWP Project/** (4 root files + subdirectories)
Your domain-specific project work:
- `TabPFN Classifier on eudirectlapse.ipynb` - Jupyter notebook applying TabPFN to eudirectlapse dataset
- `TabPFN_ausprivauto0405.R` - R analysis script
- `TabPFN_freMTPL.R` - R analysis for freMTPL dataset
- `TabPFN_on_freMTPL.ipynb` - Jupyter notebook for freMTPL
- `eudirectlapse.csv` - Dataset file
- `baselining/` subdirectory (see below)

### 2. **ADSWP Project/baselining/** (Extensive experimental work)
Contains two major subcomponents:
- **tabpfn/** - Complete TabPFN codebase copy (~120+ files)
- **tabpfn-extensions/** - Extended features package (~80+ files)

### 3. **BaselineExperiments/** (Major addition - ~40-50 files)
Your baseline experiment framework including:

**Jupyter Notebooks (~10 files):**
- `baseline_claim_classification.ipynb`
- `baselining_notebook.ipynb`
- `baselining_summary_notebook.ipynb`
- `finetuning_notebook.ipynb`
- `finetuning_regression_notebook.ipynb`
- `data_generation_exploration.ipynb`
- `usautoBI_fit.ipynb`
- And others...

**Python Scripts (~10 files):**
- `model_training.py` - Main training script
- `data_loader.py` - Data loading utilities
- `evaluation_metrics.py` - Evaluation functions
- `cleanup_outputs.py` - Cleanup utilities
- And supporting files...

**Documentation (~20+ markdown files):**
- `ARTICLE_REVISED_COMPLETE.md`
- `BEFORE_AFTER_COMPARISON.md`
- `FINETUNING_SUMMARY.md`
- `INDEX_DOCUMENTATION.md`
- `QUICK_REFERENCE.md`
- `STREAMLINING_SUMMARY.md`
- `TECHNICAL_COMPANION.md`
- `SECURITY_INCIDENT_RESOLVED.md`
- `STATUS_REPORT_FINAL.md`
- And others...

**Data & Configuration:**
- `model_output.csv` - Model outputs
- Virtual environment directory: `.venv/`

---

## Key Observations

1. **No changes to upstream code** - The root-level `src/`, `examples/`, `tests/`, and project config files (`pyproject.toml`, etc.) appear to be unchanged from upstream.

2. **Work is isolated** - Your custom work is contained in:
   - `BaselineExperiments/` directory
   - `ADSWP Project/` directory

3. **Duplicate TabPFN copies** - You have TabPFN code in multiple locations:
   - Root `/src/` (from upstream)
   - `ADSWP Project/baselining/tabpfn/` (working copy)
   - `ADSWP Project/baselining/tabpfn-extensions/` (extensions)

This could lead to maintenance issues if you update upstream code.

---

## Recommendations for Separation

To cleanly separate your work from the upstream repo:

1. **Keep only what's necessary from upstream** - Consolidate the TabPFN copies
2. **Isolate experiments** - Keep `BaselineExperiments/` and `ADSWP Project/` separate
3. **Consider a cleanup**: Remove duplicate TabPFN installations
4. **Update documentation** - Document which files are yours vs. inherited

Would you like me to help identify which of the 321 files are most critical to your work, or create a structured plan to clean up the duplicate content?
