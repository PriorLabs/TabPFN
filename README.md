# TabPFN Work Repository

Repository for ADSWP (Actuarial Data Science Working Party) TabPFN analysis and baseline experiments.

## Directory Structure

```
TabPFN-work-scott/
├── data/                           # All data files
│   └── raw/                        # Raw input datasets
│       ├── eudirectlapse.csv       # EU direct lapse data
│       └── freMTPL2freq.csv        # French MTPL frequency data
│
├── src/                            # Python source code
│   ├── __init__.py
│   ├── data_loader.py              # Data loading utilities
│   ├── evaluation_metrics.py       # Evaluation functions
│   ├── model_training.py           # Model training pipeline
│   └── cleanup_outputs.py          # Output cleanup utilities
│
├── notebooks/                      # Jupyter notebooks (organized by project)
│   ├── adswp_project/              # ADSWP domain-specific applications
│   │   ├── 01_TabPFN_classifier_eudirectlapse.ipynb
│   │   ├── 02_TabPFN_freMTPL.ipynb
│   │   ├── 03_usautoBI_fit.ipynb
│   │   └── 04_tabpfn_embedding_workflow.ipynb
│   │
│   └── baseline_experiments/       # Baseline experiments and analysis
│       ├── 01_baseline_claim_classification.ipynb
│       ├── 02_baselining_notebook.ipynb
│       ├── 03_finetuning_notebook.ipynb
│       ├── 04_finetuning_regression.ipynb
│       └── 05_data_generation_exploration.ipynb
│
├── outputs/                        # Model outputs and results
│   ├── current/                    # Latest results
│   │   ├── figures/                # PNG figures (Figure1, Figure2, etc.)
│   │   └── tables/                 # Summary tables (CSV format)
│   ├── archive/                    # Historical/versioned results
│   ├── shap/                       # SHAP analysis outputs
│   └── catboost_info/              # CatBoost training metadata
│
├── docs/                           # Documentation
│   ├── papers/                     # Paper templates and style files
│   │   ├── The_humble_logistic_regression_model.sty
│   │   └── Theres_Life_in_the_Old_GLM_Yet.sty
│   ├── reports/                    # Analysis reports
│   │   ├── ARTICLE_REVISED_COMPLETE.md
│   │   ├── TECHNICAL_COMPANION.md
│   │   ├── FINETUNING_SUMMARY.md
│   │   ├── UNIFIED_PAPER_FINAL.md
│   │   └── BEFORE_AFTER_COMPARISON.md
│   ├── analyses/                   # Analysis summaries
│   │   ├── class_imbalance_analysis_summary.md
│   │   └── baselining_notebook_summary.md
│   └── status/                     # Status and historical docs
│       ├── STATUS_REPORT_FINAL.md
│       ├── SECURITY_INCIDENT_RESOLVED.md
│       └── CLEANUP_COMPLETE.md
│
├── legacy/                         # Deprecated/archived items
│   ├── adswp_project_scripts/      # Legacy R analysis scripts
│   │   ├── TabPFN_ausprivauto0405.R
│   │   └── TabPFN_freMTPL.R
│   └── archived_results/           # Historical experiment outputs
│
└── README.md                       # This file
```

## Getting Started

### 1. Data Setup
All datasets are in `data/raw/`:
- `eudirectlapse.csv` - EU direct lapse dataset
- `freMTPL2freq.csv` - French MTPL frequency data

### 2. Python Environment
Install dependencies and set up your environment:
```bash
pip install -r requirements.txt
```

### 3. Run Notebooks
Start with numbered notebooks in order:
```bash
# ADSWP Project applications
jupyter notebook notebooks/adswp_project/01_TabPFN_classifier_eudirectlapse.ipynb

# Baseline experiments
jupyter notebook notebooks/baseline_experiments/01_baseline_claim_classification.ipynb
```

## Project Organization

### ADSWP Project (`notebooks/adswp_project/`)
Domain-specific TabPFN applications:
- **01**: TabPFN classifier on eudirectlapse data
- **02**: TabPFN on freMTPL dataset
- **03**: US Auto BI fitting
- **04**: Embedding workflow analysis

### Baseline Experiments (`notebooks/baseline_experiments/`)
Experimental framework for baseline model comparison:
- **01**: Claim classification baseline
- **02**: Comprehensive baselining analysis
- **03**: Model finetuning (primary)
- **04**: Regression finetuning
- **05**: Data generation exploration

## Output Files

### Current Results (`outputs/current/`)
- **figures/**: PNG exports of analysis (Figures 1-6)
- **tables/**: Summary tables (Table1-Table4)

### Archive (`outputs/archive/`)
Historical versioned results from experiments. Use only for reference.

### SHAP Analysis (`outputs/shap/`)
Model interpretability outputs:
- `tabpfn_shap_inputs.parquet`
- `tabpfn_shap_values.npy`

## Python Modules (`src/`)

### data_loader.py
Load and preprocess datasets.

### evaluation_metrics.py
Compute evaluation metrics (AUC, accuracy, calibration, etc.).

### model_training.py
Primary model training pipeline.

### cleanup_outputs.py
Utilities for cleaning up experimental outputs.

## Documentation

### Reports (`docs/reports/`)
Read these for comprehensive analysis:
- `ARTICLE_REVISED_COMPLETE.md` - Full article
- `TECHNICAL_COMPANION.md` - Technical details
- `FINETUNING_SUMMARY.md` - Finetuning results

### Analyses (`docs/analyses/`)
Specific analysis summaries (class imbalance, baselining approach)

### Status (`docs/status/`)
Project status, incidents, and historical information

## Naming Conventions

### Notebooks
- Format: `NN_description_of_notebook.ipynb`
- Use underscores instead of spaces
- Number sequentially within each project

### Python Modules
- Use snake_case for file/function names
- Include docstrings

### CSV/Data Files
- Use underscores and dates: `model_comparison_YYYYMMDD_HHMMSS.csv`
- Keep only essential versions (archive old runs)

### Figures
- Format: `FigureN_Description.png`
- Example: `Figure1_Model_Performance_Comparison.png`

### Tables
- Format: `TableN_Description.csv`
- Example: `Table1_Model_Performance.csv`

## .gitignore Strategy

The `.gitignore` file excludes:
- `__pycache__/` and `.pyc` files
- `*.pkl` and `*.pickle` files (non-reproducible models)
- `outputs/archive/` (versioned experimental runs)
- `.env` files with credentials
- IDE configuration files

Keep in version control:
- Source code (`src/`, `notebooks/`)
- Data (`data/raw/`)
- Current outputs (`outputs/current/`)
- Documentation (`docs/`)

## Contributing

When adding new work:
1. Create notebooks in appropriate `notebooks/` subdirectory
2. Use consistent naming: `NN_description.ipynb`
3. Add summary documentation in `docs/` if significant
4. Archive old experiment outputs to `outputs/archive/`
5. Keep latest results in `outputs/current/`

## Cleanup & Maintenance

### Remove Old Outputs
```bash
# Archive versioned experimental runs
mv outputs/*.csv outputs/archive/
mv outputs/*.pkl outputs/archive/
```

### Clean Cache Files
```bash
find . -type d -name __pycache__ -exec rm -r {} +
find . -type f -name "*.pyc" -delete
```

## References

- Original TabPFN code: See `TabPFN-upstream/` repository
- Legacy R scripts: See `legacy/adswp_project_scripts/`
- Historical results: See `outputs/archive/`

## Contact & Status

- Repository: TabPFN-work-scott (forked from upstream)
- Branch: Based on work from eda/baselining_notebook
- Last organized: March 29, 2026

---

**Note**: This repository contains ONLY custom work and new analyses. Original TabPFN code is maintained separately in `TabPFN-upstream/`.
