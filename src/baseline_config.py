"""
Configuration module for baseline experiments.

Centralized configuration for all baseline experiment notebooks.
This avoids scattered constants and makes it easy to change parameters globally.
"""

import os
from pathlib import Path

# ============================================================================
# PROJECT PATHS
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "current"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"

# Ensure output directories exist
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Data file paths
EUDIRECTLAPSE_PATH = DATA_DIR / "eudirectlapse.csv"
FREMTPL_PATH = DATA_DIR / "freMTPL2freq.csv"

# ============================================================================
# RANDOM SEED & REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 42
NUMPY_SEED = 42
SKLEARN_SEED = 42
PANDAS_SEED = 42

# ============================================================================
# DATA PREPROCESSING
# ============================================================================

# Columns that should be integer encoded
CATEGORICAL_COLS = [
    "Insured_Sex",
    "Marital_Status",
    "Education",
    "Occupation",
    "Credit_Score",
    "Policy_Type",
]

# Columns to drop or exclude
COLS_TO_DROP = ["ID", "Unnamed: 0", "index"]

# Train/test split configuration
TEST_SIZE = 0.2
TRAIN_SIZE = 1.0 - TEST_SIZE

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# TabPFN Configuration (compatible with tabpfn>=7.0.0 / TabPFN model v2.6)
# n_estimators was called n_ensemble in older versions (<7.x)
TABPFN_CONFIG = {
    "n_estimators": 8,
    "device": "auto",
}

# Post-hoc Calibration Methods
CALIBRATION_METHODS = ["sigmoid", "isotonic"]

# Baseline Models
BASELINE_MODELS = {
    "logistic_regression": {"max_iter": 1000},
    "catboost": {"iterations": 100, "verbose": 0},
    "xgboost": {"n_estimators": 100, "random_state": RANDOM_SEED},
    "random_forest": {"n_estimators": 100, "random_state": RANDOM_SEED},
}

# ============================================================================
# EVALUATION METRICS
# ============================================================================

PRIMARY_METRICS = ["auc", "accuracy", "f1"]
CALIBRATION_METRICS = ["brier_score", "calibration_error", "max_calibration_error"]
ALL_METRICS = PRIMARY_METRICS + CALIBRATION_METRICS

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

# Capping percentiles for outliers
CAPPING_PERCENTILES = {
    "annual_income": (5, 95),
    "age": (1, 99),
    "years_employed": (1, 99),
}

# Scaling method
SCALING_METHOD = "standard"  # or "minmax", "robust"

# ============================================================================
# LOGGING & VERBOSITY
# ============================================================================

VERBOSE = True
LOG_LEVEL = "INFO"

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

def set_random_seeds():
    """Set all random seeds for reproducibility."""
    import numpy as np
    import random
    
    random.seed(RANDOM_SEED)
    np.random.seed(NUMPY_SEED)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(RANDOM_SEED)
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(RANDOM_SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(RANDOM_SEED)
    except ImportError:
        pass


if __name__ == "__main__":
    # Print configuration for verification
    print("=" * 70)
    print("BASELINE EXPERIMENTS CONFIGURATION")
    print("=" * 70)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"\nRandom Seed: {RANDOM_SEED}")
    print(f"Test Size: {TEST_SIZE}")
    print(f"Categorical Columns: {CATEGORICAL_COLS}")
    print(f"Calibration Methods: {CALIBRATION_METHODS}")
    print(f"All Metrics: {ALL_METRICS}")
    print("=" * 70)
