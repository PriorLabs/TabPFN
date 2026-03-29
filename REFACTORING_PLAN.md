# Notebook Refactoring Action Plan

## Overview
This document provides a step-by-step roadmap to refactor the baseline experiments notebooks from the current fragmented state to a clean, maintainable codebase.

---

## PHASE 1: CRITICAL (2-3 hours) 🔴

### 1.1 Create Shared Configuration File
**Time: 30 min | File: `notebooks/baseline_experiments/config.py`**

**Action:**
Create this file with all shared constants:

```python
"""
Configuration for baseline experiments notebooks.

Edit this file to control:
- Data paths (must be valid for your environment)
- Random seed (for reproducibility)
- Model training parameters
- Processing options
"""

from pathlib import Path

# ============================================================================
# DATA PATHS - UPDATE THESE FOR YOUR ENVIRONMENT
# ============================================================================

# Option 1: Use absolute path (if notebooks always run from same location)
# DATA_DIR = Path("/Users/Scott/Documents/Data Science/ADSWP/TabPFN/BaselineExperiments/data")

# Option 2: Use home directory (more portable - RECOMMENDED)
DATA_DIR = Path.home() / "Documents/Data Science/ADSWP/TabPFN/BaselineExperiments/data"

# Data files
EUDIRECTLAPSE_PATH = DATA_DIR / "eudirectlapse.csv"
FREMTPL2FREQ_PATH = DATA_DIR / "freMTPL2freq.csv"

# Output directories
OUTPUT_DIR = Path(__file__).parent / "outputs"
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

# ============================================================================
# REPRODUCIBILITY
# ============================================================================

RANDOM_SEED = 943321  # Fixed seed for consistent results across runs

# ============================================================================
# MODEL TRAINING CONSTRAINTS
# ============================================================================

# Fair comparison: ALL models (including baselines) train on same number of samples
GLOBAL_MAX_TRAIN = 10_000  # TabPFN client API limit

# TabPFN-specific cap (only used if GLOBAL_MAX_TRAIN is None)
TABPFN_MAX_TRAIN = 10_000

# ============================================================================
# DATA PROCESSING OPTIONS
# ============================================================================

# Train/test split
TEST_SIZE = 0.20  # 80/20 split (standard)
STRATIFIED = True  # Preserve class distribution in splits

# Data augmentation / rebalancing
REBALANCE_MODE = None  # Options: None | "undersample" | "oversample" | "smote"
USE_CLASS_WEIGHTS = True  # Apply class_weight='balanced' to models

# ============================================================================
# BACKEND SELECTION
# ============================================================================

# TabPFN backend: "client" (remote API) or "local" (GPU required)
BACKEND = "client"  # Change to "local" if GPU available

# HuggingFace authentication
# For client backend: requires HF token (run: hf auth login)
# For local backend: uses local GPU (CUDA or MPS)

# ============================================================================
# VALIDATION - Check paths exist
# ============================================================================

if not EUDIRECTLAPSE_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {EUDIRECTLAPSE_PATH}")

# Create output directories if they don't exist
TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
```

**Update checklist:**
- ✓ Verify `DATA_DIR` path is correct for your machine
- ✓ Test that `EUDIRECTLAPSE_PATH.exists()` returns True
- ✓ Confirm output directories can be created

---

### 1.2 Create Shared Utilities Module
**Time: 60 min | File: `notebooks/baseline_experiments/baseline_utils.py`**

**Action:**
Create module with reusable functions:

```python
"""
Shared utilities for baseline experiments.

Provides functions used across multiple notebooks:
- Data loading and preprocessing
- Integer encoding for TabPFN
- Standard metric computation
- Feature preprocessing pipelines
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Union
import warnings

# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def integer_encode_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert mixed-type DataFrame to integer-encoded version.
    
    Numeric columns: passed through as-is
    Categorical columns: converted to integer codes via pd.factorize()
    
    Args:
        df: Input DataFrame with mixed types
        
    Returns:
        pd.DataFrame: Integer-encoded copy with same index
        
    Example:
        >>> df = pd.DataFrame({"A": [1.0, 2.0], "B": ["cat", "dog"]})
        >>> integer_encode_df(df)
           A  B
        0  1.0  0
        1  2.0  1
    """
    out = pd.DataFrame(index=df.index)
    
    for col in df.columns:
        if df[col].dtype.kind in 'ifb' or np.issubdtype(df[col].dtype, np.number):
            # Numeric column: keep as-is
            out[col] = pd.to_numeric(df[col])
        else:
            # Categorical column: convert to integer codes
            codes, _ = pd.factorize(df[col].astype(str), sort=True)
            out[col] = codes
    
    return out


def load_insurance_data(
    csv_path: Union[str, Path],
    target_column: str = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Load insurance dataset and auto-detect target column.
    
    Args:
        csv_path: Path to CSV file
        target_column: If None, auto-detects from column names
        random_state: Random seed for reproducibility
        
    Returns:
        Tuple of (X, y) where X is features and y is target
        
    Raises:
        FileNotFoundError: If CSV file doesn't exist
        ValueError: If target column can't be auto-detected
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    
    # Auto-detect target column if not specified
    if target_column is None or target_column not in df.columns:
        candidates = ["target", "label", "y", "class", "claimnb", "lapse"]
        col_map = {c.lower(): c for c in df.columns}
        
        detected = None
        for cand in candidates:
            if cand in col_map:
                detected = col_map[cand]
                break
        
        if detected is None:
            # Fallback: fuzzy match
            for col in df.columns:
                if "claim" in col.lower() or "lapse" in col.lower():
                    detected = col
                    break
        
        if detected is None:
            raise ValueError(
                f"Could not auto-detect target column. "
                f"Available columns: {df.columns.tolist()}"
            )
        
        target_column = detected
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y


# ============================================================================
# METRIC COMPUTATION
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred_binary: np.ndarray = None,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Compute standard classification metrics.
    
    Args:
        y_true: True binary labels (0/1)
        y_pred_proba: Predicted probabilities (floats in [0,1])
        y_pred_binary: Predicted binary labels (optional, computed from proba if not provided)
        model_name: Name for logging
        
    Returns:
        Dict with keys: 'roc_auc', 'pr_auc', 'accuracy', 'brier', 'f1'
    """
    from sklearn.metrics import (
        roc_auc_score, average_precision_score, accuracy_score,
        brier_score_loss, f1_score
    )
    
    y_true = np.asarray(y_true).ravel()
    y_pred_proba = np.asarray(y_pred_proba).ravel()
    
    if y_pred_binary is None:
        y_pred_binary = (y_pred_proba >= 0.5).astype(int)
    else:
        y_pred_binary = np.asarray(y_pred_binary).ravel()
    
    metrics = {
        'roc_auc': float(roc_auc_score(y_true, y_pred_proba)),
        'pr_auc': float(average_precision_score(y_true, y_pred_proba)),
        'accuracy': float(accuracy_score(y_true, y_pred_binary)),
        'brier': float(brier_score_loss(y_true, y_pred_proba)),
        'f1': float(f1_score(y_true, y_pred_binary)),
    }
    
    return metrics


# ============================================================================
# DATA SPLITTING AND PROCESSING
# ============================================================================

def prepare_training_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
    max_train: int = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training and test data with optional capping.
    
    Args:
        X: Feature DataFrame
        y: Target Series
        test_size: Fraction for test set (e.g., 0.20 for 80/20 split)
        random_state: Random seed
        max_train: If set, cap training set at this size (for fair comparison)
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test) as numpy arrays
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    
    # Convert y to integer labels
    if isinstance(y, pd.Series):
        y_arr = y.values
    else:
        y_arr = np.asarray(y)
    
    if y_arr.dtype.kind not in 'iu':
        le = LabelEncoder()
        y_arr = le.fit_transform(y_arr)
    
    # Ensure X is 2D array
    if isinstance(X, pd.DataFrame):
        X_arr = X.values
    else:
        X_arr = np.asarray(X)
    
    if X_arr.ndim == 1:
        X_arr = X_arr.reshape(-1, 1)
    
    # Stratified split to preserve class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X_arr, y_arr,
        test_size=test_size,
        stratify=y_arr,
        random_state=random_state
    )
    
    # Cap training set if requested
    if max_train is not None and len(X_train) > max_train:
        rng = np.random.RandomState(random_state)
        idx = rng.choice(len(X_train), size=max_train, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
    
    return X_train, X_test, y_train, y_test


# ============================================================================
# LOGGING UTILITIES
# ============================================================================

def log_section(title: str, level: int = 1):
    """
    Print formatted section header.
    
    Args:
        title: Section title
        level: Heading level (1=H1, 2=H2, etc.)
    """
    from IPython.display import display, Markdown
    
    marker = "#" * level
    display(Markdown(f"{marker} {title}"))


def log_checkpoint(message: str, status: str = "OK"):
    """
    Print checkpoint message with status.
    
    Args:
        message: Checkpoint description
        status: "OK", "WARN", "ERROR"
    """
    emoji = {"OK": "✅", "WARN": "⚠️", "ERROR": "❌"}.get(status, "•")
    print(f"{emoji} {message}")


if __name__ == "__main__":
    # Quick test of utilities
    print("baseline_utils module loaded successfully")
    print(f"integer_encode_df: {integer_encode_df}")
    print(f"load_insurance_data: {load_insurance_data}")
    print(f"compute_metrics: {compute_metrics}")
```

**Update checklist:**
- ✓ Copy/paste code into file
- ✓ Test imports: `from baseline_utils import *`
- ✓ Run quick manual test of each function

---

### 1.3 Fix Notebook 01
**Time: 15 min | File: `01_baseline_claim_classification.ipynb`**

**Changes:**
1. Add at top of notebook after imports:
   ```python
   from config import *
   from baseline_utils import *
   ```

2. Replace Cell 3 hardcoded path:
   ```python
   # OLD:
   csv_path = Path("/Users/Scott/Documents Data Science/ADSWP/TabPFN/ADSWP Project/baselining/data/freMTPL2freq.csv")
   
   # NEW:
   csv_path = FREMTPL2FREQ_PATH  # from config.py
   ```

3. Define missing `fit_and_eval()` function OR remove the call if unused

4. Remove duplicate imports (keep only first occurrence)

**Update checklist:**
- ✓ Test that notebook doesn't raise NameError on imports
- ✓ Verify CSV path resolves correctly

---

### 1.4 Update Notebook 02
**Time: 45 min | File: `02_baselining_notebook.ipynb`**

**Changes:**

1. **Cell 1:** Add config import at very top (before other imports):
   ```python
   from config import *
   from baseline_utils import *
   ```

2. **Remove/replace hardcoded constants:**
   - `RANDOM_SEED = 943321` → already in config
   - `GLOBAL_MAX_TRAIN = 10000` → already in config
   - `LOCAL_CSV_PATH = Path(...)` → use `EUDIRECTLAPSE_PATH` from config

3. **Consolidate column transformer:**
   - Remove duplicate definition
   - Use once at top, reuse throughout

4. **Replace integer encoding:**
   - OLD: Copy/paste of integer_encode_df() code
   - NEW: `X_int = integer_encode_df(X)`

5. **Replace data loading:**
   - OLD: Manual pd.read_csv() + auto-detect logic
   - NEW: `X, y = load_insurance_data(EUDIRECTLAPSE_PATH)`

**Update checklist:**
- ✓ Verify notebook runs import without errors
- ✓ Test that EUDIRECTLAPSE_PATH resolves
- ✓ Confirm all cells still execute

---

## PHASE 2: IMPORTANT (4-6 hours) 🟡

### 2.1 Create DataLoader Class
**Time: 60 min | File: `notebooks/baseline_experiments/data_loader.py`**

**Action:**
Create class to encapsulate data loading, encoding, splitting:

```python
"""
DataLoader class for insurance datasets.

Handles:
- Loading data from CSV
- Integer encoding
- Train/test splitting
- Optional rebalancing
- Capping for fair comparison
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple
from baseline_utils import integer_encode_df


class InsuranceDataLoader:
    """Load and preprocess insurance data with consistent pipeline."""
    
    def __init__(self, csv_path, target_column=None, random_state=42):
        """
        Initialize loader.
        
        Args:
            csv_path: Path to CSV file
            target_column: Target column name (auto-detected if None)
            random_state: Random seed
        """
        self.csv_path = Path(csv_path)
        self.random_state = random_state
        self._validate_path()
        
        self.X, self.y = self._load_data(target_column)
        
    def _validate_path(self):
        """Check that CSV file exists."""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.csv_path}")
    
    def _load_data(self, target_column):
        """Load and detect target column."""
        df = pd.read_csv(self.csv_path)
        
        if target_column is None or target_column not in df.columns:
            # Auto-detect (same logic as baseline_utils.load_insurance_data)
            candidates = ["target", "label", "y", "class", "claimnb", "lapse"]
            col_map = {c.lower(): c for c in df.columns}
            
            for cand in candidates:
                if cand in col_map:
                    target_column = col_map[cand]
                    break
        
        if target_column not in df.columns:
            raise ValueError(f"Target column '{target_column}' not found")
        
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        return X, y
    
    def split(self, test_size=0.20, stratify=True):
        """
        Split data into train/test.
        
        Returns:
            (X_train, X_test, y_train, y_test) as DataFrames
        """
        from sklearn.model_selection import train_test_split
        
        if stratify:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y,
                test_size=test_size,
                stratify=self.y,
                random_state=self.random_state
            )
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y,
                test_size=test_size,
                random_state=self.random_state
            )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        return X_train, X_test, y_train, y_test
    
    def encode_integers(self):
        """
        Convert to integer-encoded numpy arrays.
        
        Returns:
            (X_train_int, X_test_int, y_train_arr, y_test_arr)
        """
        # Encode features
        X_train_int = integer_encode_df(self.X_train).values
        X_test_int = integer_encode_df(self.X_test).values
        
        # Convert y to numpy
        y_train_arr = np.asarray(self.y_train).ravel()
        y_test_arr = np.asarray(self.y_test).ravel()
        
        return X_train_int, X_test_int, y_train_arr, y_test_arr
    
    def cap_training(self, max_samples):
        """
        Cap training set size for fair comparison.
        
        Args:
            max_samples: Maximum training samples
        """
        if len(self.X_train) > max_samples:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(self.X_train), size=max_samples, replace=False)
            self.X_train = self.X_train.iloc[idx]
            self.y_train = self.y_train.iloc[idx]
    
    def rebalance(self, mode):
        """
        Rebalance training data.
        
        Args:
            mode: 'undersample', 'oversample', or 'smote'
        """
        if mode is None:
            return
        
        from sklearn.utils import resample
        from imblearn.over_sampling import SMOTE
        
        y_arr = np.asarray(self.y_train).ravel()
        
        if mode == 'undersample':
            # Undersample majority class
            idx_0 = np.where(y_arr == 0)[0]
            idx_1 = np.where(y_arr == 1)[0]
            n_minority = len(idx_1)
            
            rng = np.random.RandomState(self.random_state)
            idx_0_sampled = rng.choice(idx_0, size=n_minority, replace=False)
            idx_rebalanced = np.concatenate([idx_0_sampled, idx_1])
            
            self.X_train = self.X_train.iloc[idx_rebalanced]
            self.y_train = self.y_train.iloc[idx_rebalanced]
        
        elif mode == 'oversample':
            # Oversample minority class
            idx_0 = np.where(y_arr == 0)[0]
            idx_1 = np.where(y_arr == 1)[0]
            n_majority = len(idx_0)
            
            rng = np.random.RandomState(self.random_state)
            idx_1_sampled = rng.choice(idx_1, size=n_majority, replace=True)
            idx_rebalanced = np.concatenate([idx_0, idx_1_sampled])
            
            self.X_train = self.X_train.iloc[idx_rebalanced]
            self.y_train = self.y_train.iloc[idx_rebalanced]
        
        elif mode == 'smote':
            # SMOTE (requires imbalanced-learn)
            X_arr = np.asarray(self.X_train)
            y_arr = np.asarray(self.y_train).ravel()
            
            smote = SMOTE(random_state=self.random_state)
            X_balanced, y_balanced = smote.fit_resample(X_arr, y_arr)
            
            self.X_train = pd.DataFrame(X_balanced, columns=self.X_train.columns)
            self.y_train = pd.Series(y_balanced)


if __name__ == "__main__":
    # Example usage
    from config import EUDIRECTLAPSE_PATH
    
    loader = InsuranceDataLoader(EUDIRECTLAPSE_PATH)
    print(f"✓ Loaded dataset: {loader.X.shape}")
    
    X_train, X_test, y_train, y_test = loader.split()
    print(f"✓ Split: {X_train.shape} train, {X_test.shape} test")
    
    loader.cap_training(10000)
    print(f"✓ Capped: {loader.X_train.shape} train")
    
    X_tr_int, X_te_int, y_tr, y_te = loader.encode_integers()
    print(f"✓ Encoded: {X_tr_int.shape} integer features")
```

**Update checklist:**
- ✓ Test DataLoader with actual dataset
- ✓ Verify split preserves class distribution
- ✓ Test cap and rebalance operations

---

### 2.2 Modularize Notebook 02
**Time: 90 min | Multiple changes**

**Changes:**

1. **Extract diagnostic function** (replace 200+ line cell):
   ```python
   def run_dataset_diagnostics(X, y, sample_name="Training"):
       """Run comprehensive dataset diagnostics."""
       # (move all the diagnostic code here)
       return diagnostics_dict
   ```

2. **Extract model training loop** (replace 150+ line cell):
   ```python
   def train_all_models(X_train_capped, y_train_capped, X_test, y_test, models_dict):
       """Train all models and return results."""
       results = []
       for name, model in models_dict.items():
           results.append(train_and_evaluate_model(name, model, ...))
       return pd.DataFrame(results)
   ```

3. **Extract TabPFN training** (separate from baselines):
   ```python
   def train_tabpfn(X_train_int, y_train, X_test_int, y_test):
       """Train TabPFN and return predictions/metrics."""
       # (isolated TabPFN training logic)
       return tab, proba, metrics
   ```

4. **Extract calibration** (into function):
   ```python
   def apply_calibration(raw_probs, y_train_cal, method='isotonic'):
       """Apply calibration to raw probabilities."""
       if method == 'isotonic':
           iso = IsotonicRegression(out_of_bounds='clip')
           iso.fit(raw_probs_cal, y_train_cal)
           return iso.predict(raw_probs)
   ```

5. **Extract reporting** (tables/figures):
   ```python
   def generate_tables(res_df, metrics_dict):
       """Generate and save Table 1-4."""
       # (movement code for csv saving)
   ```

**Update checklist:**
- ✓ Verify all extracted functions work in isolation
- ✓ Test end-to-end notebook execution
- ✓ Ensure outputs match original

---

### 2.3 Update Notebooks 03 and 05
**Time: 30 min each**

**Changes (same for both):**

1. Add import statements at top:
   ```python
   from config import *
   from baseline_utils import *
   # from data_loader import InsuranceDataLoader  # if 2.1 completed
   ```

2. Replace hardcoded paths with config constants
3. Replace integer_encode_df() calls with imports
4. Replace metric computation with `compute_metrics()` from utils

**Update checklist:**
- ✓ Test imports
- ✓ Verify paths resolve
- ✓ Execute at least one cell to confirm

---

## PHASE 3: NICE-TO-HAVE (2-3 hours) 🟢

### 3.1 Add Docstrings
**Time: 45 min**

Add docstrings to functions lacking them:
- `train_and_evaluate_model()`
- `build_models()`
- `build_logistic_pipeline()`
- Any others defined in notebooks

Format:
```python
def function_name(arg1, arg2) -> output_type:
    """
    Brief one-liner description.
    
    Detailed multi-line explanation if needed.
    
    Args:
        arg1: Description of arg1
        arg2: Description of arg2
        
    Returns:
        Description of return value
        
    Raises:
        ExceptionType: When this exception occurs
    """
    # implementation
```

---

### 3.2 Add Dependency Documentation
**Time: 20 min**

Add markdown cell at TOP of each notebook:

```markdown
## 📋 Notebook Dependencies

### Inputs Required
- `config.py` (shared configuration)
- `baseline_utils.py` (shared utilities)
- `data/eudirectlapse.csv` (22k rows)
- Optional: HuggingFace token (for TabPFN client)

### Outputs Generated
- CSV files: `outputs/tables/*.csv` (if 02)
- PNG files: `outputs/figures/*.png` (if 02)
- CSV files: Timestamped results (if 03)

### Execution Order
1. **02_baselining_notebook** (20-30 min) - MUST RUN FIRST
2. **02b_baselining_summary** (<1 min) - Optional reference
3. **03_finetuning_notebook** (30-45 min) - Optional optimization
4. **01, 05** - Can run anytime (independent)

### Time Estimates
| Notebook | Time | Backend | Notes |
|----------|------|---------|-------|
| 02 | 20-30 min | Client (fast), Local GPU (slow) | Primary |
| 02b | <1 min | N/A | Reads 02 outputs |
| 03 | 30-45 min | Client API | Builds on 02 |
| 01 | 5-10 min | Client or Local | Standalone |
| 05 | 10-20 min | Client API | Standalone |
```

---

### 3.3 Clean Up Output
**Time: 30 min**

1. Remove excessive `print()` statements
2. Consolidate `display(Markdown())` calls
3. Add clear section breaks
4. Use consistent formatting

---

## PHASE 4: OPTIONAL FUTURE WORK 🌟

### 4.1 Create Experiment Runner Script
**File: `run_experiments.py`**

```bash
# Usage:
python run_experiments.py --backend client --seed 943321 --run 02,03
# Loads notebooks, executes in order, saves results
```

### 4.2 Consolidate 02 and 02b
**Status:** Future consideration
- Merge 02 + 02b into single comprehensive notebook
- Keep distinction between "full run" and "summary only" modes

### 4.3 Add Unit Tests
**Status:** Future consideration
- Test `baseline_utils.py` functions
- Test `DataLoader` class
- Quick validation before running expensive notebooks

### 4.4 Create CI/CD Pipeline
**Status:** Future consideration
- GitHub Actions to auto-run notebooks
- Validate outputs match expected schema
- Alert on breaking changes

---

## VERIFICATION CHECKLIST

After completing each phase, verify:

### Phase 1 Verification ✓
- [ ] `config.py` created and all paths valid
- [ ] `baseline_utils.py` created with all functions
- [ ] `from config import *` works in notebooks
- [ ] `from baseline_utils import integer_encode_df` works
- [ ] Notebook 01 runs without NameError
- [ ] Notebook 02 first 3 cells run without error
- [ ] EUDIRECTLAPSE_PATH resolves correctly

### Phase 2 Verification ✓
- [ ] `DataLoader` class works end-to-end
- [ ] Extracted functions run in isolation
- [ ] Notebook 02 still produces same outputs as before
- [ ] Notebook 03 imports and runs first cell
- [ ] Notebook 05 imports and runs first cell

### Phase 3 Verification ✓
- [ ] Docstrings added to all custom functions
- [ ] Dependency docs added to all notebooks
- [ ] Output reduced and consolidated
- [ ] Section breaks are clear

---

## TROUBLESHOOTING

### Common Issues & Fixes

**Issue:** `ModuleNotFoundError: No module named 'config'`
- **Fix:** Ensure `config.py` is in same directory as notebooks
- **Verify:** `ls notebooks/baseline_experiments/config.py`

**Issue:** `FileNotFoundError: Dataset not found`
- **Fix:** Update `DATA_DIR` path in `config.py` to your actual path
- **Verify:** `python -c "from config import EUDIRECTLAPSE_PATH; print(EUDIRECTLAPSE_PATH.exists())"`

**Issue:** `ImportError: cannot import name 'integer_encode_df'`
- **Fix:** Add `from baseline_utils import integer_encode_df` to notebook
- **Verify:** `from baseline_utils import *` works

**Issue:** Notebook runs but data path wrong
- **Fix:** `config.py` uses `Path.home()` which may not match your setup
- **Solution:** Print `print(Path.home())` and update if needed

---

## DELIVERABLES BY PHASE

### Phase 1 Deliverables
- [ ] `config.py` (150 lines)
- [ ] `baseline_utils.py` (200 lines)
- [ ] Updated notebooks 01, 02 (import config/utils)

### Phase 2 Deliverables
- [ ] `data_loader.py` (200 lines)
- [ ] Refactored notebook 02 (extracted functions)
- [ ] Updated notebooks 03, 05 (use utils)

### Phase 3 Deliverables
- [ ] Updated docstrings (all functions)
- [ ] Dependency documentation (all notebooks)
- [ ] Cleaned output (removed noise)

### Phase 4 Deliverables (Optional)
- [ ] `run_experiments.py` (experiment runner)
- [ ] `tests/test_baseline_utils.py` (unit tests)
- [ ] `.github/workflows/validate_notebooks.yml` (CI/CD)

---

## TIME BREAKDOWN

| Phase | Hours | Priority | Status |
|-------|-------|----------|--------|
| Phase 1 | 2-3 | CRITICAL | ⭐⭐⭐ |
| Phase 2 | 4-6 | HIGH | ⭐⭐⭐ |
| Phase 3 | 2-3 | MEDIUM | ⭐⭐ |
| Phase 4 | 3-5 | LOW | ⭐ |
| **TOTAL** | **11-17** | - | - |

**Recommended:** Complete Phases 1 & 2 (6-9 hours) for solid ROI. Phases 3 & 4 are optional but recommended for long-term maintenance.

---

## MAINTENANCE GOING FORWARD

After refactoring, maintain these practices:

1. **Always update config.py first** when adding new parameters
2. **Use baseline_utils functions** instead of repeated code
3. **Keep notebooks modular** - extract complex logic to functions
4. **Document dependencies** in markdown cell at top
5. **Use DataLoader for data ops** instead of manual pd.read_csv()
6. **Run phases 1-2 only** when onboarding new team members

---

