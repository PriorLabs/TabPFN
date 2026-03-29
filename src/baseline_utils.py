"""
Utility functions for baseline experiments.

Centralizes commonly used functions to avoid duplication across notebooks.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    brier_score_loss,
)
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


class BaselineConfig:
    """Configuration wrapper for easier access in functions."""
    
    def __init__(self, config_module):
        self.config = config_module
        self.set_seeds()
    
    def set_seeds(self):
        """Set random seeds for reproducibility."""
        self.config.set_random_seeds()


# ============================================================================
# DATA LOADING & PREPROCESSING
# ============================================================================

def load_data(
    filepath: str,
    drop_cols: Optional[list] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Load CSV data and perform basic cleaning.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    drop_cols : list, optional
        Columns to drop from the dataset
    verbose : bool
        Print loading information
        
    Returns
    -------
    pd.DataFrame
        Loaded and cleaned data
    """
    df = pd.read_csv(filepath)
    
    if verbose:
        print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")
    
    if drop_cols:
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])
        if verbose:
            print(f"Dropped {len(drop_cols)} columns")
    
    return df


def integer_encode_df(df_in: pd.DataFrame) -> pd.DataFrame:
    """
    Integer encode categorical columns in DataFrame.
    
    Converts string/object columns to integer codes.
    
    Parameters
    ----------
    df_in : pd.DataFrame
        Input DataFrame with categorical columns
        
    Returns
    -------
    pd.DataFrame
        DataFrame with integer-encoded categorical columns
    """
    out = pd.DataFrame(index=df_in.index)
    
    for col in df_in.columns:
        if df_in[col].dtype == "object":
            out[col] = pd.factorize(df_in[col])[0]
        else:
            out[col] = df_in[col]
    
    return out


def cap_outliers(
    df: pd.DataFrame,
    percentiles: Dict[str, Tuple[float, float]],
) -> pd.DataFrame:
    """
    Cap outliers using percentile thresholds.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    percentiles : dict
        Dictionary with column names as keys and (lower, upper) percentiles as values.
        Example: {"age": (5, 95), "income": (1, 99)}
        
    Returns
    -------
    pd.DataFrame
        DataFrame with outliers capped
    """
    df_capped = df.copy()
    
    for col, (lower, upper) in percentiles.items():
        if col in df_capped.columns:
            p_lower = df_capped[col].quantile(lower / 100)
            p_upper = df_capped[col].quantile(upper / 100)
            df_capped[col] = df_capped[col].clip(p_lower, p_upper)
    
    return df_capped


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    method: str = "standard",
) -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Scale features using specified method.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    method : str
        Scaling method: 'standard', 'minmax', or 'robust'
        
    Returns
    -------
    tuple
        (X_train_scaled, X_test_scaled, scaler)
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index,
    )
    
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index,
    )
    
    return X_train_scaled, X_test_scaled, scaler


# ============================================================================
# EVALUATION METRICS
# ============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities (shape: (n_samples,) for binary)
    y_pred : np.ndarray, optional
        Hard predictions. If None, computed from y_pred_proba using 0.5 threshold
        
    Returns
    -------
    dict
        Dictionary with metric names and values
    """
    if y_pred is None:
        y_pred = (y_pred_proba >= 0.5).astype(int)
    
    metrics = {
        "auc": roc_auc_score(y_true, y_pred_proba),
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "brier_score": brier_score_loss(y_true, y_pred_proba),
    }
    
    return metrics


def calibration_error(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute expected calibration error (ECE).
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels
    y_pred_proba : np.ndarray
        Predicted probabilities
        
    Returns
    -------
    float
        Expected calibration error
    """
    n_bins = 10
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    
    ece = 0
    for i in range(n_bins):
        mask = (y_pred_proba >= bin_edges[i]) & (y_pred_proba < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_acc = (y_true[mask] == (y_pred_proba[mask] >= 0.5)).mean()
            bin_conf = y_pred_proba[mask].mean()
            ece += np.abs(bin_acc - bin_conf) * mask.sum() / len(y_true)
    
    return ece


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def fit_and_evaluate(
    model,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model",
) -> Dict[str, float]:
    """
    Fit model and evaluate on test set.
    
    Parameters
    ----------
    model : sklearn-compatible model
        Model with fit() and predict_proba() methods
    X_train : pd.DataFrame
        Training features
    X_test : pd.DataFrame
        Test features
    y_train : np.ndarray
        Training labels
    y_test : np.ndarray
        Test labels
    model_name : str
        Name of the model (for logging)
        
    Returns
    -------
    dict
        Dictionary with evaluation metrics
    """
    try:
        # Fit model
        model.fit(X_train, y_train)
        
        # Get predictions
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        
        # Compute metrics
        metrics = compute_metrics(y_test, y_pred_proba, y_pred)
        metrics["model_name"] = model_name
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error fitting {model_name}: {str(e)}")
        raise


def fit_and_eval(
    model_name: str,
    model,
    X_train: Optional[np.ndarray] = None,
    X_test: Optional[np.ndarray] = None,
    y_train: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
) -> Dict:
    """
    Fit and evaluate model, returning predictions in dict format.
    
    Useful for TabPFN and other models that may accept numpy arrays.
    Returns both probabilities and binary predictions.
    
    Parameters
    ----------
    model_name : str
        Name of the model
    model : sklearn-compatible model
        Model with fit() and predict_proba() or predict() methods
    X_train : np.ndarray or pd.DataFrame, optional
        Training features. If None, uses global X_train (notebook context)
    X_test : np.ndarray or pd.DataFrame, optional
        Test features. If None, uses global X_test (notebook context)
    y_train : np.ndarray, optional
        Training labels. If None, uses global y_train (notebook context)
    y_test : np.ndarray, optional
        Test labels. If None, uses global y_test (notebook context)
        
    Returns
    -------
    dict
        Dictionary with keys:
        - 'proba': predicted probabilities for positive class
        - 'preds_bin': binary predictions (0/1)
        - 'model_name': name of the model
    """
    # Ensure required data is provided explicitly
    if X_train is None or X_test is None or y_train is None or y_test is None:
        raise ValueError(
            "fit_and_eval requires X_train, X_test, y_train, and y_test to be "
            "provided explicitly. None values are not supported."
        )
    try:
        # Fit model
        model.fit(X_train, y_train)
        
        # Get predictions
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_proba = model.predict(X_test)
        
        if hasattr(model, 'predict'):
            y_pred_bin = model.predict(X_test)
        else:
            y_pred_bin = (y_pred_proba >= 0.5).astype(int)
        
        return {
            'proba': y_pred_proba,
            'preds_bin': y_pred_bin,
            'model_name': model_name,
        }
        
    except Exception as e:
        logger.error(f"Error fitting {model_name}: {str(e)}")
        raise


# ============================================================================
# VISUALIZATION HELPERS
# ============================================================================

def create_metric_summary_table(
    results_dict: Dict[str, Dict[str, float]],
    round_decimals: int = 4,
) -> pd.DataFrame:
    """
    Create summary table from model evaluation results.
    
    Parameters
    ----------
    results_dict : dict
        Dictionary with model names as keys and metrics dicts as values
    round_decimals : int
        Number of decimals to round to
        
    Returns
    -------
    pd.DataFrame
        Summary table with models as rows and metrics as columns
    """
    df = pd.DataFrame(results_dict).T
    return df.round(round_decimals)


def save_results(
    df: pd.DataFrame,
    filepath: str,
    verbose: bool = True,
) -> None:
    """
    Save DataFrame to CSV.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to save
    filepath : str
        Output file path
    verbose : bool
        Print confirmation message
    """
    df.to_csv(filepath)
    if verbose:
        print(f"Saved results to: {filepath}")


if __name__ == "__main__":
    print("Baseline utilities module loaded successfully")
    print("Available functions:")
    print("  - load_data()")
    print("  - integer_encode_df()")
    print("  - cap_outliers()")
    print("  - scale_features()")
    print("  - compute_metrics()")
    print("  - calibration_error()")
    print("  - fit_and_evaluate()")
    print("  - create_metric_summary_table()")
    print("  - save_results()")
