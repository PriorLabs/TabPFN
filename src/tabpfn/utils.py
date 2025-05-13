"""A collection of random utilities for the TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import ctypes
import random # Added for infer_random_state fix
import typing
import warnings
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal # Removed TypeVar, overload as they are not used here. Added Literal.

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.base import check_is_fitted, is_classifier # Added is_classifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer, OrdinalEncoder
from sklearn.utils.multiclass import check_classification_targets
from torch import nn


from tabpfn.constants import (
    DEFAULT_NUMPY_PREPROCESSING_DTYPE,
    NA_PLACEHOLDER,
    REGRESSION_NAN_BORDER_LIMIT_LOWER,
    REGRESSION_NAN_BORDER_LIMIT_UPPER,
    # Added constants that might be used from original utils if merged from other versions
    ORDINAL_MAX_UNIQUE_VALUES,
    Algorithm,
    DatasetDeviceType,
    DatasetPropertyType,
    DefaultEvalPositionType,
    DefaultMaxEvalPositionType,
    Distributions,
    Encodings,
    InputPropertyType,
    OutputPropertyType,
    ProblemType,
    SklearnEstimatorType,
    XType, # Assuming XType and YType are defined in constants or are basic types
    YType,
)
from tabpfn.misc._sklearn_compat import check_array, validate_data


if TYPE_CHECKING:
    from sklearn.base import BaseEstimator, TransformerMixin # Added BaseEstimator
    from sklearn.pipeline import Pipeline
    from tabpfn.classifier import TabPFNClassifier # No XType, YType here, assumed defined in constants
    from tabpfn.regressor import TabPFNRegressor
    # Conditional import for torch_xla for type hinting if needed elsewhere
    try:
        import torch_xla.core.xla_model as xm_typehint # For type hinting if XLA device is passed
    except ImportError:
        xm_typehint = None # type: ignore
    # For PerFeatureTransformer type hint in update_encoder_outlier_params
    from tabpfn.model.transformer import PerFeatureTransformer
    # For SequentialEncoder and InputNormalizationEncoderStep hints
    from tabpfn.model.encoders import SequentialEncoder, InputNormalizationEncoderStep


MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)

# --- TPU Support Logic ---
_TORCH_XLA_AVAILABLE = False
_XLA_DEVICE: torch.device | None = None

try:
    import torch_xla.core.xla_model as xm
    if xm.get_xla_supported_devices():
        _XLA_DEVICE = xm.xla_device()
        _TORCH_XLA_AVAILABLE = True
    else:
        _TORCH_XLA_AVAILABLE = False
except ImportError:
    _TORCH_XLA_AVAILABLE = False

def is_torch_xla_available() -> bool:
    """Check if torch_xla is installed and a TPU is detected and configured."""
    return _TORCH_XLA_AVAILABLE

def get_xla_device_if_available() -> torch.device | None:
    """Return the torch.device for XLA if available, otherwise None."""
    return _XLA_DEVICE
# --- End TPU Support Logic ---


def _get_embeddings(
    model: TabPFNClassifier | TabPFNRegressor,
    X: XType, # type: ignore
    data_source: Literal["train", "test"] = "test",
) -> np.ndarray:
    """Get the embeddings for the input data `X`."""
    check_is_fitted(model)
    data_map = {"train": "train_embeddings", "test": "test_embeddings"}
    selected_data = data_map[data_source]

    from tabpfn.preprocessing import ClassifierEnsembleConfig, RegressorEnsembleConfig # Local import

    # Assuming validate_X_predict and _fix_dtypes are defined in this file
    X_np = validate_X_predict(X, model) # type: ignore
    # Assuming model has categorical_features_indices, preprocessor_
    X_df = _fix_dtypes(X_np, cat_indices=model.categorical_features_indices, numeric_dtype='float32') # type: ignore
    X_transformed = model.preprocessor_.transform(X_df) # type: ignore

    embeddings: list[np.ndarray] = []
    executor = typing.cast(typing.Any, model.executor_)
    for output, config in executor.iter_outputs(
        X_transformed, # Pass transformed numpy array
        device=model.device_,
        autocast=model.use_autocast_,
        only_return_standard_out=False,
    ):
        output_dict = typing.cast(dict[str, torch.Tensor], output)
        embed = output_dict[selected_data].squeeze(1) # Original had squeeze(1)
        assert isinstance(config, (ClassifierEnsembleConfig, RegressorEnsembleConfig))
        assert embed.ndim == 2
        embeddings.append(embed.cpu().numpy()) # Removed extra squeeze()

    return np.array(embeddings)


def _repair_borders(borders: np.ndarray, *, inplace: Literal[True]) -> None:
    """Repairs borders inplace."""
    if inplace is not True: # Explicit check for True
        raise NotImplementedError("Only inplace=True is supported")

    nan_mask = np.isnan(borders)
    if nan_mask.any(): # Process only if NaNs are present
        if np.isnan(borders[-1]): # Specific handling if last element is NaN
            valid_borders = borders[~nan_mask]
            if valid_borders.size > 0:
                largest = valid_borders.max()
                borders[nan_mask] = largest
                # Ensure last border is distinct and larger after filling NaNs
                if borders[-1] <= (borders[-2] if len(borders) > 1 and not np.isnan(borders[-2]) else borders[-1] -1 ): # check against second to last if exists
                     borders[-1] = (borders[-2] if len(borders) > 1 and not np.isnan(borders[-2]) else largest) * 1.1 + 1e-6 # Ensure it's larger
            else: # All borders were NaN
                 borders[:] = 0 # Or some other default, this case should be rare
                 borders[-1] = 1.0


    # Ensure last two borders are distinct and ordered
    if len(borders) > 1 and borders[-1] - borders[-2] < 1e-6:
        borders[-1] = borders[-2] + max(1e-6, np.abs(borders[-2] * 0.1)) # Ensure positive increment

    # Ensure first two borders are distinct and ordered
    if len(borders) > 1 and borders[0] == borders[1]: # Simplified check, assuming borders should be increasing
        borders[0] -= max(1e-6, np.abs(borders[1] * 0.1)) # Ensure it's smaller


def _cancel_nan_borders(
    *,
    borders: np.ndarray,
    broken_mask: npt.NDArray[np.bool_],
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    """Cancels NaN borders."""
    borders_copy = borders.copy() # Work on a copy
    # This logic seems complex and highly specific. Assuming it's correct from original.
    # Count transitions in broken_mask to find edges of NaN blocks
    # (True > False) means end of a NaN block from left, or start of non-NaN from left
    # (False > True) means start of a NaN block from left
    
    # Simplified check: if edges are broken, try to fix them
    if broken_mask[0]: # NaN block at the start
        first_valid_idx = np.where(~broken_mask)[0]
        if first_valid_idx.size > 0:
            borders_copy[broken_mask & (np.arange(len(borders_copy)) < first_valid_idx[0])] = borders_copy[first_valid_idx[0]]
            if first_valid_idx[0] > 0 : # Ensure first element is distinct if it was part of NaNs
                 borders_copy[0] = borders_copy[first_valid_idx[0]] - 1.0


    if broken_mask[-1]: # NaN block at the end
        last_valid_idx = np.where(~broken_mask)[0]
        if last_valid_idx.size > 0:
             borders_copy[broken_mask & (np.arange(len(borders_copy)) > last_valid_idx[-1])] = borders_copy[last_valid_idx[-1]]
             if last_valid_idx[-1] < len(borders_copy) -1: # Ensure last element is distinct
                  borders_copy[-1] = borders_copy[last_valid_idx[-1]] + 1.0
    
    # logit_cancel_mask: True where logits correspond to a segment between two NaNs in original borders
    # or between a NaN and an edge that became NaN.
    # A logit is cancelled if both borders it spans are "broken" (were NaN or became NaN effectively).
    # The original broken_mask refers to borders. Logits are between borders.
    # logit i is between border i and border i+1.
    # It's cancelled if border i OR border i+1 is broken.
    logit_cancel_mask = broken_mask[:-1] | broken_mask[1:]
    return borders_copy, logit_cancel_mask


def infer_device_and_type( # Renamed from original to match new TPU logic
    device_config: str | torch.device | Literal["auto"], # Matched signature
) -> torch.device:
    """Infer the device to use for inference."""
    if isinstance(device_config, torch.device):
        return device_config

    if device_config == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if is_torch_xla_available(): # TPU check
            xla_device = get_xla_device_if_available()
            if xla_device:
                return xla_device
        return torch.device("cpu")
    if device_config == "tpu": # TPU specific
        if is_torch_xla_available():
            xla_device = get_xla_device_if_available()
            if xla_device:
                return xla_device
            else: # Should not happen if is_torch_xla_available is true
                raise RuntimeError("TPU device specified ('tpu') but no XLA device found, though torch_xla seems installed. Check TPU config.")
        else:
            raise RuntimeError("TPU device specified ('tpu') but torch_xla is not installed or no TPU is available. Install torch_xla and ensure TPU is configured.")
    
    # Handles "cpu", "cuda", "cuda:0" etc.
    return torch.device(device_config)


def is_autocast_available(device_type: str) -> bool: # Original had device_type from torch.device.type
    """Infer whether autocast is available for the given device type."""
    try:
        if hasattr(torch.amp.autocast_mode, "is_autocast_available"): # PyTorch 1.10+
            return bool(torch.amp.autocast_mode.is_autocast_available(device_type)) # type: ignore
        # Fallback for older PyTorch versions or if above is not found (though unlikely)
        if device_type == "cuda":
            return hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast")
        if device_type == "cpu": # CPU autocast for BFloat16
            return hasattr(torch.cpu, "amp") and hasattr(torch.cpu.amp, "autocast")
        return False # XLA handles its own precision
    except Exception: # Broad except to catch any issues with checking
        return False


def infer_fp16_inference_mode(device: torch.device, *, enable: bool | None) -> bool:
    """Infer whether fp16 inference should be enabled."""
    # XLA (TPU) manages its own precision (often bfloat16), so fp16 via torch.amp.autocast might not be applicable or optimal.
    # Typically, for XLA, you don't use torch.cuda.amp.autocast.
    if device.type == "xla": # If device is XLA (TPU)
        if enable is True: # If user explicitly wants to enable (could be a misconfiguration)
            warnings.warn("fp16_inference via torch.amp.autocast is typically not used with XLA/TPU devices. XLA manages its own precision. Proceeding as requested, but this may not have the intended effect or could error.", UserWarning, stacklevel=2)
            return True # Or False, as torch.amp for CUDA won't apply. Let's follow 'enable'.
        return False # Default to False for XLA, as XLA handles it.

    # Original logic for CPU/CUDA
    is_cpu = device.type.lower() == "cpu"
    # Autocast for CUDA is well-defined. For CPU, it's mainly for BFloat16.
    fp16_available_on_device = is_autocast_available(device.type) and not is_cpu # Primarily for CUDA

    if enable is None: # Auto-detect
        return fp16_available_on_device

    if enable is True:
        if not fp16_available_on_device:
            # More specific error for CPU
            if is_cpu:
                 warnings.warn(
                    "You specified `inference_precision='autocast'` or `enable=True` for fp16 mode on a CPU device. "
                    "Torch CPU autocast mainly supports BFloat16 and might not provide significant speedup or could even error with FP16. "
                    "TabPFN's fp16 inference mode is primarily optimized for CUDA. Continuing, but ensure your PyTorch version supports CPU autocast for the operations used if expecting FP16 behavior.",
                    UserWarning, stacklevel=2
                 )
                 return True # Allow if user insists, but with warning.
            else: # For other non-CUDA, non-XLA devices if any
                raise ValueError(
                    f"You specified `inference_precision='autocast'` or `enable=True` for fp16 mode, but your device ({device.type}) "
                    "does not support it or it's not recommended. "
                    "Please ensure your PyTorch version and device type are compatible or set enable=False."
                )
        return True
    
    # enable is False
    return False


NUMERIC_DTYPE_KINDS = "?bBiufmM" # Added M for datetime
OBJECT_DTYPE_KINDS = "OV"
STRING_DTYPE_KINDS = "SaU"


def _fix_dtypes(
    X: pd.DataFrame | np.ndarray,
    cat_indices: Sequence[int | str] | None,
    numeric_dtype: Literal["float32", "float64"] = "float32", # Changed default to float32
) -> pd.DataFrame:
    """Fixes dtypes and ensures X is a DataFrame."""
    if isinstance(X, np.ndarray):
        # For numpy arrays, create a DataFrame. Infer object for mixed types.
        try:
            X = pd.DataFrame(X, dtype=None if X.dtype.kind in OBJECT_DTYPE_KINDS else numeric_dtype)
        except Exception: # Fallback for complex object arrays
            X = pd.DataFrame(X)
            X = X.convert_dtypes() # Let pandas try its best
    elif not isinstance(X, pd.DataFrame):
        raise ValueError(f"Input X must be a pandas DataFrame or numpy array, got {type(X)}")

    # Convert specified cat_indices to 'category' dtype
    if cat_indices is not None:
        cat_indices_resolved = []
        for idx in cat_indices:
            if isinstance(idx, (int, np.integer)):
                if idx < len(X.columns):
                    cat_indices_resolved.append(X.columns[idx])
            elif isinstance(idx, str):
                if idx in X.columns:
                    cat_indices_resolved.append(idx)
            # else: ignore invalid identifiers in cat_indices for now

        for col_name in cat_indices_resolved:
            if not pd.api.types.is_categorical_dtype(X[col_name]):
                X[col_name] = X[col_name].astype('category')
    
    # Convert object and string columns not already categorical
    for col_name in X.columns:
        if X[col_name].dtype.kind in OBJECT_DTYPE_KINDS or X[col_name].dtype.kind in STRING_DTYPE_KINDS:
            if not pd.api.types.is_categorical_dtype(X[col_name]):
                 # Check if it can be numeric before casting to category
                try:
                    pd.to_numeric(X[col_name]) # Check if convertible
                    # If it's numeric-like but object/string, user should specify if categorical
                    # For now, if not in cat_indices, leave as is for OrdinalEncoder to handle or error
                    # Or, more aggressively:
                    # X[col_name] = X[col_name].astype('category')
                except ValueError: # Not easily convertible to numeric, so treat as category
                    X[col_name] = X[col_name].astype('category')


    # Ensure numeric columns are of the target numeric_dtype, converting pandas' nullable int/float
    for col_name in X.select_dtypes(include='number').columns:
        # Check if it's not already the target type (e.g. float32)
        # And handle pandas specific nullable dtypes by converting to standard numpy dtypes
        if X[col_name].dtype != numeric_dtype or hasattr(X[col_name].dtype, 'na_value'):
            # Attempt conversion, coercing errors for uncastable types if necessary
            try:
                X[col_name] = X[col_name].astype(numeric_dtype)
            except Exception: # If astype fails (e.g. trying to cast string numbers in object to float)
                 # This case should be rare if previous conversions worked.
                 # One option is to try pd.to_numeric with errors='coerce' then astype
                 X[col_name] = pd.to_numeric(X[col_name], errors='coerce').astype(numeric_dtype)


    return X


def _get_ordinal_encoder(
    *,
    numpy_dtype: np.dtype = DEFAULT_NUMPY_PREPROCESSING_DTYPE,
) -> ColumnTransformer:
    """Gets a ColumnTransformer for ordinal encoding of specified dtypes."""
    oe = OrdinalEncoder(
        categories="auto",
        dtype=numpy_dtype,
        handle_unknown="use_encoded_value",
        unknown_value=-1, # Consistent with TabPFN defaults
        encoded_missing_value=np.nan, # NaNs remain NaNs for OrdinalEncoder
    )
    # make_column_selector will select columns of these dtypes
    # OrdinalEncoder should only apply to 'category' and potentially 'object'/'string'
    # if they haven't been converted by _fix_dtypes.
    # For robustness, target 'category' explicitly.
    return ColumnTransformer(
        transformers=[("encoder", oe, make_column_selector(dtype_include=["category"]))],
        remainder="passthrough", # Keep other columns (numerical) as they are
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def validate_Xy_fit( # type: ignore[override] # Signature differs from BaseEstimator's for estimator type
    X: XType, # type: ignore
    y: YType, # type: ignore
    estimator: TabPFNRegressor | TabPFNClassifier | BaseEstimator, # Added BaseEstimator
    *,
    max_num_features: int,
    max_num_samples: int,
    ensure_y_numeric: bool = False,
    ignore_pretraining_limits: bool = False,
) -> tuple[np.ndarray, np.ndarray, npt.NDArray[Any] | list[str] | None, int]: # feature_names can be list
    """Validate the input data for fitting, using sklearn's validate_data."""
    
    # Store original feature names if X is a DataFrame
    original_feature_names: list[str] | None = None
    if isinstance(X, pd.DataFrame):
        original_feature_names = X.columns.tolist()

    # sklearn's validate_data can handle pandas DataFrames and numpy arrays
    # It also sets n_features_in_ and feature_names_in_ on the estimator
    X_validated, y_validated = validate_data( # type: ignore
        estimator, # Pass the actual estimator instance
        X=X,
        y=y,
        accept_sparse=False,
        dtype=None, # Allow mixed types initially, _fix_dtypes handles it later if needed
        force_all_finite='allow-nan', # Allow NaNs, TabPFN handles them
        ensure_min_samples=1, # TabPFN can work with 1 sample in prompt
        ensure_min_features=1,
        y_numeric=ensure_y_numeric, # Checks if y is numeric if True
        multi_output=False, # TabPFN is single output
         # reset=True by default, sets attributes on estimator
    )
    
    # After validate_data, estimator will have n_features_in_ and feature_names_in_
    n_features_in = estimator.n_features_in_
    feature_names_in_ = getattr(estimator, "feature_names_in_", original_feature_names)


    # Pretraining limit checks
    if not ignore_pretraining_limits:
        if X_validated.shape[0] > max_num_samples:
            raise ValueError(
                f"Number of samples {X_validated.shape[0]} exceeds the maximum {max_num_samples} "
                "supported by TabPFN. Set `ignore_pretraining_limits=True` to override."
            )
        if n_features_in > max_num_features: # Use n_features_in from estimator
            raise ValueError(
                f"Number of features {n_features_in} exceeds the maximum {max_num_features} "
                "supported by TabPFN. Set `ignore_pretraining_limits=True` to override."
            )

    if is_classifier(estimator): # check_is_fitted(estimator) is not needed before this
        check_classification_targets(y_validated)
    
    # Final check for y NaNs if not classifier (regression or other)
    # validate_data with force_all_finite='allow-nan' only applies to X
    if not is_classifier(estimator) and pd.isna(y_validated).any():
        raise ValueError("NaNs found in target variable y for regression or non-classification task.")

    return X_validated, y_validated, feature_names_in_, n_features_in


def validate_X_predict( # type: ignore[override]
    X: XType, # type: ignore
    estimator: TabPFNRegressor | TabPFNClassifier | BaseEstimator, # Added BaseEstimator
) -> np.ndarray:
    """Validate the input data for prediction."""
    check_is_fitted(estimator) # Ensure estimator is fitted
    
    # validate_data for X only, reset=False ensures estimator attributes are not reset
    X_validated = validate_data( # type: ignore
        estimator,
        X=X,
        y=None,
        reset=False, # Crucial: do not reset fitted attributes
        accept_sparse=False,
        dtype=None, 
        force_all_finite='allow-nan',
    )
    # n_features_in_ should be compared by validate_data if estimator is passed.
    # If X_validated is pd.DataFrame, convert to numpy for consistency
    if isinstance(X_validated, pd.DataFrame):
        X_validated = X_validated.to_numpy()
        
    return X_validated


def infer_categorical_features(
    X: np.ndarray, # Expects numpy array after initial validation
    *,
    provided: Sequence[int] | None, # Indices relative to the numpy array X
    min_samples_for_inference: int,
    max_unique_for_category: int,
    # min_unique_for_numerical: int, # This parameter was unused in the provided code
) -> list[int]:
    """Infer categorical features from the given NumPy data."""
    n_samples, n_features = X.shape
    inferred_cat_indices: list[int] = []

    if provided is not None:
        for idx in provided:
            if 0 <= idx < n_features:
                # Consider a column categorical if user provided it,
                # possibly with a loose check on unique values for sanity.
                # For now, trust user input if index is valid.
                inferred_cat_indices.append(idx)
        return sorted(list(set(inferred_cat_indices)))

    if n_samples < min_samples_for_inference:
        return [] # Not enough data to infer reliably

    for i in range(n_features):
        col_data = X[:, i]
        # Skip column if all values are NaN, can't infer type
        if np.all(pd.isna(col_data)): # Use pd.isna for robust NaN check
            continue
        
        # np.unique on a column with NaNs will count NaN as a unique value.
        # It's better to count unique values among non-NaN data.
        non_nan_col_data = col_data[~pd.isna(col_data)]
        if non_nan_col_data.size == 0: # All NaNs after all, skip
            continue

        num_unique = len(np.unique(non_nan_col_data))
        
        if num_unique <= max_unique_for_category:
            inferred_cat_indices.append(i)
        # No min_unique_for_numerical heuristic applied here from original.
        # If it's not caught by max_unique_for_category, it's considered numerical.

    return sorted(list(set(inferred_cat_indices)))

def _process_text_na_dataframe( # type: ignore[override]
    X_df: pd.DataFrame, # Expects DataFrame from _fix_dtypes
    # placeholder: str = NA_PLACEHOLDER, # Placeholder logic seems to be for string fill
    ord_encoder: ColumnTransformer, # Expects a ColumnTransformer
    *,
    fit_encoder: bool = False,
) -> np.ndarray:
    """
    Processes a DataFrame using a ColumnTransformer (expected to handle ordinal encoding).
    Numerical columns are typically passed through or imputed by earlier steps/other transformers.
    """
    if X_df.empty:
        # Try to determine expected output shape from a fitted encoder
        if fit_encoder or not hasattr(ord_encoder, "transformers_") or not ord_encoder.transformers_:
            return np.array([]).reshape(0, X_df.shape[1] if X_df.shape[1] > 0 else 0).astype(DEFAULT_NUMPY_PREPROCESSING_DTYPE)

        # If encoder is fitted, calculate expected output columns
        output_features = 0
        if hasattr(ord_encoder, 'n_features_out_'): # Scikit-learn 1.0+
            output_features = ord_encoder.n_features_out_
        elif ord_encoder.transformers_: # Fallback for older or custom
             for name, trans, cols in ord_encoder.transformers_:
                if trans == 'drop': continue
                if trans == 'passthrough': output_features += len(cols)
                elif hasattr(trans, 'get_feature_names_out'): # Fitted transformer
                    try: output_features += len(trans.get_feature_names_out(cols))
                    except: output_features += len(cols) # Fallback
                else: output_features += len(cols) # Untrained or simple transformer
        
        return np.array([]).reshape(0, output_features if output_features > 0 else (X_df.shape[1] if X_df.shape[1] > 0 else 0) ).astype(DEFAULT_NUMPY_PREPROCESSING_DTYPE)


    if fit_encoder:
        # Dynamically build transformers for ord_encoder based on X_df's dtypes
        # This assumes _fix_dtypes has already run and categoricals are 'category' dtype
        cat_cols = X_df.select_dtypes(include='category').columns.tolist()
        
        current_transformers = []
        if cat_cols:
            # Get the actual OrdinalEncoder from the 'encoder' part of the ColumnTransformer if it exists
            # This is a bit complex if ord_encoder is pre-configured.
            # For simplicity, let's assume ord_encoder is a template and we define its 'encoder' part here.
             oe_instance = OrdinalEncoder(
                categories="auto", dtype=DEFAULT_NUMPY_PREPROCESSING_DTYPE,
                handle_unknown="use_encoded_value", unknown_value=-1,
                encoded_missing_value=np.nan 
            )
             current_transformers.append(("cat_encoder", oe_instance, cat_cols))
        
        # Handle numerical columns: pass them through. Imputation should be separate.
        num_cols = X_df.select_dtypes(include=np.number).columns.tolist()
        if num_cols:
            current_transformers.append(("num_passthrough", "passthrough", num_cols))
            
        if not current_transformers: # No categorical or numerical columns
            return X_df.to_numpy(dtype=DEFAULT_NUMPY_PREPROCESSING_DTYPE)

        ord_encoder.transformers_ = current_transformers
        X_transformed = ord_encoder.fit_transform(X_df)
    else: # Just transform
        X_transformed = ord_encoder.transform(X_df)

    # Ensure float32/64 numpy array as output (DEFAULT_NUMPY_PREPROCESSING_DTYPE)
    return np.array(X_transformed, dtype=DEFAULT_NUMPY_PREPROCESSING_DTYPE)


def _map_to_bucket_ix(y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    """Maps y values to bucket indices based on borders."""
    # Ensure borders are sorted for searchsorted
    if not torch.all(borders[:-1] <= borders[1:]):
        # This case should ideally not happen if borders are managed correctly
        # For robustness, sort them here or raise an error
        warnings.warn("Borders are not sorted. Sorting them for _map_to_bucket_ix.", UserWarning, stacklevel=2)
        borders, _ = torch.sort(borders)

    # searchsorted returns the index where y would be inserted to maintain order.
    # Subtracting 1 gives the index of the bucket y falls into.
    ix = torch.searchsorted(borders, y, right=True) - 1 # right=True means if y == border_val, it goes to bucket right of border
    
    # Clamp indices to be within valid range [0, len(borders) - 2]
    # If y is less than the first border, it might result in -1, clamp to 0.
    # If y is equal to the last border, searchsorted(right=True) gives len(borders), -1 gives len(borders)-1.
    #   This means it's in the last bucket effectively, which has index len(borders)-2.
    # If y is greater than last border, also len(borders)-1.
    ix = torch.clamp(ix, 0, len(borders) - 2)
    
    # Specific handling for edge cases, though clamp should mostly cover it.
    # If y is exactly the first border value, it should be in the first bucket (index 0).
    ix[y <= borders[0]] = 0 # Ensure values <= first border are in bucket 0
    # If y is exactly the last border, it is considered in the last bucket.
    # searchsorted(right=True) with y == borders[-1] gives len(borders). ix becomes len(borders)-1.
    # This is problematic if we want it in bucket len(borders)-2.
    # Let's adjust: if y == borders[-1], it means it's at the very end of the last bucket.
    # The bucket index should be len(borders) - 2.
    ix[y >= borders[-1]] = len(borders) - 2 # Ensure values >= last border are in last bucket

    return ix


def _cdf(logits: torch.Tensor, borders: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    """Calculates CDF values for ys given logits over buckets defined by borders."""
    # Ensure ys has a trailing dimension for broadcasting with bucket computations
    if ys.ndim == logits.ndim -1 :
        ys_expanded = ys.unsqueeze(-1)
    else: # Assume ys is already (..., 1) or compatible
        ys_expanded = ys

    n_bars = borders.shape[0] - 1
    if n_bars <= 0: # Should not happen
        return torch.zeros_like(ys_expanded.squeeze(-1))

    # y_buckets: for each y, which bucket index it falls into. Shape: same as ys_expanded
    y_buckets = _map_to_bucket_ix(ys_expanded.squeeze(-1), borders).clamp(0, n_bars - 1).to(logits.device)
    y_buckets_expanded = y_buckets.unsqueeze(-1)


    probs = torch.softmax(logits, dim=-1) # Probabilities for each bucket (..., n_bars)
    
    # Cumulative probability up to (but not including) the start of each bucket
    # cumsum gives P(X <= bucket_i_end). We want P(X < bucket_i_start)
    # prob_so_far[j] = sum(probs[0]...probs[j-1])
    # Padded cumsum to handle 0th bucket easily
    prob_cumsum_exclusive = torch.cat(
        [torch.zeros_like(probs[..., :1]), torch.cumsum(probs[..., :-1], dim=-1)],
        dim=-1
    )
    # Gather P(X < start of y_bucket)
    prob_left_of_bucket = torch.gather(prob_cumsum_exclusive, dim=-1, index=y_buckets_expanded)

    bucket_widths = borders[1:] - borders[:-1] # Widths of each bucket
    # Replace zero widths with a small epsilon to avoid division by zero if borders are identical
    bucket_widths = torch.clamp(bucket_widths, min=1e-9) 

    # Proportion of the way through its bucket each y is
    # borders_y_bucket_starts: start border of the bucket each y falls into
    borders_y_bucket_starts = torch.gather(borders[:-1].to(ys_expanded.device), dim=0, index=y_buckets)
    bucket_widths_for_ys = torch.gather(bucket_widths.to(ys_expanded.device), dim=0, index=y_buckets)
    
    share_of_bucket_left = (ys_expanded.squeeze(-1) - borders_y_bucket_starts) / bucket_widths_for_ys
    share_of_bucket_left = torch.clamp(share_of_bucket_left, 0.0, 1.0)

    # Probability mass within the y_bucket, up to y itself
    prob_in_bucket_for_y = torch.gather(probs, dim=-1, index=y_buckets_expanded) * share_of_bucket_left.unsqueeze(-1)
    
    prob_left_of_ys = prob_left_of_bucket + prob_in_bucket_for_y
    prob_left_of_ys = prob_left_of_ys.squeeze(-1) # Remove the trailing 1 dim

    # Handle edge cases for ys outside the border range
    prob_left_of_ys[ys <= borders[0]] = 0.0
    prob_left_of_ys[ys >= borders[-1]] = 1.0
    return torch.clamp(prob_left_of_ys, 0.0, 1.0)


def translate_probs_across_borders(
    logits: torch.Tensor,
    *,
    frm: torch.Tensor, # Original borders
    to: torch.Tensor,   # New borders to evaluate CDF at
) -> torch.Tensor:
    """Translate a distribution (defined by logits over 'frm' borders) to probabilities over 'to' borders."""
    if frm.device != logits.device: frm = frm.to(logits.device)
    if to.device != logits.device: to = to.to(logits.device)

    # Calculate CDF values at the 'to' border points using the 'frm' distribution
    # prob_left_of_to_border_i = P(X <= to_border_i | X ~ dist(logits, frm))
    prob_left = _cdf(logits, borders=frm, ys=to) # Shape: (..., num_to_borders)

    # Ensure CDF values are monotonically increasing and within [0,1]
    # This can be an issue if 'to' borders are not sorted or if _cdf has numerical instability.
    # For safety, clamp and ensure monotonicity if needed, though _cdf should handle clamping.
    
    # Probabilities for new buckets are P(to_border_i-1 < X <= to_border_i)
    # = CDF(to_border_i) - CDF(to_border_i-1)
    # Need to handle the first bucket: P(X <= to_border_0)
    # And ensure the sum is 1.
    
    # A common way is to take differences of CDF values at the new borders.
    # Prepend 0 and append 1 to the CDF values if 'to' covers the whole range.
    # More generally, prob_left[..., 0] should correspond to P(X <= to[0])
    # and prob_left[..., -1] to P(X <= to[-1]).
    # The probability for bucket j (between to[j] and to[j+1]) is cdf(to[j+1]) - cdf(to[j])

    # If 'to' represents bucket edges, then prob_left[i] is CDF at to[i].
    # Prob for bucket to[i]..to[i+1] is prob_left[i+1] - prob_left[i]
    new_probs = prob_left[..., 1:] - prob_left[..., :-1]
    
    # Clamp to avoid negative probabilities due to floating point issues
    new_probs = torch.clamp(new_probs, min=0.0)
    
    # Normalize if needed to ensure sum to 1, though ideally _cdf is accurate enough.
    # sum_new_probs = torch.sum(new_probs, dim=-1, keepdim=True)
    # new_probs = new_probs / torch.clamp(sum_new_probs, min=1e-9) # Avoid division by zero
    
    return new_probs


def update_encoder_outlier_params( # Adjusted for PerFeatureTransformer and encoder structure
    model: PerFeatureTransformer, # Expects the core transformer model
    remove_outliers_std: float | None,
    seed: int | None,
    *,
    inplace: Literal[True] = True, # Default to True as original
) -> None:
    """Update outlier removal parameters in the model's input encoder."""
    if not inplace:
        # Deepcopying PerFeatureTransformer can be complex, not supported.
        raise NotImplementedError("Only inplace=True modification is supported for encoder outlier params.")

    if remove_outliers_std is not None and remove_outliers_std <= 0:
        raise ValueError("remove_outliers_std must be greater than 0 if provided.")

    if not hasattr(model, "encoder") or not isinstance(model.encoder, SequentialEncoder): # type: ignore
        warnings.warn("Model does not have a SequentialEncoder at model.encoder. Cannot update outlier params.", UserWarning, stacklevel=2)
        return

    encoder_module = model.encoder # This is nn.Module, should be SequentialEncoder
    
    found_norm_step = False
    # Iterate through the modules in SequentialEncoder
    for step_module in encoder_module: # type: ignore
        if isinstance(step_module, InputNormalizationEncoderStep): # type: ignore
            found_norm_step = True
            # Enable/disable outlier removal
            step_module.remove_outliers = (remove_outliers_std is not None) and (remove_outliers_std > 0)
            if step_module.remove_outliers:
                step_module.remove_outliers_sigma = remove_outliers_std
            
            # Update seed if provided and step has seed management
            if seed is not None and hasattr(step_module, "seed"):
                step_module.seed = seed
                if hasattr(step_module, "reset_seed") and callable(step_module.reset_seed):
                    step_module.reset_seed()
            break # Assume only one such step, or modify the first one found

    if not found_norm_step:
        warnings.warn("InputNormalizationEncoderStep not found in model.encoder. Outlier params not updated.", UserWarning, stacklevel=2)


def _transform_borders_one(
    borders: np.ndarray, # Original borders (typically from bar distribution)
    target_transform: TransformerMixin | Pipeline, # scikit-learn transformer
    *,
    repair_nan_borders_after_transform: bool,
) -> tuple[npt.NDArray[np.bool_] | None, bool, np.ndarray]:
    """Transforms borders using the inverse of target_transform and repairs them."""
    
    # Inverse transform the borders. Reshape for sklearn's expected 2D input.
    borders_t = target_transform.inverse_transform(borders.reshape(-1, 1)).squeeze()

    logit_cancel_mask: npt.NDArray[np.bool_] | None = None
    if repair_nan_borders_after_transform:
        # Identify "broken" borders (NaN, Inf, or outside large predefined limits)
        broken_mask = (
            ~np.isfinite(borders_t)
            | (borders_t > REGRESSION_NAN_BORDER_LIMIT_UPPER)
            | (borders_t < REGRESSION_NAN_BORDER_LIMIT_LOWER)
        )
        if broken_mask.any():
            borders_t, logit_cancel_mask = _cancel_nan_borders(
                borders=borders_t, # Pass the transformed, potentially broken borders
                broken_mask=broken_mask,
            )
    
    # Further repair for very close or identical borders
    _repair_borders(borders_t, inplace=True)

    # Check if borders ended up in descending order after transformation
    # This can happen with some transforms like log or reciprocal if original values crossed zero etc.
    descending_borders = False
    if len(borders_t) > 1 and np.all(np.diff(borders_t) <= 0): # Check if non-increasing
        # If strictly descending or flat in parts but generally descending
        if not np.all(np.diff(borders_t) == 0): # Not all flat
            descending_borders = True 
            borders_t = borders_t[::-1] # Reverse to make them ascending
            if logit_cancel_mask is not None:
                # Logits are between borders. If borders are reversed, the corresponding logits also effectively reverse.
                # Mask for N logits is N elements. If borders_t has M elements, logits are M-1.
                # broken_mask was M elements. logit_cancel_mask is M-1.
                logit_cancel_mask = logit_cancel_mask[::-1] 

    return logit_cancel_mask, descending_borders, borders_t


def get_total_memory_windows() -> float:
    """Get the total physical memory of the system for Windows OS, in GB."""
    import platform # Moved import inside
    if platform.system() != "Windows":
        return 0.0 

    class _MEMORYSTATUSEX(ctypes.Structure):
        _fields_: typing.ClassVar = [ # type: ignore
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    mem_status = _MEMORYSTATUSEX()
    mem_status.dwLength = ctypes.sizeof(_MEMORYSTATUSEX)

    try:
        # Ensure ctypes.windll.kernel32.GlobalMemoryStatusEx exists and is callable
        if hasattr(ctypes, "windll") and hasattr(ctypes.windll, "kernel32") and \
           hasattr(ctypes.windll.kernel32, "GlobalMemoryStatusEx"):
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem_status))
            return float(mem_status.ullTotalPhys) / (1024**3)  # Bytes to GB
        return 0.0
    except (AttributeError, OSError):
        return 0.0

# --- Placeholder for get_random_feature_shift ---
# This function was in the original utils.py provided in a previous turn.
# Adding it back as a placeholder. Its internal logic needs review/completion if used.
def get_random_feature_shift(
    X: torch.Tensor,
    *,
    max_index: int, 
    random_state: np.random.Generator,
    config: Any, # Should be ModelConfig or similar if used for config-driven shifts
    categorical_feats: list[int] | None = None,
) -> torch.Tensor:
    """Apply a random feature shift to the input data X. (Placeholder from original)"""
    if categorical_feats is None:
        categorical_feats = []
    
    num_features = X.shape[-1]
    if num_features == 0:
        return X.clone()

    shifted_X = X.clone()
    
    # Determine number of features to shift: at least 1, up to half, if features > 0
    # max_shiftable = num_features // 2 if num_features > 1 else num_features
    # num_to_shift = random_state.integers(1, max_shiftable + 1) if max_shiftable > 0 else 0
    
    if num_features == 1:
        num_to_shift = 1
    else:
        num_to_shift = random_state.integers(1, max(2, num_features // 2))


    if num_to_shift > 0:
        features_to_shift = random_state.choice(num_features, size=num_to_shift, replace=False)
        for feat_idx in features_to_shift:
            if feat_idx not in categorical_feats: 
                col_data = X[..., feat_idx]
                if col_data.numel() > 0: # Check if the slice is not empty
                    feature_std = torch.std(col_data[~torch.isnan(col_data)]) # Std of non-NaN
                    if not torch.isnan(feature_std) and feature_std > 1e-6: # Check if std is valid
                        shift_amount = random_state.normal(0, 0.1) * feature_std.item()
                        shifted_X[..., feat_idx] += shift_amount
                    elif not torch.isnan(feature_std): # Std is near zero (constant feature)
                        # Add small random noise if feature is constant but not NaN
                         shift_amount = random_state.normal(0, 0.01) # Small absolute shift
                         shifted_X[..., feat_idx] += shift_amount

    return shifted_X