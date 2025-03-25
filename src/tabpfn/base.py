"""Common logic for TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    overload,
)

import torch

# --- TabPFN imports ---
from tabpfn.constants import (
    AUTOCAST_DTYPE_BYTE_SIZE,
    DEFAULT_DTYPE_BYTE_SIZE,
)
from tabpfn.inference import (
    InferenceEngine,
    InferenceEngineCacheKV,
    InferenceEngineCachePreprocessing,
    InferenceEngineOnDemand,
)
from tabpfn.model.loading import (
    load_model_criterion_config,
)
from tabpfn.utils import (
    infer_fp16_inference_mode,
)

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from tabpfn.misc.compile_to_onnx import ONNXModelWrapper
    from tabpfn.model.bar_distribution import FullSupportBarDistribution
    from tabpfn.model.config import InferenceConfig
    from tabpfn.model.transformer import PerFeatureTransformer


@overload
def initialize_tabpfn_model(
    model_path: Path,
    which: Literal["regressor"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    static_seed: int,
) -> tuple[PerFeatureTransformer, InferenceConfig, FullSupportBarDistribution]: ...


@overload
def initialize_tabpfn_model(
    model_path: Path,
    which: Literal["classifier"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    static_seed: int,
) -> tuple[PerFeatureTransformer, InferenceConfig, None]: ...


def initialize_tabpfn_model(
    model_path: Path,
    which: Literal["classifier", "regressor"],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    static_seed: int,
) -> tuple[PerFeatureTransformer, InferenceConfig, FullSupportBarDistribution | None]:
    """Common logic to load the TabPFN model, set up the random state,
    and optionally download the model.

    Args:
        model_path: Path or directive ("auto") to load the pre-trained model from.
        which: Which TabPFN model to load.
        fit_mode: Determines caching behavior.
        static_seed: Random seed for reproducibility logic.

    Returns:
        model: The loaded TabPFN model.
        config: The configuration object associated with the loaded model.
        bar_distribution: The BarDistribution for regression (`None` if classifier).
    """
    # Handle auto model_path
    download = True

    # Load model with potential caching
    if which == "classifier":
        # The classifier's bar distribution is not used;
        # pass check_bar_distribution_criterion=False
        model, _, config_ = load_model_criterion_config(
            model_path=model_path,
            check_bar_distribution_criterion=False,
            cache_trainset_representation=(fit_mode == "fit_with_cache"),
            which="classifier",
            version="v2",
            download=download,
            model_seed=static_seed,
        )
        bar_distribution = None
    else:
        # The regressor's bar distribution is required
        model, bardist, config_ = load_model_criterion_config(
            model_path=model_path,
            check_bar_distribution_criterion=True,
            cache_trainset_representation=(fit_mode == "fit_with_cache"),
            which="regressor",
            version="v2",
            download=download,
            model_seed=static_seed,
        )
        bar_distribution = bardist

    return model, config_, bar_distribution


def load_onnx_model(
    model_path: Path,
    device: torch.device,
) -> ONNXModelWrapper:
    """Load a TabPFN model in ONNX format.

    Args:
        model_path: Path to the ONNX model file.
        which: Which TabPFN model to load.
        version: The version of the model.
        device: The device to run the model on.

    Returns:
        The loaded ONNX model wrapped in a PyTorch-compatible interface.

    Raises:
        ImportError: If onnxruntime is not installed.
        FileNotFoundError: If the model file doesn't exist.
    """
    try:
        from tabpfn.misc.compile_to_onnx import ONNXModelWrapper
    except ImportError as err:
        raise ImportError(
            "onnxruntime is required to load ONNX models. "
            "Install it with: pip install onnxruntime-gpu"
            "or pip install onnxruntime",
        ) from err

    if not model_path.exists():
        raise FileNotFoundError(
            f"ONNX model not found at: {model_path}, "
            "please compile the model by running "
            "`from tabpfn.misc.compile_to_onnx import compile_onnx_models; "
            "compile_onnx_models()`"
            "or change `model_path`.",
        )

    return ONNXModelWrapper(model_path, device)


def determine_precision(
    inference_precision: torch.dtype | Literal["autocast", "auto"],
    device_: torch.device,
) -> tuple[bool, torch.dtype | None, int]:
    """Decide whether to use autocast or a forced precision dtype.

    Args:
        inference_precision:

            - If `"auto"`, decide automatically based on the device.
            - If `"autocast"`, explicitly use PyTorch autocast (mixed precision).
            - If a `torch.dtype`, force that precision.

        device_: The device on which inference is run.

    Returns:
        use_autocast_:
            True if mixed-precision autocast will be used.
        forced_inference_dtype_:
            If not None, the forced precision dtype for the model.
        byte_size:
            The byte size per element for the chosen precision.
    """
    if inference_precision in ["autocast", "auto"]:
        use_autocast_ = infer_fp16_inference_mode(
            device=device_,
            enable=True if (inference_precision == "autocast") else None,
        )
        forced_inference_dtype_ = None
        byte_size = (
            AUTOCAST_DTYPE_BYTE_SIZE if use_autocast_ else DEFAULT_DTYPE_BYTE_SIZE
        )
    elif isinstance(inference_precision, torch.dtype):
        use_autocast_ = False
        forced_inference_dtype_ = inference_precision
        byte_size = inference_precision.itemsize
    else:
        raise ValueError(f"Unknown inference_precision={inference_precision}")

    return use_autocast_, forced_inference_dtype_, byte_size


def create_inference_engine(  # noqa: PLR0913
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    model: PerFeatureTransformer | ONNXModelWrapper,
    ensemble_configs: Any,
    cat_ix: list[int],
    fit_mode: Literal["low_memory", "fit_preprocessors", "fit_with_cache"],
    device_: torch.device,
    rng: np.random.Generator,
    n_jobs: int,
    byte_size: int,
    forced_inference_dtype_: torch.dtype | None,
    memory_saving_mode: bool | Literal["auto"] | float | int,
    use_autocast_: bool,
    use_onnx: bool = False,
) -> InferenceEngine:
    """Creates the appropriate TabPFN inference engine based on `fit_mode`.

    Each execution mode will perform slightly different operations based on the mode
    specified by the user. In the case where preprocessors will be fit after `prepare`,
    we will use them to further transform the associated borders with each ensemble
    config member.

    Args:
        X_train: Training features
        y_train: Training target
        model: The loaded TabPFN model.
        ensemble_configs: The ensemble configurations to create multiple "prompts".
        cat_ix: Indices of inferred categorical features.
        fit_mode: Determines how we prepare inference (pre-cache or not).
        device_: The device for inference.
        rng: Numpy random generator.
        n_jobs: Number of parallel CPU workers.
        byte_size: Byte size for the chosen inference precision.
        forced_inference_dtype_: If not None, the forced dtype for inference.
        memory_saving_mode: GPU/CPU memory saving settings.
        use_autocast_: Whether we use torch.autocast for inference.
        use_onnx: Whether to use ONNX runtime for model inference.
    """
    engine: (
        InferenceEngineOnDemand
        | InferenceEngineCachePreprocessing
        | InferenceEngineCacheKV
    )
    if fit_mode == "low_memory":
        engine = InferenceEngineOnDemand.prepare(
            X_train=X_train,
            y_train=y_train,
            cat_ix=cat_ix,
            ensemble_configs=ensemble_configs,
            rng=rng,
            model=model,
            n_workers=n_jobs,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            use_onnx=use_onnx,
        )
    elif fit_mode == "fit_preprocessors":
        engine = InferenceEngineCachePreprocessing.prepare(
            X_train=X_train,
            y_train=y_train,
            cat_ix=cat_ix,
            ensemble_configs=ensemble_configs,
            n_workers=n_jobs,
            model=model,
            rng=rng,
            dtype_byte_size=byte_size,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            use_onnx=use_onnx,
        )
    elif fit_mode == "fit_with_cache":
        engine = InferenceEngineCacheKV.prepare(
            X_train=X_train,
            y_train=y_train,
            cat_ix=cat_ix,
            model=model,
            ensemble_configs=ensemble_configs,
            n_workers=n_jobs,
            device=device_,
            dtype_byte_size=byte_size,
            rng=rng,
            force_inference_dtype=forced_inference_dtype_,
            save_peak_mem=memory_saving_mode,
            autocast=use_autocast_,
            use_onnx=use_onnx,
        )
    else:
        raise ValueError(f"Invalid fit_mode: {fit_mode}")

    return engine


def check_cpu_warning(
    device: str | torch.device, X: np.ndarray | torch.Tensor | pd.DataFrame
) -> None:
    """Check if using CPU with large datasets and warn or error appropriately.

    Args:
        device: The torch device being used
        X: The input data (NumPy array, Pandas DataFrame, or Torch Tensor)
    """
    allow_cpu_override = os.getenv("TABPFN_ALLOW_CPU_LARGE_DATASET", "0") == "1"

    # Determine number of samples
    try:
        num_samples = X.shape[0]
    except AttributeError:
        return

    if device == torch.device("cpu") or device == "cpu" or "cpu" in device:
        if num_samples > 1000:
            if not allow_cpu_override:
                raise RuntimeError(
                    "Running on CPU with more than 1000 samples is not allowed "
                    "by default due to slow performance.\n"
                    "To override this behavior, set the environment variable "
                    "TABPFN_ALLOW_CPU_LARGE_DATASET=1.\n"
                    "Alternatively, consider using a GPU or the tabpfn-client API: "
                    "https://github.com/PriorLabs/tabpfn-client"
                )
        elif num_samples > 200:
            warnings.warn(
                "Running on CPU with more than 200 samples may be slow.\n"
                "Consider using a GPU or the tabpfn-client API: "
                "https://github.com/PriorLabs/tabpfn-client",
                stacklevel=2,
            )
