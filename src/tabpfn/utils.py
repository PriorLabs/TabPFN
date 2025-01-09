"""A collection of random utilities for the TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import os
import sys
import warnings
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, overload

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.base import check_array, is_classifier
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OrdinalEncoder
from sklearn.utils.multiclass import check_classification_targets
from torch import nn

from tabpfn.constants import (
    DEFAULT_NUMPY_PREPROCESSING_DTYPE,
    REGRESSION_NAN_BORDER_LIMIT_LOWER,
    REGRESSION_NAN_BORDER_LIMIT_UPPER,
)
from tabpfn.model.bar_distribution import FullSupportBarDistribution
from tabpfn.model.loading import download_model, load_model

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin
    from sklearn.pipeline import Pipeline

    from tabpfn.classifier import TabPFNClassifier, XType, YType
    from tabpfn.model.config import InferenceConfig
    from tabpfn.model.transformer import PerFeatureTransformer
    from tabpfn.regressor import TabPFNRegressor

MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)


def _repair_borders(borders: np.ndarray, *, inplace: Literal[True]) -> None:
    # Try to repair a broken transformation of the borders:
    #   This is needed when a transformation of the ys leads to very extreme values
    #   in the transformed borders, since the borders spanned a very large range in
    #   the original space.
    #   Borders that were transformed to extreme values are all set to the same
    #   value, the maximum of the transformed borders. Thus probabilities predicted
    #   in these buckets have no effects. The outhermost border is set to the
    #   maximum of the transformed borders times 2, so still allow for some weight
    #   in the long tailed distribution and avoid infinite loss.
    if inplace is not True:
        raise NotImplementedError("Only inplace is supported")

    if np.isnan(borders[-1]):
        nans = np.isnan(borders)
        largest = borders[~nans].max()
        borders[nans] = largest
        borders[-1] = borders[-1] * 2

    if borders[-1] - borders[-2] < 1e-6:
        borders[-1] = borders[-1] * 1.1

    if borders[0] == borders[1]:
        borders[0] -= np.abs(borders[0] * 0.1)


def _cancel_nan_borders(
    *,
    borders: np.ndarray,
    broken_mask: npt.NDArray[np.bool_],
) -> tuple[np.ndarray, npt.NDArray[np.bool_]]:
    # OPTIM: You could do one check at a time
    # assert it is consecutive areas starting from both ends
    borders = borders.copy()
    num_right_borders = (broken_mask[:-1] > broken_mask[1:]).sum()
    num_left_borders = (broken_mask[1:] > broken_mask[:-1]).sum()
    assert num_left_borders <= 1
    assert num_right_borders <= 1

    if num_right_borders:
        assert bool(broken_mask[0]) is True
        rightmost_nan_of_left = np.where(broken_mask[:-1] > broken_mask[1:])[0][0] + 1
        borders[:rightmost_nan_of_left] = borders[rightmost_nan_of_left]
        borders[0] = borders[1] - 1.0

    if num_left_borders:
        assert bool(broken_mask[-1]) is True
        leftmost_nan_of_right = np.where(broken_mask[1:] > broken_mask[:-1])[0][0]
        borders[leftmost_nan_of_right + 1 :] = borders[leftmost_nan_of_right]
        borders[-1] = borders[-2] + 1.0

    # logit mask, mask out the nan positions, the borders are 1 more than logits
    logit_cancel_mask = broken_mask[1:] | broken_mask[:-1]
    return borders, logit_cancel_mask


def infer_device_and_type(device: str | torch.device | None) -> torch.device:
    """Infer the device and data type from the given device string.

    Args:
        device: The device to infer the type from.

    Returns:
        The inferred device
    """
    if (device is None) or (isinstance(device, str) and device == "auto"):
        device_type_ = "cuda" if torch.cuda.is_available() else "cpu"
        return torch.device(device_type_)
    if isinstance(device, str):
        return torch.device(device)

    if isinstance(device, torch.device):
        return device

    raise ValueError(f"Invalid device: {device}")


def is_autocast_available(device_type: str) -> bool:
    """Infer whether autocast is available for the given device type.

    Args:
        device_type: The device type to check for autocast availability.

    Returns:
        Whether autocast is available for the given device type.
    """
    # Try to use PyTorch's built-in function first
    try:
        from torch.amp.autocast_mode import (
            is_autocast_available as torch_is_autocast_available,
        )

        return bool(torch_is_autocast_available(device_type))
    except (ImportError, AttributeError):
        # Fall back to custom implementation if the function isn't available
        return bool(
            hasattr(torch.cuda, "amp")
            and hasattr(torch.cuda.amp, "autocast")
            and (
                device_type == torch.device("cuda").type
                or (
                    device_type == torch.device("cpu").type
                    and hasattr(torch.cpu, "amp")
                )
            ),
        )


def infer_fp16_inference_mode(device: torch.device, *, enable: bool | None) -> bool:
    """Infer whether fp16 inference should be enabled.

    Args:
        device: The device to validate against.
        enable:
            Whether it should be enabled, `True` or `False`, otherwise if `None`,
            detect if it's possible and use it if so.

    Returns:
        Whether to use fp16 inference or not.

    Raises:
        ValueError: If fp16 inference was enabled and device type does not support it.
    """
    is_cpu = device.type.lower() == "cpu"
    fp16_available = (
        not is_cpu  # CPU can show enabled, yet it kills inference speed
        and is_autocast_available(device.type)
    )

    if enable is None:
        return fp16_available

    if enable is True:
        if not fp16_available:
            raise ValueError(
                "You specified `fp16_inference=True`, however"
                "`torch.amp.autocast_mode.is_autocast_available()`"
                f" reported that your used device ({device=})"
                " does not support it."
                "\nPlease ensure your version of torch and device type"
                " are compatible with torch.autocast()`"
                " or set `fp16_inference=False`.",
            )
        return True

    if enable is False:
        return False

    raise ValueError(f"Unrecognized argument '{enable}'")


def _user_cache_dir(platform: str, appname: str = "tabpfn") -> Path:
    use_instead_path = (Path.cwd() / ".tabpfn_models").resolve()

    # https://docs.python.org/3/library/sys.html#sys.platform
    if platform == "win32":
        # Honestly, I don't want to do what `platformdirs` does:
        # https://github.com/tox-dev/platformdirs/blob/b769439b2a3b70769a93905944a71b3e63ef4823/src/platformdirs/windows.py#L252-L265
        APPDATA_PATH = os.environ.get("APPDATA", "")
        if APPDATA_PATH.strip() != "":
            return Path(APPDATA_PATH) / appname

        warnings.warn(
            "Could not find APPDATA environment variable to get user cache dir,"
            " but detected platform 'win32'."
            f" Defaulting to a path '{use_instead_path}'."
            " If you would prefer, please specify a directory when creating"
            " the model.",
            UserWarning,
            stacklevel=2,
        )
        return use_instead_path

    if platform == "darwin":
        return Path.home() / "Library" / "Caches" / appname

    # TODO: Not entirely sure here, Python doesn't explicitly list
    # all of these and defaults to the underlying operating system
    # if not sure.
    linux_likes = ("freebsd", "linux", "netbsd", "openbsd")
    if any(platform.startswith(linux) for linux in linux_likes):
        # The reason to use "" as default is that the env var could exist but be empty.
        # We catch all this with the `.strip() != ""` below
        XDG_CACHE_HOME = os.environ.get("XDG_CACHE_HOME", "")
        if XDG_CACHE_HOME.strip() != "":
            return Path(XDG_CACHE_HOME) / appname
        return Path.home() / ".cache" / appname

    warnings.warn(
        f"Unknown platform '{platform}' to get user cache dir."
        f" Defaulting to a path at the execution site '{use_instead_path}'."
        " If you would prefer, please specify a directory when creating"
        " the model.",
        UserWarning,
        stacklevel=2,
    )
    return use_instead_path


@overload
def load_model_criterion_config(
    model_path: str | Path | None,
    *,
    check_bar_distribution_criterion: Literal[False],
    cache_trainset_representation: bool,
    version: Literal["v2"],
    which: Literal["classifier"],
    download: bool,
    model_seed: int,
) -> tuple[
    PerFeatureTransformer,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss,
    InferenceConfig,
]: ...


@overload
def load_model_criterion_config(
    model_path: str | Path | None,
    *,
    check_bar_distribution_criterion: Literal[True],
    cache_trainset_representation: bool,
    version: Literal["v2"],
    which: Literal["regressor"],
    download: bool,
    model_seed: int,
) -> tuple[PerFeatureTransformer, FullSupportBarDistribution, InferenceConfig]: ...


def load_model_criterion_config(
    model_path: None | str | Path,
    *,
    check_bar_distribution_criterion: bool,
    cache_trainset_representation: bool,
    which: Literal["regressor", "classifier"],
    version: Literal["v2"] = "v2",
    download: bool,
    model_seed: int,
) -> tuple[
    PerFeatureTransformer,
    nn.BCEWithLogitsLoss | nn.CrossEntropyLoss | FullSupportBarDistribution,
    InferenceConfig,
]:
    """Load the model, criterion, and config from the given path.

    Args:
        model_path: The path to the model.
        check_bar_distribution_criterion:
            Whether to check if the criterion
            is a FullSupportBarDistribution, which is the expected criterion
            for models trained for regression.
        cache_trainset_representation:
            Whether the model should know to cache the trainset representation.
        which: Whether the model is a regressor or classifier.
        version: The version of the model.
        download: Whether to download the model if it doesn't exist.
        model_seed: The seed of the model.

    Returns:
        The model, criterion, and config.
    """
    if model_path is None:
        USER_TABPFN_CACHE_DIR_LOCATION = os.environ.get("TABPFN_MODEL_CACHE_DIR", "")
        if USER_TABPFN_CACHE_DIR_LOCATION.strip() != "":
            model_dir = Path(USER_TABPFN_CACHE_DIR_LOCATION)
        else:
            model_dir = _user_cache_dir(platform=sys.platform, appname="tabpfn")

        model_name = f"tabpfn-{version}-{which}.ckpt"
        model_path = model_dir / model_name
    else:
        if not isinstance(model_path, (str, Path)):
            raise ValueError(f"Invalid model_path: {model_path}")

        model_path = Path(model_path)
        model_dir = model_path.parent
        model_name = model_path.name

    model_dir.mkdir(parents=True, exist_ok=True)
    if not model_path.exists():
        if not download:
            raise ValueError(
                f"Model path does not exist and downloading is disabled"
                f"\nmodel path: {model_path}",
            )

        # NOTE: We use warnings as:
        # * Logging is only visible if the user has logging enabled,
        #   which for the majority of people using Python, this is not
        #   the case.
        # * `print` has no way to easily be disabled from the outside.
        warnings.warn(
            f"Downloading model to {model_path}.",
            UserWarning,
            stacklevel=2,
        )
        res = download_model(
            model_path,
            version=version,
            which=which,
            model_name=model_name,
        )
        if res != "ok":
            raise RuntimeError(f"Failed to download model to {model_path}!") from res[0]

    loaded_model, criterion, config = load_model(path=model_path, model_seed=model_seed)
    loaded_model.cache_trainset_representation = cache_trainset_representation
    if check_bar_distribution_criterion and not isinstance(
        criterion,
        FullSupportBarDistribution,
    ):
        raise ValueError(
            f"The model loaded, '{model_path}', was expected to have a"
            " FullSupportBarDistribution criterion, but instead "
            f" had a {type(criterion).__name__} criterion.",
        )
    return loaded_model, criterion, config


# https://numpy.org/doc/2.1/reference/arrays.dtypes.html#checking-the-data-type
NUMERIC_DTYPE_KINDS = "?bBiufm"
OBJECT_DTYPE_KINDS = "OV"
STRING_DTYPE_KINDS = "SaU"
UNSUPPORTED_DTYPE_KINDS = "cM"  # Not needed, just for completeness


def _fix_dtypes(
    X: pd.DataFrame | np.ndarray,
    cat_indices: Sequence[int | str] | None,
    numeric_dtype: Literal["float32", "float64"] = "float64",
) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        # This will help us get better dtype inference later
        convert_dtype = True
    elif isinstance(X, np.ndarray):
        if X.dtype.kind in NUMERIC_DTYPE_KINDS:
            # It's a numeric type, just wrap the array in pandas with the correct dtype
            X = pd.DataFrame(X, copy=False, dtype=numeric_dtype)
            convert_dtype = False
        elif X.dtype.kind in OBJECT_DTYPE_KINDS:
            # If numpy and object dype, we rely on pandas to handle introspection
            # of columns and rows to determine the dtypes.
            X = pd.DataFrame(X, copy=True)
            convert_dtype = True
        elif X.dtype.kind in STRING_DTYPE_KINDS:
            raise ValueError(
                f"String dtypes are not supported. Got dtype: {X.dtype}",
            )
        else:
            raise ValueError(f"Invalid dtype for X: {X.dtype}")
    else:
        raise ValueError(f"Invalid type for X: {type(X)}")

    if cat_indices is not None:
        # So annoyingly, things like AutoML Benchmark may sometimes provide
        # numeric indices for categoricals, while providing named columns in the
        # dataframe. Equally, dataframes loaded from something like a csv may just have
        # integer column names, and so it makes sense to access them just like you would
        # string columns.
        # Hence, we check if the types match and decide whether to use `iloc` to select
        # columns, or use the indices as column names...
        is_numeric_indices = all(isinstance(i, (int, np.integer)) for i in cat_indices)
        columns_are_numeric = all(
            isinstance(col, (int, np.integer)) for col in X.columns.tolist()
        )
        use_iloc = is_numeric_indices and not columns_are_numeric
        if use_iloc:
            X.iloc[:, cat_indices] = X.iloc[:, cat_indices].astype("category")
        else:
            X[cat_indices] = X[cat_indices].astype("category")

    # Alright, pandas can have a few things go wrong.
    #
    # 1. Of course, object dtypes, `convert_dtypes()` will handle this for us if
    #   possible. This will raise later if can't convert.
    # 2. String dtypes can still exist, OrdinalEncoder will do something but
    #   it's not ideal. We should probably check unique counts at the expense of doing
    #   so.
    # 3. For all dtypes relating to timeseries and other _exotic_ types not supported by
    #   numpy, we leave them be and let the pipeline error out where it will.
    # 4. Pandas will convert dtypes to Int64Dtype/Float64Dtype, which include
    #   `pd.NA`. Sklearn's Ordinal encoder treats this differently than `np.nan`.
    #   We can fix this one by converting all numeric columns to float64, which uses
    #   `np.nan` instead of `pd.NA`.
    #
    if convert_dtype:
        X = X.convert_dtypes()

    integer_columns = X.select_dtypes(include=["number"]).columns
    if len(integer_columns) > 0:
        X[integer_columns] = X[integer_columns].astype(numeric_dtype)
    return X


def _get_ordinal_encoder(
    *,
    numpy_dtype: np.floating = DEFAULT_NUMPY_PREPROCESSING_DTYPE,  # type: ignore
) -> ColumnTransformer:
    oe = OrdinalEncoder(
        # TODO: Could utilize the categorical dtype values directly instead of "auto"
        categories="auto",
        dtype=numpy_dtype,  # type: ignore
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=np.nan,  # Missing stays missing
    )

    # Documentation of sklearn, deferring to pandas is misleading here. It's done
    # using a regex on the type of the column, and using `object`, `"object"` and
    # `np.object` will not pick up strings.
    to_convert = ["category", "string"]
    return ColumnTransformer(
        transformers=[("encoder", oe, make_column_selector(dtype_include=to_convert))],
        remainder="passthrough",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )


def validate_Xy_fit(
    X: XType,
    y: YType,
    estimator: TabPFNRegressor | TabPFNClassifier,
    *,
    max_num_features: int,
    max_num_samples: int,
    ensure_y_numeric: bool = False,
    ignore_pretraining_limits: bool = False,
) -> tuple[np.ndarray, np.ndarray, npt.NDArray[Any] | None, int]:
    """Validate the input data for fitting."""
    # Calls `BaseEstimator._validate_data()` with specification
    X, y = estimator._validate_data(
        X,
        y,
        # Parameters to `check_X_y()`
        accept_sparse=False,
        dtype=None,  # This is handled later in `fit()`
        force_all_finite="allow-nan",
        ensure_min_samples=2,
        ensure_min_features=1,
        y_numeric=ensure_y_numeric,
        estimator=estimator,
    )

    if X.shape[1] > max_num_features:
        if not ignore_pretraining_limits:
            raise ValueError(
                f"Number of features {X.shape[1]} in the input data is greater than "
                f"the maximum number of features {max_num_features} officially "
                "supported by the TabPFN model. Set `ignore_pretraining_limits=True` "
                "to override this error!",
            )

        warnings.warn(
            f"Number of features {X.shape[1]} is greater than the maximum "
            f"Number of features {max_num_features} supported by the model."
            " You may see degraded performance.",
            UserWarning,
            stacklevel=2,
        )
    if X.shape[0] > max_num_samples:
        if not ignore_pretraining_limits:
            raise ValueError(
                f"Number of samples {X.shape[0]} in the input data is greater than "
                f"the maximum number of samples {max_num_samples} officially supported"
                f" by TabPFN. Set `ignore_pretraining_limits=True` to override this "
                f"error!",
            )
        warnings.warn(
            f"Number of samples {X.shape[0]} is greater than the maximum "
            f"Number of samples {max_num_samples} supported by the model."
            " You may see degraded performance.",
            UserWarning,
            stacklevel=2,
        )

    if is_classifier(estimator):
        check_classification_targets(y)
    # Annoyingly, the `force_all_finite` above only applies to `X` and
    # there is no way to specify this for `y`. The validation check above
    # will also only check for NaNs in `y` if `multi_output=True` which is
    # something we don't want. Hence, we run another check on `y` here.
    # However we also have to consider if ther dtype is a string type,
    # then

    y = check_array(
        y,
        accept_sparse=False,
        force_all_finite=True,
        dtype=None,  # type: ignore
        ensure_2d=False,
    )

    # NOTE: Theoretically we don't need to return the feature names and number,
    # but it makes it clearer in the calling code that these variables now exist
    # and can be set on the estimator.
    return X, y, getattr(estimator, "feature_names_in_", None), estimator.n_features_in_


def validate_X_predict(
    X: XType,
    estimator: TabPFNRegressor | TabPFNClassifier,
) -> np.ndarray:
    """Validate the input data for prediction."""
    return estimator._validate_data(  # type: ignore
        X,
        # NOTE: Important that reset is False, i.e. doesn't reset estimator
        reset=False,
        #
        # Parameters to `check_X_y()`
        accept_sparse=False,
        dtype=None,
        force_all_finite="allow-nan",
        estimator=estimator,
    )


def infer_categorical_features(
    X: np.ndarray,
    *,
    provided: Sequence[int] | None,
    min_samples_for_inference: int,
    max_unique_for_category: int,
    min_unique_for_numerical: int,
) -> list[int]:
    """Infer the categorical features from the given data.

    !!! note

        This function may infer particular columns to not be categorical
        as defined by what suits the model predictions and it's pre-training.

    Args:
        X: The data to infer the categorical features from.
        provided: Any user provided indices of what is considered categorical.
        min_samples_for_inference:
            The minimum number of samples required
            for automatic inference of features which were not provided
            as categorical.
        max_unique_for_category:
            The maximum number of unique values for a
            feature to be considered categorical.
        min_unique_for_numerical:
            The minimum number of unique values for a
            feature to be considered numerical.

    Returns:
        The indices of inferred categorical features.
    """
    # We presume everything is numerical and go from there
    maybe_categoricals = () if provided is None else provided
    large_enough_x_to_infer_categorical = X.shape[0] > min_samples_for_inference
    indices = []

    for ix, col in enumerate(X.T):
        if ix in maybe_categoricals:
            if len(np.unique(col)) <= max_unique_for_category:
                indices.append(ix)
        elif (
            large_enough_x_to_infer_categorical
            and len(np.unique(col)) < min_unique_for_numerical
        ):
            indices.append(ix)

    return indices


def infer_random_state(
    random_state: int | np.random.RandomState | np.random.Generator | None,
) -> tuple[int, np.random.Generator]:
    """Infer the random state from the given input.

    Args:
        random_state: The random state to infer.

    Returns:
        A static integer seed and a random number generator.
    """
    if isinstance(random_state, (int, np.integer)):
        np_rng = np.random.default_rng(random_state)
        static_seed = int(random_state)
    elif isinstance(random_state, np.random.RandomState):
        static_seed = int(random_state.randint(0, MAXINT_RANDOM_SEED))
        np_rng = np.random.default_rng(static_seed)
    elif isinstance(random_state, np.random.Generator):
        np_rng = random_state
        static_seed = int(np_rng.integers(0, MAXINT_RANDOM_SEED))
    elif random_state is None:
        np_rng = np.random.default_rng()
        static_seed = int(np_rng.integers(0, MAXINT_RANDOM_SEED))
    else:
        raise ValueError(f"Invalid random_state {random_state}")

    return static_seed, np_rng


def _map_to_bucket_ix(y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    ix = torch.searchsorted(sorted_sequence=borders, input=y) - 1
    ix[y == borders[0]] = 0
    ix[y == borders[-1]] = len(borders) - 2
    return ix


# TODO (eddiebergman): Can probably put this back to the Bar distribution.
# However we don't really need the full BarDistribution class and this was
# put here to make that a bit more obvious in terms of what was going on.
def _cdf(logits: torch.Tensor, borders: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    ys = ys.repeat(logits.shape[:-1] + (1,))
    n_bars = len(borders) - 1
    y_buckets = _map_to_bucket_ix(ys, borders).clamp(0, n_bars - 1).to(logits.device)

    probs = torch.softmax(logits, dim=-1)
    prob_so_far = torch.cumsum(probs, dim=-1) - probs
    prob_left_of_bucket = prob_so_far.gather(index=y_buckets, dim=-1)

    bucket_widths = borders[1:] - borders[:-1]
    share_of_bucket_left = ys - borders[y_buckets] / bucket_widths[y_buckets]
    share_of_bucket_left = share_of_bucket_left.clamp(0.0, 1.0)

    prob_in_bucket = probs.gather(index=y_buckets, dim=-1) * share_of_bucket_left
    prob_left_of_ys = prob_left_of_bucket + prob_in_bucket

    prob_left_of_ys[ys <= borders[0]] = 0.0
    prob_left_of_ys[ys >= borders[-1]] = 1.0
    return prob_left_of_ys.clip(0.0, 1.0)


def translate_probs_across_borders(
    logits: torch.Tensor,
    *,
    frm: torch.Tensor,
    to: torch.Tensor,
) -> torch.Tensor:
    """Translate the probabilities across the borders.

    Args:
        logits: The logits defining the distribution to translate.
        frm: The borders to translate from.
        to: The borders to translate to.

    Returns:
        The translated probabilities.
    """
    prob_left = _cdf(logits, borders=frm, ys=to)
    prob_left[..., 0] = 0.0
    prob_left[..., -1] = 1.0

    return (prob_left[..., 1:] - prob_left[..., :-1]).clamp_min(0.0)


def update_encoder_outlier_params(
    model: nn.Module,
    remove_outliers_std: float | None,
    seed: int | None,
    *,
    inplace: Literal[True],
) -> None:
    """Update the encoder to handle outliers in the model.

    !!! warning

        This only happens inplace.

    Args:
        model: The model to update.
        remove_outliers_std: The standard deviation to remove outliers.
        seed: The seed to use, if any.
        inplace: Whether to do the operation inplace.

    Raises:
        ValueError: If `inplace` is not `True`.
    """
    if not inplace:
        raise ValueError("Only inplace is supported")

    if remove_outliers_std is not None and remove_outliers_std <= 0:
        raise ValueError("remove_outliers_std must be greater than 0")

    if not hasattr(model, "encoder"):
        return

    encoder = model.encoder
    norm_layer = next(
        e for e in encoder if "InputNormalizationEncoderStep" in str(e.__class__)
    )
    norm_layer.remove_outliers = (remove_outliers_std is not None) and (
        remove_outliers_std > 0
    )
    if norm_layer.remove_outliers:
        norm_layer.remove_outliers_sigma = remove_outliers_std

    norm_layer.seed = seed
    norm_layer.reset_seed()


def _transform_borders_one(
    borders: np.ndarray,
    target_transform: TransformerMixin | Pipeline,
    *,
    repair_nan_borders_after_transform: bool,
) -> tuple[npt.NDArray[np.bool_] | None, bool, np.ndarray]:
    """Transforms the borders used for the bar distribution for regression.

    Args:
        borders: The borders to transform.
        target_transform: The target transformer to use.
        repair_nan_borders_after_transform:
            Whether to repair any borders that are NaN after the transformation.

    Returns:
        logit_cancel_mask:
            The mask of the logit values to ignore,
            those that mapped to NaN borders.
        descending_borders: Whether the borders are descending after transformation
        borders_t: The transformed borders themselves.
    """
    borders_t = target_transform.inverse_transform(borders.reshape(-1, 1)).squeeze()  # type: ignore

    logit_cancel_mask: npt.NDArray[np.bool_] | None = None
    if repair_nan_borders_after_transform:
        broken_mask = (
            ~np.isfinite(borders_t)
            | (borders_t > REGRESSION_NAN_BORDER_LIMIT_UPPER)
            | (borders_t < REGRESSION_NAN_BORDER_LIMIT_LOWER)
        )
        if broken_mask.any():
            borders_t, logit_cancel_mask = _cancel_nan_borders(
                borders=borders_t,
                broken_mask=broken_mask,
            )

    _repair_borders(borders_t, inplace=True)

    reversed_order = np.arange(len(borders_t) - 1, -1, -1)
    descending_borders = (np.argsort(borders_t) == reversed_order).all()
    if descending_borders:
        borders_t = borders_t[::-1]
        logit_cancel_mask = (
            logit_cancel_mask[::-1] if logit_cancel_mask is not None else None
        )

    return logit_cancel_mask, descending_borders, borders_t
