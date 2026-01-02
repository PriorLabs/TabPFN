"""A collection of random utilities for the TabPFN models."""

#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import contextlib
import os
import typing
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from sklearn.base import (
    TransformerMixin,
    check_is_fitted,
    is_classifier,
)
from sklearn.utils.multiclass import check_classification_targets

from tabpfn.architectures.base.encoders import (
    MulticlassClassificationTargetEncoder,
    SequentialEncoder,
)
from tabpfn.constants import (
    NA_PLACEHOLDER,
    REGRESSION_NAN_BORDER_LIMIT_LOWER,
    REGRESSION_NAN_BORDER_LIMIT_UPPER,
)
from tabpfn.misc._sklearn_compat import check_array, validate_data

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline

    from tabpfn.architectures.interface import Architecture
    from tabpfn.classifier import TabPFNClassifier, XType, YType
    from tabpfn.regressor import TabPFNRegressor

MAXINT_RANDOM_SEED = int(np.iinfo(np.int32).max)


def get_autocast_context(
    device: torch.device, *, enabled: bool
) -> contextlib.AbstractContextManager:
    """Returns a torch.autocast context manager, disabling it for MPS devices.

    Args:
        device: The torch device being used.
        enabled: Whether to enable autocast.

    Returns:
        A context manager for autocasting.
    """
    if device.type == "mps":
        return contextlib.nullcontext()
    return torch.autocast(device.type, enabled=enabled)


def get_embeddings(
    model: TabPFNClassifier | TabPFNRegressor,
    X: XType,
    data_source: Literal["train", "test"] = "test",
) -> np.ndarray:
    """Extract embeddings from a fitted TabPFN model.

    Args:
        model : TabPFNClassifier | TabPFNRegressor
            The fitted classifier or regressor.
        X : XType
            The input data.
        data_source : {"train", "test"}, default="test"
            Select the transformer output to return. Use ``"train"`` to obtain
            embeddings from the training tokens and ``"test"`` for the test tokens.

    Returns:
        np.ndarray
            The computed embeddings for each fitted estimator.
            When ``n_estimators > 1`` the returned array has shape
            ``(n_estimators, n_samples, embedding_dim)``. You can average over the
            first axis or reshape to concatenate the estimators, e.g.:

                emb = get_embeddings(model, X)
                emb_avg = emb.mean(axis=0)
                emb_concat = emb.reshape(emb.shape[1], -1)
    """
    check_is_fitted(model)

    data_map = {"train": "train_embeddings", "test": "test_embeddings"}

    selected_data = data_map[data_source]

    # Avoid circular imports
    from tabpfn.preprocessing import (  # noqa: PLC0415
        ClassifierEnsembleConfig,
        RegressorEnsembleConfig,
    )

    X = validate_X_predict(X, model)
    X = fix_dtypes(X, cat_indices=model.categorical_features_indices)
    X = model.preprocessor_.transform(X)

    embeddings: list[np.ndarray] = []

    # Cast executor to Any to bypass the iter_outputs signature check
    for output, config in model.executor_.iter_outputs(
        X,
        autocast=model.use_autocast_,
        only_return_standard_out=False,
    ):
        # Cast output to Any to allow dict-like access
        output_dict = typing.cast("dict[str, torch.Tensor]", output)
        embed = output_dict[selected_data].squeeze(1)
        assert isinstance(config, (ClassifierEnsembleConfig, RegressorEnsembleConfig))
        assert embed.ndim == 2
        embeddings.append(embed.squeeze().cpu().numpy())

    return np.array(embeddings)


def _repair_borders(borders: np.ndarray, *, inplace: Literal[True]) -> None:
    # Try to repair a broken transformation of the borders:
    #   This is needed when a transformation of the ys leads to very extreme values
    #   in the transformed borders, since the borders spanned a very large range in
    #   the original space.
    #   Borders that were transformed to extreme values are all set to the same
    #   value, the maximum of the transformed borders. Thus probabilities predicted
    #   in these buckets have no effects. The outermost border is set to the
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


DevicesSpecification = Union[
    torch.device, str, Sequence[Union[torch.device, str]], Literal["auto"]
]


def infer_devices(devices: DevicesSpecification) -> tuple[torch.device, ...]:
    """Selects the appropriate PyTorch devices for inference.

    If `device` is "auto" then the devices are selected as follows:
    1. If CUDA is available and not excluded, returns the first "cuda" device
    2. Otherwise, if MPS is available and not excluded, returns the "mps" device
    3. Otherwise, returns the "cpu" device

    CUDA and MPS can be excluded from the "auto" selection by specifying the
    TABPFN_EXCLUDE_DEVICES environment variable. This can be either "cuda", "mps", or
    "cuda,mps". This is useful for excluding device classes in CI.

    Args:
        devices: The device specification. One of:
            - "auto": the device will be selected as described above
            - a PyTorch device string like "cuda", "mps", or "cpu": this single device
                will be selected by parsing the string to a torch.device
            - a torch.device: this single device will be selected
            - a list of PyTorch device strings or torch.device: each item will be
                converted to a torch.device, and all of the devices selected

    Returns:
        A tuple of at least one device.
    """
    exclude_devices = {
        d.strip()
        for d in os.getenv("TABPFN_EXCLUDE_DEVICES", "").split(",")
        if d.strip()
    }

    if devices == "auto":
        if "cuda" not in exclude_devices and torch.cuda.is_available():
            return (torch.device("cuda:0"),)

        if _is_mps_supported() and "mps" not in exclude_devices:
            return (torch.device("mps"),)

        return (torch.device("cpu"),)

    if isinstance(devices, (str, torch.device)):
        devices = (devices,)

    devices = tuple(_parse_device(device) for device in devices)

    if len(set(devices)) != len(devices):
        raise ValueError(
            "The list of devices for inference cannot contain the same device more "
            f"than once. It contained: {devices}"
        )

    if not _is_mps_supported() and any(d.type == "mps" for d in devices):
        raise ValueError(
            "The MPS device was selected, "
            "but this is not supported by TabPFN before PyTorch 2.5. "
            'Set `device="cpu"` instead.'
        )

    return devices


def _parse_device(device: str | torch.device) -> torch.device:
    # This is safe because torch.device(torch.device(...)) is a noop.
    # torch.device(device) returns a fairly informative error message if `device` is not
    # a valid device, thus do no extra error reporting.
    device = torch.device(device)

    # In older versions of PyTorch, some torch.cuda functions fail if the device has no
    # index. 0 is implicit if no index is specified, so add it.
    if device == torch.device("cuda"):
        device = torch.device("cuda:0")

    return device


def _is_mps_supported() -> bool:
    """Return True if the MPS device is supported, otherwise False.

    We have found that using MPS can lead to poor accuracy on PyTorch <2.5. See
    https://github.com/PriorLabs/TabPFN/pull/619
    """
    return torch.__version__ >= "2.5" and torch.backends.mps.is_available()


def is_autocast_available(device_type: str) -> bool:
    """Infer whether autocast is available for the given device type.

    Args:
        device_type: The device type to check for autocast availability.

    Returns:
        Whether autocast is available for the given device type.
    """
    # Try to use PyTorch's built-in function first
    try:
        # Check if the function is available in torch
        if hasattr(torch.amp.autocast_mode, "is_autocast_available"):
            # Use function directly
            torch_is_autocast_available = torch.amp.autocast_mode.is_autocast_available
            return bool(torch_is_autocast_available(device_type))
        # Fall back to custom implementation
        raise AttributeError("is_autocast_available not found")
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


def infer_fp16_inference_mode(
    devices: Sequence[torch.device], *, enable: bool | None
) -> bool:
    """Infer whether fp16 inference should be enabled.

    Args:
        devices: The devices to validate against.
        enable:
            Whether it should be enabled, `True` or `False`, otherwise if `None`,
            detect if it's possible and use it if so.

    Returns:
        Whether to use fp16 inference or not.

    Raises:
        ValueError: If fp16 inference was enabled and any of the selected devices do
            not support it.
    """
    is_cpu = any(device.type.lower() == "cpu" for device in devices)
    fp16_available = (
        not is_cpu  # CPU can show enabled, yet it kills inference speed
        and any(is_autocast_available(device.type) for device in devices)
    )

    if enable is None:
        return fp16_available

    if enable is True:
        if not fp16_available:
            raise ValueError(
                "You specified `fp16_inference=True`, however"
                "`torch.amp.autocast_mode.is_autocast_available()`"
                f" reported that one or more of the selected devices ({devices=})"
                " does not support it."
                "\nPlease ensure your version of torch and device type"
                " are compatible with torch.autocast()`"
                " or set `fp16_inference=False`.",
            )
        return True

    if enable is False:
        return False

    raise ValueError(f"Unrecognized argument '{enable}'")


# https://numpy.org/doc/2.1/reference/arrays.dtypes.html#checking-the-data-type
NUMERIC_DTYPE_KINDS = "?bBiufm"
OBJECT_DTYPE_KINDS = "OV"
STRING_DTYPE_KINDS = "SaU"
UNSUPPORTED_DTYPE_KINDS = "cM"  # Not needed, just for completeness


def fix_dtypes(  # noqa: D103
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
    # Calls `validate_data()` with specification

    # Checks that we do not call validate_data() in case
    # the Prompttuning is enabled, since it is not differentiable.
    # TODO: update then Prompttuning is enabled for diffable models
    if not is_classifier(estimator) or (
        is_classifier(estimator) and not estimator.differentiable_input
    ):
        X, y = validate_data(
            estimator,
            X=X,
            y=y,
            # Parameters to `check_X_y()`
            accept_sparse=False,
            dtype=None,  # This is handled later in `fit()`
            ensure_all_finite="allow-nan",
            ensure_min_samples=2,
            ensure_min_features=1,
            y_numeric=ensure_y_numeric,
            estimator=estimator,
        )
    else:  # Quick check for tensor input for diffable classifier
        assert isinstance(X, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert len(X) == len(y)
        assert len(X.shape) == 2
        estimator.n_features_in_ = X.shape[1]

    if X.shape[1] > max_num_features and not ignore_pretraining_limits:
        raise ValueError(
            f"Number of features {X.shape[1]} in the input data is greater than "
            f"the maximum number of features {max_num_features} officially "
            "supported by the TabPFN model. Set `ignore_pretraining_limits=True` "
            "to override this error!",
        )
    if X.shape[0] > max_num_samples and not ignore_pretraining_limits:
        raise ValueError(
            f"Number of samples {X.shape[0]} in the input data is greater than "
            f"the maximum number of samples {max_num_samples} officially supported"
            f" by TabPFN. Set `ignore_pretraining_limits=True` to override this "
            f"error!",
        )

    if is_classifier(estimator) and not estimator.differentiable_input:
        check_classification_targets(y)
        # Annoyingly, the `ensure_all_finite` above only applies to `X` and
        # there is no way to specify this for `y`. The validation check above
        # will also only check for NaNs in `y` if `multi_output=True` which is
        # something we don't want. Hence, we run another check on `y` here.
        # However we also have to consider if ther dtype is a string type,
        # then
        y = check_array(
            y,
            accept_sparse=False,
            ensure_all_finite=True,
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
    result = validate_data(
        estimator,
        X=X,
        # NOTE: Important that reset is False, i.e. doesn't reset estimator
        reset=False,
        # Parameters to `check_X_y()`
        accept_sparse=False,
        dtype=None,
        ensure_all_finite="allow-nan",
        estimator=estimator,
    )
    return typing.cast("np.ndarray", result)


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
        # Calculate total distinct values once, treating NaN as a category.
        try:
            s = pd.Series(col)
            # counts NaN/None as a category
            num_distinct = s.nunique(dropna=False)
        except TypeError as e:
            # e.g. "unhashable type: 'dict'" when object arrays contain dicts
            raise TypeError(
                "argument must be a string or a number"
                "(columns must only contain strings or numbers)"
            ) from e
        if ix in maybe_categoricals:
            if num_distinct <= max_unique_for_category:
                indices.append(ix)
        elif (
            large_enough_x_to_infer_categorical
            and num_distinct < min_unique_for_numerical
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


def process_text_na_dataframe(
    X: pd.DataFrame,
    placeholder: str = NA_PLACEHOLDER,
    ord_encoder: ColumnTransformer | None = None,
    *,
    fit_encoder: bool = False,
) -> np.ndarray:
    """Convert `X` to float64, replacing NA with NaN in string cells.

    If `ord_encoder` is not None, then it will be used to encode `X` before the
    conversion to float64.

    Note that this function sometimes mutates its input.
    """
    # Replace NAN values in X, for dtypes, which the OrdinalEncoder cannot handle
    # with placeholder NAN value. Later placeholder NAN values are transformed to np.nan
    string_cols = X.select_dtypes(include=["string", "object"]).columns
    if len(string_cols) > 0:
        X[string_cols] = X[string_cols].fillna(placeholder)

    if fit_encoder and ord_encoder is not None:
        X_encoded = ord_encoder.fit_transform(X)
    elif ord_encoder is not None:
        X_encoded = ord_encoder.transform(X)
    else:
        X_encoded = X

    string_cols_ix = [X.columns.get_loc(col) for col in string_cols]
    placeholder_mask = X[string_cols] == placeholder
    X_encoded[:, string_cols_ix] = np.where(
        placeholder_mask,
        np.nan,
        X_encoded[:, string_cols_ix],
    )
    return X_encoded.astype(np.float64)


def _map_to_bucket_ix(y: torch.Tensor, borders: torch.Tensor) -> torch.Tensor:
    ix = torch.searchsorted(sorted_sequence=borders, input=y) - 1
    ix[y == borders[0]] = 0
    ix[y == borders[-1]] = len(borders) - 2
    return ix


# TODO (eddiebergman): Can probably put this back to the Bar distribution.
# However we don't really need the full BarDistribution class and this was
# put here to make that a bit more obvious in terms of what was going on.
def _cdf(logits: torch.Tensor, borders: torch.Tensor, ys: torch.Tensor) -> torch.Tensor:
    ys = ys.repeat((*logits.shape[:-1], 1))
    n_bars = len(borders) - 1
    y_buckets = _map_to_bucket_ix(ys, borders).clamp(0, n_bars - 1).to(logits.device)

    probs = torch.softmax(logits, dim=-1)
    prob_so_far = torch.cumsum(probs, dim=-1) - probs
    prob_left_of_bucket = prob_so_far.gather(index=y_buckets, dim=-1)

    bucket_widths = borders[1:] - borders[:-1]
    share_of_bucket_left = (ys - borders[y_buckets]) / bucket_widths[y_buckets]
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


def update_encoder_params(
    models: list[Architecture],
    remove_outliers_std: float | None,
    seed: int | None,
    *,
    differentiable_input: bool = False,
) -> None:
    """Update the loaded encoder elements and setting to be compatible with inference
    requirements. This concerns handling outliers in the model and also removes
    non-differentiable steps from the label encoder.

    !!! warning

        This only happens inplace.

    Args:
        models: The models to update.
        remove_outliers_std: The standard deviation to remove outliers.
        seed: The seed to use, if any.
        inplace: Whether to do the operation inplace.
        differentiable_input: Whether the entire model including forward pass should
            be differentiable with pt autograd. This disables non-differentiable
            encoder steps.
    """
    if remove_outliers_std is not None and remove_outliers_std <= 0:
        raise ValueError("remove_outliers_std must be greater than 0")

    for model in models:
        # TODO: find a less hacky way to change settings during training
        # and inference
        if not hasattr(model, "encoder"):
            raise ValueError(
                "Model does not have an encoder, this breaks the TabPFN sklearn "
                "wrapper."
            )

        encoder = model.encoder

        # TODO: maybe check that norm_layer even exists
        norm_layer = next(
            e for e in encoder if "InputNormalizationEncoderStep" in str(e.__class__)
        )
        if not hasattr(norm_layer, "remove_outliers"):
            raise ValueError(
                "InputNormalizationEncoderStep does not have a remove_outliers "
                "attribute, this will break the TabPFN sklearn wrapper."
            )
        norm_layer.remove_outliers = (remove_outliers_std is not None) and (
            remove_outliers_std > 0
        )
        if norm_layer.remove_outliers:
            norm_layer.remove_outliers_sigma = remove_outliers_std

        norm_layer.seed = seed
        norm_layer.reset_seed()

        if differentiable_input:
            diffable_steps = []  # only differentiable encoder steps.
            for module in model.y_encoder:
                if isinstance(module, MulticlassClassificationTargetEncoder):
                    pass
                else:
                    diffable_steps.append(module)

            model.y_encoder = SequentialEncoder(*diffable_steps)


def transform_borders_one(
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


def pad_tensors(
    tensor_list: list[torch.Tensor],
    padding_val: float | None = 0,
    *,
    labels: bool = False,
) -> list[torch.Tensor]:
    """Pad tensors to maximum dims at the last dimensions.
    if labels=False, 2d tensors are expected, if labels=True, one 1d
    vectors are expected as inputs.

    Args:
        tensor_list: List of tensors to be padded.
        padding_val: what value to use for padding.
        labels: If true, the tensor list should contain 1D
            tensors that are padded only along this dimension.
            If false, rows and feature dimensions are padded.
    """
    max_size_clms = max([item.size(-1) for item in tensor_list])
    if not labels:
        max_size_rows = max([item.size(-2) for item in tensor_list])
    ret_list = []
    for item in tensor_list:
        pad_seqence = [0, max_size_clms - item.size(-1)]
        if not labels:
            pad_seqence.extend([0, max_size_rows - item.size(-2)])
        padded_item = torch.nn.functional.pad(
            item, pad_seqence, mode="constant", value=padding_val
        )
        ret_list.append(padded_item)
    return ret_list


def balance_probas_by_class_counts(
    probas: torch.Tensor,
    class_counts: np.ndarray,
) -> torch.Tensor:
    """Balance probabilities by class counts.

    Args:
        probas: The probabilities to balance.
        class_counts: The class counts to use for balancing.

    Returns:
        The balanced probabilities.
    """
    class_prob_in_train = class_counts / class_counts.sum()
    balanced_probas = probas / torch.from_numpy(class_prob_in_train).float().to(
        probas.device
    )
    return balanced_probas / balanced_probas.sum(dim=-1, keepdim=True)
