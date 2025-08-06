#  Copyright (c) Prior Labs GmbH 2025.

from __future__ import annotations

import contextlib
import hashlib
from abc import abstractmethod
from collections import UserList
from collections.abc import Sequence
from copy import deepcopy
from typing import TYPE_CHECKING, Any, Literal, NamedTuple, TypeVar
from typing_extensions import Self, override

import numpy as np
import torch
from scipy import optimize
from scipy.stats import shapiro
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.decomposition import TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)

from tabpfn.utils import infer_random_state

if TYPE_CHECKING:
    from sklearn.base import TransformerMixin


try:
    from kditransform import KDITransformer

    # This import fails on some systems, due to problems with numba
except ImportError:
    KDITransformer = PowerTransformer  # fallback to avoid error


class KDITransformerWithNaN(KDITransformer):
    """KDI transformer that can handle NaN values. It performs KDI with NaNs replaced by
    mean values and then fills the NaN values with NaNs after the transformation.
    """

    def _more_tags(self) -> dict:
        return {"allow_nan": True}

    def fit(self, X: torch.Tensor | np.ndarray, y: Any | None = None) -> Self:
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        return super().fit(X, y)  # type: ignore

    def transform(self, X: torch.Tensor | np.ndarray) -> np.ndarray:
        # if tensor convert to numpy
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()

        # Calculate the NaN mask for the current dataset
        nan_mask = np.isnan(X)

        # Replace NaNs with the mean of columns
        imputation = np.nanmean(X, axis=0)
        imputation = np.nan_to_num(imputation, nan=0)
        X = np.nan_to_num(X, nan=imputation)

        # Apply the transformation
        X = super().transform(X)

        # Reintroduce NaN values based on the current dataset's mask
        X[nan_mask] = np.nan

        return X  # type: ignore


class AdaptiveQuantileTransformer(QuantileTransformer):
    """A QuantileTransformer that automatically adapts the 'n_quantiles' parameter
    based on the number of samples provided during the 'fit' method.

    This fixes an issue in older versions of scikit-learn where the 'n_quantiles'
    parameter could not exceed the number of samples in the input data.

    This code prevents errors that occur when the requested 'n_quantiles' is
    greater than the number of available samples in the input data (X).
    This situation can arises because we first initialize the transformer
    based on total samples and then subsample.
    """

    def __init__(self, *, n_quantiles: int = 1000, **kwargs: Any) -> None:
        # Store the user's desired n_quantiles to use as an upper bound
        self._user_n_quantiles = n_quantiles
        # Initialize parent with this, but it will be adapted in fit
        super().__init__(n_quantiles=n_quantiles, **kwargs)

    def fit(
        self, X: np.ndarray, y: np.ndarray | None = None
    ) -> AdaptiveQuantileTransformer:
        n_samples = X.shape[0]

        # Adapt n_quantiles for this fit: min of user's preference and available samples
        # Ensure n_quantiles is at least 1
        effective_n_quantiles = max(
            1, min(self._user_n_quantiles, n_samples, self.subsample)
        )

        # Set self.n_quantiles to the effective value BEFORE calling super().fit()
        # This ensures the parent class uses the adapted value for fitting
        # and self.n_quantiles will reflect the value used for the fit.
        self.n_quantiles = effective_n_quantiles

        # Convert Generator to RandomState if needed for sklearn compatibility
        if isinstance(self.random_state, np.random.Generator):
            # Generate a random integer to use as seed for RandomState
            seed = int(self.random_state.integers(0, 2**32))
            self.random_state = np.random.RandomState(seed)
        elif hasattr(self.random_state, "bit_generator"):
            # Handle other Generator-like objects
            raise ValueError(
                f"Unsupported random state type: {type(self.random_state)}. "
                "Please provide an integer seed or np.random.RandomState object."
            )

        return super().fit(X, y)


ALPHAS = (
    0.05,
    0.1,
    0.2,
    0.25,
    0.3,
    0.4,
    0.5,
    0.6,
    0.8,
    1.0,
    1.2,
    1.5,
    1.8,
    2.0,
    2.5,
    3.0,
    5.0,
)


def get_all_kdi_transformers() -> dict[str, KDITransformerWithNaN]:
    try:
        all_preprocessors = {
            "kdi": KDITransformerWithNaN(alpha=1.0, output_distribution="normal"),
            "kdi_uni": KDITransformerWithNaN(
                alpha=1.0,
                output_distribution="uniform",
            ),
        }
        for alpha in ALPHAS:
            all_preprocessors[f"kdi_alpha_{alpha}"] = KDITransformerWithNaN(
                alpha=alpha,
                output_distribution="normal",
            )
            all_preprocessors[f"kdi_alpha_{alpha}_uni"] = KDITransformerWithNaN(
                alpha=alpha,
                output_distribution="uniform",
            )
        return all_preprocessors
    except Exception:  # noqa: BLE001
        return {}


# this is taken from https://github.com/scipy/scipy/pull/18852
# which fix overflow issues
# we can directly import from scipy once we drop support for scipy < 1.12
def _yeojohnson(x, lmbda=None):
    x = np.asarray(x)
    if x.size == 0:
        # changed from scipy from return x
        return (x, None) if lmbda is None else x

    if np.issubdtype(x.dtype, np.complexfloating):
        raise ValueError(
            "Yeo-Johnson transformation is not defined for " "complex numbers."
        )

    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float64, copy=False)

    if lmbda is not None:
        return _yeojohnson_transform(x, lmbda)

    # if lmbda=None, find the lmbda that maximizes the log-likelihood function.
    lmax = _yeojohnson_normmax(x)
    y = _yeojohnson_transform(x, lmax)

    return y, lmax


def _yeojohnson_transform(x, lmbda):
    """Returns `x` transformed by the Yeo-Johnson power transform with given
    parameter `lmbda`.
    """
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    out = np.zeros_like(x, dtype=dtype)
    pos = x >= 0  # binary mask

    # when x >= 0
    if abs(lmbda) < np.spacing(1.0):
        out[pos] = np.log1p(x[pos])
    else:  # lmbda != 0
        # more stable version of: ((x + 1) ** lmbda - 1) / lmbda
        out[pos] = np.expm1(lmbda * np.log1p(x[pos])) / lmbda

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.0):
        out[~pos] = -np.expm1((2 - lmbda) * np.log1p(-x[~pos])) / (2 - lmbda)
    else:  # lmbda == 2
        out[~pos] = -np.log1p(-x[~pos])

    return out


def _yeojohnson_llf(lmb, data):
    r"""The yeojohnson log-likelihood function."""
    data = np.asarray(data)
    n_samples = data.shape[0]

    if n_samples == 0:
        return np.nan

    trans = _yeojohnson_transform(data, lmb)
    trans_var = trans.var(axis=0)
    loglike = np.empty_like(trans_var)

    # Avoid RuntimeWarning raised by np.log when the variance is too low
    tiny_variance = trans_var < np.finfo(trans_var.dtype).tiny
    loglike[tiny_variance] = np.inf

    loglike[~tiny_variance] = -n_samples / 2 * np.log(trans_var[~tiny_variance])
    loglike[~tiny_variance] += (lmb - 1) * (np.sign(data) * np.log1p(np.abs(data))).sum(
        axis=0
    )
    return loglike


def _yeojohnson_normmax(x, brack=None):
    """Compute optimal Yeo-Johnson transform parameter.
    Compute optimal Yeo-Johnson transform parameter for input data, using
    maximum likelihood estimation.

    """

    def _neg_llf(lmbda, data):
        llf = _yeojohnson_llf(lmbda, data)
        # reject likelihoods that are inf which are likely due to small
        # variance in the transformed space
        llf[np.isinf(llf)] = -np.inf
        return -llf

    with np.errstate(invalid="ignore"):
        if not np.all(np.isfinite(x)):
            raise ValueError("Yeo-Johnson input must be finite.")
        if np.all(x == 0):
            return 1.0
        if brack is not None:
            return optimize.brent(_neg_llf, brack=brack, args=(x,))
        x = np.asarray(x)
        dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
        # Allow values up to 20 times the maximum observed value to be safely
        # transformed without over- or underflow.
        log1p_max_x = np.log1p(20 * np.max(np.abs(x)))
        # Use half of floating point's exponent range to allow safe computation
        # of the variance of the transformed data.
        log_eps = np.log(np.finfo(dtype).eps)
        log_tiny_float = (np.log(np.finfo(dtype).tiny) - log_eps) / 2
        log_max_float = (np.log(np.finfo(dtype).max) + log_eps) / 2
        # Compute the bounds by approximating the inverse of the Yeo-Johnson
        # transform on the smallest and largest floating point exponents, given
        # the largest data we expect to observe. See [1] for further details.
        # [1] https://github.com/scipy/scipy/pull/18852#issuecomment-1630286174
        lb = log_tiny_float / log1p_max_x
        ub = log_max_float / log1p_max_x
        # Convert the bounds if all or some of the data is negative.
        if np.all(x < 0):
            lb, ub = 2 - ub, 2 - lb
        elif np.any(x < 0):
            lb, ub = max(2 - ub, lb), min(2 - lb, ub)
        # Match `optimize.brent`'s tolerance.
        tol_brent = 1.48e-08
        return optimize.fminbound(_neg_llf, lb, ub, args=(x,), xtol=tol_brent)


# we created this inspired by the scipy change for transform
# https://github.com/scipy/scipy/pull/18852
# this is not in scipy even 1.12
def _yeojohnson_inverse_transform(x, lmbda):
    """Return inverse-transformed input x following Yeo-Johnson inverse
    transform with parameter lambda.
    """
    dtype = x.dtype if np.issubdtype(x.dtype, np.floating) else np.float64
    x_inv = np.zeros_like(x, dtype=dtype)
    pos = x >= 0

    # when x >= 0
    if abs(lmbda) < np.spacing(1.0):
        x_inv[pos] = np.expm1(x[pos])
    else:  # lmbda != 0
        # more stable version of: (x * lmbda + 1) ** (1 / lmbda) - 1
        x_inv[pos] = np.expm1(np.log(x[pos] * lmbda + 1) / lmbda)

    # when x < 0
    if abs(lmbda - 2) > np.spacing(1.0):
        # more stable version of: 1 - (-(2 - lmbda) * x + 1) ** (1 / (2 - lmbda))
        x_inv[~pos] = -np.expm1(np.log(-(2 - lmbda) * x[~pos] + 1) / (2 - lmbda))
    else:  # lmbda == 2
        x_inv[~pos] = -np.expm1(-x[~pos])

    return x_inv


class SafePowerTransformer(PowerTransformer):
    """Variant of PowerTransformer that uses the scipy yeo-johnson functions
    which have been fixed to avoid overflow issues.
    """

    def __init__(
        self,
        method: str = "yeo-johnson",
        *,
        standardize: bool = True,
        copy: bool = True,
    ) -> None:
        super().__init__(method=method, standardize=standardize, copy=copy)

    # requires scipy >= 1.9
    # this is the default in scikit-learn main for versions > 1.7
    # see https://github.com/scikit-learn/scikit-learn/pull/31227
    def _yeo_johnson_optimize(self, x):
        # the computation of lambda is influenced by NaNs so we need to
        # get rid of them
        x = x[~np.isnan(x)]
        _, lmbda = _yeojohnson(x, lmbda=None)
        return lmbda

    def _yeo_johnson_transform(self, x, lmbda):
        return _yeojohnson_transform(x, lmbda)

    def _yeo_johnson_inverse_transform(self, x, lmbda):
        return _yeojohnson_inverse_transform(x, lmbda)


def skew(x: np.ndarray) -> float:
    """skewness: 3 * (mean - median) / std."""
    return float(3 * (np.nanmean(x, 0) - np.nanmedian(x, 0)) / np.std(x, 0))


def _inf_to_nan_func(x: np.ndarray) -> np.ndarray:
    return np.nan_to_num(x, nan=np.nan, neginf=np.nan, posinf=np.nan)


def _exp_minus_1(x: np.ndarray) -> np.ndarray:
    return np.exp(x) - 1  # type: ignore


T = TypeVar("T")


def _identity(x: T) -> T:
    return x


inf_to_nan_transformer = FunctionTransformer(
    func=_inf_to_nan_func,
    inverse_func=_identity,
    check_inverse=False,
)
nan_impute_transformer = SimpleImputer(
    missing_values=np.nan,
    strategy="mean",
    # keep empty features for inverse to function
    keep_empty_features=True,
)
nan_impute_transformer.inverse_transform = (
    _identity  # do not inverse np.nan values.  # type: ignore
)

_make_finite_transformer = [
    ("inf_to_nan", inf_to_nan_transformer),
    ("nan_impute", nan_impute_transformer),
]


def make_standard_scaler_safe(
    _name_scaler_tuple: tuple[str, TransformerMixin],
    *,
    no_name: bool = False,
) -> Pipeline:
    # Make sure that all data that enters and leaves a scaler is finite.
    # This is needed in edge cases where, for example, a division by zero
    # occurs while scaling or when the input contains not number values.
    return Pipeline(
        steps=[
            *[(n + "_pre ", deepcopy(t)) for n, t in _make_finite_transformer],
            ("placeholder", _name_scaler_tuple) if no_name else _name_scaler_tuple,
            *[(n + "_post", deepcopy(t)) for n, t in _make_finite_transformer],
        ],
    )


def make_box_cox_safe(input_transformer: TransformerMixin | Pipeline) -> Pipeline:
    """Make box cox save.

    The Box-Cox transformation can only be applied to strictly positive data.
    With first MinMax scaling, we achieve this without loss of function.
    Additionally, for test data, we also need clipping.
    """
    return Pipeline(
        steps=[
            ("mm", MinMaxScaler(feature_range=(0.1, 1), clip=True)),
            ("box_cox", input_transformer),
        ],
    )


def add_safe_standard_to_safe_power_without_standard(
    input_transformer: TransformerMixin,
) -> Pipeline:
    """In edge cases PowerTransformer can create inf values and similar. Then, the post
    standard scale crashes. This fixes this issue.
    """
    return Pipeline(
        steps=[
            ("input_transformer", input_transformer),
            ("standard", make_standard_scaler_safe(("standard", StandardScaler()))),
        ],
    )


class _TransformResult(NamedTuple):
    X: np.ndarray
    categorical_features: list[int]


# TODO(eddiebergman): I'm sure there's a way to handle this when using dataframes.
class FeaturePreprocessingTransformerStep:
    """Base class for feature preprocessing steps.

    It's main abstraction is really just to provide categorical indices along the
    pipeline.
    """

    categorical_features_after_transform_: list[int]

    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> _TransformResult:
        self.fit(X, categorical_features)
        # TODO(eddiebergman): If we could get rid of this... anywho, needed for
        # the AddFingerPrint
        result = self._transform(X, is_test=False)
        return _TransformResult(result, self.categorical_features_after_transform_)

    @abstractmethod
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        """Underlying method of the preprocessor to implement by subclassses.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical feature.

        Returns:
            list of indices of categorical features after the transform.
        """
        raise NotImplementedError

    def fit(self, X: np.ndarray, categorical_features: list[int]) -> Self:
        """Fits the preprocessor.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical feature.
        """
        self.categorical_features_after_transform_ = self._fit(X, categorical_features)
        assert self.categorical_features_after_transform_ is not None, (
            "_fit should have returned a list of the indexes of the categorical"
            "features after the transform."
        )
        return self

    @abstractmethod
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        """Underlying method of the preprocessor to implement by subclassses.

        Args:
            X: 2d array of shape (n_samples, n_features)
            is_test: Should be removed, used for the `AddFingerPrint` step.

        Returns:
            2d np.ndarray of shape (n_samples, new n_features)
        """
        raise NotImplementedError

    def transform(self, X: np.ndarray) -> _TransformResult:
        """Transforms the data.

        Args:
            X: 2d array of shape (n_samples, n_features).
        """
        # TODO: Get rid of this, it's always test in `transform`
        result = self._transform(X, is_test=True)
        return _TransformResult(result, self.categorical_features_after_transform_)


class SequentialFeatureTransformer(UserList):
    """A transformer that applies a sequence of feature preprocessing steps.
    This is very related to sklearn's Pipeline, but it is designed to work with
    categorical_features lists that are always passed on.

    Currently this class is only used once, thus this could also be made
    less general if needed.
    """

    def __init__(self, steps: Sequence[FeaturePreprocessingTransformerStep]):
        super().__init__(steps)
        self.steps = steps
        self.categorical_features_: list[int] | None = None

    def fit_transform(
        self,
        X: np.ndarray | torch.tensor,
        categorical_features: list[int],
    ) -> _TransformResult:
        """Fit and transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical features.
        """
        for step in self.steps:
            X, categorical_features = step.fit_transform(X, categorical_features)
            assert isinstance(categorical_features, list), (
                f"The {step=} must return list of categorical features,"
                f" but {type(step)} returned {categorical_features}"
            )

        self.categorical_features_ = categorical_features
        return _TransformResult(X, categorical_features)

    def fit(
        self, X: np.ndarray | torch.tensor, categorical_features: list[int]
    ) -> Self:
        """Fit all the steps in the pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features)
            categorical_features: list of indices of categorical feature.
        """
        assert (
            len(self) > 0
        ), "The SequentialFeatureTransformer must have at least one step."
        self.fit_transform(X, categorical_features)
        return self

    def transform(self, X: np.ndarray) -> _TransformResult:
        """Transform the data using the fitted pipeline.

        Args:
            X: 2d array of shape (n_samples, n_features).
        """
        assert (
            len(self) > 0
        ), "The SequentialFeatureTransformer must have at least one step."
        assert self.categorical_features_ is not None, (
            "The SequentialFeatureTransformer must be fit before it"
            " can be used to transform."
        )
        categorical_features = []
        for step in self:
            X, categorical_features = step.transform(X)

        assert categorical_features == self.categorical_features_, (
            f"Expected categorical features {self.categorical_features_},"
            f"but got {categorical_features}"
        )
        return _TransformResult(X, categorical_features)


class RemoveConstantFeaturesStep(FeaturePreprocessingTransformerStep):
    """Remove features that are constant in the training data."""

    def __init__(self) -> None:
        super().__init__()
        self.sel_: list[bool] | None = None

    @override
    def _fit(
        self, X: np.ndarray | torch.Tensor, categorical_features: list[int]
    ) -> list[int]:
        if isinstance(X, torch.Tensor):
            sel_ = torch.max(X[0:1, :] != X, dim=0)[0].cpu()
        else:
            sel_ = ((X[0:1, :] == X).mean(axis=0) < 1.0).tolist()

        if not any(sel_):
            raise ValueError(
                "All features are constant and would have been removed!"
                " Unable to predict using TabPFN.",
            )
        self.sel_ = sel_

        return [
            new_idx
            for new_idx, idx in enumerate(np.where(sel_)[0])
            if idx in categorical_features
        ]

    @override
    def _transform(
        self, X: np.ndarray | torch.Tensor, *, is_test: bool = False
    ) -> np.ndarray:
        assert self.sel_ is not None, "You must call fit first"
        return X[:, self.sel_]


_CONSTANT = 10**12


def float_hash_arr(arr: np.ndarray) -> float:
    _hash = int(hashlib.sha256(arr.tobytes()).hexdigest(), 16)
    return _hash % _CONSTANT / _CONSTANT


class AddFingerprintFeaturesStep(FeaturePreprocessingTransformerStep):
    """Adds a fingerprint feature to the features based on hash of each row.

    If `is_test = True`, it keeps the first hash even if there are collisions.
    If `is_test = False`, it handles hash collisions by counting up and rehashing
    until a unique hash is found.
    The idea is basically to add a random feature to help the model distinguish between
    identical rows. We use hashing to make sure the result does not depend on the order
    of the rows.
    """

    def __init__(self, random_state: int | np.random.Generator | None = None):
        super().__init__()
        self.random_state = random_state

    @override
    def _fit(
        self, X: np.ndarray | torch.Tensor, categorical_features: list[int]
    ) -> list[int]:
        _, rng = infer_random_state(self.random_state)
        self.rnd_salt_ = int(rng.integers(0, 2**16))
        return [*categorical_features]

    @override
    def _transform(
        self, X: np.ndarray | torch.Tensor, *, is_test: bool = False
    ) -> np.ndarray | torch.Tensor:
        X_det = X.detach().cpu().numpy() if isinstance(X, torch.Tensor) else X

        # no detach necessary for numpy
        X_h = np.zeros(X.shape[0], dtype=X_det.dtype)
        if is_test:
            # Keep the first hash even if there are collisions
            salted_X = X_det + self.rnd_salt_
            for i, row in enumerate(salted_X):
                h = float_hash_arr(row + self.rnd_salt_)
                X_h[i] = h
        else:
            # Handle hash collisions by counting up and rehashing
            seen_hashes = set()
            salted_X = X_det + self.rnd_salt_
            for i, row in enumerate(salted_X):
                h = float_hash_arr(row)
                add_to_hash = 0
                while h in seen_hashes and not np.isnan(row).all():
                    add_to_hash += 1
                    h = float_hash_arr(row + add_to_hash)
                X_h[i] = h
                seen_hashes.add(h)

        if isinstance(X, torch.Tensor):
            return torch.cat(
                [X, torch.from_numpy(X_h).float().reshape(-1, 1).to(X.device)], dim=1
            )
        else:  # noqa: RET505
            return np.concatenate([X, X_h.reshape(-1, 1)], axis=1)


class ShuffleFeaturesStep(FeaturePreprocessingTransformerStep):
    """Shuffle the features in the data."""

    def __init__(
        self,
        shuffle_method: Literal["shuffle", "rotate"] | None = "rotate",
        shuffle_index: int = 0,
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()
        self.random_state = random_state
        self.shuffle_method = shuffle_method
        self.shuffle_index = shuffle_index

        self.index_permutation_: list[int] | None = None

    @override
    def _fit(
        self, X: np.ndarray | torch.tensor, categorical_features: list[int]
    ) -> list[int]:
        _, rng = infer_random_state(self.random_state)
        if self.shuffle_method == "rotate":
            index_permutation = np.roll(
                np.arange(X.shape[1]),
                self.shuffle_index,
            ).tolist()
        elif self.shuffle_method == "shuffle":
            index_permutation = rng.permutation(X.shape[1]).tolist()
        elif self.shuffle_method is None:
            index_permutation = np.arange(X.shape[1]).tolist()
        else:
            raise ValueError(f"Unknown shuffle method {self.shuffle_method}")
        if isinstance(X, torch.Tensor):
            self.index_permutation_ = torch.tensor(index_permutation, dtype=torch.long)
        else:
            self.index_permutation_ = index_permutation

        return [
            new_idx
            for new_idx, idx in enumerate(index_permutation)
            if idx in categorical_features
        ]

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        assert self.index_permutation_ is not None, "You must call fit first"
        assert (
            len(self.index_permutation_) == X.shape[1]
        ), "The number of features must not change after fit"
        return X[:, self.index_permutation_]


class ReshapeFeatureDistributionsStep(FeaturePreprocessingTransformerStep):
    """Reshape the feature distributions using different transformations."""

    APPEND_TO_ORIGINAL_THRESHOLD = 500

    @staticmethod
    def get_column_types(X: np.ndarray) -> list[str]:
        """Returns a list of column types for the given data, that indicate how
        the data should be preprocessed.
        """
        # TODO(eddiebergman): Bad to keep calling skew again and again here...
        column_types = []
        for col in range(X.shape[1]):
            if np.unique(X[:, col]).size < 10:
                column_types.append(f"ordinal_{col}")
            elif (
                skew(X[:, col]) > 1.1
                and np.min(X[:, col]) >= 0
                and np.max(X[:, col]) <= 1
            ):
                column_types.append(f"skewed_pos_1_0_{col}")
            elif skew(X[:, col]) > 1.1 and np.min(X[:, col]) > 0:
                column_types.append(f"skewed_pos_{col}")
            elif skew(X[:, col]) > 1.1:
                column_types.append(f"skewed_{col}")
            elif shapiro(X[0:3000, col]).statistic > 0.95:
                column_types.append(f"normal_{col}")
            else:
                column_types.append(f"other_{col}")
        return column_types

    @staticmethod
    def get_adaptive_preprocessors(
        num_examples: int = 100,
        random_state: int | None = None,
    ) -> dict[str, ColumnTransformer]:
        """Returns a dictionary of adaptive column transformers that can be used to
        preprocess the data. Adaptive column transformers are used to preprocess the
        data based on the column type, they receive a pandas dataframe with column
        names, that indicate the column type. Column types are not datatypes,
        but rather a string that indicates how the data should be preprocessed.

        Args:
            num_examples: The number of examples in the dataset.
            random_state: The random state to use for the transformers.
        """
        return {
            "adaptive": ColumnTransformer(
                [
                    (
                        "skewed_pos_1_0",
                        FunctionTransformer(
                            func=np.exp,
                            inverse_func=np.log,
                            check_inverse=False,
                        ),
                        make_column_selector("skewed_pos_1_0*"),
                    ),
                    (
                        "skewed_pos",
                        make_box_cox_safe(
                            add_safe_standard_to_safe_power_without_standard(
                                SafePowerTransformer(
                                    standardize=False,
                                    method="box-cox",
                                ),
                            ),
                        ),
                        make_column_selector("skewed_pos*"),
                    ),
                    (
                        "skewed",
                        add_safe_standard_to_safe_power_without_standard(
                            SafePowerTransformer(
                                standardize=False,
                                method="yeo-johnson",
                            ),
                        ),
                        make_column_selector("skewed*"),
                    ),
                    (
                        "other",
                        AdaptiveQuantileTransformer(
                            output_distribution="normal",
                            n_quantiles=max(num_examples // 10, 2),
                            random_state=random_state,
                        ),
                        # "other" or "ordinal"
                        make_column_selector("other*"),
                    ),
                    (
                        "ordinal",
                        # default FunctionTransformer yields the identity function
                        FunctionTransformer(),
                        # "other" or "ordinal"
                        make_column_selector("ordinal*"),
                    ),
                    (
                        "normal",
                        # default FunctionTransformer yields the identity function
                        FunctionTransformer(),
                        make_column_selector("normal*"),
                    ),
                ],
                remainder="passthrough",
            ),
        }

    @staticmethod
    def get_all_preprocessors(
        num_examples: int,
        random_state: int | None = None,
    ) -> dict[str, TransformerMixin | Pipeline]:
        all_preprocessors = {
            "power": add_safe_standard_to_safe_power_without_standard(
                PowerTransformer(standardize=False),
            ),
            "safepower": add_safe_standard_to_safe_power_without_standard(
                SafePowerTransformer(standardize=False),
            ),
            "power_box": make_box_cox_safe(
                add_safe_standard_to_safe_power_without_standard(
                    PowerTransformer(standardize=False, method="box-cox"),
                ),
            ),
            "safepower_box": make_box_cox_safe(
                add_safe_standard_to_safe_power_without_standard(
                    SafePowerTransformer(standardize=False, method="box-cox"),
                ),
            ),
            "log": FunctionTransformer(
                func=np.log,
                inverse_func=np.exp,
                check_inverse=False,
            ),
            "1_plus_log": FunctionTransformer(
                func=np.log1p,
                inverse_func=_exp_minus_1,
                check_inverse=False,
            ),
            "exp": FunctionTransformer(
                func=np.exp,
                inverse_func=np.log,
                check_inverse=False,
            ),
            "quantile_uni_coarse": AdaptiveQuantileTransformer(
                output_distribution="uniform",
                n_quantiles=max(num_examples // 10, 2),
                random_state=random_state,
            ),
            "quantile_norm_coarse": AdaptiveQuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(num_examples // 10, 2),
                random_state=random_state,
            ),
            "quantile_uni": AdaptiveQuantileTransformer(
                output_distribution="uniform",
                n_quantiles=max(num_examples // 5, 2),
                random_state=random_state,
            ),
            "quantile_norm": AdaptiveQuantileTransformer(
                output_distribution="normal",
                n_quantiles=max(num_examples // 5, 2),
                random_state=random_state,
            ),
            "quantile_uni_fine": AdaptiveQuantileTransformer(
                output_distribution="uniform",
                n_quantiles=num_examples,
                random_state=random_state,
            ),
            "quantile_norm_fine": AdaptiveQuantileTransformer(
                output_distribution="normal",
                n_quantiles=num_examples,
                random_state=random_state,
            ),
            "robust": RobustScaler(unit_variance=True),
            # default FunctionTransformer yields the identity function
            "none": FunctionTransformer(),
            **get_all_kdi_transformers(),
        }

        with contextlib.suppress(Exception):
            all_preprocessors["norm_and_kdi"] = FeatureUnion(
                [
                    (
                        "norm",
                        AdaptiveQuantileTransformer(
                            output_distribution="normal",
                            n_quantiles=max(num_examples // 10, 2),
                            random_state=random_state,
                        ),
                    ),
                    (
                        "kdi",
                        KDITransformerWithNaN(alpha=1.0, output_distribution="uniform"),
                    ),
                ],
            )

        all_preprocessors.update(
            ReshapeFeatureDistributionsStep.get_adaptive_preprocessors(
                num_examples,
                random_state=random_state,
            ),
        )

        return all_preprocessors

    def get_all_global_transformers(
        self,
        num_examples: int,
        num_features: int,
        random_state: int | None = None,
    ) -> dict[str, FeatureUnion | Pipeline]:
        return {
            "scaler": make_standard_scaler_safe(("standard", StandardScaler())),
            "svd": FeatureUnion(
                [
                    # default FunctionTransformer yields the identity function
                    ("passthrough", FunctionTransformer()),
                    (
                        "svd",
                        Pipeline(
                            steps=[
                                (
                                    "save_standard",
                                    make_standard_scaler_safe(
                                        ("standard", StandardScaler(with_mean=False)),
                                    ),
                                ),
                                (
                                    "svd",
                                    TruncatedSVD(
                                        algorithm="arpack",
                                        n_components=max(
                                            1,
                                            min(
                                                num_examples // 10 + 1,
                                                num_features // 2,
                                            ),
                                        ),
                                        random_state=random_state,
                                    ),
                                ),
                            ],
                        ),
                    ),
                ],
            ),
        }

    def __init__(
        self,
        *,
        transform_name: str = "safepower",
        apply_to_categorical: bool = False,
        append_to_original: bool | Literal["auto"] = False,
        subsample_features: float = -1,
        global_transformer_name: str | None = None,
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()
        self.transform_name = transform_name
        self.apply_to_categorical = apply_to_categorical
        self.append_to_original = append_to_original
        self.random_state = random_state
        self.subsample_features = float(subsample_features)
        self.global_transformer_name = global_transformer_name
        self.transformer_: Pipeline | ColumnTransformer | None = None

    def _set_transformer_and_cat_ix(  # noqa: PLR0912
        self,
        n_samples: int,
        n_features: int,
        categorical_features: list[int],
    ) -> tuple[Pipeline | ColumnTransformer, list[int]]:
        if "adaptive" in self.transform_name:
            raise NotImplementedError("Adaptive preprocessing raw removed.")

        static_seed, rng = infer_random_state(self.random_state)

        if (
            self.global_transformer_name is not None
            and self.global_transformer_name != "None"
            and not (self.global_transformer_name == "svd" and n_features < 2)
        ):
            global_transformer_ = self.get_all_global_transformers(
                n_samples,
                n_features,
                random_state=static_seed,
            )[self.global_transformer_name]
        else:
            global_transformer_ = None

        all_preprocessors = self.get_all_preprocessors(
            n_samples,
            random_state=static_seed,
        )
        if self.subsample_features > 0:
            subsample_features = int(self.subsample_features * n_features) + 1
            # sampling more features than exist
            replace = subsample_features > n_features
            self.subsampled_features_ = rng.choice(
                list(range(n_features)),
                subsample_features,
                replace=replace,
            )
            categorical_features = [
                new_idx
                for new_idx, idx in enumerate(self.subsampled_features_)
                if idx in categorical_features
            ]
            n_features = subsample_features
        else:
            self.subsampled_features_ = np.arange(n_features)

        all_feats_ix = list(range(n_features))
        transformers = []

        numerical_ix = [i for i in range(n_features) if i not in categorical_features]

        append_decision = n_features < self.APPEND_TO_ORIGINAL_THRESHOLD
        self.append_to_original = (
            append_decision
            if self.append_to_original == "auto"
            else self.append_to_original
        )

        # -------- Append to original ------
        # If we append to original, all the categorical indices are kept in place
        # as the first transform is a passthrough on the whole X as it is above
        if self.append_to_original and self.apply_to_categorical:
            trans_ixs = categorical_features + numerical_ix
            transformers.append(("original", "passthrough", all_feats_ix))
            cat_ix = categorical_features  # Exist as they are in original

        elif self.append_to_original and not self.apply_to_categorical:
            trans_ixs = numerical_ix
            # Includes the categoricals passed through
            transformers.append(("original", "passthrough", all_feats_ix))
            cat_ix = categorical_features  # Exist as they are in original

        # -------- Don't append to original ------
        # We only have categorical indices if we don't transform them
        # The first transformer will be a passthrough on the categorical indices
        # Making them the first
        elif not self.append_to_original and self.apply_to_categorical:
            trans_ixs = categorical_features + numerical_ix
            cat_ix = []  # We have none left, they've been transformed

        elif not self.append_to_original and not self.apply_to_categorical:
            trans_ixs = numerical_ix
            transformers.append(("cats", "passthrough", categorical_features))
            cat_ix = list(range(len(categorical_features)))  # They are at start

        else:
            raise ValueError(
                f"Unrecognized combination of {self.apply_to_categorical=}"
                f" and {self.append_to_original=}",
            )

        # NOTE: No need to keep track of categoricals here, already done above
        if self.transform_name != "per_feature":
            _transformer = all_preprocessors[self.transform_name]
            transformers.append(("feat_transform", _transformer, trans_ixs))
        else:
            preprocessors = list(all_preprocessors.values())
            transformers.extend(
                [
                    (f"transformer_{i}", rng.choice(preprocessors), [i])  # type: ignore
                    for i in trans_ixs
                ],
            )

        transformer = ColumnTransformer(
            transformers,
            remainder="drop",
            sparse_threshold=0.0,  # No sparse
        )

        # Apply a global transformer which accepts the entire dataset instead of
        # one column
        # NOTE: We assume global_transformer does not destroy the semantic meaning of
        # categorical_features_.
        if global_transformer_:
            transformer = Pipeline(
                [
                    ("preprocess", transformer),
                    ("global_transformer", global_transformer_),
                ],
            )

        self.transformer_ = transformer

        return transformer, cat_ix

    @override
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        n_samples, n_features = X.shape
        transformer, cat_ix = self._set_transformer_and_cat_ix(
            n_samples,
            n_features,
            categorical_features,
        )
        transformer.fit(X[:, self.subsampled_features_])
        self.categorical_features_after_transform_ = cat_ix
        self.transformer_ = transformer
        return cat_ix

    @override
    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> _TransformResult:
        n_samples, n_features = X.shape
        transformer, cat_ix = self._set_transformer_and_cat_ix(
            n_samples,
            n_features,
            categorical_features,
        )
        Xt = transformer.fit_transform(X[:, self.subsampled_features_])
        self.categorical_features_after_transform_ = cat_ix
        self.transformer_ = transformer
        return _TransformResult(Xt, cat_ix)  # type: ignore

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        assert self.transformer_ is not None, "You must call fit first"
        return self.transformer_.transform(X[:, self.subsampled_features_])  # type: ignore


class EncodeCategoricalFeaturesStep(FeaturePreprocessingTransformerStep):
    def __init__(
        self,
        categorical_transform_name: str = "ordinal",
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()
        self.categorical_transform_name = categorical_transform_name
        self.random_state = random_state

        self.categorical_transformer_ = None

    @staticmethod
    def get_least_common_category_count(x_column: np.ndarray) -> int:
        if len(x_column) == 0:
            return 0
        counts = np.unique(x_column, return_counts=True)[1]
        return int(counts.min())

    def _get_transformer(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> tuple[ColumnTransformer | None, list[int]]:
        if self.categorical_transform_name.startswith("ordinal"):
            name = self.categorical_transform_name[len("ordinal") :]
            # Create a column transformer
            if name.startswith("_common_categories"):
                name = name[len("_common_categories") :]
                categorical_features = [
                    i
                    for i, col in enumerate(X.T)
                    if i in categorical_features
                    and self.get_least_common_category_count(col) >= 10
                ]
            elif name.startswith("_very_common_categories"):
                name = name[len("_very_common_categories") :]
                categorical_features = [
                    i
                    for i, col in enumerate(X.T)
                    if i in categorical_features
                    and self.get_least_common_category_count(col) >= 10
                    and len(np.unique(col)) < (len(X) // 10)  # type: ignore
                ]

            assert name in ("_shuffled", ""), (
                "unknown categorical transform name, should be 'ordinal'"
                f" or 'ordinal_shuffled' it was {self.categorical_transform_name}"
            )

            ct = ColumnTransformer(
                [
                    (
                        "ordinal_encoder",
                        OrdinalEncoder(
                            handle_unknown="use_encoded_value",
                            unknown_value=np.nan,
                        ),  # 'sparse' has been deprecated
                        categorical_features,
                    ),
                ],
                # The column numbers to be transformed
                remainder="passthrough",  # Leave the rest of the columns untouched
            )
            return ct, categorical_features

        if self.categorical_transform_name == "onehot":
            # Create a column transformer
            ct = ColumnTransformer(
                [
                    (
                        "one_hot_encoder",
                        OneHotEncoder(
                            drop="if_binary",
                            sparse_output=False,
                            handle_unknown="ignore",
                        ),
                        categorical_features,
                    ),
                ],
                # The column numbers to be transformed
                remainder="passthrough",  # Leave the rest of the columns untouched
            )
            return ct, categorical_features

        if self.categorical_transform_name in ("numeric", "none"):
            return None, categorical_features
        raise ValueError(
            f"Unknown categorical transform {self.categorical_transform_name}",
        )

    def _fit(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> list[int]:
        ct, categorical_features = self._get_transformer(X, categorical_features)
        if ct is None:
            self.categorical_transformer_ = None
            return categorical_features

        _, rng = infer_random_state(self.random_state)

        if self.categorical_transform_name.startswith("ordinal"):
            ct.fit(X)
            categorical_features = list(range(len(categorical_features)))

            self.random_mappings_ = {}
            if self.categorical_transform_name.endswith("_shuffled"):
                for col_ix in categorical_features:
                    col_cats = len(
                        ct.named_transformers_["ordinal_encoder"].categories_[col_ix],
                    )
                    perm = rng.permutation(col_cats)
                    self.random_mappings_[col_ix] = perm

        elif self.categorical_transform_name == "onehot":
            Xt = ct.fit_transform(X)
            if Xt.size >= 1_000_000:
                ct = None
            else:
                categorical_features = list(range(Xt.shape[1]))[
                    ct.output_indices_["one_hot_encoder"]
                ]
        else:
            raise ValueError(
                f"Unknown categorical transform {self.categorical_transform_name}",
            )

        self.categorical_transformer_ = ct
        return categorical_features

    def _fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> tuple[np.ndarray, list[int]]:
        ct, categorical_features = self._get_transformer(X, categorical_features)
        if ct is None:
            self.categorical_transformer_ = None
            return X, categorical_features

        _, rng = infer_random_state(self.random_state)

        if self.categorical_transform_name.startswith("ordinal"):
            Xt = ct.fit_transform(X)
            categorical_features = list(range(len(categorical_features)))

            self.random_mappings_ = {}
            if self.categorical_transform_name.endswith("_shuffled"):
                for col_ix in categorical_features:
                    col_cats = len(
                        ct.named_transformers_["ordinal_encoder"].categories_[col_ix],
                    )
                    perm = rng.permutation(col_cats)
                    self.random_mappings_[col_ix] = perm

                    Xcol: np.ndarray = Xt[:, col_ix]  # type: ignore
                    not_nan_mask = ~np.isnan(Xcol)
                    Xcol[not_nan_mask] = perm[Xcol[not_nan_mask].astype(int)].astype(
                        Xcol.dtype,
                    )

        elif self.categorical_transform_name == "onehot":
            Xt = ct.fit_transform(X)
            if Xt.size >= 1_000_000:
                ct = None
                Xt = X
            else:
                categorical_features = list(range(Xt.shape[1]))[
                    ct.output_indices_["one_hot_encoder"]
                ]
        else:
            raise ValueError(
                f"Unknown categorical transform {self.categorical_transform_name}",
            )

        self.categorical_transformer_ = ct
        return Xt, categorical_features  # type: ignore

    @override
    def fit_transform(
        self,
        X: np.ndarray,
        categorical_features: list[int],
    ) -> _TransformResult:
        Xt, cat_ix = self._fit_transform(X, categorical_features)
        self.categorical_features_after_transform_ = cat_ix
        return _TransformResult(Xt, cat_ix)

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        if self.categorical_transformer_ is None:
            return X

        import warnings

        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", message=".*Found unknown categories in col.*"
            )  # These warnings are expected when transforming test data
            transformed = self.categorical_transformer_.transform(X)
        if self.categorical_transform_name.endswith("_shuffled"):
            for col, mapping in self.random_mappings_.items():
                not_nan_mask = ~np.isnan(transformed[:, col])  # type: ignore
                transformed[:, col][not_nan_mask] = mapping[
                    transformed[:, col][not_nan_mask].astype(int)
                ].astype(transformed[:, col].dtype)
        return transformed  # type: ignore


class NanHandlingPolynomialFeaturesStep(FeaturePreprocessingTransformerStep):
    def __init__(
        self,
        *,
        max_features: int | None = None,
        random_state: int | np.random.Generator | None = None,
    ):
        super().__init__()

        self.max_poly_features = max_features
        self.random_state = random_state

        self.poly_factor_1_idx: np.ndarray | None = None
        self.poly_factor_2_idx: np.ndarray | None = None

        self.standardizer = StandardScaler(with_mean=False)

    @override
    def _fit(self, X: np.ndarray, categorical_features: list[int]) -> list[int]:
        assert len(X.shape) == 2, "Input data must be 2D, i.e. (n_samples, n_features)"
        _, rng = infer_random_state(self.random_state)

        if X.shape[0] == 0 or X.shape[1] == 0:
            return [*categorical_features]

        # How many polynomials can we create?
        n_polynomials = (X.shape[1] * (X.shape[1] - 1)) // 2 + X.shape[1]
        n_polynomials = (
            min(self.max_poly_features, n_polynomials)
            if self.max_poly_features
            else n_polynomials
        )

        X = self.standardizer.fit_transform(X)

        # Randomly select the indices of the factors
        self.poly_factor_1_idx = rng.choice(
            np.arange(0, X.shape[1]),
            size=n_polynomials,
            replace=True,
        )
        self.poly_factor_2_idx = np.ones_like(self.poly_factor_1_idx) * -1
        for i in range(len(self.poly_factor_1_idx)):
            while self.poly_factor_2_idx[i] == -1:
                poly_factor_1_ = self.poly_factor_1_idx[i]
                # indices of the factors that have already been used
                used_indices = self.poly_factor_2_idx[
                    self.poly_factor_1_idx == poly_factor_1_
                ]
                # remaining indices, only factors with higher index can be selected
                # to avoid duplicates
                indices_ = set(range(poly_factor_1_, X.shape[1])) - set(
                    used_indices.tolist(),
                )
                if len(indices_) == 0:
                    self.poly_factor_1_idx[i] = rng.choice(
                        np.arange(0, X.shape[1]),
                        size=1,
                    )
                    continue
                self.poly_factor_2_idx[i] = rng.choice(list(indices_), size=1)

        return categorical_features

    @override
    def _transform(self, X: np.ndarray, *, is_test: bool = False) -> np.ndarray:
        assert len(X.shape) == 2, "Input data must be 2D, i.e. (n_samples, n_features)"

        if X.shape[0] == 0 or X.shape[1] == 0:
            return X

        X = self.standardizer.transform(X)  # type: ignore

        poly_features_xs = X[:, self.poly_factor_1_idx] * X[:, self.poly_factor_2_idx]

        return np.hstack((X, poly_features_xs))


class DifferentiableZNormStep(FeaturePreprocessingTransformerStep):
    def __init__(self):
        super().__init__()

        self.means = torch.tensor([])
        self.stds = torch.tensor([])

    def _fit(self, X: torch.Tensor, categorical_features: list[int]) -> list[int]:
        self.means = X.mean(dim=0, keepdim=True)
        self.stds = X.std(dim=0, keepdim=True)
        return categorical_features

    def _transform(self, X: torch.Tensor, *, is_test=False):  # noqa: ARG002
        assert X.shape[1] == self.means.shape[1]
        assert X.shape[1] == self.stds.shape[1]
        return (X - self.means) / self.stds
