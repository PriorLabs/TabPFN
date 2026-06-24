#  Copyright (c) Prior Labs GmbH 2026.

"""Tests for mode (vs mean) imputation of categorical features.

Covers the leaf transformers that perform NaN imputation and accept a
``categorical_features`` argument: the safe-scaler imputer, KDITransformerWithNaN
and TorchSafeStandardScaler. Numerical columns must keep mean imputation.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.architectures.tabpfn_v2_5 import (
    _pad_and_reshape_feature_groups,
    _remove_constant_features,
)
from tabpfn.architectures.tabpfn_v3 import _impute_nan_and_inf_with_mean
from tabpfn.preprocessing.steps.kdi_transformer import KDITransformerWithNaN
from tabpfn.preprocessing.steps.utils import _NoInverseImputer, mode
from tabpfn.preprocessing.torch.ops import (
    categorical_mask_from_inds,
    categorical_mode_fill,
    grouped_inds_from_mask,
)
from tabpfn.preprocessing.torch.steps import TorchAddSVDFeaturesStep
from tabpfn.preprocessing.torch.torch_standard_scaler import TorchStandardScaler
from tabpfn.preprocessing.torch.torch_svd import TorchSafeStandardScaler


def _data() -> np.ndarray:
    # col 0 (categorical): mode is 1.0 (appears 3x), mean of [1,1,1,4] is 1.75
    # col 1 (numerical): mean of finite values is 20.0
    return np.array(
        [
            [1.0, 10.0],
            [1.0, np.nan],
            [1.0, 30.0],
            [4.0, np.nan],
            [np.nan, 20.0],
        ],
    )


def test__no_inverse_imputer__mode_for_categorical_mean_for_numerical():
    x = _data()
    imp = _NoInverseImputer(strategy="mean", categorical_features=[0]).fit(x)
    # col 0 imputed with mode (1.0), not mean (1.5); col 1 stays mean (20.0).
    assert imp.statistics_[0] == 1.0
    assert imp.statistics_[1] == 20.0


def test__no_inverse_imputer__defaults_to_mean():
    x = _data()
    imp = _NoInverseImputer(strategy="mean").fit(x)
    np.testing.assert_allclose(imp.statistics_, [1.75, 20.0])


def test__kdi__mode_for_categorical():
    x = _data()
    kdi = KDITransformerWithNaN(categorical_features=[0]).fit(x)
    assert kdi.imputation_values_[0] == 1.0
    assert kdi.imputation_values_[1] == 20.0

    kdi_mean = KDITransformerWithNaN().fit(x)
    assert kdi_mean.imputation_values_[0] == 1.75


def test__torch_safe_standard_scaler__mode_fill_for_categorical():
    x = torch.from_numpy(_data()).to(torch.float32)
    cache = TorchSafeStandardScaler(categorical_features=[0]).fit(x)
    # impute_fill is the categorical-aware fill: categorical -> mode (1.0).
    assert cache["impute_fill"][0].item() == pytest.approx(1.0)
    assert cache["impute_fill"][1].item() == pytest.approx(20.0)
    # "mean" stays the true mean of the mode-imputed data ([1,1,1,4,1] -> 1.6).
    assert cache["mean"][0].item() == pytest.approx(1.6)
    assert cache["mean"][1].item() == pytest.approx(20.0)

    mean_cache = TorchSafeStandardScaler().fit(x)
    assert "impute_fill" not in mean_cache
    assert mean_cache["mean"][0].item() == pytest.approx(1.75)


def test__torch_svd_step__threads_mode_to_scaler():
    # The GPU SVD step's pre-SVD scaler should mode-impute the categorical col.
    # x shape for _fit is [num_train_rows, batch_size, num_cols].
    x = torch.from_numpy(_data()).to(torch.float32).unsqueeze(1)  # batch=1
    cache = TorchAddSVDFeaturesStep(categorical_features=[0])._fit(x)
    assert cache["scaler_impute_fill"][0].item() == pytest.approx(1.0)  # mode
    assert cache["scaler_impute_fill"][1].item() == pytest.approx(20.0)  # mean
    assert cache["scaler_mean"][0].item() == pytest.approx(1.6)  # mean of imputed

    mean_cache = TorchAddSVDFeaturesStep()._fit(x)
    assert mean_cache["scaler_impute_fill"][0].item() == pytest.approx(1.75)  # mean


def test__all_nan_categorical_falls_back_to_mean():
    # A categorical column with no finite values must not break imputation.
    # (The real pipeline converts inf->nan before the imputer, so an all-missing
    # column reaches the imputer as all-NaN.)
    x = np.array([[np.nan, 1.0], [np.nan, 2.0]])
    imp = _NoInverseImputer(
        strategy="mean", keep_empty_features=True, categorical_features=[0]
    ).fit(x)
    assert np.isfinite(imp.statistics_[0])  # kept mean's (finite) fallback
    # sanity: mode() itself reports NaN for the all-non-finite column
    assert np.isnan(mode(x)[0])


def _x_rbc() -> torch.Tensor:
    # (R, B, C): col 0 categorical (mode 1.0, mean 1.75), col 1 numerical (mean 20).
    return torch.from_numpy(_data()).to(torch.float32).unsqueeze(1)  # batch=1


def _route_through_grouping(
    categorical_inds: list[list[int]],
    column_mask: torch.Tensor,
    num_features_per_group: int,
) -> list[list[int]]:
    """Route a categorical mask through the real v2.5 constant-removal + grouping."""
    batch_size, num_columns = column_mask.shape
    mask = categorical_mask_from_inds(
        categorical_inds, batch_size, num_columns, torch.zeros(1)
    )
    mask, _ = _remove_constant_features(x_RiBC=mask, column_mask=column_mask)
    mask, _ = _pad_and_reshape_feature_groups(mask, num_features_per_group)
    return grouped_inds_from_mask(mask)


def test__mask_routing__maps_through_pad_and_group():
    # B=1, C=3, F=2: cols pad to 4 -> 2 groups of 2. Original cats {0, 2} land at
    # group0/pos0 and group1/pos0. No constant removal (mask all True).
    grouped = _route_through_grouping(
        [[0, 2]], torch.ones(1, 3, dtype=torch.bool), num_features_per_group=2
    )
    assert grouped == [[0], [0]]


def test__mask_routing__shifts_with_constant_removal():
    # Dropping a leading non-categorical column (B=1) shifts later categoricals.
    # cols=[0:num, 1:cat, 2:num], drop col0 -> kept [cat, num] -> pad to F=2 ->
    # one group [cat@0, num@1]; categorical is at group0/pos0.
    grouped = _route_through_grouping(
        [[1]], torch.tensor([[False, True, True]]), num_features_per_group=2
    )
    assert grouped == [[0]]


def test__categorical_mode_fill__overwrites_only_categorical_columns():
    x = _x_rbc()
    mean_fill = torch.tensor([[1.75, 20.0]])  # (B, C)
    fill = categorical_mode_fill(
        x, num_train_rows=x.shape[0], categorical_inds=[[0]], mean_fill_BC=mean_fill
    )
    assert fill[0, 0].item() == pytest.approx(1.0)  # categorical -> mode
    assert fill[0, 1].item() == pytest.approx(20.0)  # numerical -> mean (untouched)


def test__torch_standard_scaler__separate_mode_fill_keeps_mean_centering():
    x = _x_rbc()  # (R, B=1, C=2): col0 categorical (mode 1.0), col1 numerical (mean 20)
    cache = TorchStandardScaler().fit(x, categorical_inds=[[0]])
    # impute_fill: mode for categorical, mean for numerical.
    assert cache["impute_fill"][0, 0].item() == pytest.approx(1.0)  # mode
    assert cache["impute_fill"][0, 1].item() == pytest.approx(20.0)  # mean
    # "mean" (used for centering) stays the mean of the mode-imputed col:
    # [1,1,1,4,1] -> 1.6, not the mode.
    assert cache["mean"][0, 0].item() == pytest.approx(1.6)
    assert cache["mean"][0, 1].item() == pytest.approx(20.0)

    mean_cache = TorchStandardScaler().fit(x)
    assert "impute_fill" not in mean_cache
    assert mean_cache["mean"][0, 0].item() == pytest.approx(1.75)  # mean of [1,1,1,4]


def test__v3_impute__mode_for_categorical_mean_otherwise():
    x = _x_rbc()  # col0 has a NaN at row 4, col1 NaNs at rows 1 and 3

    out, _ = _impute_nan_and_inf_with_mean(
        x, num_train_rows=x.shape[0], categorical_inds=[[0]]
    )
    assert out[4, 0, 0].item() == pytest.approx(1.0)  # categorical filled with mode
    assert out[1, 0, 1].item() == pytest.approx(20.0)  # numerical filled with mean

    # Without categorical_inds the categorical cell falls back to the mean.
    out_mean, _ = _impute_nan_and_inf_with_mean(x, num_train_rows=x.shape[0])
    assert out_mean[4, 0, 0].item() == pytest.approx(1.75)  # mean of [1,1,1,4]


def test__v3_impute__cache_path_uses_mode_from_scaler_cache():
    # The fix: on the cache path the mode comes from the scaler cache (baked in by
    # TorchStandardScaler.fit), so no per-call recompute is needed.
    x = _x_rbc()
    cache = TorchStandardScaler().fit(x, categorical_inds=[[0]])
    out, _ = _impute_nan_and_inf_with_mean(
        x, num_train_rows=x.shape[0], scaler_cache=cache
    )
    assert out[4, 0, 0].item() == pytest.approx(1.0)  # categorical -> cached mode
    assert out[1, 0, 1].item() == pytest.approx(20.0)  # numerical -> cached mean
