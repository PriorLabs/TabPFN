#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.preprocessing.steps import KDITransformerWithNaN

kditransform = pytest.importorskip("kditransform")


def test__kdi_transformer_fit__with_nan_integration():
    """Tests KDITransformerWithNaN handles NaNs and maintains mask."""
    # Create data with NaNs and a torch tensor to test both features
    X = torch.tensor(
        [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [np.nan, 8.0, 9.0]],
        dtype=torch.float32,
    )

    transformer = KDITransformerWithNaN(alpha=1.0, output_distribution="normal")

    # Test fit
    transformer.fit(X)
    assert hasattr(transformer, "imputation_values_")

    # Test transform
    Xt = transformer.transform(X)

    # Verify type and shape
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == X.shape

    # Verify NaNs are preserved in the exact same positions
    mask = torch.isnan(X).numpy()
    assert np.all(np.isnan(Xt) == mask)

    # Verify non-NaN values are actual numbers (transformed)
    assert np.all(np.isfinite(Xt[~mask]))


def test__kdi_transformer_fit_transform__with_nan_integration():
    """Tests KDITransformerWithNaN handles NaNs and maintains mask."""
    # Create data with NaNs and a torch tensor to test both features
    X = torch.tensor(
        [[1.0, np.nan, 3.0], [4.0, 5.0, np.nan], [np.nan, 8.0, 9.0]],
        dtype=torch.float32,
    )

    transformer = KDITransformerWithNaN(alpha=1.0, output_distribution="normal")

    # Test fit
    Xt = transformer.fit_transform(X)
    assert hasattr(transformer, "imputation_values_")

    # Verify type and shape
    assert isinstance(Xt, np.ndarray)
    assert Xt.shape == X.shape

    # Verify NaNs are preserved in the exact same positions
    mask = torch.isnan(X).numpy()
    assert np.all(np.isnan(Xt) == mask)

    # Verify non-NaN values are actual numbers (transformed)
    assert np.all(np.isfinite(Xt[~mask]))


# ---------------------------------------------------------------------------
# The overridden fit and transform against the kditransform class they mirror
# ---------------------------------------------------------------------------


def _kdi_cases() -> list[np.ndarray]:
    rng = np.random.default_rng(0)
    plain = rng.standard_normal((2165, 11)) * rng.uniform(0.1, 50, 11)
    ties = np.round(rng.standard_normal((598, 6)) * 3)
    with_nan = plain[:700].copy()
    with_nan[rng.random(with_nan.shape) < 0.05] = np.nan
    constant = plain[:300].copy()
    constant[:, 2] = 4.0
    small = rng.standard_normal((37, 3))
    single = rng.standard_normal((1132, 1)).astype(np.float32).astype(np.float64)
    return [plain, ties, with_nan, constant, small, single]


@pytest.mark.parametrize("alpha", [0.3, 1.0, 3.0])
@pytest.mark.parametrize("output_distribution", ["normal", "uniform"])
def test__kdi_transformer__matches_kditransform_bit_for_bit(
    alpha: float, output_distribution: str
) -> None:
    """Quantiles, forward and inverse transforms equal the upstream implementation."""
    for X in _kdi_cases():
        X_imputed = np.nan_to_num(X, nan=np.nan_to_num(np.nanmean(X, axis=0), nan=0))
        reference = kditransform.KDITransformer(
            alpha=alpha, output_distribution=output_distribution
        )
        ours = KDITransformerWithNaN(
            alpha=alpha, output_distribution=output_distribution
        )
        # Fit on a copy: the upstream fit centres and un-centres its input in place,
        # which moves it by a rounding error, while `KDITransformerWithNaN.fit` works
        # on the imputed copy and leaves the input alone.
        reference.fit(X_imputed.copy())
        expected = reference.transform(X_imputed.copy())
        result = ours.fit_transform(X)
        np.testing.assert_array_equal(ours.quantiles_, reference.quantiles_)
        np.testing.assert_array_equal(result[~np.isnan(X)], expected[~np.isnan(X)])
        assert np.isnan(result[np.isnan(X)]).all()
        X_new = X[: len(X) // 2] * 1.1 + 0.3
        np.testing.assert_array_equal(
            ours.transform(np.nan_to_num(X_new, nan=0.0)),
            reference.transform(np.nan_to_num(X_new, nan=0.0)),
        )
        np.testing.assert_array_equal(
            ours.inverse_transform(expected.copy()),
            reference.inverse_transform(expected.copy()),
        )
