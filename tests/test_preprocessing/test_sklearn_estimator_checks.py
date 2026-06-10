#  Copyright (c) Prior Labs GmbH 2026.

"""sklearn's standard estimator checks, run on all sklearn-style transformers."""

from __future__ import annotations

import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from tabpfn.preprocessing.steps import (
    AdaptiveQuantileTransformer,
    SafePowerTransformer,
    SquashingScaler,
)
from tabpfn.preprocessing.steps.kdi_transformer import KDITransformerWithNaN

# Input-validation gaps inherited from the upstream `kditransform` package.
_KNOWN_DEVIATIONS = {
    ("KDITransformerWithNaN", "check_n_features_in_after_fitting"),
    ("KDITransformerWithNaN", "check_complex_data"),
    ("KDITransformerWithNaN", "check_dtype_object"),
    ("KDITransformerWithNaN", "check_estimator_sparse_tag"),
    ("KDITransformerWithNaN", "check_estimator_sparse_array"),
    ("KDITransformerWithNaN", "check_estimator_sparse_matrix"),
    ("KDITransformerWithNaN", "check_estimator_sparse_data"),  # sklearn < 1.5 name
    ("KDITransformerWithNaN", "check_transformer_data_not_an_array"),
    # KDI's KDE bandwidth optimization cannot bracket on a single sample.
    ("KDITransformerWithNaN", "check_fit2d_1sample"),
}


@parametrize_with_checks(
    [
        AdaptiveQuantileTransformer(),
        SafePowerTransformer(),
        SquashingScaler(),
        KDITransformerWithNaN(alpha=1.0, output_distribution="uniform"),
    ]
)
def test__sklearn_estimator_checks(estimator, check) -> None:
    """All transformers must satisfy sklearn's estimator contract (clone,
    set_params, no param mutation in fit, pickling, ...).
    """
    check_name = getattr(check, "func", check).__name__
    if (type(estimator).__name__, check_name) in _KNOWN_DEVIATIONS:
        pytest.xfail("input validation gap inherited from kditransform")
    check(estimator)
