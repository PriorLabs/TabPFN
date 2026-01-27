from __future__ import annotations

from typing import Literal

import numpy as np
import pytest

from tabpfn.preprocessing.steps import ReshapeFeatureDistributionsStep


def test__preprocessing_large_dataset():
    num_samples = 150000
    num_features = 2
    input_shape = (num_samples, num_features)
    rng = np.random.default_rng(42)
    X = rng.random(input_shape)

    preprocessing_step = ReshapeFeatureDistributionsStep(
        transform_name="quantile_uni",
        apply_to_categorical=False,
        append_to_original=False,
        max_features_per_estimator=500,
        global_transformer_name=None,
        random_state=42,
    )

    result = preprocessing_step.fit_transform(X, categorical_features=[])

    assert result is not None
    X_transformed = result.X
    assert X_transformed.shape == input_shape
    assert X_transformed.dtype == X.dtype


@pytest.mark.parametrize(
    ("append_to_original_setting", "num_features", "expected_output_features"),
    [
        # Test 'auto' mode below the threshold: should append original features
        pytest.param("auto", 10, 20, id="auto_below_threshold_appends"),
        # Test 'auto' mode above the threshold: should NOT append original features
        pytest.param("auto", 600, 500, id="auto_above_threshold_replaces"),
        # If n features more than half of max_features_per_estimator we do not append
        pytest.param("auto", 300, 300, id="auto_below_half_threshold_replaces"),
        # True: always append after capping (600 → capped 500 → doubled)
        pytest.param(True, 600, 1000, id="true_always_appends"),
        # Test False: should never append
        pytest.param(False, 10, 10, id="false_never_appends"),
    ],
)
def test_reshape_step_append_original_logic(
    append_to_original_setting: bool | Literal["auto"],
    num_features: int,
    expected_output_features: int,
):
    """Tests the `append_to_original` logic, including the "auto" mode which
    depends on the APPEND_TO_ORIGINAL_THRESHOLD class constant (500).
    """
    num_samples = 100
    rng = np.random.default_rng(42)
    X = rng.random((num_samples, num_features))

    preprocessing_step = ReshapeFeatureDistributionsStep(
        transform_name="quantile_uni",
        append_to_original=append_to_original_setting,
        random_state=42,
        max_features_per_estimator=500,
    )

    X_transformed, _ = preprocessing_step.fit_transform(X, categorical_features=[])

    assert X_transformed.shape[0] == num_samples
    assert X_transformed.shape[1] == expected_output_features
