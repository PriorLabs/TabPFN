from __future__ import annotations

import numpy as np

from tabpfn.model.preprocessing import ReshapeFeatureDistributionsStep


def test_preprocessing_large_dataset():
    # Generate a synthetic dataset with more than 10,000 samples
    num_samples = 15000
    num_features = 10
    X = np.random.rand(num_samples, num_features)

    # Create an instance of ReshapeFeatureDistributionsStep
    preprocessing_step = ReshapeFeatureDistributionsStep(
        transform_name="quantile_norm",
        apply_to_categorical=False,
        append_to_original=False,
        subsample_features=-1,
        global_transformer_name=None,
        random_state=42,
    )

    # Define categorical features (empty in this case)
    categorical_features = []

    # Run the preprocessing step
    result = preprocessing_step.fit_transform(X, categorical_features)

    # Assert the result is not None
    assert result is not None
