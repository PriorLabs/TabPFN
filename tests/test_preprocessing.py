from __future__ import annotations

import numpy as np
import pytest
import torch

from tabpfn.model.preprocessing import (
    DifferentiableZNormStep,
    ReshapeFeatureDistributionsStep,
)

# --- Fixtures ---


@pytest.fixture
def sample_data():
    """Provides a simple 2D torch tensor for testing."""
    return torch.tensor(
        [[1.0, 10.0, -5.0], [3.0, 12.0, -3.0], [5.0, 14.0, -1.0]], dtype=torch.float32
    )


def test_preprocessing_large_dataset():
    # Generate a synthetic dataset with more than 10,000 samples
    num_samples = 150000
    num_features = 2
    rng = np.random.default_rng()
    X = rng.random((num_samples, num_features))

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


@pytest.fixture
def data_with_zero_std():
    """Provides data where one column has zero standard deviation."""
    return torch.tensor(
        [[1.0, 5.0, -5.0], [3.0, 5.0, -3.0], [5.0, 5.0, -1.0]], dtype=torch.float32
    )


@pytest.fixture
def categorical_features_list():
    """Provides a sample list of categorical feature indices."""
    return [1, 2]


# --- Test Functions ---


def test_diff_znorm_initialization():
    """Test initialization with empty means and stds."""
    step = DifferentiableZNormStep()
    assert isinstance(step.means, torch.Tensor)
    assert step.means.numel() == 0
    assert isinstance(step.stds, torch.Tensor)
    assert step.stds.numel() == 0


def test_diff_znorm_fit(sample_data, categorical_features_list):
    """Test _fit calculates and stores mean/std correctly."""
    step = DifferentiableZNormStep()
    expected_means = torch.mean(sample_data, dim=0, keepdim=True)
    expected_stds = torch.std(sample_data, dim=0, keepdim=True)

    returned_cat_features = step._fit(sample_data, categorical_features_list)

    assert torch.allclose(step.means, expected_means)
    assert torch.allclose(step.stds, expected_stds)
    assert step.means.shape == (1, sample_data.shape[1])
    assert step.stds.shape == (1, sample_data.shape[1])
    assert returned_cat_features == categorical_features_list


def test_diff_znorm_transform(sample_data, categorical_features_list):
    """Test _transform applies Z-norm correctly."""
    step = DifferentiableZNormStep()
    step._fit(sample_data, categorical_features_list)  # Fit first

    expected_output = (sample_data - step.means) / step.stds
    transformed_data = step._transform(sample_data)

    assert isinstance(transformed_data, torch.Tensor)
    assert transformed_data.shape == sample_data.shape
    assert torch.allclose(transformed_data, expected_output)

    # Verify properties of transformed data
    mean_transformed = torch.mean(transformed_data, dim=0)
    std_transformed = torch.std(transformed_data, dim=0)
    assert torch.allclose(
        mean_transformed, torch.zeros(sample_data.shape[1]), atol=1e-6
    )
    assert torch.allclose(std_transformed, torch.ones(sample_data.shape[1]), atol=1e-6)


def test_diff_znorm_fit_transform_integration(sample_data, categorical_features_list):
    """Test fit and transform used together via base class methods."""
    step = DifferentiableZNormStep()
    step.fit(sample_data, categorical_features_list)
    result = step.transform(sample_data)
    transformed_data = result.X
    returned_cat_features = result.categorical_features

    mean_transformed = torch.mean(transformed_data, dim=0)
    std_transformed = torch.std(transformed_data, dim=0)
    assert torch.allclose(
        mean_transformed, torch.zeros(sample_data.shape[1]), atol=1e-6
    )
    assert torch.allclose(std_transformed, torch.ones(sample_data.shape[1]), atol=1e-6)
    assert returned_cat_features == categorical_features_list


def test_diff_znorm_transform_shape_mismatch(sample_data, categorical_features_list):
    """Test transform raises AssertionError on input shape mismatch."""
    step = DifferentiableZNormStep()
    step._fit(sample_data, categorical_features_list)  # Fit with 3 features

    mismatched_data = torch.tensor(
        [[1.0, 10.0], [3.0, 12.0]], dtype=torch.float32
    )  # 2 features

    with pytest.raises(AssertionError):
        step._transform(mismatched_data)


def test_diff_znorm_transform_with_zero_std(
    data_with_zero_std, categorical_features_list
):
    """Test transform behavior with zero std deviation column."""
    step = DifferentiableZNormStep()
    step._fit(data_with_zero_std, categorical_features_list)

    assert torch.isclose(step.stds[0, 1], torch.tensor(0.0))

    transformed_data = step._transform(data_with_zero_std)

    # Expect NaN for division by zero
    assert torch.isnan(transformed_data[:, 1]).all()
    assert not torch.isnan(transformed_data[:, 0]).any()
    assert not torch.isnan(transformed_data[:, 2]).any()


@pytest.mark.parametrize(
    ("append_to_original_setting", "num_features", "expected_output_features"),
    [
        # Test 'auto' mode below the threshold: should append original features
        pytest.param("auto", 10, 20, id="auto_below_threshold_appends"),
        # Test 'auto' mode above the threshold: should NOT append original features
        pytest.param("auto", 600, 600, id="auto_above_threshold_replaces"),
        # Test True: should always append, regardless of threshold
        pytest.param(True, 600, 1200, id="true_always_appends"),
        # Test False: should never append
        pytest.param(False, 10, 10, id="false_never_appends"),
    ],
)
def test_reshape_step_append_original_logic(
    append_to_original_setting, num_features, expected_output_features
):
    """Tests the `append_to_original` logic, including the "auto" mode which
    depends on the APPEND_TO_ORIGINAL_THRESHOLD class constant (500).
    """
    # ARRANGE: Create a dataset with the specified number of features
    num_samples = 100
    rng = np.random.default_rng(42)
    X = rng.random((num_samples, num_features))

    # ARRANGE: Instantiate the step with the specified setting
    preprocessing_step = ReshapeFeatureDistributionsStep(
        transform_name="quantile_norm",
        append_to_original=append_to_original_setting,
        random_state=42,
    )

    # ACT: Run the preprocessing
    Xt, _ = preprocessing_step.fit_transform(X, categorical_features=[])

    # ASSERT: Check if the number of output features matches the expected outcome
    assert Xt.shape[0] == num_samples
    assert Xt.shape[1] == expected_output_features
