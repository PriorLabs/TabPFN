from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.preprocessing import LabelEncoder
from torch.torch_version import TorchVersion

from tabpfn import TabPFNClassifier
from tabpfn.constants import NA_PLACEHOLDER
from tabpfn.inference_config import InferenceConfig
from tabpfn.preprocessing.steps.preprocessing_helpers import get_ordinal_encoder
from tabpfn.utils import (
    balance_probas_by_class_counts,
    fix_dtypes,
    infer_categorical_features,
    infer_devices,
    process_text_na_dataframe,
    validate_Xy_fit,
)


def test_infer_categorical_with_str_and_nan_provided_included():
    X = np.array([[np.nan, "NA"]], dtype=object).reshape(-1, 1)
    out = infer_categorical_features(
        X,
        provided=[0],
        min_samples_for_inference=0,
        max_unique_for_category=2,
        min_unique_for_numerical=5,
    )
    assert out == [0]


def test_infer_categorical_with_str_and_nan_multiple_rows_provided_included():
    X = np.array([[np.nan], ["NA"], ["NA"]], dtype=object)
    out = infer_categorical_features(
        X,
        provided=[0],
        min_samples_for_inference=0,
        max_unique_for_category=2,
        min_unique_for_numerical=5,
    )
    assert out == [0]


def test_infer_categorical_auto_inference_blocked_when_not_enough_samples():
    X = np.array([[1.0], [1.0], [np.nan]])
    out = infer_categorical_features(
        X,
        provided=None,
        min_samples_for_inference=3,
        max_unique_for_category=2,
        min_unique_for_numerical=4,
    )
    assert out == []


def test_infer_categorical_auto_inference_enabled_with_enough_samples():
    X = np.array([[1.0, 0.0], [1.0, 1.0], [2.0, 2.0], [2.0, 3.0], [np.nan, 9.0]])
    out = infer_categorical_features(
        X,
        provided=None,
        min_samples_for_inference=3,
        max_unique_for_category=3,
        min_unique_for_numerical=4,
    )
    assert out == [0]


def test_infer_categorical_provided_column_excluded_if_exceeds_max_unique():
    X = np.array([[0], [1], [2], [3], [np.nan]], dtype=float)
    out = infer_categorical_features(
        X,
        provided=[0],
        min_samples_for_inference=0,
        max_unique_for_category=3,
        min_unique_for_numerical=2,
    )
    assert out == []


def test_infer_categorical_with_dict_raises_error():
    X = np.array([[{"a": 1}], [{"b": 2}]], dtype=object)
    with pytest.raises(TypeError):
        infer_categorical_features(
            X,
            provided=None,
            min_samples_for_inference=0,
            max_unique_for_category=2,
            min_unique_for_numerical=2,
        )


def test__infer_devices__auto__cuda_and_mps_not_available__selects_cpu(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "")
    mocker.patch("torch.cuda").is_available.return_value = False
    mocker.patch("torch.backends.mps").is_available.return_value = False
    assert infer_devices(devices="auto") == (torch.device("cpu"),)


def test__infer_devices__auto__single_cuda_gpu_available__selects_it(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "")
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mocker.patch("torch.backends.mps").is_available.return_value = True
    assert infer_devices(devices="auto") == (torch.device("cuda:0"),)


def test__infer_devices__auto__multiple_cuda_gpus_available__selects_first(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "")
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 3
    mocker.patch("torch.backends.mps").is_available.return_value = True

    assert infer_devices(devices="auto") == (torch.device("cuda:0"),)


def test__infer_devices__auto__cuda_and_mps_available_but_excluded__selects_cpu(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TABPFN_EXCLUDE_DEVICES", "mps,cuda")
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 1
    mocker.patch("torch.backends.mps").is_available.return_value = True
    assert infer_devices(devices="auto") == (torch.device("cpu"),)


def test__infer_devices__auto__mps_available_but_torch_too_old__selects_cpu(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch, "__version__", TorchVersion("2.4.0"))
    mocker.patch("torch.cuda").is_available.return_value = False
    mocker.patch("torch.backends.mps").is_available.return_value = True
    assert infer_devices(devices="auto") == (torch.device("cpu"),)


def test__infer_devices__device_specified__selects_it(
    mocker: MagicMock,
) -> None:
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 2
    mocker.patch("torch.backends.mps").is_available.return_value = True

    assert infer_devices(devices="cuda:0") == (torch.device("cuda:0"),)


def test__infer_devices__multiple_devices_specified___selects_them(
    mocker: MagicMock,
) -> None:
    mock_cuda = mocker.patch("torch.cuda")
    mock_cuda.is_available.return_value = True
    mock_cuda.device_count.return_value = 3
    mocker.patch("torch.backends.mps").is_available.return_value = False

    inferred = set(infer_devices(devices=["cuda:0", "cuda:1", "cuda:4"]))
    expected = {torch.device("cuda:0"), torch.device("cuda:1"), torch.device("cuda:4")}
    assert inferred == expected


def test__infer_devices__device_specified_twice__raises() -> None:
    with pytest.raises(
        ValueError,
        match="The list of devices for inference cannot contain the same device more ",
    ):
        infer_devices(devices=["cpu", "cpu"])


def test__infer_devices__mps_specified_but_torch_too_old__raises(
    mocker: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(torch, "__version__", TorchVersion("2.4.0"))
    mocker.patch("torch.backends.mps").is_available.return_value = True
    with pytest.raises(ValueError, match="The MPS device was selected"):
        infer_devices(devices="mps")


# --- Test Data for the "test_process_text_na_dataframe" test ---
test_cases = [
    {
        # Mixed dtypes & None / pd.Na
        "df": pd.DataFrame(
            {
                "ratio": [0.4, 0.5, 0.6],
                "risk": ["High", None, "Low"],
                "height": ["Low", "Low", "Low"],
                "amount": [10.2, 20.4, 20.5],
                "type": ["guest", "member", pd.NA],
            }
        ),
        "categorical_indices": [1, 2, 4],
        "ground_truth": np.array(
            [
                [0.4, 0, 0, 10.2, 0],
                [0.5, np.nan, 0, 20.4, 1],
                [0.6, 1, 0, 20.5, np.nan],
            ]
        ),
    },
    {
        # Mixed dtypes & no missing values
        "df": pd.DataFrame(
            {
                "ratio": [0.4, 0.5, 0.6],
                "risk": ["High", "Medium", "Low"],
                "height": ["Low", "Low", "High"],
                "amount": [10.2, 20.4, np.nan],
                "type": ["guest", "member", "vip"],
            }
        ),
        "categorical_indices": [1, 2, 4],
        "ground_truth": np.array(
            [
                [0.4, 0, 1, 10.2, 0],
                [0.5, 2, 1, 20.4, 1],
                [0.6, 1, 0, np.nan, 2],
            ]
        ),
    },
    {
        # All numerical no nan
        "df": pd.DataFrame(
            {
                "ratio": [0.1, 0.2, 0.3],
                "amount": [5.0, 15.5, 25.0],
                "score": [1.0, 2.5, 3.5],
            }
        ),
        "categorical_indices": [],
        "ground_truth": np.array(
            [
                [0.1, 5.0, 1.0],
                [0.2, 15.5, 2.5],
                [0.3, 25.0, 3.5],
            ]
        ),
    },
    {
        # all categorical no nan
        "df": pd.DataFrame(
            {
                "risk": ["High", "High", "High"],
                "height": ["Low", "Low", "Low"],
                "type": ["guest", "guest", "guest"],
            }
        ),
        "categorical_indices": [0, 1, 2],
        "ground_truth": np.array(
            [
                [0, 0, 0],
                [0, 0, 0],
                [0, 0, 0],
            ]
        ),
    },
]


# --- Fixture for the "test_process_text_na_dataframe" test ---
# prepare the DataFrame
@pytest.fixture(params=test_cases)
def prepared_tabpfn_data(request):
    temp_df = request.param["df"].copy()
    categorical_idx = request.param["categorical_indices"]
    # Dummy target, as tests do not need a target
    y = np.array([0, 1, 0])

    cls = TabPFNClassifier()

    X, y, feature_names_in, n_features_in = validate_Xy_fit(
        temp_df,
        y,
        estimator=cls,
        ensure_y_numeric=False,
        max_num_samples=InferenceConfig.MAX_NUMBER_OF_SAMPLES,
        max_num_features=InferenceConfig.MAX_NUMBER_OF_FEATURES,
        ignore_pretraining_limits=False,
    )

    if feature_names_in is not None:
        cls.feature_names_in_ = feature_names_in
    cls.n_features_in_ = n_features_in

    if not cls.differentiable_input:
        _, counts = np.unique(y, return_counts=True)
        cls.class_counts_ = counts
        cls.label_encoder_ = LabelEncoder()
        y = cls.label_encoder_.fit_transform(y)
        cls.classes_ = cls.label_encoder_.classes_
        cls.n_classes_ = len(cls.classes_)
    else:
        cls.label_encoder_ = None
        if not hasattr(cls, "n_classes_"):
            cls.n_classes_ = int(torch.max(torch.tensor(y)).item()) + 1
        cls.classes_ = torch.arange(cls.n_classes_)

    cls.inferred_categorical_indices_ = infer_categorical_features(
        X=X,
        provided=categorical_idx,
        min_samples_for_inference=InferenceConfig.MIN_NUMBER_SAMPLES_FOR_CATEGORICAL_INFERENCE,
        max_unique_for_category=InferenceConfig.MAX_UNIQUE_FOR_CATEGORICAL_FEATURES,
        min_unique_for_numerical=InferenceConfig.MIN_UNIQUE_FOR_NUMERICAL_FEATURES,
    )
    return (
        fix_dtypes(X, cat_indices=cls.inferred_categorical_indices_),
        cls.inferred_categorical_indices_,
        request.param["ground_truth"],
    )


# --- Actual test ---
# This is a test for the OrderPreservingColumnTransformer, which is not used currently
# But might be used in the future, therefore I'll leave it in.
@pytest.mark.skip
def test_process_text_na_dataframe(prepared_tabpfn_data):
    X, categorical_idx, ground_truth = prepared_tabpfn_data  # use the fixture

    ord_encoder = get_ordinal_encoder()
    X_out = process_text_na_dataframe(
        X,
        placeholder=NA_PLACEHOLDER,
        ord_encoder=ord_encoder,
        fit_encoder=True,
    )

    assert X_out.shape[0] == ground_truth.shape[0]
    assert X_out.shape[1] == ground_truth.shape[1]

    for col_name in X.columns:
        # col_name should already be a numeric index but using get_loc for safety
        col_idx = X.columns.get_loc(col_name)
        original_col = X[col_name].to_numpy()
        output_col = X_out[:, col_idx]
        gt_col = ground_truth[:, col_idx]
        if col_idx not in categorical_idx:
            # For numeric columns, values should be preserved (within float tolerance).
            # NaNs should also be in the same positions.
            np.testing.assert_allclose(
                output_col,
                original_col,
                equal_nan=True,
                rtol=1e-5,
            )
        else:
            # OrdinalEncoder does not guarante that element order is preserved:

            # First, check if np.nan are correctly positioned
            # ! use np.isnan on outputcol -> must be numerical
            # ! use pd.isna on original col -> can be any type
            np.testing.assert_array_equal(np.isnan(output_col), pd.isna(original_col))
            # Second, check if there are as many unique non-nan values, as expected
            # e.g.: ["high", "mid", "low"] -> [0,2,1] or [2,1,0],...
            assert len(np.unique(output_col[~pd.isna(output_col)])) == len(
                np.unique(gt_col[~pd.isna(gt_col)])
            )


def test_balance_probas_by_class_counts():
    """Test balancing probabilities by class counts."""
    probas = torch.tensor([[0.2, 0.8], [0.6, 0.4], [0.5, 0.5]])
    class_counts = np.array([1, 2])

    balanced = balance_probas_by_class_counts(probas, class_counts)

    # Check that each row sums to one
    sums = balanced.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(3), rtol=1e-5, atol=1e-5)

    expected_balanced = torch.tensor([[1 / 3, 2 / 3], [0.75, 0.25], [2 / 3, 1 / 3]])
    assert torch.allclose(balanced, expected_balanced, rtol=1e-4, atol=1e-4)
