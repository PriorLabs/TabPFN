#  Copyright (c) Prior Labs GmbH 2025.
"""Example of using TabPFN for regression.

This example demonstrates how to use TabPFNRegressor on a regression task
using the diabetes dataset from scikit-learn.
"""

from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import numpy as np

from tabpfn import TabPFNRegressor

from tabpfn.config import ModelInterfaceConfig
from tabpfn.preprocessing import PreprocessorConfig

# Load data
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

# Desired upsample size
upsample_size = 1000

# Number of samples in training set
n_samples = X_train.shape[0]

# Randomly sample indices with replacement
upsample_indices = np.random.choice(n_samples, size=upsample_size, replace=True)

# Create upsampled training data
X_train = X_train[upsample_indices]
y_train = y_train[upsample_indices]

print(f"Original training size: {n_samples}")
print(f"Upsampled training size: {y_train.shape[0]}")

remove_highly_correlated = False
preprocessing_configs = [
        PreprocessorConfig(
            "quantile_uni",
            append_original="auto",
            categorical_name="ordinal_very_common_categories_shuffled",
            global_transformer_name="svd",
            remove_highly_correlated = remove_highly_correlated
        ),
        PreprocessorConfig("safepower", categorical_name="onehot", remove_highly_correlated = remove_highly_correlated),
    ]

config_dict = {
    "PREPROCESS_TRANSFORMS": preprocessing_configs
               }
inference_config= ModelInterfaceConfig.from_user_input(inference_config=config_dict)

# Initialize a regressor
# reg = TabPFNRegressor(n_estimators=8, device="cpu",ignore_pretraining_limits=False, random_state= 42)  # Use CPU
reg = TabPFNRegressor(n_estimators=8, ignore_pretraining_limits=False, random_state= 42, inference_config=inference_config)

reg.fit(X_train, y_train)

# Predict a point estimate (using the mean)
predictions = reg.predict(X_test)
print("Mean Squared Error (MSE):", mean_squared_error(y_test, predictions))
print("Mean Absolute Error (MAE):", mean_absolute_error(y_test, predictions))
print("R-squared (R^2):", r2_score(y_test, predictions))

# Predict quantiles
quantiles = [0.25, 0.5, 0.75]
quantile_predictions = reg.predict(
    X_test,
    output_type="quantiles",
    quantiles=quantiles,
)
for q, q_pred in zip(quantiles, quantile_predictions):
    print(f"Quantile {q} MAE:", mean_absolute_error(y_test, q_pred))
# Predict with mode
mode_predictions = reg.predict(X_test, output_type="mode")
print("Mode MAE:", mean_absolute_error(y_test, mode_predictions))
