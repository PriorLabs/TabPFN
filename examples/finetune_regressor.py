"""Example of fine-tuning a TabPFN regressor using the FinetunedTabPFNRegressor wrapper.

Note: We recommend running the fine-tuning scripts on a CUDA-enabled GPU, as full
support for the Apple Silicon (MPS) backend is still under development.
"""

import warnings

import sklearn.datasets
import torch
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor
from tabpfn.finetuning.finetuned_regressor import FinetunedTabPFNRegressor

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.api_core\._python_version_support",
)

# =============================================================================
# Fine-tuning Configuration
# For details and more options see FinetunedTabPFNRegressor
# =============================================================================

# Training hyperparameters
NUM_EPOCHS = 30
LEARNING_RATE = 1e-5

# Data sampling configuration (dataset dependent)
# the ratio of the total dataset to be used for validation during training
VALIDATION_SPLIT_RATIO = 0.1
# total context split into train/test
NUM_FINETUNE_CTX_PLUS_QUERY_SAMPLES = 10_000
FINETUNE_CTX_QUERY_SPLIT_RATIO = 0.5
NUM_INFERENCE_SUBSAMPLE_SAMPLES = 20_000
# to reduce memory usage during training we can use activation checkpointing,
# may not be necessary for small datasets
USE_ACTIVATION_CHECKPOINTING = True
WEIGHT_DECAY = 0.01

# Ensemble configuration
# number of estimators to use during finetuning
NUM_ESTIMATORS_FINETUNE = 8
# number of estimators to use during train time validation
NUM_ESTIMATORS_VALIDATION = 8
# number of estimators to use during final inference
NUM_ESTIMATORS_FINAL_INFERENCE = 8

# Reproducibility
RANDOM_STATE = 0


def main() -> None:
    # We use the California Housing dataset for this example.
    data = sklearn.datasets.fetch_california_housing(as_frame=True)
    X_all = data.data
    y_all = data.target

    # Take a subset for faster demonstration
    _, X_all, _, y_all = train_test_split(
        X_all,
        y_all,
        test_size=20_000,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.1, random_state=RANDOM_STATE
    )

    print(
        f"Loaded {len(X_train):,} samples for training and "
        f"{len(X_test):,} samples for testing."
    )

    # 2. Initial model evaluation on test set
    inference_config = {
        "SUBSAMPLE_SAMPLES": NUM_INFERENCE_SUBSAMPLE_SAMPLES,
    }
    base_reg = TabPFNRegressor(
        device=[f"cuda:{i}" for i in range(torch.cuda.device_count())],
        n_estimators=NUM_ESTIMATORS_FINAL_INFERENCE,
        ignore_pretraining_limits=True,
        inference_config=inference_config,
    )
    base_reg.fit(X_train, y_train)

    base_pred = base_reg.predict(X_test)
    mse = mean_squared_error(y_test, base_pred)
    r2 = r2_score(y_test, base_pred)

    print(f"📊 Default TabPFN Test MSE: {mse:.4f}")
    print(f"📊 Default TabPFN Test R²: {r2:.4f}\n")

    # 3. Initialize and run fine-tuning
    print("--- 2. Initializing and Fitting Model ---\n")

    # Instantiate the wrapper with your desired hyperparameters
    finetuned_reg = FinetunedTabPFNRegressor(
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        validation_split_ratio=VALIDATION_SPLIT_RATIO,
        n_finetune_ctx_plus_query_samples=NUM_FINETUNE_CTX_PLUS_QUERY_SAMPLES,
        finetune_ctx_query_split_ratio=FINETUNE_CTX_QUERY_SPLIT_RATIO,
        n_inference_subsample_samples=NUM_INFERENCE_SUBSAMPLE_SAMPLES,
        random_state=RANDOM_STATE,
        n_estimators_finetune=NUM_ESTIMATORS_FINETUNE,
        n_estimators_validation=NUM_ESTIMATORS_VALIDATION,
        n_estimators_final_inference=NUM_ESTIMATORS_FINAL_INFERENCE,
        weight_decay=WEIGHT_DECAY,
        use_activation_checkpointing=USE_ACTIVATION_CHECKPOINTING,
    )

    # 4. Call .fit() to start the fine-tuning process on the training data
    finetuned_reg.fit(X_train.values, y_train.values)
    print("\n")

    # 5. Evaluate the fine-tuned model
    print("--- 3. Evaluating Model on Held-out Test Set ---\n")
    y_pred = finetuned_reg.predict(X_test.values)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"📊 Finetuned TabPFN Test MSE: {mse:.4f}")
    print(f"📊 Finetuned TabPFN Test R²: {r2:.4f}")


if __name__ == "__main__":
    main()
