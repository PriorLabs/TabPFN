"""Provides a detailed example of fine-tuning a TabPFNRegressor model.

This script demonstrates the complete workflow, including data loading and preparation
for the Bike Sharing Demand dataset, model configuration, the fine-tuning loop,
and performance evaluation for a regression task.

Note: We recommend running the fine-tuning scripts on a CUDA-enabled GPU, as full
support for the Apple Silicon (MPS) backend is still under development.
"""

from functools import partial

import numpy as np
import sklearn.datasets
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator


def rps_loss(
    outputs: torch.Tensor,
    targets: torch.Tensor,
    bucket_widths: torch.Tensor,
) -> torch.Tensor:
    """
    Calculates the Ranked Probability Score (RPS) loss for binned regression with variable bin widths.

    RPS measures the difference between predicted cumulative probabilities
    and the true cumulative probabilities for ordered categorical outcomes.
    Lower RPS indicates better probabilistic forecasts.

    The formula for RPS for a single prediction and true value is:
    $$ \text{RPS} = \sum_{k=1}^{K} w_k (\hat{F}(k) - F(k))^2 $$
    where $K$ is the number of bins, $w_k$ is the width of bin $k$, $\hat{F}(k)$ is the
    predicted cumulative probability up to bin $k$, and $F(k)$ is the observed (true)
    cumulative probability up to bin $k$.

    The function first converts the true target value into a one-hot encoded vector,
    effectively creating an empirical probability mass function (PMF) where the true bin
    has a probability of 1 and all others are 0. This PMF is then converted into a
    step-like empirical cumulative distribution function (CDF).

    For variable bin widths (e.g., quantile-based bins), bucket_widths should be provided
    to properly weight the squared CDF differences.

    Args:
        outputs (torch.Tensor): Predicted probabilities for each bin. Shape (batch_size, num_bins).
        targets (torch.Tensor): True bin labels (integer indices). Shape (batch_size,).
        bucket_widths (torch.Tensor): Width of each bin. Shape (num_bins,).

    Returns:
        torch.Tensor: The mean RPS loss for the batch.
    """
    # Ensure targets are long integers for scatter operation
    targets = targets.long()
    
    pred_cdf = torch.cumsum(outputs, dim=1)
    target_one_hot = torch.zeros_like(outputs).scatter_(1, targets.unsqueeze(1), 1.)
    target_cdf = torch.cumsum(target_one_hot, dim=1)
    cdf_diff = pred_cdf - target_cdf

    # Weight by bucket widths for variable bin sizes
    bucket_widths = bucket_widths.to(outputs.device)
    weighted_squared_diff = cdf_diff**2 * bucket_widths.unsqueeze(0)
    rps = torch.mean(torch.sum(weighted_squared_diff, dim=1))

    return rps


def prepare_data(config: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Loads, subsets, and splits the California Housing dataset."""
    print("--- 1. Data Preparation ---")
    # Fetch Ames housing data from OpenML
    bike_sharing = sklearn.datasets.fetch_openml(
        name="Bike_Sharing_Demand", version=2, as_frame=True, parser="auto"
    )

    # Separate features (X) and target (y)
    X_df = bike_sharing.data
    y_df = bike_sharing.target

    # Select only numeric features for simplicity
    X_numeric = X_df.select_dtypes(include=np.number)

    X_all, y_all = X_numeric.values, y_df.values

    rng = np.random.default_rng(config["random_seed"])
    num_samples_to_use = min(config["num_samples_to_use"], len(y_all))
    indices = rng.choice(np.arange(len(y_all)), size=num_samples_to_use, replace=False)
    X, y = X_all[indices], y_all[indices]

    splitter = partial(
        train_test_split,
        test_size=config["valid_set_ratio"],
        random_state=config["random_seed"],
    )
    X_train, X_test, y_train, y_test = splitter(X, y)

    print(
        f"Loaded and split data: {X_train.shape[0]} train, {X_test.shape[0]} test samples."
    )
    print("---------------------------\n")
    return X_train, X_test, y_train, y_test


def setup_regressor(config: dict) -> tuple[TabPFNRegressor, dict]:
    """Initializes the TabPFN regressor and its configuration."""
    print("--- 2. Model Setup ---")
    regressor_config = {
        "ignore_pretraining_limits": True,
        "device": config["device"],
        "n_estimators": 2,
        "random_state": config["random_seed"],
        "inference_precision": torch.float32,
    }
    regressor = TabPFNRegressor(
        **regressor_config, fit_mode="batched", differentiable_input=False
    )

    print(f"Using device: {config['device']}")
    print("----------------------\n")
    return regressor, regressor_config


def evaluate_regressor(
    regressor: TabPFNRegressor,
    eval_config: dict,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
) -> tuple[float, float, float]:
    """Evaluates the regressor's performance on the test set."""
    eval_regressor = clone_model_for_evaluation(regressor, eval_config, TabPFNRegressor)
    eval_regressor.fit(X_train, y_train)

    try:
        predictions = eval_regressor.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
    except Exception as e:
        print(f"An error occurred during evaluation: {e}")
        mse, mae, r2 = np.nan, np.nan, np.nan

    return mse, mae, r2


def main() -> None:
    """Main function to configure and run the finetuning workflow."""
    # --- Master Configuration ---
    # This improved structure separates general settings from finetuning hyperparameters.
    config = {
        # Sets the computation device ('cuda' for GPU if available, otherwise 'cpu').
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # The total number of samples to draw from the full dataset. This is useful for
        # managing memory and computation time, especially with large datasets.
        # For very large datasets the entire dataset is preprocessed and then
        # fit in memory, potentially leading to OOM errors.
        "num_samples_to_use": 100_000,
        # A seed for random number generators to ensure that data shuffling, splitting,
        # and model initializations are reproducible.
        "random_seed": 42,
        # The proportion of the dataset to allocate to the valid set for final evaluation.
        "valid_set_ratio": 0.3,
        # During evaluation, this is the number of samples from the training set given to the
        # model as context before it makes predictions on the test set.
        "n_inference_context_samples": 10000,
    }
    config["finetuning"] = {
        # The total number of passes through the entire fine-tuning dataset.
        "epochs": 10,
        # A small learning rate is crucial for fine-tuning to avoid catastrophic forgetting.
        "learning_rate": 1.5e-6,
        # Meta Batch size for finetuning, i.e. how many datasets per batch. Must be 1 currently.
        "meta_batch_size": 1,
        # The number of samples within each training data split. It's capped by
        # n_inference_context_samples to align with the evaluation setup.
        "batch_size": int(
            min(
                config["n_inference_context_samples"],
                config["num_samples_to_use"] * (1 - config["valid_set_ratio"]),
            )
        ),
    }

    # --- Setup Data, Model, and Dataloader ---
    X_train, X_test, y_train, y_test = prepare_data(config)
    regressor, regressor_config = setup_regressor(config)

    splitter = partial(train_test_split, test_size=config["valid_set_ratio"])
    # Note: `max_data_size` corresponds to the finetuning `batch_size` in the config
    training_datasets = regressor.get_preprocessed_datasets(
        X_train, y_train, splitter, max_data_size=config["finetuning"]["batch_size"]
    )
    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=config["finetuning"]["meta_batch_size"],
        collate_fn=meta_dataset_collator,
    )

    # Optimizer must be created AFTER get_preprocessed_datasets, which initializes the model
    optimizer = Adam(regressor.model_.parameters(), lr=config["finetuning"]["learning_rate"])
    print(
        f"--- Optimizer Initialized: Adam, LR: {config['finetuning']['learning_rate']} ---\n"
    )

    # Create evaluation config, linking it to the master config
    eval_config = {
        **regressor_config,
        "inference_config": {
            "SUBSAMPLE_SAMPLES": config["n_inference_context_samples"]
        },
    }

    # --- Finetuning and Evaluation Loop ---
    print("--- 3. Starting Finetuning & Evaluation ---")
    for epoch in range(config["finetuning"]["epochs"] + 1):
        if epoch > 0:
            # Create a tqdm progress bar to iterate over the dataloader
            progress_bar = tqdm(finetuning_dataloader, desc=f"Finetuning Epoch {epoch}")
            for data_batch in progress_bar:
                optimizer.zero_grad()
                (
                    X_trains_preprocessed,
                    X_tests_preprocessed,
                    y_trains_znorm,
                    y_test_znorm,
                    cat_ixs,
                    confs,
                    raw_space_bardist_,
                    znorm_space_bardist_,
                    _,
                    _y_test_raw,
                ) = data_batch

                regressor.raw_space_bardist_ = raw_space_bardist_[0]
                regressor.bardist_ = znorm_space_bardist_[0]
                regressor.fit_from_preprocessed(
                    X_trains_preprocessed, y_trains_znorm, cat_ixs, confs
                )
                logits, _, _ = regressor.forward(X_tests_preprocessed)

                # Get bucket widths from the BarDistribution for RPS loss
                bucket_widths = znorm_space_bardist_[0].bucket_widths
                
                # Logits have shape (seq_len, batch_size, num_bars), take last sequence position
                logits = logits[-1]  # Now shape: (batch_size, num_bars)
                
                # Convert logits to probabilities for RPS loss
                probabilities = torch.softmax(logits, dim=-1)
                
                # Map continuous targets to bin indices  
                # y_test_znorm has shape (seq_len, batch_size), take last position
                y_target = y_test_znorm[-1] if y_test_znorm.dim() > 1 else y_test_znorm
                y_target_bins = znorm_space_bardist_[0].map_to_bucket_idx(
                    y_target.to(config["device"])
                ).clamp(0, znorm_space_bardist_[0].num_bars - 1)
                
                # Use RPS loss instead of default BarDistribution loss
                loss = rps_loss(probabilities, y_target_bins, bucket_widths)
                loss.backward()
                optimizer.step()

                # Set the postfix of the progress bar to show the current loss
                progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # Evaluation Step (runs before finetuning and after each epoch)
        mse, mae, r2 = evaluate_regressor(
            regressor, eval_config, X_train, y_train, X_test, y_test
        )

        status = "Initial" if epoch == 0 else f"Epoch {epoch}"
        print(
            f"📊 {status} Evaluation | Test MSE: {mse:.4f}, Test MAE: {mae:.4f}, Test R2: {r2:.4f}\n"
        )

    print("--- ✅ Finetuning Finished ---")


if __name__ == "__main__":
    main()
