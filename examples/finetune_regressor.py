"""An example for finetuning TabPFN on the California Housing Regression dataset."""

from functools import partial

import numpy as np
import sklearn
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNRegressor
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator

# TODO: refactor common logic: Data loading
# and training loop into BAsic FineTuner class
# for both regression and classification


def eval_test_regression_standard(
    reg: TabPFNRegressor,
    eval_init_args: dict,
    *,
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
):
    reg_eval = clone_model_for_evaluation(reg, eval_init_args, TabPFNRegressor)

    reg_eval.fit(X_train_raw, y_train_raw)
    predictions = reg_eval.predict(X_train_raw)
    print(f"DEBUG Train MSE: {mean_squared_error(y_train_raw, predictions)}")
    print(f"DEBUG Train MAE: {mean_absolute_error(y_train_raw, predictions)}")
    print(f"DEBUG Train R2: {r2_score(y_train_raw, predictions)}")
    try:
        predictions = reg_eval.predict(X_test_raw)
        mse = mean_squared_error(y_test_raw, predictions)
        mae = mean_absolute_error(y_test_raw, predictions)
        r2 = r2_score(y_test_raw, predictions)
    except Exception as e:  # noqa: BLE001  # TODO: Narrow exception type if possible
        print(f"Error during evaluation prediction/metric calculation: {e}")
        mse, mae, r2 = np.nan, np.nan, np.nan  # Return NaNs on error
    return mse, mae, r2


if __name__ == "__main__":
    device = "cpu"
    n_use = 100
    do_epochs = 1
    test_set_size = 0.2

    data_frame_x, data_frame_y = sklearn.datasets.fetch_california_housing(
        return_X_y=True
    )
    splitfn = partial(train_test_split, test_size=test_set_size)
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = splitfn(
        data_frame_x[:n_use], data_frame_y[:n_use]
    )

    print(X_train_raw.shape)
    print(X_test_raw.shape)

    regressor_args = {
        "ignore_pretraining_limits": True,
        "device": device,
        "n_estimators": 2,
        "random_state": 2,  # For reproducibility of internal sampling/preprocessing
        "inference_precision": torch.float32,  # Keep precision consistent
        # memory_saving_mode default is 'auto'
    }

    reg = TabPFNRegressor(
        **regressor_args, differentiable_input=False, fit_mode="batched"
    )

    res_mse, res_mae, res_r2 = eval_test_regression_standard(
        reg,
        regressor_args,
        X_train_raw=X_train_raw,
        y_train_raw=y_train_raw,
        X_test_raw=X_test_raw,
        y_test_raw=y_test_raw,
    )

    print(f"Test MSE: {res_mse:.4f}")
    print(f"Test MAE: {res_mae:.4f}")
    print(f"Test R2: {res_r2:.4f}")

    hyperparams = {
        "optimization_space": "preprocessed"  # "raw_label_space",  "preprocessed"
    }

    datasets_collection = reg.get_preprocessed_datasets(
        X_train_raw, y_train_raw, splitfn, max_data_size=150
    )

    my_dl_train = DataLoader(
        datasets_collection, batch_size=1, collate_fn=meta_dataset_collator
    )
    optim_impl = Adam(reg.model_.parameters(), lr=1e-5)

    loss_batches = []
    mse_batches = []

    # Training Loop
    for epoch in range(do_epochs):
        for data_batch in tqdm(my_dl_train):
            optim_impl.zero_grad()

            (
                X_trains_preprocessed,
                X_tests_preprocessed,
                y_trains_preprocessed,
                y_test_standardized,
                cat_ixs,
                confs,
                normalized_bardist_,
                bardist_,
                batch_x_test_raw,
                batch_y_test_raw,
            ) = data_batch

            reg.normalized_bardist_ = normalized_bardist_[0]

            reg.fit_from_preprocessed(
                X_trains_preprocessed, y_trains_preprocessed, cat_ixs, confs
            )

            averaged_pred_logits, _, _ = reg.forward(
                X_tests_preprocessed
            )  # [BatchSize, N_test, NumBars]

            # TabPFN Regressor standardises the label
            # distribution with a Z-Transform. When
            # optimizing for FineTuning we can choose
            # to optimize the difference in standardised
            # distributions in this standardised (preprocessed)
            # space or the raw label space.

            lossfn = None
            if hyperparams["optimization_space"] == "raw_label_space":
                lossfn = bardist_[0]
                y_test = batch_y_test_raw

            elif hyperparams["optimization_space"] == "preprocessed":
                lossfn = normalized_bardist_[0]
                y_test = y_test_standardized
            else:
                raise ValueError("Need to define optimization space")

            nll_loss_per_sample = lossfn(averaged_pred_logits, y_test.to(device))
            loss = nll_loss_per_sample.mean()

            print(f" Loss in EPOCH {epoch + 1}: {loss}")
            loss.backward()
            optim_impl.step()

        print(f"---- EPOCH {epoch + 1} Evaluation Results (Standard Predict): ----")

        res_mse, res_mae, res_r2 = np.nan, np.nan, np.nan
        res_mse, res_mae, res_r2 = eval_test_regression_standard(
            reg,
            regressor_args,
            X_train_raw=X_train_raw,
            y_train_raw=y_train_raw,
            X_test_raw=X_test_raw,
            y_test_raw=y_test_raw,
        )

        print(f"Test MSE: {res_mse:.4f}")
        print(f"Test MAE: {res_mae:.4f}")
        print(f"Test R2: {res_r2:.4f}")

        # TODO: implement experiment tracking
