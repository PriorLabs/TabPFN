"""Example of fine-tuning a TabPFN classifier using the FinetunedTabPFNClassifier wrapper.

Note: We recommend running the fine-tuning scripts on a CUDA-enabled GPU, as full
support for the Apple Silicon (MPS) backend is still under development.
"""

import warnings

import numpy as np
import sklearn.datasets
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier
from tabpfn.finetuning.finetuned_classifier import (
    FinetunedTabPFNClassifier,
)

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module=r"google\.api_core\._python_version_support",
)

# =============================================================================
# Fine-tuning Configuration
# For details and more options see FinetunedTabPFNClassifier
# =============================================================================

# Training hyperparameters
NUM_EPOCHS = 30
LEARNING_RATE = 3e-5
EARLY_STOPPING_PATIENCE = 5
WEIGHT_DECAY = 0.001
USE_LR_SCHEDULER = True
LR_WARMUP_ONLY = False

# Data sampling configuration
VALIDATION_SPLIT_RATIO = 0.1
NUM_FINETUNE_CTX_PLUS_QUERY_SAMPLES = 20_000
FINETUNE_CTX_QUERY_SPLIT_RATIO = 0.02
NUM_INFERENCE_SUBSAMPLE_SAMPLES = 50_000

# Ensemble configuration
NUM_ESTIMATORS_FINETUNE = 2
NUM_ESTIMATORS_VALIDATION = 4
NUM_ESTIMATORS_FINAL_INFERENCE = 8

# Reproducibility
RANDOM_STATE = 42


def calculate_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate ROC AUC with binary vs. multiclass handling."""
    if len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, y_pred_proba[:, 1])  # pyright: ignore[reportReturnType]
    return roc_auc_score(y_true, y_pred_proba, multi_class="ovr")  # pyright: ignore[reportReturnType]


def main() -> None:
    # Get test dataset
    X_all, y_all = sklearn.datasets.fetch_covtype(return_X_y=True, shuffle=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.1, random_state=RANDOM_STATE, stratify=y_all
    )

    # 2. Initial model evaluation on test set
    inference_config = {
        "SUBSAMPLE_SAMPLES": NUM_INFERENCE_SUBSAMPLE_SAMPLES,
    }
    base_clf = TabPFNClassifier(
        device=[f"cuda:{i}" for i in range(torch.cuda.device_count())],
        n_estimators=NUM_ESTIMATORS_FINAL_INFERENCE,
        ignore_pretraining_limits=True,
        inference_config=inference_config,
    )
    base_clf.fit(X_train, y_train)

    base_pred_proba = base_clf.predict_proba(X_test)
    roc_auc = calculate_roc_auc(
        y_test,
        base_pred_proba,
    )  # pyright: ignore[reportReturnType, reportArgumentType]
    log_loss_score = log_loss(y_test, base_pred_proba)

    print(f"📊 Initial Test ROC: {roc_auc:.4f}")
    print(f"📊 Initial Test Log Loss: {log_loss_score:.4f}\n")

    # 3. Initialize and run fine-tuning
    print("--- 2. Initializing and Fitting Model ---\n")

    # Instantiate the wrapper with your desired hyperparameters
    finetuned_clf = FinetunedTabPFNClassifier(
        device="cuda" if torch.cuda.is_available() else "cpu",
        epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        validation_split_ratio=VALIDATION_SPLIT_RATIO,
        n_finetune_ctx_plus_query_samples=NUM_FINETUNE_CTX_PLUS_QUERY_SAMPLES,
        finetune_ctx_query_split_ratio=FINETUNE_CTX_QUERY_SPLIT_RATIO,
        n_inference_subsample_samples=NUM_INFERENCE_SUBSAMPLE_SAMPLES,
        random_state=RANDOM_STATE,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        weight_decay=WEIGHT_DECAY,
        use_lr_scheduler=USE_LR_SCHEDULER,
        lr_warmup_only=LR_WARMUP_ONLY,
        n_estimators_finetune=NUM_ESTIMATORS_FINETUNE,
        n_estimators_validation=NUM_ESTIMATORS_VALIDATION,
        n_estimators_final_inference=NUM_ESTIMATORS_FINAL_INFERENCE,
    )

    # 4. Call .fit() to start the fine-tuning process on the training data
    finetuned_clf.fit(X_train, y_train)  # pyright: ignore[reportArgumentType]
    print("\n")

    # 5. Evaluate the fine-tuned model
    print("--- 3. Evaluating Model on Held-out Test Set ---\n")
    y_pred_proba = finetuned_clf.predict_proba(
        X_test,
    )  # pyright: ignore[reportArgumentType]

    roc_auc = calculate_roc_auc(
        y_test,
        y_pred_proba,
    )  # pyright: ignore[reportArgumentType]
    loss = log_loss(y_test, y_pred_proba)

    print(f"📊 Final Test ROC: {roc_auc:.4f}")
    print(f"📊 Final Test Log Loss: {loss:.4f}")


if __name__ == "__main__":
    main()
