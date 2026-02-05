"""Example of fine-tuning a TabPFN classifier using the FinetunedTabPFNClassifier wrapper.

Note: We recommend running the fine-tuning script on a CUDA-enabled GPU with 80Gb of VRAM.
"""

import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# =============================================================================
# Fine-tuning Configuration
# For details and more options see FinetunedTabPFNClassifier
#
# These settings work well for the Higgs dataset.
# For other datasets, you may need to adjust these settings to get good results.
# =============================================================================

# Training hyperparameters
NUM_EPOCHS = 30
LEARNING_RATE = 2e-5

# Ensemble configuration
# number of estimators to use during finetuning
NUM_ESTIMATORS_FINETUNE = 2
# number of estimators to use during trian time validation
NUM_ESTIMATORS_VALIDATION = 2
# number of estimators to use during final inference
NUM_ESTIMATORS_FINAL_INFERENCE = 2

# Reproducibility
RANDOM_STATE = 0


def calculate_roc_auc(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """Calculate ROC AUC with binary vs. multiclass handling."""
    if len(np.unique(y_true)) == 2:
        return roc_auc_score(y_true, y_pred_proba[:, 1])  # pyright: ignore[reportReturnType]
    return roc_auc_score(y_true, y_pred_proba, multi_class="ovr")  # pyright: ignore[reportReturnType]


def main() -> None:
    # We use the "Higgs" dataset (see https://www.openml.org/search?type=data&sort=runs&id=44129&status=active)
    # but only take a random subset of 100k samples for this example.
    data = sklearn.datasets.fetch_openml(data_id=44129, as_frame=True, parser="auto")
    _, X_all, _, y_all = train_test_split(
        data.data,
        data.target,
        test_size=100_000,
        random_state=RANDOM_STATE,
        stratify=data.target,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all, test_size=0.1, random_state=RANDOM_STATE, stratify=y_all
    )

    print(
        f"Loaded {len(X_train):,} samples for training and {len(X_test):,} samples for testing."
    )

    # 2. Initial model evaluation on test set
    base_clf = TabPFNClassifier(
        device=[f"cuda:{i}" for i in range(torch.cuda.device_count())],
        n_estimators=NUM_ESTIMATORS_FINAL_INFERENCE,
        ignore_pretraining_limits=True,
        inference_config={"SUBSAMPLE_SAMPLES": 50_000},
        random_state=RANDOM_STATE,
    )
    base_clf.fit(X_train, y_train)

    base_pred_proba = base_clf.predict_proba(X_test)
    roc_auc = calculate_roc_auc(y_test, base_pred_proba)
    log_loss_score = log_loss(y_test, base_pred_proba)

    print(f"📊 Default TabPFN Test ROC: {roc_auc:.4f}")
    print(f"📊 Default TabPFN Test Log Loss: {log_loss_score:.4f}\n")

    # 3. Initialize and run fine-tuning
    print("--- 2. Initializing and Fitting Model ---\n")

    # Instantiate the wrapper with your desired hyperparameters
    finetuned_clf = FinetunedTabPFNClassifier(
        device="cuda",
        epochs=NUM_EPOCHS,
        learning_rate=LEARNING_RATE,
        n_estimators_finetune=NUM_ESTIMATORS_FINETUNE,
        n_estimators_validation=NUM_ESTIMATORS_VALIDATION,
        n_estimators_final_inference=NUM_ESTIMATORS_FINAL_INFERENCE,
        random_state=RANDOM_STATE,
    )

    # 4. Call .fit() to start the fine-tuning process on the training data
    finetuned_clf.fit(X_train, y_train)
    print("\n")

    # 5. Evaluate the fine-tuned model
    print("--- 3. Evaluating Model on Held-out Test Set ---\n")
    y_pred_proba = finetuned_clf.predict_proba(X_test)

    roc_auc = calculate_roc_auc(y_test, y_pred_proba)
    loss = log_loss(y_test, y_pred_proba)

    print(f"📊 Finetuned TabPFN Test ROC: {roc_auc:.4f}")
    print(f"📊 Finetuned TabPFN Test Log Loss: {loss:.4f}")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Please run the script on a CUDA-enabled GPU."
        )
    main()
