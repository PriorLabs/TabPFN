"""An example for finetuning TabPFN on the Covertype dataset.

This code can be restructured into an Sklearn compatible classifier:
FinetunedTabPFNClassifier(sklearn.base.ClassifierMixin, sklearn.base.RegressorMixin)
def __init__(self, base_estimator: TabPFNClassifier,
 training_datasets: list[pd.DataFrame],
  evaluation_datasets: list[pd.DataFrame],
  epochs: int,

  ):
    self.base_estimator = base_estimator
    self.training_datasets = training_datasets
    self.evaluation_datasets = evaluation_datasets
    self.epochs = epochs

def fit(self):
    # below training code

def predict(self, X):
    self.base_estimator.predict(X)

def predict_proba(self, X):
    self.base_estimator.predict_proba(X)

"""

from functools import partial

import numpy as np
import sklearn
import torch
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNClassifier

# TODO: fix this import for tests
from tabpfn.finetune_utils import clone_model_for_evaluation
from tabpfn.utils import meta_dataset_collator


def eval_test(
    clf: TabPFNClassifier,
    classifier_args: dict,
    *,
    X_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    X_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
) -> tuple[float, float]:
    clf_eval = clone_model_for_evaluation(clf, classifier_args, TabPFNClassifier)
    clf_eval.fit(X_train_raw, y_train_raw)

    try:
        predictions_labels = clf_eval.predict(X_test_raw)
        predictions_proba = clf_eval.predict_proba(X_test_raw)
        accuracy = accuracy_score(y_test_raw, predictions_labels)
        ll = log_loss(y_test_raw, predictions_proba)

    except Exception as e:  # noqa: BLE001  # TODO: Narrow exception type if possible
        print(f"Error during evaluation prediction/metric calculation: {e}")
        accuracy, ll = np.nan, np.nan

    return accuracy, ll


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # n_use = 200_000
    n_use = 1000
    do_epochs = 3
    random_seed = 42
    test_set_size = 0.3

    # Load Covertype Dataset (7-way classification)
    data_frame_x, data_frame_y = sklearn.datasets.fetch_covtype(
        return_X_y=True, shuffle=True
    )
    # Use numpy Generator for reproducibility
    rng = np.random.default_rng(random_seed)
    splitfn = partial(train_test_split, test_size=test_set_size)
    indices = np.arange(len(data_frame_y))
    subset_indices = rng.choice(indices, size=n_use, replace=False)
    X_subset = data_frame_x[subset_indices]
    y_subset = data_frame_y[subset_indices]

    X_train, X_test, y_train, y_test = splitfn(
        data_frame_x[:n_use],
        data_frame_y[:n_use],
        test_size=test_set_size,
        stratify=y_subset,
        random_state=random_seed,
    )

    classifier_args = {
        "ignore_pretraining_limits": True,
        "device": device,
        "n_estimators": 2,
        "random_state": 2,
        "inference_precision": torch.float32,
    }
    clf = TabPFNClassifier(
        **classifier_args, fit_mode="batched", differentiable_input=False
    )

    datasets_list = clf.get_preprocessed_datasets(X_train, y_train, splitfn, 1000)
    datasets_list_test = clf.get_preprocessed_datasets(X_test, y_test, splitfn, 1000)
    my_dl_train = DataLoader(
        datasets_list, batch_size=1, collate_fn=meta_dataset_collator
    )

    optim_impl = Adam(clf.model_.parameters(), lr=1e-4)
    lossfn = torch.nn.NLLLoss()
    loss_batches: list[float] = []
    acc_batches: list[float] = []

    res_acc, ll = eval_test(
        clf,
        classifier_args,
        X_train_raw=X_train,
        y_train_raw=y_train,
        X_test_raw=X_test,
        y_test_raw=y_test,
    )
    print("Initial accuracy:", res_acc)
    print("Initial Test Log Loss:", ll)

    # Training Loop
    for epoch in range(do_epochs):
        for data_batch in tqdm(my_dl_train):
            optim_impl.zero_grad()
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            clf.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = clf.forward(X_tests)
            loss = lossfn(torch.log(preds), y_tests.to(device))
            loss.backward()
            optim_impl.step()

        res_acc, ll = eval_test(
            clf,
            classifier_args,
            X_train_raw=X_train,
            y_train_raw=y_train,
            X_test_raw=X_test,
            y_test_raw=y_test,
        )
        print(f"---- EPOCH {epoch}: ----")
        print("Test Acc:", res_acc)
        print("Test Log Loss:", ll)

        # TODO: implement experiment tracking
