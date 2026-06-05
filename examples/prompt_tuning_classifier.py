#  Copyright (c) Prior Labs GmbH 2026.

"""Example of prompt-tuning a TabPFN classifier.

Prompt tuning keeps the TabPFN weights frozen and instead optimizes the
in-context examples themselves: a small set of (X, y) "prompt" samples is
treated as differentiable parameters and updated by gradient descent so that
predictions on held-out queries improve.

This relies on ``differentiable_input=True``, which lets gradients flow from a
downstream loss back through ``fit_with_differentiable_input`` into the prompt
tensors. The example runs on CPU but is considerably faster on a CUDA GPU.
"""

import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from tabpfn import TabPFNClassifier


def eval_test(
    clf: TabPFNClassifier,
    prompt_x: torch.Tensor,
    prompt_y: torch.Tensor,
    my_dl_test: DataLoader,
    lossfn: torch.nn.Module,
    device: str,
) -> tuple[float, float]:
    """Evaluate the current prompt on the held-out test set."""
    with torch.no_grad():
        loss_sum = 0.0
        acc_sum = 0.0
        acc_items = 0
        clf.fit_with_differentiable_input(prompt_x, prompt_y.flatten())
        for data_batch in my_dl_test:
            X_tests, y_tests = data_batch
            predictions = clf.forward(X_tests, use_inference_mode=True)
            loss_sum += lossfn(torch.log(predictions), y_tests.to(device)).item()
            acc_sum += (
                accuracy_score(
                    y_tests.flatten().cpu(),
                    predictions.argmax(dim=1).cpu(),
                )
                * y_tests.numel()
            )
            acc_items += y_tests.numel()

        res_accuracy = acc_sum / acc_items
        return loss_sum, res_accuracy


def main() -> None:
    n_prompt_samples = 200  # samples used for prompt tuning
    n_total_samples = 2000  # total samples
    n_classes = 3
    n_features = 20
    do_epochs = 5
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Device: {device}")
    print(f"Prompt samples: {n_prompt_samples}")
    print(f"Total samples: {n_total_samples}")
    print(f"Classes: {n_classes}")

    torch.manual_seed(42)

    # Create synthetic dataset
    X, y = make_classification(
        n_samples=n_total_samples,
        n_features=n_features,
        n_informative=10,
        n_redundant=5,
        n_classes=n_classes,
        n_clusters_per_class=2,
        random_state=42,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    clf = TabPFNClassifier(
        ignore_pretraining_limits=True,
        device=device,
        n_estimators=1,
        random_state=2,
        inference_precision=torch.float32,
        differentiable_input=True,
    )
    # fit_with_differentiable_input does not infer n_classes_ from y (y is a
    # differentiable tensor, not discrete labels), so set it explicitly.
    clf.n_classes_ = len(np.unique(y))

    # The prompt: the in-context examples we will optimize.
    prompt_x_tensor = torch.tensor(X_train[:n_prompt_samples], dtype=torch.float).to(
        device
    )
    prompt_y_tensor = torch.tensor(y_train[:n_prompt_samples], dtype=torch.float).to(
        device
    )

    # The query examples used to drive the prompt-tuning loss.
    train_x_tensor = torch.tensor(X_train[n_prompt_samples:], dtype=torch.float).to(
        device
    )
    train_y_tensor = torch.tensor(y_train[n_prompt_samples:], dtype=torch.long).to(
        device
    )

    test_x_tensor = torch.tensor(X_test, dtype=torch.float).to(device)
    test_y_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

    dataset_train = TensorDataset(train_x_tensor, train_y_tensor)
    dataloader_train = DataLoader(dataset_train, batch_size=128, shuffle=True)
    dataset_test = TensorDataset(test_x_tensor, test_y_tensor)
    dataloader_test = DataLoader(dataset_test, batch_size=512, shuffle=False)

    lossfn = torch.nn.NLLLoss()

    # Compute initial accuracy
    loss_test, res_acc = eval_test(
        clf,
        prompt_x_tensor,
        prompt_y_tensor,
        dataloader_test,
        lossfn,
        device,
    )
    print(f"Initial accuracy: {res_acc:.4f}")

    # Mark the prompt tensors as the parameters to optimize.
    prompt_x_tensor.requires_grad_(requires_grad=True)
    prompt_y_tensor.requires_grad_(requires_grad=True)

    optim_impl = Adam([prompt_x_tensor, prompt_y_tensor], lr=1e-2)

    for epoch in range(do_epochs):
        total_loss = 0.0
        for data_batch in tqdm(
            dataloader_train, desc=f"Epoch {epoch + 1}", leave=False
        ):
            input_x_batch, input_y_batch = data_batch
            optim_impl.zero_grad()
            clf.fit_with_differentiable_input(
                prompt_x_tensor, prompt_y_tensor.flatten()
            )
            # use_inference_mode=True for correct output format; gradients still
            # flow because differentiable_input=True.
            predictions = clf.forward(input_x_batch, use_inference_mode=True)
            loss = lossfn(torch.log(predictions), input_y_batch.to(device))
            loss.backward()
            optim_impl.step()
            total_loss += loss.item()

        loss_test, res_acc = eval_test(
            clf,
            prompt_x_tensor,
            prompt_y_tensor,
            dataloader_test,
            lossfn,
            device,
        )
        print(
            f"Epoch {epoch + 1}/{do_epochs}: "
            f"Test Acc={res_acc:.4f}, Test Loss={loss_test:.4f}"
        )


if __name__ == "__main__":
    main()
