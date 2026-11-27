#  Copyright (c) Prior Labs GmbH 2026.

"""Extract gradients of TabPFN predictions with respect to the input data.

With ``differentiable_input=True``, TabPFN keeps the autograd graph intact all
the way from the input tensors to the predicted probabilities. Backpropagating
a prediction through the frozen model therefore yields ``d prediction / d X``
for every test sample and feature — a simple saliency map showing which
feature values drive each prediction.

Run:
    python examples/input_gradients.py
"""

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier
from tabpfn.utils import infer_devices


def main() -> None:
    """Compute d P(benign) / d X on the breast-cancer dataset."""
    device = infer_devices("auto")[0]
    print(f"Device: {device}")

    data = load_breast_cancer()

    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=0, stratify=data.target
    )

    # Standardize features so gradient magnitudes are comparable across features.
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    clf = TabPFNClassifier(
        n_estimators=1,
        random_state=0,
        device=device,
        inference_precision=torch.float32,
        differentiable_input=True,
    )
    # fit_with_differentiable_input does not infer n_classes_ from y (y is a
    # differentiable tensor, not discrete labels), so set it explicitly.
    clf.n_classes_ = len(np.unique(y_train))

    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(
        X_test, dtype=torch.float32, device=device, requires_grad=True
    )

    clf.fit_with_differentiable_input(X_train_t, y_train_t)

    # use_inference_mode=True gives (n_samples, n_classes) probabilities; gradients
    # still flow because differentiable_input=True.
    probs = clf.forward(X_test_t, use_inference_mode=True)
    accuracy = (probs.argmax(dim=1).cpu().numpy() == y_test).mean()
    print(f"Test accuracy: {accuracy:.4f}")

    # Gradient of the summed benign-class probability w.r.t. every input value.
    # Summing over samples is just a trick to get all per-sample gradients in one
    # backward pass; sample i's row of `grads` only depends on prediction i.
    (grads,) = torch.autograd.grad(probs[:, 1].sum(), X_test_t)
    print(f"Gradient matrix shape: {tuple(grads.shape)}")
    print(grads)


if __name__ == "__main__":
    main()
