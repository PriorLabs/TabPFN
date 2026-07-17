#  Copyright (c) Prior Labs GmbH 2026.

"""Extract gradients of TabPFN predictions with respect to the input data.

With ``differentiable_input=True``, TabPFN keeps the autograd graph intact all
the way from the input tensors to the predicted probabilities. Backpropagating
a prediction through the frozen model therefore yields ``d prediction / d X``
for every test sample and feature — a simple saliency map showing which
feature values drive each prediction.

Run:
    pip install seaborn  # plotting only; not a TabPFN dependency
    python examples/input_gradients.py
"""

from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tabpfn import TabPFNClassifier
from tabpfn.utils import infer_devices

# Prior Labs brand palette.
PRIOR_BLUE = "#101075"
PRIOR_YELLOW = "#ffd400"
PRIOR_INK = "#101010"
PRIOR_GRID = "#cccccc"

PRIOR_DIVERGING = LinearSegmentedColormap.from_list(
    "prior_diverging", [PRIOR_BLUE, "#ffffff", PRIOR_YELLOW]
)


def _apply_brand_style(ax: plt.Axes, *, grid_axis: Literal["x", "y"] | None) -> None:
    """White field, dashed grid, ink spines without the top/right ones."""
    ax.set_facecolor("#ffffff")
    if grid_axis is not None:
        ax.grid(
            axis=grid_axis, color=PRIOR_GRID, linestyle="--", linewidth=0.6, alpha=0.8
        )
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(PRIOR_INK)
        ax.spines[spine].set_linewidth(1.0)
    ax.tick_params(colors=PRIOR_INK)
    ax.title.set_color(PRIOR_INK)
    ax.xaxis.label.set_color(PRIOR_INK)
    ax.yaxis.label.set_color(PRIOR_INK)


def main(output_path: str = "input_gradients.png", *, show: bool = True) -> None:
    """Compute and plot d P(benign) / d X on the breast-cancer dataset."""
    device = infer_devices("auto")[0]
    print(f"Device: {device}")

    data = load_breast_cancer()
    feature_names = data.feature_names

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
    grads = grads.cpu().numpy()
    print(f"Gradient matrix shape: {grads.shape}")

    # Global saliency: mean absolute gradient per feature, most salient first.
    saliency = np.abs(grads).mean(axis=0)
    order = np.argsort(saliency)[::-1]

    sns.set_theme(style="white")
    # Use the brand font when it is installed; fall back to the default otherwise.
    if any(f.name == "Saans" for f in font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = ["Saans", "DejaVu Sans"]

    fig, (ax_bar, ax_map) = plt.subplots(
        1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1, 1.2]}
    )
    fig.set_facecolor("#ffffff")
    fig.suptitle(
        "TabPFN input gradients — d P(benign) / d X, breast-cancer dataset",
        fontsize=14,
        color=PRIOR_INK,
        fontweight="medium",
    )

    sns.barplot(x=saliency[order], y=feature_names[order], color=PRIOR_BLUE, ax=ax_bar)
    _apply_brand_style(ax_bar, grid_axis="x")
    ax_bar.set_xlabel("mean |gradient| over test samples")
    ax_bar.set_title("Global feature saliency")
    ax_bar.tick_params(axis="y", labelsize=8)

    # Signed gradients for the most salient features on a handful of test samples.
    n_show, n_top = 15, 15
    top = order[:n_top]
    max_abs = np.abs(grads[:n_show, top]).max()
    sns.heatmap(
        grads[:n_show, top].T,
        cmap=PRIOR_DIVERGING,
        vmin=-max_abs,
        vmax=max_abs,
        linewidths=0.6,
        linecolor="#ffffff",
        yticklabels=feature_names[top],
        xticklabels=[f"{p:.2f}" for p in probs[:n_show, 1].detach().cpu().numpy()],
        cbar_kws={"label": "d P(benign) / d x"},
        ax=ax_map,
    )
    _apply_brand_style(ax_map, grid_axis=None)
    for spine in ax_map.spines.values():
        spine.set_visible(False)
    ax_map.set_xlabel("test sample, labelled by predicted P(benign)")
    ax_map.set_title(f"Signed gradients, top {n_top} features, {n_show} samples")
    ax_map.tick_params(axis="both", labelsize=8)

    fig.tight_layout()
    fig.savefig(
        output_path, dpi=130, bbox_inches="tight", facecolor=fig.get_facecolor()
    )
    print(f"Saved {output_path}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
