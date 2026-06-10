#  Copyright (c) Prior Labs GmbH 2026.
"""Visualise the predicted distribution for a single test point.

Run:
    python examples/plot_regression_distribution.py
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor
from tabpfn.visualisation import plot_regression_distribution


def main(
    output_path: str = "regression_distribution.png", *, show: bool = True
) -> None:
    """Fit a regressor and plot the predicted distribution for three test points."""
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    reg = TabPFNRegressor(n_estimators=4)
    reg.fit(X_train, y_train)

    # Pick the test points with the lowest, median and highest predicted target, so the
    # three panels span the range of predictions the model makes on this dataset.
    preds = reg.predict(X_test)
    order = np.argsort(preds)
    selected = [order[0], order[len(order) // 2], order[-1]]
    titles = ["Row A", "Row B", "Row C"]

    # Predict the three points in one batch and plot each panel from the shared output.
    out = reg.predict(X_test[selected], output_type="full")

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle("TabPFN predicted distributions — diabetes dataset", fontsize=13)

    for i, (ax, idx, title) in enumerate(zip(axes, selected, titles, strict=True)):
        plot_regression_distribution(out, sample_idx=i, ax=ax)
        true_val = y_test[idx]
        true_line = ax.axvline(
            true_val, color="purple", ls="-.", lw=1.4, label=f"true = {true_val:.0f}"
        )
        leg = ax.get_legend()
        handles = getattr(leg, "legend_handles", None) or getattr(
            leg, "legendHandles", []
        )
        ax.legend(handles=[*handles, true_line], fontsize=9)
        ax.set_title(title)

    fig.tight_layout()
    fig.savefig(output_path, dpi=130, bbox_inches="tight")
    print(f"Saved {output_path}")
    if show:
        plt.show()


if __name__ == "__main__":
    main()
