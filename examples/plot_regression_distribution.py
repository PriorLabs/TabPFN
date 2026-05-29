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
from tabpfn.visualization import plot_regression_distribution

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0
)

reg = TabPFNRegressor(n_estimators=4)
reg.fit(X_train, y_train)

# Pick three test points with different predicted values for comparison.
preds = reg.predict(X_test)
low_idx = int(np.argmin(preds))
mid_idx = int(np.argsort(preds)[len(preds) // 2])
high_idx = int(np.argmax(preds))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("TabPFN predicted distributions — diabetes dataset", fontsize=13)

for ax, idx, label in zip(
    axes,
    [low_idx, mid_idx, high_idx],
    ["low prediction", "median prediction", "high prediction"],
):
    plot_regression_distribution(reg, X_test[idx], ax=ax)
    true_val = y_test[idx]
    true_line = ax.axvline(true_val, color="purple", ls="-.", lw=1.4, label=f"true = {true_val:.0f}")
    leg = ax.get_legend()
    handles = getattr(leg, "legend_handles", None) or getattr(leg, "legendHandles", [])
    ax.legend(handles=[*handles, true_line], fontsize=9)
    ax.set_title(label)

plt.tight_layout()
plt.savefig("regression_distribution.png", dpi=130, bbox_inches="tight")
print("Saved regression_distribution.png")
plt.show()
