"""Plot the predicted target distribution of a TabPFN regressor for one sample."""

#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from tabpfn import TabPFNRegressor

_STAT_STYLES = {
    "mean": ("#d62728", "-"),
    "median": ("#2ca02c", "--"),
    "mode": ("#ff7f0e", ":"),
}


def _as_single_row(x: object) -> np.ndarray | pd.DataFrame:
    """Coerce a single sample (1d array, Series or 1-row frame) to a 2d input."""
    if isinstance(x, pd.Series):
        return x.to_frame().T
    if isinstance(x, pd.DataFrame):
        if len(x) != 1:
            raise ValueError(f"Expected a single row, got {len(x)} rows.")
        return x
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.shape[0] != 1:
        raise ValueError(f"Expected a single row, got shape {arr.shape}.")
    return arr


def plot_regression_distribution(
    regressor: TabPFNRegressor,
    x: object,
    *,
    statistics: Sequence[str] = ("mean", "median", "mode"),
    quantile_interval: tuple[float, float] | None = (0.1, 0.9),
    zoom_quantile: float | None = 0.99,
    smooth: float = 0.005,
    ax: plt.Axes | None = None,
    color: str = "#1f77b4",
) -> plt.Axes:
    """Plot the predicted target distribution for a single sample.

    Args:
        regressor: A fitted ``TabPFNRegressor``.
        x: One sample, as a 1d array, a ``pandas.Series`` or a 1-row frame.
        statistics: Point statistics to mark with a vertical line. Any of
            ``"mean"``, ``"median"``, ``"mode"``.
        quantile_interval: Central interval to shade, e.g. ``(0.1, 0.9)`` for the
            80% interval. Pass ``None`` to disable.
        zoom_quantile: Fraction of probability mass to keep in view, centred on the
            median. Pass ``None`` to show the full support.
        smooth: Width of the display-only moving average over the density, as a
            fraction of the number of bars. Pass ``0`` to show the raw bar density.
        ax: Existing axes to draw on. A new figure is created if omitted.
        color: Base colour of the density curve.

    Returns:
        The matplotlib axes containing the plot.
    """
    # Local import because matplotlib is an optional dependency.
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.patches import Patch  # noqa: PLC0415
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. "
            'Install it with `pip install "tabpfn[viz]"`'
        ) from err

    out = regressor.predict(_as_single_row(x), output_type="full")
    logits, criterion = out["logits"], out["criterion"]

    widths = criterion.bucket_widths.cpu()
    centers = (criterion.borders[:-1].cpu() + widths / 2).numpy()
    density = (logits.softmax(-1).squeeze(0).cpu() / widths).numpy()
    if smooth:
        density = uniform_filter1d(density, max(1, round(smooth * len(density))))

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4.5))

    ax.fill_between(centers, density, color=color, alpha=0.18, lw=0)
    ax.plot(centers, density, color=color, lw=1.8)

    legend_handles = []
    if quantile_interval is not None:
        lo, hi = (criterion.icdf(logits, q).item() for q in quantile_interval)
        band = (centers >= lo) & (centers <= hi)
        pct = round((quantile_interval[1] - quantile_interval[0]) * 100)
        ax.fill_between(centers[band], density[band], color=color, alpha=0.3, lw=0)
        legend_handles.append(
            Patch(facecolor=color, alpha=0.5, lw=0, label=f"{pct}% interval")
        )

    for name in statistics:
        value = float(np.atleast_1d(out[name])[0])
        c, ls = _STAT_STYLES[name]
        legend_handles.append(
            ax.axvline(value, color=c, ls=ls, lw=1.6, label=f"{name} = {value:.3g}")
        )

    if zoom_quantile is not None:
        tail = (1 - zoom_quantile) / 2
        ax.set_xlim(
            criterion.icdf(logits, tail).item(),
            criterion.icdf(logits, 1 - tail).item(),
        )

    visible = density[(centers >= ax.get_xlim()[0]) & (centers <= ax.get_xlim()[1])]
    ax.set_ylim(0, visible.max() * 1.1 if visible.size else None)
    ax.margins(x=0)
    ax.set_xlabel("Predicted target")
    ax.set_ylabel("Probability density")
    ax.set_title("TabPFN predicted distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(handles=legend_handles, fontsize=9)
    return ax
