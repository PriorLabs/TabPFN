"""Plot the predicted target distribution of a TabPFN regressor for one sample."""

#  Copyright (c) Prior Labs GmbH 2026.

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from scipy.ndimage import uniform_filter1d

if TYPE_CHECKING:
    import matplotlib.pyplot as plt

    from tabpfn.regressor import FullOutputDict

_STAT_STYLES = {
    "mean": ("#d62728", "-"),
    "median": ("#2ca02c", "--"),
    "mode": ("#ff7f0e", ":"),
}


def _validate_args(
    prediction: FullOutputDict,
    sample_idx: int,
    statistics: Sequence[str],
    quantile_interval: tuple[float, float] | None,
    zoom_quantile: float | None,
    smooth: float,
) -> None:
    if not {"logits", "criterion"} <= prediction.keys():
        raise ValueError(
            'prediction must be the output of predict(..., output_type="full").'
        )
    unknown = [name for name in statistics if name not in _STAT_STYLES]
    if unknown:
        raise ValueError(
            f"Unknown statistics {unknown}; choose from {list(_STAT_STYLES)}."
        )
    if quantile_interval is not None:
        lo_q, hi_q = quantile_interval
        if not 0 <= lo_q < hi_q <= 1:
            raise ValueError(
                "quantile_interval must be (low, high) with 0 <= low < high <= 1."
            )
    if zoom_quantile is not None and not 0 < zoom_quantile <= 1:
        raise ValueError("zoom_quantile must be in (0, 1].")
    if smooth < 0:
        raise ValueError("smooth must be non-negative.")
    n_samples = prediction["logits"].shape[0]
    if not 0 <= sample_idx < n_samples:
        raise ValueError(
            f"sample_idx {sample_idx} is out of range for {n_samples} sample(s)."
        )


def plot_regression_distribution(
    prediction: FullOutputDict,
    *,
    sample_idx: int = 0,
    statistics: Sequence[str] = ("mean", "median", "mode"),
    quantile_interval: tuple[float, float] | None = (0.1, 0.9),
    zoom_quantile: float | None = 0.99,
    smooth: float = 0.005,
    ax: plt.Axes | None = None,
    color: str = "#1f77b4",
) -> plt.Axes:
    """Plot the predicted target distribution for a single sample.

    Args:
        prediction: Output of ``regressor.predict(X, output_type="full")``. It may
            hold several samples; pick the one to plot with ``sample_idx``.
        sample_idx: Index of the sample to plot within ``prediction``.
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
    _validate_args(
        prediction, sample_idx, statistics, quantile_interval, zoom_quantile, smooth
    )

    # Local import because matplotlib is an optional dependency.
    try:
        import matplotlib.pyplot as plt  # noqa: PLC0415
        from matplotlib.patches import Patch  # noqa: PLC0415
    except ModuleNotFoundError as err:
        raise ModuleNotFoundError(
            "matplotlib is required for plotting. "
            'Install it with `pip install "tabpfn[viz]"`'
        ) from err

    logits = prediction["logits"][sample_idx : sample_idx + 1]
    criterion = prediction["criterion"]

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
        value = float(np.atleast_1d(prediction[name])[sample_idx])
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
