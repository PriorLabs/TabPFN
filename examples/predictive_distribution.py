#  Copyright (c) Prior Labs GmbH 2026.
"""Example: visualizing TabPFN's full predictive distribution for regression.

TabPFN regressors predict a *bar distribution* — a discrete probability
distribution over the target axis — rather than just a point estimate.
This lets you extract means, medians, quantiles, and the full density.

This example shows how to:
  1. Fit a TabPFNRegressor on a toy regression dataset.
  2. Retrieve the bar distribution via ``predict(X, output_type="full")``.
  3. Derive summary statistics (mean, median, 90% credible interval).
  4. Visualize the full predictive density as a per-sample heatmap using
     ``plot_bar_distribution``, including coarse-merge and log-density views.

Requirements: pip install tabpfn matplotlib torch

CLI usage::

    # Standard run (saves two PNGs to the current directory):
    python predictive_distribution.py

    # Quick headless run for testing:
    python predictive_distribution.py --n-train 60 --n-test 40 --n-estimators 1 --no-show
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from matplotlib.colors import Colormap

from tabpfn import TabPFNRegressor


# ─────────────────────────────── plotting helpers ─────────────────────────────


def heatmap_with_box_sizes(
    ax: Axes,
    data: torch.Tensor,
    x_starts: torch.Tensor,
    x_ends: torch.Tensor,
    y_starts: torch.Tensor,
    y_ends: torch.Tensor,
    cmap: Colormap | None = None,
    set_lims: bool = True,
    threshold_i: float = 0.0,
    y_min: float | None = None,
    y_max: float | None = None,
    per_col_normalize: bool = False,
) -> None:
    """Draw a variable-cell-size heatmap via pcolormesh.

    ``data`` shape is ``(n_rows, n_cols)``. Both x and y grids must be
    contiguous (``x_ends[i] == x_starts[i+1]``, ``y_ends[j] == y_starts[j+1]``).

    Args:
        ax: Matplotlib axis to draw on.
        data: 2-D tensor of shape ``(n_rows, n_cols)``.
        x_starts: Left edge of each column, length ``n_cols``.
        x_ends: Right edge of each column, length ``n_cols``.
        y_starts: Bottom edge of each row, length ``n_rows``.
        y_ends: Top edge of each row, length ``n_rows``.
        cmap: Matplotlib colormap. Defaults to ``"Blues"``.
        set_lims: Auto-set axis limits from data extents.
        threshold_i: Normalised intensity below which cells appear blank.
        y_min: Override lower y-axis limit (only when ``set_lims=True``).
        y_max: Override upper y-axis limit (only when ``set_lims=True``).
        per_col_normalize: Normalise each column independently.
    """
    if cmap is None:
        cmap = plt.colormaps["Blues"]

    if y_starts.ndim != 1 or y_ends.shape != y_starts.shape:
        raise ValueError("y_starts and y_ends must be 1-D tensors of equal length.")

    if per_col_normalize:
        col_min = data.min(0, keepdim=True).values
        col_max = data.max(0, keepdim=True).values
        data = (data - col_min) / (col_max - col_min).clamp(min=1e-10)
    else:
        d_min, d_max = data.min(), data.max()
        data = (data - d_min) / (d_max - d_min).clamp(min=1e-10)

    data = ((data - threshold_i) / (1.0 - threshold_i)).clamp(min=0.0)

    x_edges = torch.cat([x_starts, x_ends[-1:]]).numpy()
    y_edges = torch.cat([y_starts, y_ends[-1:]]).numpy()
    mesh = ax.pcolormesh(x_edges, y_edges, data.numpy(), cmap=cmap, vmin=0.0, vmax=1.0)
    mesh.set_rasterized(True)

    if set_lims:
        ax.set_xlim(x_starts[0].item(), x_ends[-1].item())
        if y_min is not None and y_max is not None:
            ax.set_ylim(y_min, y_max)
        else:
            ax.set_ylim(y_starts[0].item(), y_ends[-1].item())


def plot_bar_distribution(
    ax: Axes,
    x: torch.Tensor,
    bar_borders: torch.Tensor,
    logits: torch.Tensor,
    merge_bars: int | None = None,
    restrict_to_range: tuple[float, float] | None = None,
    plot_log_probs: bool = False,
    **kwargs: Any,
) -> None:
    """Plot TabPFN's per-sample bar distribution as a vertical density heatmap.

    Each test sample is rendered as a vertical column coloured by the model's
    predictive density at that x-position.

    Args:
        ax: Matplotlib axis to draw on.
        x: 1-D x-positions of shape ``(n_test,)`` for the test samples.
        bar_borders: Border positions from ``preds["criterion"].borders``.
        logits: Raw logits of shape ``(n_test, n_bars)`` from
            ``preds["logits"]``.
        merge_bars: If set, merge this many adjacent bars into one for a
            faster, coarser render (e.g. ``merge_bars=10``).
        restrict_to_range: ``(y_min, y_max)`` to crop the y-axis to a
            sub-range of target values.
        plot_log_probs: If True, plot log-densities. Useful when a few bars
            dominate and low-probability regions are invisible on a linear scale.
        **kwargs: Forwarded to :func:`heatmap_with_box_sizes`
            (e.g. ``cmap``, ``threshold_i``, ``per_col_normalize``).
    """
    x = x.flatten()
    if logits.ndim == 3 and logits.shape[0] == 1:
        logits = logits.squeeze(0)
    predictions = logits.softmax(-1)
    if predictions.shape != (len(x), len(bar_borders) - 1):
        raise ValueError(
            f"Shape mismatch: predictions {predictions.shape} vs "
            f"expected ({len(x)}, {len(bar_borders) - 1})."
        )

    if merge_bars is not None and merge_bars > 1:
        new_border_inds = torch.arange(0, len(bar_borders), merge_bars)
        if new_border_inds[-1] != len(bar_borders) - 1:
            new_border_inds = torch.cat([new_border_inds, torch.tensor([len(bar_borders) - 1])])
        bar_borders = bar_borders[new_border_inds]
        pred_cumsum = torch.cat(
            [predictions.new_zeros(len(predictions), 1), predictions.cumsum(-1)], dim=-1
        )
        predictions = pred_cumsum[:, new_border_inds[1:]] - pred_cumsum[:, new_border_inds[:-1]]

    if restrict_to_range is not None:
        y_min_r, y_max_r = restrict_to_range
        # Include every bar whose interval overlaps [y_min_r, y_max_r].
        overlap = (bar_borders[:-1] < y_max_r) & (bar_borders[1:] > y_min_r)
        if not overlap.any():
            return
        idx = overlap.nonzero(as_tuple=True)[0]
        first_i, last_i = idx[0].item(), idx[-1].item()
        predictions = predictions[:, first_i : last_i + 1]
        bar_borders = bar_borders[first_i : last_i + 2]

    x, order = x.sort(0)

    bar_widths = bar_borders[1:] - bar_borders[:-1]
    predictions = predictions[order] / bar_widths.clamp(min=1e-10)
    predictions[~torch.isfinite(predictions)] = 0.0
    predictions[:, bar_widths < 1e-10] = 0.0

    if plot_log_probs:
        predictions = predictions.log()
        finite = predictions[torch.isfinite(predictions)]
        if finite.numel() > 0:
            predictions[~torch.isfinite(predictions)] = finite.min()

    x_starts = torch.cat([x[:1], (x[1:] + x[:-1]) / 2])
    x_ends = torch.cat([(x[1:] + x[:-1]) / 2, x[-1:]])

    heatmap_with_box_sizes(
        ax,
        predictions.T,
        x_starts,
        x_ends,
        bar_borders[:-1],
        bar_borders[1:],
        **kwargs,
    )


# ─────────────────────────────────── main ─────────────────────────────────────


def main(
    n_train: int = 400,
    n_test: int = 200,
    n_estimators: int = 4,
    no_show: bool = False,
    out_dir: Path | None = None,
) -> None:
    """Run the predictive-distribution example end-to-end."""
    if out_dir is None:
        out_dir = Path(".")
    # ── toy dataset: three uncertainty regimes along x ────────────────────────
    # x in [0, 4]:  tight unimodal noise
    # x in [4, 7]:  variance grows (heteroscedastic)
    # x in [7, 10]: two equally-likely modes (bimodal)
    # A single point estimate fails to capture all three; the bar distribution
    # shows the full picture.

    rng = np.random.default_rng(0)

    x_train = rng.uniform(0.0, 10.0, n_train).astype(np.float32)
    y_clean = np.sin(0.6 * x_train)

    bimodal = x_train > 7.0
    sigma = np.where(
        bimodal,
        0.20,
        np.clip(0.10 + np.maximum(0.0, x_train - 4.0) * 0.25, 0.10, 0.85),
    ).astype(np.float32)

    mode_offset = np.where(
        bimodal, np.where(rng.random(n_train) < 0.5, 1.5, -1.5), 0.0
    ).astype(np.float32)

    y_train = (y_clean + mode_offset + rng.normal(0.0, sigma)).astype(np.float32)
    X_train = x_train.reshape(-1, 1)

    # Dense test grid for a smooth heatmap.
    X_test = np.linspace(0.0, 10.0, n_test, dtype=np.float32).reshape(-1, 1)

    # ── fit and retrieve full predictive distribution ─────────────────────────
    reg = TabPFNRegressor(n_estimators=n_estimators, random_state=0)
    reg.fit(X_train, y_train)

    # output_type="full" returns logits and the BarDistribution criterion so
    # you can compute any statistic without re-running inference.
    preds = reg.predict(X_test, output_type="full")

    criterion = preds["criterion"].to("cpu")
    logits = preds["logits"].detach().cpu()
    borders = criterion.borders

    print(f"Bar distribution: {logits.shape[1]} bars over {len(X_test)} test points")
    print(f"Border range: [{borders[0]:.2f}, {borders[-1]:.2f}]")

    # ── derive custom statistics from the raw distribution ────────────────────
    # All of these are also available directly in preds, but computing them
    # manually shows how to use the criterion API for custom summaries.
    mean_vals = criterion.mean(logits).numpy()
    median_vals = criterion.median(logits).numpy()
    q05 = criterion.icdf(logits, 0.05).numpy()
    q95 = criterion.icdf(logits, 0.95).numpy()

    # These should match the precomputed preds shortcuts up to small numerical
    # jitter. The median is a step/quantile inversion, so it can differ by up to
    # one bar width — hence we report the deviation rather than assert on it.
    mean_diff = float(np.abs(mean_vals - preds["mean"]).max())
    median_diff = float(np.abs(median_vals - preds["median"]).max())
    print(f"mean   vs preds['mean']:   max |Δ| = {mean_diff:.2e}  (expect ~0, exact)")
    print(f"median vs preds['median']: max |Δ| = {median_diff:.2e}  (expect <1 bar width)")

    Y_MIN, Y_MAX = -4.0, 4.0
    x_test_1d = torch.tensor(X_test[:, 0])

    # ── figure 1: full predictive density with point estimates overlaid ───────
    fig, ax = plt.subplots(figsize=(13, 6))

    plot_bar_distribution(ax, x_test_1d, borders, logits, restrict_to_range=(Y_MIN, Y_MAX))

    ax.scatter(x_train, y_train, s=16, alpha=0.6, color="black", edgecolors="none", label="train data", zorder=2)
    ax.plot(X_test[:, 0], preds["mean"],   color="C0", lw=2.0, label="mean",   zorder=3)
    ax.plot(X_test[:, 0], preds["median"], color="C1", lw=2.0, ls="--", label="median", zorder=3)
    ax.fill_between(X_test[:, 0], q05, q95, color="C0", alpha=0.12, label="90% CI", zorder=1)

    for x_edge in (4.0, 7.0):
        ax.axvline(x_edge, color="gray", lw=1, ls=":", alpha=0.7, zorder=0)
    ax.text(2.0,  Y_MAX * 0.92, "tight unimodal",  ha="center", fontsize=12, color="gray")
    ax.text(5.5,  Y_MAX * 0.92, "heteroscedastic", ha="center", fontsize=12, color="gray")
    ax.text(8.5,  Y_MAX * 0.92, "bimodal",         ha="center", fontsize=12, color="gray")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_xlim(0.0, 10.0)
    ax.set_ylim(Y_MIN, Y_MAX)
    ax.set_title(
        "TabPFN predictive distribution\n"
        "heatmap = full bar distribution   |   lines = derived point estimates"
    )
    ax.legend(loc="lower left")
    fig.tight_layout()

    out1 = out_dir / "tabpfn_predictive_distribution.png"
    fig.savefig(out1, dpi=150)
    print(f"Saved {out1}")

    # ── figure 2: plot_bar_distribution variants ──────────────────────────────
    # Left: default linear density.  Centre: coarser merge (merge_bars=10).
    # Right: log-density — reveals low-probability structure in the tails.
    fig2, (ax_lin, ax_coarse, ax_log) = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)

    plot_bar_distribution(ax_lin,    x_test_1d, borders, logits, restrict_to_range=(Y_MIN, Y_MAX))
    plot_bar_distribution(ax_coarse, x_test_1d, borders, logits, restrict_to_range=(Y_MIN, Y_MAX), merge_bars=10)
    plot_bar_distribution(ax_log,    x_test_1d, borders, logits, restrict_to_range=(Y_MIN, Y_MAX), plot_log_probs=True)

    titles = [
        "Default (linear density)",
        "Coarse merge (merge_bars=10)",
        "Log density (plot_log_probs=True)",
    ]
    for axi, title in zip((ax_lin, ax_coarse, ax_log), titles):
        axi.set_xlim(0.0, 10.0)
        axi.set_ylim(Y_MIN, Y_MAX)
        axi.set_xlabel("x")
        axi.set_ylabel("y")
        axi.set_title(title)

    fig2.suptitle("plot_bar_distribution variants")
    fig2.tight_layout()

    out2 = out_dir / "tabpfn_predictive_distribution_variants.png"
    fig2.savefig(out2, dpi=150)
    print(f"Saved {out2}")

    if not no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--n-train",      type=int,  default=400, metavar="N", help="Training samples (default: 400).")
    parser.add_argument("--n-test",       type=int,  default=200, metavar="N", help="Test-grid points (default: 200).")
    parser.add_argument("--n-estimators", type=int,  default=4,   metavar="N", help="TabPFN estimators; lower = faster (default: 4).")
    parser.add_argument("--no-show",      action="store_true",                 help="Skip plt.show().")
    parser.add_argument("--out-dir",      type=Path, default=Path("."),        help="Directory for saved PNGs (default: current dir).")
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    main(
        n_train=args.n_train,
        n_test=args.n_test,
        n_estimators=args.n_estimators,
        no_show=args.no_show,
        out_dir=args.out_dir,
    )
