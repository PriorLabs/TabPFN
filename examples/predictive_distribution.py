#  Copyright (c) Prior Labs GmbH 2026.
"""Visualize TabPFN's predictive bar distribution for regression.

CLI usage::

    python predictive_distribution.py
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
    """Draw a variable-cell-size heatmap via pcolormesh."""
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
    """Plot TabPFN's per-sample bar distribution as a vertical density heatmap."""
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
    heatmap_with_box_sizes(ax, predictions.T, x_starts, x_ends, bar_borders[:-1], bar_borders[1:], **kwargs)


def main(
    n_train: int = 400,
    n_test: int = 200,
    n_estimators: int = 4,
    no_show: bool = False,
    out_dir: Path | None = None,
) -> None:
    if out_dir is None:
        out_dir = Path(".")
    rng = np.random.default_rng(0)
    x_train = rng.uniform(0.0, 10.0, n_train).astype(np.float32)
    y_clean = np.sin(0.6 * x_train)
    bimodal = x_train > 7.0
    sigma = np.where(
        bimodal, 0.20, np.clip(0.10 + np.maximum(0.0, x_train - 4.0) * 0.25, 0.10, 0.85)
    ).astype(np.float32)
    mode_offset = np.where(bimodal, np.where(rng.random(n_train) < 0.5, 1.5, -1.5), 0.0).astype(np.float32)
    y_train = (y_clean + mode_offset + rng.normal(0.0, sigma)).astype(np.float32)
    X_train = x_train.reshape(-1, 1)
    X_test = np.linspace(0.0, 10.0, n_test, dtype=np.float32).reshape(-1, 1)

    reg = TabPFNRegressor(n_estimators=n_estimators, random_state=0)
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test, output_type="full")

    criterion = preds["criterion"].to("cpu")
    logits = preds["logits"].detach().cpu()
    borders = criterion.borders
    x_test_1d = torch.tensor(X_test[:, 0])
    Y_MIN, Y_MAX = -4.0, 4.0

    q05 = criterion.icdf(logits, 0.05).numpy()
    q95 = criterion.icdf(logits, 0.95).numpy()

    # Figure 1: full predictive density with point estimates
    fig, ax = plt.subplots(figsize=(13, 6))
    plot_bar_distribution(ax, x_test_1d, borders, logits, restrict_to_range=(Y_MIN, Y_MAX))
    ax.scatter(x_train, y_train, s=16, alpha=0.6, color="black", edgecolors="none", label="train data", zorder=2)
    ax.plot(X_test[:, 0], preds["mean"],   color="C0", lw=2.0, label="mean",   zorder=3)
    ax.plot(X_test[:, 0], preds["median"], color="C1", lw=2.0, ls="--", label="median", zorder=3)
    ax.fill_between(X_test[:, 0], q05, q95, color="C0", alpha=0.12, label="90% CI", zorder=1)
    for x_edge in (4.0, 7.0):
        ax.axvline(x_edge, color="dimgray", lw=1.2, ls="-.", alpha=0.8, zorder=4)
    for xpos, label in [(2.0, "tight unimodal"), (5.5, "heteroscedastic"), (8.5, "bimodal")]:
        ax.text(xpos, Y_MAX * 0.92, label, ha="center", fontsize=12, color="gray")
    ax.set(xlabel="x", ylabel="y", xlim=(0.0, 10.0), ylim=(Y_MIN, Y_MAX),
           title="TabPFN predictive distribution\nheatmap = full bar distribution   |   lines = derived point estimates")
    ax.legend(loc="lower left")
    fig.tight_layout()
    out1 = out_dir / "tabpfn_predictive_distribution.png"
    fig.savefig(out1, dpi=150)
    print(f"Saved {out1}")

    # Figure 2: linear / coarse / log-density variants
    fig2, (ax_lin, ax_coarse, ax_log) = plt.subplots(1, 3, figsize=(18, 6), sharex=True, sharey=True)
    plot_bar_distribution(ax_lin,    x_test_1d, borders, logits, restrict_to_range=(Y_MIN, Y_MAX))
    plot_bar_distribution(ax_coarse, x_test_1d, borders, logits, restrict_to_range=(Y_MIN, Y_MAX), merge_bars=10)
    plot_bar_distribution(ax_log,    x_test_1d, borders, logits, restrict_to_range=(Y_MIN, Y_MAX), plot_log_probs=True)
    for axi, title in zip(
        (ax_lin, ax_coarse, ax_log),
        ("Default (linear density)", "Coarse merge (merge_bars=10)", "Log density (plot_log_probs=True)"),
    ):
        axi.set(xlim=(0.0, 10.0), ylim=(Y_MIN, Y_MAX), xlabel="x", ylabel="y", title=title)
    fig2.suptitle("plot_bar_distribution variants")
    fig2.tight_layout()
    out2 = out_dir / "tabpfn_predictive_distribution_variants.png"
    fig2.savefig(out2, dpi=150)
    print(f"Saved {out2}")

    if not no_show:
        plt.show()
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-train",      type=int,  default=400,      metavar="N")
    parser.add_argument("--n-test",       type=int,  default=200,      metavar="N")
    parser.add_argument("--n-estimators", type=int,  default=4,        metavar="N")
    parser.add_argument("--no-show",      action="store_true")
    parser.add_argument("--out-dir",      type=Path, default=Path("."))
    args = parser.parse_args()
    if args.no_show:
        plt.switch_backend("Agg")
    main(n_train=args.n_train, n_test=args.n_test, n_estimators=args.n_estimators,
         no_show=args.no_show, out_dir=args.out_dir)
