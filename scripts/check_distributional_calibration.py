"""Diagnostic script to verify whether softmax_temperature and average_before_softmax
affect distributional calibration metrics.

Hypothesis (from Jonas Landsgesell paper + email thread):
- softmax_temperature=0.9 sharpens distributions -> hurts log-score / CRPS / CRLS
- averaging probabilities vs logits before ensembling may matter for calibration

Metrics:
  NLL   - Negative log-likelihood (log score), sensitive to sharpness
  CRPS  - Continuous Ranked Probability Score, integral over quantile losses
  CRLS  - Continuous Ranked Log Score = CRPS + NLL / 2 (Brehmer & Gneiting 2021)
  IS95  - Interval Score at 95%: penalises width + miscoverage (lower = better)
  Cov95 - Empirical coverage of 95% prediction interval (target: 0.95)
  Sharp - Mean width of 95% PI (lower = sharper, but only good if well-calibrated)
  MACE  - Mean Absolute Calibration Error from PIT (lower = better)
  KS_p  - p-value of KS test for PIT uniformity (higher = better calibrated)
  RMSE  - Root mean squared error of the mean prediction
"""

from __future__ import annotations

import numpy as np
import torch
from scipy.stats import kstest

from tabpfn import TabPFNRegressor


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_pit(criterion, logits: torch.Tensor, y: np.ndarray) -> np.ndarray:
    """P(Y <= y_true) under predicted distribution. Uniform => calibrated."""
    y_t = torch.as_tensor(y, dtype=logits.dtype, device=logits.device).unsqueeze(-1)
    return criterion.cdf(logits, y_t).squeeze(-1).cpu().detach().numpy()


def compute_nll(criterion, logits: torch.Tensor, y: np.ndarray) -> float:
    """Mean negative log-likelihood (log score)."""
    y_t = torch.as_tensor(y, dtype=logits.dtype, device=logits.device)
    return criterion(logits, y_t).mean().item()


def compute_crps(criterion, logits: torch.Tensor, y: np.ndarray) -> float:
    """CRPS via quantile decomposition: E_q[(F^{-1}(q) - y)*(q - 1{y<=F^{-1}(q)})]."""
    quantile_levels = np.linspace(0.01, 0.99, 99)
    crps_sum = 0.0
    for q in quantile_levels:
        q_pred = criterion.icdf(logits, q).cpu().detach().numpy()
        indicator = (y <= q_pred).astype(float)
        crps_sum += np.mean((indicator - q) ** 2)
    return crps_sum / len(quantile_levels)


def compute_crls(crps: float, nll: float) -> float:
    """Continuous Ranked Log Score (Brehmer & Gneiting 2021).
    Combines sharpness of log score with calibration of CRPS:
      CRLS = (CRPS + NLL) / 2
    """
    return (crps + nll) / 2


def compute_is95(criterion, logits: torch.Tensor, y: np.ndarray) -> tuple[float, float, float]:
    """Interval Score at 95% PI.
    IS_alpha = (u - l) + (2/alpha) * [max(0, l-y) + max(0, y-u)]
    """
    alpha = 0.05
    l = criterion.icdf(logits, alpha / 2).cpu().detach().numpy()
    u = criterion.icdf(logits, 1 - alpha / 2).cpu().detach().numpy()
    width = u - l
    penalty = (2 / alpha) * (np.maximum(0, l - y) + np.maximum(0, y - u))
    is95 = np.mean(width + penalty)
    coverage = np.mean((y >= l) & (y <= u))
    sharpness = np.mean(width)
    return is95, coverage, sharpness


def compute_mace(pit: np.ndarray, n_bins: int = 10) -> float:
    """Mean Absolute Calibration Error from PIT histogram."""
    expected = 1.0 / n_bins
    counts, _ = np.histogram(pit, bins=n_bins, range=(0, 1))
    observed = counts / len(pit)
    return np.mean(np.abs(observed - expected))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_config(X_train, y_train, X_test, y_test, *, softmax_temperature, average_before_softmax, ensemble_temperature=1.0):
    reg = TabPFNRegressor(
        n_estimators=8,
        softmax_temperature=softmax_temperature,
        average_before_softmax=average_before_softmax,
        ensemble_temperature=ensemble_temperature,
        random_state=42,
    )
    reg.fit(X_train, y_train)
    result = reg.predict(X_test, output_type="full")

    criterion = result["criterion"]
    logits = result["logits"]

    pit  = compute_pit(criterion, logits, y_test)
    nll  = compute_nll(criterion, logits, y_test)
    crps = compute_crps(criterion, logits, y_test)
    crls = compute_crls(crps, nll)
    is95, cov95, sharp = compute_is95(criterion, logits, y_test)
    mace = compute_mace(pit)
    _, ks_p = kstest(pit, "uniform")
    rmse = np.sqrt(np.mean((result["mean"] - y_test) ** 2))

    return dict(nll=nll, crps=crps, crls=crls, is95=is95,
                cov95=cov95, sharp=sharp, mace=mace, ks_p=ks_p, rmse=rmse)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(42)
    n_train, n_test, n_features = 100, 200, 2

    X = rng.normal(0, 1, (n_train + n_test, n_features))
    w = rng.normal(0, 1, n_features)
    y = X @ w + rng.normal(0, 1, n_train + n_test)

    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    configs = [
        {"softmax_temperature": 0.9, "average_before_softmax": False, "ensemble_temperature": 1.0},        # old default
        {"softmax_temperature": 1.0, "average_before_softmax": False, "ensemble_temperature": 1.0},        # no temp scaling
        {"softmax_temperature": 0.9, "average_before_softmax": False, "ensemble_temperature": 1/0.9},      # NEW default
        {"softmax_temperature": 1.0, "average_before_softmax": False, "ensemble_temperature": 1.0},        # fully neutral
    ]

    cols = ["NLL", "CRPS", "CRLS", "IS95", "Cov95", "Sharp", "MACE", "KS_p", "RMSE"]
    header = f"{'Config':<45}" + "".join(f"{c:>8}" for c in cols)
    print(header)
    print("-" * len(header))

    for cfg in configs:
        label = f"sm_t={cfg['softmax_temperature']}, ens_t={cfg['ensemble_temperature']:.3f}"
        m = evaluate_config(X_train, y_train, X_test, y_test, **cfg)
        print(
            f"{label:<45}"
            f"{m['nll']:>8.4f}"
            f"{m['crps']:>8.4f}"
            f"{m['crls']:>8.4f}"
            f"{m['is95']:>8.4f}"
            f"{m['cov95']:>8.4f}"
            f"{m['sharp']:>8.4f}"
            f"{m['mace']:>8.4f}"
            f"{m['ks_p']:>8.4f}"
            f"{m['rmse']:>8.4f}"
        )


if __name__ == "__main__":
    main()
