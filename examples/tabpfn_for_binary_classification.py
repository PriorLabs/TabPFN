import argparse
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNRegressor

seed = 0
BASELINE_FILE = Path("baseline_preds.npy")


def _set_seeds() -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def _max_diff(a: np.ndarray, b: np.ndarray) -> float:
    return np.max(np.abs(a - b) / (np.abs(a) + 1e-12))


def run_single_mode(fit_mode: str, X_train, X_test, y_train) -> np.ndarray:
    """Runs a single experiment and returns predictions."""
    print(f"\nRunning fit_mode = {fit_mode}")

    _set_seeds()
    reg = TabPFNRegressor(
        fit_mode=fit_mode,
        inference_precision=torch.float64,
        random_state=seed,
        n_estimators=8,
    )
    reg.fit(X_train, y_train)
    preds = reg.predict(X_test)
    return preds


def ensure_baseline(X_train, X_test, y_train):
    """Creates baseline if missing, returns baseline predictions."""
    if BASELINE_FILE.exists():
        print("Loading existing baseline predictions...")
        return np.load(BASELINE_FILE)

    print("Baseline file missing — generating baseline using fit_mode='low_memory'...")
    baseline_preds = run_single_mode("low_memory", X_train, X_test, y_train)

    print("Saving baseline predictions...")
    np.save(BASELINE_FILE, baseline_preds)
    return baseline_preds


def run_experiments(fit_modes):
    # Load data once
    X, y = load_diabetes(return_X_y=True)
    X_train, X_test, y_train, _ = train_test_split(
        X, y, test_size=0.33, random_state=seed
    )

    # Ensure baseline exists
    baseline_preds = ensure_baseline(X_train, X_test, y_train)

    # Loop over modes
    for mode in fit_modes:
        preds = run_single_mode(mode, X_train, X_test, y_train)
        diff = _max_diff(baseline_preds, preds)
        print(f"Max relative diff vs baseline (low_memory): {diff}")

    # Remove baseline after completion
    if BASELINE_FILE.exists():
        print("\nRemoving baseline file...")
        BASELINE_FILE.unlink()
        print("Baseline file removed.")


def main():
    parser = argparse.ArgumentParser(description="Compare multiple TabPFN fit modes.")
    parser.add_argument(
        "--fit-modes",
        nargs="+",
        type=str,
        required=True,
        choices=["low_memory", "fit_preprocessors", "fit_with_cache"],
        help="One or more fit modes (space separated).",
    )
    args = parser.parse_args()

    run_experiments(args.fit_modes)


if __name__ == "__main__":
    main()
