"""Pilot timing script for TabPFN fine-tuning.

Usage examples:
  python scripts/pilot_timing.py --n_pilot 1000
  python scripts/pilot_timing.py --csv data/processed/my_features.csv --target target_col --n_pilot 500

Notes:
- Installs: pip install tabpfn torch pandas scikit-learn
- If no CSV provided, a synthetic dataset will be used.
"""

import argparse
import time
import os
import re
import textwrap
import numpy as np
import pandas as pd
import torch

try:
    from tabpfn import TabPFNClassifier
except Exception:
    TabPFNClassifier = None

from sklearn.datasets import make_classification


def load_data_from_csv(path, target_col=None, nrows=None):
    df = pd.read_csv(path, nrows=nrows)
    if target_col is None:
        target_col = df.columns[-1]
    y = df[target_col].values
    X = df.drop(columns=[target_col])

    # Simple preprocessing: factorize object columns
    for col in X.select_dtypes(include=[object, "category"]).columns:
        X[col] = pd.factorize(X[col])[0]

    # Fill missing
    X = X.fillna(0)
    return X.values.astype(np.float32), y


def synthetic_data(n_samples=1000, n_features=20, n_classes=2, random_state=42):
    X, y = make_classification(
        n_samples=n_samples, n_features=n_features, n_informative=min(10, n_features),
        n_redundant=0, n_classes=n_classes, random_state=random_state,
    )
    return X.astype(np.float32), y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, default=None, help="Path to CSV with target column")
    parser.add_argument("--target", type=str, default=None, help="Target column name (if CSV)")
    parser.add_argument("--n_pilot", type=int, default=1000, help="Number of pilot rows to time")
    parser.add_argument("--device", type=str, default=None, help="Device: cuda, cpu, or auto")
    parser.add_argument("--n_estimators", type=int, default=2, help="TabPFN n_estimators")
    parser.add_argument("--hf_token", type=str, default=None, help="Hugging Face token (or set HF_TOKEN env var)")
    args = parser.parse_args()

    # Allow passing a Hugging Face token on the CLI; falls back to HF_TOKEN env var if not provided
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.csv:
        try:
            X, y = load_data_from_csv(args.csv, target_col=args.target)
            print(f"Loaded CSV {args.csv} with shape {X.shape}")
        except Exception as e:
            print(f"Error loading CSV: {e}. Falling back to synthetic data.")
            X, y = synthetic_data(n_samples=args.n_pilot)
    else:
        X, y = synthetic_data(n_samples=max(args.n_pilot, 1000))
        print(f"Using synthetic dataset: X.shape={X.shape}")

    # Sample pilot rows
    rng = np.random.default_rng(42)
    if X.shape[0] >= args.n_pilot:
        idx = rng.choice(np.arange(X.shape[0]), size=args.n_pilot, replace=False)
        X_pilot = X[idx]
        y_pilot = y[idx]
    else:
        X_pilot = X
        y_pilot = y

    if TabPFNClassifier is None:
        print("TabPFN is not installed or failed import. Install with: pip install tabpfn")
        return

    hf_token_present = bool(os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN"))
    if hf_token_present:
        print("Hugging Face token found in environment (using for gated model downloads).")
    else:
        print("No Hugging Face token found in environment. If the model is gated you will need to authenticate to download it.")

    clf = TabPFNClassifier(device=device, n_estimators=args.n_estimators)

    print(f"Running pilot: device={device}, n_pilot={X_pilot.shape[0]}, n_estimators={args.n_estimators}")
    t0 = time.perf_counter()
    try:
        clf.fit(X_pilot, y_pilot)
    except Exception as e:
        err = str(e)
        print(f"Error during TabPFN fit: {err}")

        # Provide actionable instructions if this appears to be a Hugging Face gated-model / auth issue
        if any(tok in err for tok in ("HuggingFace", "gated", "Failed to download", "accept", "Prior-Labs/")):
            m = re.search(r"([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)", err)
            model_url = f"https://huggingface.co/{m.group(1)}" if m else "https://huggingface.co/Prior-Labs/tabpfn_2_6"
            msg = textwrap.dedent(f"""
            It looks like the TabPFN model download failed due to a gated Hugging Face model or missing authentication.

            Quick steps to resolve:
              1) Visit the model page and accept terms: {model_url}
              2) Authenticate locally so the script can download the model:
                 - Install the HF CLI (if needed): `pip install huggingface-hub`
                 - Login interactively: `huggingface-cli login`  OR  `hf auth login`
                 - Or create a read token on https://huggingface.co/settings/tokens and set it in your shell:
                     export HF_TOKEN=hf_xxx   # (zsh/bash)
              3) Re-run this script after completing steps 1 & 2.

            Notes:
              - Ensure you accepted the model's terms on the model page while logged into your HF account.
              - If you need commercial access/licensing, contact Prior Labs: sales@priorlabs.ai
            """)
            print(msg)
        return
    elapsed = time.perf_counter() - t0

    time_per_sample = elapsed / max(1, X_pilot.shape[0])
    print(f"Pilot elapsed: {elapsed:.2f}s — time_per_sample: {time_per_sample:.6f}s")

    # Example cost estimation helper (adjust hourly_rate outside)
    hourly_rate = 1.4  # USD/hr estimated; replace with your chosen provider rate
    N_full = 10_000
    epochs = 10
    estimated_seconds = time_per_sample * N_full * epochs
    estimated_hours = estimated_seconds / 3600.0
    estimated_cost = estimated_hours * hourly_rate

    print("\nEstimated run for: N_full=10_000, epochs=10")
    print(f"Estimated GPU hours: {estimated_hours:.2f} hr")
    print(f"Estimated compute cost (@ ${hourly_rate}/hr): ${estimated_cost:.2f}")


if __name__ == "__main__":
    main()
