"""Finetune pilot: measures per-sample time for the TabPFN fine-tuning loop.

This script runs a minimal finetuning loop matching the example in
`examples/finetune_classifier.py` to measure training time per sample.

Usage:
  python scripts/finetune_pilot.py --n_pilot 2000 --device cpu

Notes:
- Set `HF_TOKEN` or pass `--hf_token` if you need access to gated models.
- Installs: `pip install tabpfn torch pandas scikit-learn tqdm`
"""

import argparse
import os
import time
import textwrap
import numpy as np
import torch
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

try:
    from tabpfn import TabPFNClassifier
except Exception:
    TabPFNClassifier = None


def synthetic_data(n_samples=2000, n_features=20, n_classes=2, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=min(10, n_features),
        n_redundant=0,
        n_classes=n_classes,
        random_state=random_state,
    )
    return X.astype(np.float32), y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_pilot", type=int, default=2000)
    parser.add_argument("--n_features", type=int, default=20)
    parser.add_argument("--n_classes", type=int, default=2)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--n_estimators", type=int, default=2)
    parser.add_argument("--hf_token", type=str, default=None)
    args = parser.parse_args()

    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if TabPFNClassifier is None:
        print("TabPFN not installed. Install with: pip install tabpfn")
        return

    print(f"Building synthetic dataset: n={args.n_pilot}, features={args.n_features}")
    X, y = synthetic_data(args.n_pilot, args.n_features, args.n_classes)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    train_n = int(len(X_train))

    # Finetuning config (aligned with upstream example)
    finetune_config = {
        "epochs": args.epochs,
        "learning_rate": 1e-5,
        "meta_batch_size": 1,
    }

    batch_size = args.batch_size or min(10000, int(len(X_train)))

    clf = TabPFNClassifier(
        ignore_pretraining_limits=True,
        device=device,
        n_estimators=args.n_estimators,
        random_state=42,
        inference_precision=torch.float32,
        fit_mode="batched",
        differentiable_input=False,
    )
    # initialize variables (matches example)
    try:
        clf._initialize_model_variables()
    except Exception:
        # Some backends may initialize lazily; continue
        pass

    # Build training datasets using TabPFN finetuning helper
    try:
        from tabpfn.finetuning.data_util import get_preprocessed_dataset_chunks
    except Exception as e:
        print(f"Failed to import finetuning helpers: {e}")
        return

    split_fn = lambda X, y, stratify=None: train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    try:
        training_datasets = get_preprocessed_dataset_chunks(
            calling_instance=clf,
            X_raw=X_train,
            y_raw=y_train,
            split_fn=split_fn,
            max_data_size=None,
            model_type="classifier",
            equal_split_size=True,
            data_shuffle_seed=42,
            preprocessing_random_state=42,
            shuffle=True,
        )
    except Exception as e:
        print(f"Failed to build preprocessed datasets: {e}")
        return

    from torch.optim import Adam
    from torch.utils.data import DataLoader
    from tabpfn.finetuning.data_util import meta_dataset_collator

    optimizer = Adam(clf.model_.parameters(), lr=finetune_config["learning_rate"]) if hasattr(clf, "model_") else None

    dataloader = DataLoader(training_datasets, batch_size=finetune_config["meta_batch_size"], collate_fn=meta_dataset_collator)

    loss_fn = torch.nn.CrossEntropyLoss()

    total_train_samples = 0
    total_train_time = 0.0

    print(f"Starting finetune pilot on device={device} batches={len(dataloader)} batch_size(meta)={finetune_config['meta_batch_size']}")

    for epoch in range(finetune_config["epochs"]):
        for batch_item in dataloader:
            # The collator returns a ClassifierBatch dataclass
            X_train_batch = batch_item.X_context
            X_test_batch = batch_item.X_query
            y_train_batch = batch_item.y_context
            y_test_batch = batch_item.y_query
            cat_ixs = batch_item.cat_indices
            confs = batch_item.configs

            # y_train_batch is a list (per-estimator) of tensors; compute samples as length of first tensor
            # Determine number of training samples in this batch (per-estimator tensors)
            n_samples = 1
            try:
                first_X = X_train_batch[0]
                if hasattr(first_X, "shape"):
                    n_samples = int(first_X.shape[0])
                else:
                    n_samples = len(first_X)
            except Exception:
                try:
                    n_samples = int(y_train_batch[0].shape[0])
                except Exception:
                    n_samples = 1

            start = time.perf_counter()
            # fit_from_preprocessed performs the internal forward/backward steps for TabPFN
            try:
                clf.fit_from_preprocessed(X_train_batch, y_train_batch, cat_ixs, confs)
            except Exception as e:
                print(f"fit_from_preprocessed failed: {e}")
                return

            # perform an explicit forward+loss+backward+step if a PyTorch model is exposed
            if optimizer is not None and hasattr(clf, "forward"):
                try:
                    preds = clf.forward(X_test_batch, return_logits=True)
                    loss = loss_fn(preds, y_test_batch.to(device))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                except Exception:
                    # Some wrappers may not support this exact flow; ignore
                    pass

            elapsed = time.perf_counter() - start
            total_train_time += elapsed
            total_train_samples += max(1, n_samples)

    if total_train_samples == 0:
        print("No training samples processed; cannot compute timing")
        return

    # Use the actual number of training rows for per-sample estimate to avoid
    # differences in internal batching/chunking representations.
    time_per_sample = total_train_time / max(1, train_n)
    print(f"Finetune pilot: total_time={total_train_time:.2f}s, internal_batches={total_train_samples}, train_rows={train_n}, time_per_sample={time_per_sample:.6f}s")

    # Extrapolate to full run
    N_full = 10_000
    epochs = 10
    estimated_seconds = time_per_sample * N_full * epochs
    estimated_hours = estimated_seconds / 3600.0

    # Provide a small set of hourly rates for quick reference
    example_rates = {
        "RTX4090_spot": 0.35,
        "A100_40GB_spot": 0.9,
        "A100_80GB_spot": 1.4,
        "H100_spot": 3.0,
    }

    print("\nEstimated full finetune (N=10k, epochs=10):")
    print(f"  Estimated GPU hours: {estimated_hours:.2f} hr")
    for name, rate in example_rates.items():
        print(f"  Cost @ {name} (${rate}/hr): ${estimated_hours * rate:.2f}")


if __name__ == "__main__":
    main()
