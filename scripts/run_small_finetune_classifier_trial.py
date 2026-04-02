"""Run a small TabPFN classifier fine-tuning smoke test on a real dataset.

This script is designed for local readiness checks before larger fine-tuning runs.
It prefers the local upstream TabPFN source tree so repo-specific fine-tuning APIs
are available even if the installed package differs.
"""

from __future__ import annotations

import argparse
import csv
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader


def configure_import_path(prefer_upstream_src: bool) -> Path | None:
    work_repo_root = Path(__file__).resolve().parents[1]
    upstream_src = work_repo_root.parent / "TabPFN-upstream" / "src"
    if prefer_upstream_src and upstream_src.exists():
        sys.path.insert(0, str(upstream_src))
        return upstream_src
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small TabPFN classifier fine-tune smoke test")
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/coil2000.csv"),
        help="CSV dataset path, relative to TabPFN-work-scott root unless absolute",
    )
    parser.add_argument("--target-col", type=str, default="CARAVAN")
    parser.add_argument("--rows", type=int, default=300)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--context-samples", type=int, default=64)
    parser.add_argument("--n-estimators", type=int, default=2)
    parser.add_argument("--max-finetune-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--save-path",
        type=Path,
        default=None,
        help="Optional path for the saved fine-tuned model (.tabpfn_fit). Defaults to outputs/current/models/<timestamp>_...tabpfn_fit",
    )
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("outputs/current/tables/tabpfn_finetune_trial_results.csv"),
        help="CSV file to append structured trial results to",
    )
    parser.add_argument("--prefer-upstream-src", action="store_true", default=True)
    parser.add_argument("--no-prefer-upstream-src", dest="prefer_upstream_src", action="store_false")
    return parser.parse_args()


def resolve_data_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / path


def resolve_log_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / path


def resolve_save_path(path: Path | None, target_col: str, device: str, rows: int) -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    if path is not None:
        return path if path.is_absolute() else repo_root / path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    file_name = f"{timestamp}_tabpfn_finetune_{target_col}_{device}_{rows}.tabpfn_fit"
    return repo_root / "outputs" / "current" / "models" / file_name


def load_subset(data_path: Path, target_col: str, rows: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(data_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {data_path}")
    sampled_parts = []
    for _, group in df.groupby(target_col, sort=False):
        group_rows = max(1, round(rows * len(group) / len(df)))
        sampled_parts.append(group.sample(n=min(group_rows, len(group)), random_state=seed))
    sampled = pd.concat(sampled_parts, axis=0)
    sampled = sampled.sample(n=min(rows, len(sampled)), random_state=seed)
    y = sampled[target_col].to_numpy()
    X = pd.get_dummies(sampled.drop(columns=[target_col]), drop_first=False).to_numpy()
    return X, y


def safe_train_test_split(features: np.ndarray, target: np.ndarray, seed: int):
    _, counts = np.unique(target, return_counts=True)
    stratify_target = target if len(counts) > 1 and counts.min() >= 2 else None
    return train_test_split(
        features,
        target,
        test_size=0.3,
        random_state=seed,
        stratify=stratify_target,
    )


def evaluate_model(
    classifier: Any,
    classifier_config: dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    context_samples: int,
):
    from tabpfn import TabPFNClassifier
    from tabpfn.finetune_utils import clone_model_for_evaluation

    eval_config = {
        **classifier_config,
        "inference_config": {"SUBSAMPLE_SAMPLES": context_samples},
    }
    eval_classifier = clone_model_for_evaluation(classifier, eval_config, TabPFNClassifier)
    eval_classifier.fit(X_train, y_train)
    probabilities = eval_classifier.predict_proba(X_test)
    return roc_auc_score(y_test, probabilities[:, 1]), log_loss(y_test, probabilities)


def append_result_row(log_path: Path, row: dict[str, Any]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        with log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with log_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        existing_rows = list(reader)
        existing_fieldnames = reader.fieldnames or []

    merged_fieldnames = list(existing_fieldnames)
    for key in row.keys():
        if key not in merged_fieldnames:
            merged_fieldnames.append(key)

    if merged_fieldnames != existing_fieldnames:
        with log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
            writer.writeheader()
            for existing_row in existing_rows:
                normalized_row = {field: existing_row.get(field, "") for field in merged_fieldnames}
                writer.writerow(normalized_row)

    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
        normalized_row = {field: row.get(field, "") for field in merged_fieldnames}
        writer.writerow(normalized_row)


def main() -> None:
    args = parse_args()
    wall_start = time.perf_counter()
    upstream_src = configure_import_path(args.prefer_upstream_src)

    from tabpfn import TabPFNClassifier, save_fitted_tabpfn_model
    from tabpfn.utils import meta_dataset_collator

    data_path = resolve_data_path(args.data_path)
    log_path = resolve_log_path(args.log_path)
    save_path = resolve_save_path(args.save_path, args.target_col, args.device, args.rows)
    X, y = load_subset(data_path, args.target_col, args.rows, args.seed)
    X_train, X_test, y_train, y_test = safe_train_test_split(X, y, args.seed)

    classifier_config = {
        "ignore_pretraining_limits": True,
        "device": args.device,
        "n_estimators": args.n_estimators,
        "random_state": args.seed,
        "inference_precision": torch.float32,
    }

    classifier = TabPFNClassifier(
        **classifier_config,
        fit_mode="batched",
        differentiable_input=False,
    )
    classifier._initialize_model_variables()
    optimizer = Adam(classifier.model_.parameters(), lr=1e-5)
    loss_function = torch.nn.CrossEntropyLoss()

    splitter = lambda features, target: safe_train_test_split(features, target, args.seed)
    training_datasets = classifier.get_preprocessed_datasets(
        X_train,
        y_train,
        splitter,
        min(args.context_samples, len(X_train)),
    )
    finetuning_dataloader = DataLoader(
        training_datasets,
        batch_size=1,
        collate_fn=meta_dataset_collator,
    )

    roc_before, log_loss_before = evaluate_model(
        classifier,
        classifier_config,
        X_train,
        y_train,
        X_test,
        y_test,
        args.context_samples,
    )

    steps_executed = 0
    last_loss = None
    for X_train_batch, X_test_batch, y_train_batch, y_test_batch, cat_ixs, confs in finetuning_dataloader:
        if len(np.unique(y_train_batch)) != len(np.unique(y_test_batch)):
            continue
        optimizer.zero_grad()
        classifier.fit_from_preprocessed(X_train_batch, y_train_batch, cat_ixs, confs)
        predictions = classifier.forward(X_test_batch, return_logits=True)
        loss = loss_function(predictions, y_test_batch.to(args.device))
        loss.backward()
        optimizer.step()
        last_loss = float(loss.item())
        steps_executed += 1
        if steps_executed >= args.max_finetune_steps:
            break

    roc_after, log_loss_after = evaluate_model(
        classifier,
        classifier_config,
        X_train,
        y_train,
        X_test,
        y_test,
        args.context_samples,
    )

    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_fitted_tabpfn_model(classifier, save_path)

    wall_time_sec = time.perf_counter() - wall_start
    max_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    result_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script_name": Path(__file__).name,
        "data_path": str(data_path),
        "target_col": args.target_col,
        "rows": len(X),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "device": args.device,
        "prefer_upstream_src": args.prefer_upstream_src,
        "upstream_src": str(upstream_src) if upstream_src is not None else "",
        "context_samples": args.context_samples,
        "n_estimators": args.n_estimators,
        "max_finetune_steps": args.max_finetune_steps,
        "seed": args.seed,
        "save_path": str(save_path),
        "finetune_steps_executed": steps_executed,
        "last_step_loss": f"{last_loss:.6f}" if last_loss is not None else "",
        "initial_roc_auc": f"{roc_before:.6f}",
        "initial_log_loss": f"{log_loss_before:.6f}",
        "post_step_roc_auc": f"{roc_after:.6f}",
        "post_step_log_loss": f"{log_loss_after:.6f}",
        "wall_time_sec": f"{wall_time_sec:.6f}",
        "max_rss_bytes": max_rss_bytes,
    }
    append_result_row(log_path, result_row)

    print("=== Small TabPFN Classifier Fine-Tune Trial ===")
    print(f"data_path={data_path}")
    print(f"target_col={args.target_col}")
    print(f"rows={len(X)} train_rows={len(X_train)} test_rows={len(X_test)}")
    print(f"device={args.device}")
    print(f"prefer_upstream_src={args.prefer_upstream_src}")
    if upstream_src is not None:
        print(f"upstream_src={upstream_src}")
    print(f"log_path={log_path}")
    print(f"save_path={save_path}")
    print(f"context_samples={args.context_samples} n_estimators={args.n_estimators}")
    print(f"initial_eval roc_auc={roc_before:.4f} log_loss={log_loss_before:.4f}")
    print(f"finetune_steps_executed={steps_executed}")
    if last_loss is not None:
        print(f"last_step_loss={last_loss:.4f}")
    print(f"post_step_eval roc_auc={roc_after:.4f} log_loss={log_loss_after:.4f}")
    print(f"wall_time_sec={wall_time_sec:.4f}")
    print(f"max_rss_bytes={max_rss_bytes}")


if __name__ == "__main__":
    main()