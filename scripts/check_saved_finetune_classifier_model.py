"""Reload and validate a saved TabPFN fine-tuned classifier artifact.

By default this script reads the latest saved-model row from the fine-tune trial
results CSV, reconstructs the same sampled dataset and holdout split, reloads the
`.tabpfn_fit` artifact, and evaluates it on the original holdout partition.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


def configure_import_path() -> Path | None:
    work_repo_root = Path(__file__).resolve().parents[1]
    upstream_src = work_repo_root.parent / "TabPFN-upstream" / "src"
    if upstream_src.exists():
        sys.path.insert(0, str(upstream_src))
        return upstream_src
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a saved TabPFN fine-tuned classifier model")
    parser.add_argument(
        "--results-log-path",
        type=Path,
        default=Path("outputs/current/tables/tabpfn_finetune_trial_results.csv"),
        help="CSV containing saved fine-tune trial metadata",
    )
    parser.add_argument(
        "--check-log-path",
        type=Path,
        default=Path("outputs/current/tables/tabpfn_finetune_reload_checks.csv"),
        help="CSV file to append reload validation results to",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional explicit .tabpfn_fit path. If omitted, use the latest saved-path row from the results log.",
    )
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args()


def resolve_repo_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    repo_root = Path(__file__).resolve().parents[1]
    return repo_root / path


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
                writer.writerow({field: existing_row.get(field, "") for field in merged_fieldnames})

    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fieldnames)
        writer.writerow({field: row.get(field, "") for field in merged_fieldnames})


def load_latest_saved_trial(results_log_path: Path) -> dict[str, str]:
    with results_log_path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    saved_rows = [row for row in rows if row.get("save_path")]
    if not saved_rows:
        raise ValueError(f"No rows with save_path found in {results_log_path}")
    return saved_rows[-1]


def reconstruct_split(data_path: Path, target_col: str, rows: int, seed: int):
    df = pd.read_csv(data_path)
    sampled_parts = []
    for _, group in df.groupby(target_col, sort=False):
        group_rows = max(1, round(rows * len(group) / len(df)))
        sampled_parts.append(group.sample(n=min(group_rows, len(group)), random_state=seed))
    sampled = pd.concat(sampled_parts, axis=0)
    sampled = sampled.sample(n=min(rows, len(sampled)), random_state=seed)
    y = sampled[target_col].to_numpy()
    X = pd.get_dummies(sampled.drop(columns=[target_col]), drop_first=False).to_numpy()
    return train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)


def main() -> None:
    args = parse_args()
    upstream_src = configure_import_path()
    results_log_path = resolve_repo_path(args.results_log_path)
    check_log_path = resolve_repo_path(args.check_log_path)

    trial_row = load_latest_saved_trial(results_log_path)
    save_path = resolve_repo_path(args.model_path) if args.model_path else Path(trial_row["save_path"])
    data_path = Path(trial_row["data_path"])
    target_col = trial_row["target_col"]
    rows = int(trial_row["rows"])
    seed = int(trial_row["seed"])

    from tabpfn import TabPFNClassifier
    from tabpfn.finetune_utils import clone_model_for_evaluation
    from tabpfn.model_loading import load_fitted_tabpfn_model

    X_train, X_test, y_train, y_test = reconstruct_split(data_path, target_col, rows, seed)
    loaded_model = load_fitted_tabpfn_model(save_path, device=args.device)
    eval_config = {
        "ignore_pretraining_limits": True,
        "device": args.device,
        "n_estimators": int(trial_row["n_estimators"]),
        "random_state": seed,
        "inference_precision": getattr(__import__("torch"), "float32"),
        "inference_config": {"SUBSAMPLE_SAMPLES": int(trial_row["context_samples"])}
    }
    eval_model = clone_model_for_evaluation(loaded_model, eval_config, TabPFNClassifier)
    eval_model.fit(X_train, y_train)
    probabilities = eval_model.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, probabilities[:, 1])
    ll = log_loss(y_test, probabilities)

    result_row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source_results_log": str(results_log_path),
        "model_path": str(save_path),
        "data_path": str(data_path),
        "target_col": target_col,
        "rows": rows,
        "seed": seed,
        "device": args.device,
        "upstream_src": str(upstream_src) if upstream_src is not None else "",
        "reload_eval_roc_auc": f"{roc_auc:.6f}",
        "reload_eval_log_loss": f"{ll:.6f}",
        "logged_post_step_roc_auc": trial_row.get("post_step_roc_auc", ""),
        "logged_post_step_log_loss": trial_row.get("post_step_log_loss", ""),
    }
    append_result_row(check_log_path, result_row)

    print("=== Saved TabPFN Fine-Tune Reload Check ===")
    print(f"model_path={save_path}")
    print(f"data_path={data_path}")
    print(f"target_col={target_col}")
    print(f"rows={rows} seed={seed}")
    print(f"device={args.device}")
    if upstream_src is not None:
        print(f"upstream_src={upstream_src}")
    print(f"check_log_path={check_log_path}")
    print(f"reload_eval roc_auc={roc_auc:.4f} log_loss={ll:.4f}")
    if trial_row.get("post_step_roc_auc") and trial_row.get("post_step_log_loss"):
        print(
            "logged_post_step_eval "
            f"roc_auc={float(trial_row['post_step_roc_auc']):.4f} "
            f"log_loss={float(trial_row['post_step_log_loss']):.4f}"
        )


if __name__ == "__main__":
    main()