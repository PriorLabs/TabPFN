"""Stage A pilot runner for insurance domain fine-tuning comparisons.

This executes a controlled first-pass experiment from
`docs/reports/INSURANCE_DOMAIN_FINETUNING_METHOD_PROTOCOL.md`:
- held-out target insurance dataset
- domain fine-tune pool from other insurance datasets
- model arms: raw TabPFN, domain-fine-tuned TabPFN, GLM, RandomForest,
  and CatBoost when available

Outputs are appended to:
- outputs/current/tables/domain_finetune_study_runs.csv
"""

from __future__ import annotations

import argparse
import csv
import resource
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    target_col: str


DATASETS = {
    "eudirectlapse": DatasetSpec(
        name="eudirectlapse",
        path=Path("data/raw/eudirectlapse.csv"),
        target_col="lapse",
    ),
    "coil2000": DatasetSpec(
        name="coil2000",
        path=Path("data/raw/coil2000.csv"),
        target_col="CARAVAN",
    ),
    "ausprivauto0405": DatasetSpec(
        name="ausprivauto0405",
        path=Path("data/raw/ausprivauto0405.csv"),
        target_col="ClaimOcc",
    ),
    "freMTPL2freq_binary": DatasetSpec(
        name="freMTPL2freq_binary",
        path=Path("data/raw/freMTPL2freq_binary.csv"),
        target_col="ClaimIndicator",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage A insurance domain fine-tune pilot")
    parser.add_argument(
        "--target-dataset",
        choices=list(DATASETS.keys()),
        default="eudirectlapse",
        help="Held-out target dataset for evaluation",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-rows", type=int, default=4000)
    parser.add_argument("--pool-rows-per-dataset", type=int, default=2000)
    parser.add_argument("--test-size", type=float, default=0.3)
    parser.add_argument("--tabpfn-device", type=str, default="cpu")
    parser.add_argument("--tabpfn-context-samples", type=int, default=64)
    parser.add_argument("--tabpfn-n-estimators", type=int, default=2)
    parser.add_argument("--tabpfn-max-finetune-steps", type=int, default=1)
    parser.add_argument(
        "--log-path",
        type=Path,
        default=Path("outputs/current/tables/domain_finetune_study_runs.csv"),
    )
    parser.add_argument(
        "--logbook-path",
        type=Path,
        default=Path("outputs/current/logs/domain_finetune_logbook.md"),
        help="Markdown logbook file to append run interpretation and notes",
    )
    parser.add_argument(
        "--observations",
        type=str,
        default="",
        help="Free-text observations from this run",
    )
    parser.add_argument(
        "--comments",
        type=str,
        default="",
        help="Free-text analyst comments for this run",
    )
    parser.add_argument("--prefer-upstream-src", action="store_true", default=True)
    parser.add_argument("--no-prefer-upstream-src", dest="prefer_upstream_src", action="store_false")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def configure_import_path(prefer_upstream_src: bool) -> Path | None:
    upstream_src = repo_root().parent / "TabPFN-upstream" / "src"
    if prefer_upstream_src and upstream_src.exists():
        sys.path.insert(0, str(upstream_src))
        return upstream_src
    return None


def resolve_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return repo_root() / path


def read_and_sample(spec: DatasetSpec, rows: int, seed: int) -> pd.DataFrame:
    path = resolve_path(spec.path)
    frame = pd.read_csv(path)
    if spec.target_col not in frame.columns:
        raise ValueError(f"Target column '{spec.target_col}' not in {path}")
    frame = frame.dropna(subset=[spec.target_col]).copy()

    class_counts = frame[spec.target_col].value_counts(dropna=False)
    if class_counts.min() < 2 or len(class_counts) < 2:
        sampled = frame.sample(n=min(rows, len(frame)), random_state=seed)
    else:
        sampled_parts = []
        for _, group in frame.groupby(spec.target_col, sort=False):
            n_group = max(1, round(rows * len(group) / len(frame)))
            sampled_parts.append(group.sample(n=min(n_group, len(group)), random_state=seed))
        sampled = pd.concat(sampled_parts, axis=0)
        sampled = sampled.sample(n=min(rows, len(sampled)), random_state=seed)
    return sampled.reset_index(drop=True)


def safe_split(X: np.ndarray, y: np.ndarray, test_size: float, seed: int):
    _, counts = np.unique(y, return_counts=True)
    stratify_target = y if len(counts) > 1 and counts.min() >= 2 else None
    return train_test_split(X, y, test_size=test_size, random_state=seed, stratify=stratify_target)


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        mask = (y_prob >= lo) & (y_prob < hi if i < bins - 1 else y_prob <= hi)
        if not np.any(mask):
            continue
        conf = float(np.mean(y_prob[mask]))
        acc = float(np.mean(y_true[mask]))
        ece += abs(acc - conf) * (np.sum(mask) / len(y_true))
    return float(ece)


def ensure_binary_int(y: pd.Series | np.ndarray) -> np.ndarray:
    vals = np.asarray(y)
    if vals.dtype.kind in {"i", "u", "b"}:
        uniq = np.unique(vals)
        if len(uniq) == 2:
            return vals.astype(int)
    unique_vals = pd.Series(vals).dropna().unique().tolist()
    if len(unique_vals) != 2:
        raise ValueError(f"Expected binary target, found classes={unique_vals}")
    mapper = {unique_vals[0]: 0, unique_vals[1]: 1}
    return np.array([mapper[v] for v in vals], dtype=int)


def evaluate_probs(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y_prob = np.clip(y_prob, 1e-8, 1.0 - 1e-8)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(np.mean((y_prob - y_true) ** 2)),
        "log_loss": float(log_loss(y_true, y_prob)),
        "ece": ece_score(y_true, y_prob),
    }


def append_rows(log_path: Path, rows: list[dict[str, Any]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if not log_path.exists():
        with log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return

    with log_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        existing = list(reader)
        fieldnames = reader.fieldnames or []

    merged = list(fieldnames)
    for row in rows:
        for key in row.keys():
            if key not in merged:
                merged.append(key)

    if merged != fieldnames:
        with log_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=merged)
            writer.writeheader()
            for old_row in existing:
                writer.writerow({k: old_row.get(k, "") for k in merged})

    with log_path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged)
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in merged})


def append_logbook_entry(
    logbook_path: Path,
    args: argparse.Namespace,
    target_dataset: str,
    run_rows: list[dict[str, Any]],
) -> None:
    logbook_path.parent.mkdir(parents=True, exist_ok=True)

    metric_rows = [row for row in run_rows if row.get("roc_auc", "") != ""]
    raw_row = next((row for row in metric_rows if row.get("model_variant") == "raw"), None)
    tuned_row = next((row for row in metric_rows if row.get("model_variant") == "domain_finetuned"), None)

    interpretation_lines: list[str] = []
    if raw_row is not None and tuned_row is not None:
        d_roc = float(tuned_row["roc_auc"]) - float(raw_row["roc_auc"])
        d_pr = float(tuned_row["pr_auc"]) - float(raw_row["pr_auc"])
        d_brier = float(tuned_row["brier"]) - float(raw_row["brier"])
        d_logloss = float(tuned_row["log_loss"]) - float(raw_row["log_loss"])
        interpretation_lines.append(
            f"- Domain-finetuned minus raw TabPFN: ROC AUC {d_roc:+.4f}, PR AUC {d_pr:+.4f}, Brier {d_brier:+.4f}, LogLoss {d_logloss:+.4f}."
        )
        if d_brier < 0 and d_logloss < 0:
            interpretation_lines.append("- Primary calibration endpoints improved for domain fine-tuned TabPFN on this target.")
        elif d_brier > 0 and d_logloss > 0:
            interpretation_lines.append("- Primary calibration endpoints degraded for domain fine-tuned TabPFN on this target.")
        else:
            interpretation_lines.append("- Mixed calibration endpoint movement; keep this target in follow-up runs.")
    else:
        interpretation_lines.append("- Could not compute raw-vs-domain-finetuned delta for this run.")

    lines = [
        f"## Run {datetime.now(timezone.utc).isoformat()}",
        "",
        "### Configuration",
        f"- Stage: A",
        f"- Target dataset: {target_dataset}",
        f"- Seed: {args.seed}",
        f"- Target rows: {args.target_rows}",
        f"- Pool rows per dataset: {args.pool_rows_per_dataset}",
        f"- Device/context/steps: {args.tabpfn_device}/{args.tabpfn_context_samples}/{args.tabpfn_max_finetune_steps}",
        "",
        "### Results",
        "| Model Variant | ROC AUC | PR AUC | Brier | LogLoss | ECE | Notes |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in run_rows:
        roc = row.get("roc_auc", "")
        pr = row.get("pr_auc", "")
        brier = row.get("brier", "")
        ll = row.get("log_loss", "")
        ece = row.get("ece", "")
        notes = row.get("notes", "")
        lines.append(
            "| "
            f"{row.get('model_variant','')} | "
            f"{f'{float(roc):.4f}' if roc != '' else ''} | "
            f"{f'{float(pr):.4f}' if pr != '' else ''} | "
            f"{f'{float(brier):.4f}' if brier != '' else ''} | "
            f"{f'{float(ll):.4f}' if ll != '' else ''} | "
            f"{f'{float(ece):.4f}' if ece != '' else ''} | "
            f"{notes} |"
        )

    lines.append("")
    lines.append("### Interpretation")
    lines.extend(interpretation_lines)
    lines.append("")
    lines.append("### Observations")
    lines.append(f"- {args.observations if args.observations else 'No observations supplied at runtime.'}")
    lines.append("")
    lines.append("### Comments")
    lines.append(f"- {args.comments if args.comments else 'No comments supplied at runtime.'}")
    lines.append("\n")

    with logbook_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def run_tabpfn_raw(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    tabpfn_device: str,
    n_estimators: int,
    seed: int,
) -> np.ndarray:
    from tabpfn import TabPFNClassifier

    model = TabPFNClassifier(
        ignore_pretraining_limits=True,
        device=tabpfn_device,
        n_estimators=n_estimators,
        random_state=seed,
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    return probs


def run_tabpfn_finetuned(
    X_pool: np.ndarray,
    y_pool: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    tabpfn_device: str,
    context_samples: int,
    n_estimators: int,
    max_finetune_steps: int,
    seed: int,
) -> tuple[np.ndarray, int]:
    import torch
    from torch.optim import Adam
    from torch.utils.data import DataLoader

    from tabpfn import TabPFNClassifier
    from tabpfn.finetune_utils import clone_model_for_evaluation
    from tabpfn.utils import meta_dataset_collator

    cfg = {
        "ignore_pretraining_limits": True,
        "device": tabpfn_device,
        "n_estimators": n_estimators,
        "random_state": seed,
        "inference_precision": torch.float32,
    }
    model = TabPFNClassifier(**cfg, fit_mode="batched", differentiable_input=False)
    model._initialize_model_variables()

    optimizer = Adam(model.model_.parameters(), lr=1e-5)
    loss_fn = torch.nn.CrossEntropyLoss()

    splitter = lambda features, target: safe_split(features, target, test_size=0.3, seed=seed)
    datasets = model.get_preprocessed_datasets(
        X_pool,
        y_pool,
        splitter,
        min(context_samples, len(X_pool)),
    )
    loader = DataLoader(datasets, batch_size=1, collate_fn=meta_dataset_collator)

    steps = 0
    for xb, xvb, yb, yvb, cix, conf in loader:
        if len(np.unique(yb)) != len(np.unique(yvb)):
            continue
        optimizer.zero_grad()
        model.fit_from_preprocessed(xb, yb, cix, conf)
        logits = model.forward(xvb, return_logits=True)
        loss = loss_fn(logits, yvb.to(tabpfn_device))
        loss.backward()
        optimizer.step()
        steps += 1
        if steps >= max_finetune_steps:
            break

    eval_cfg = {**cfg, "inference_config": {"SUBSAMPLE_SAMPLES": context_samples}}
    eval_model = clone_model_for_evaluation(model, eval_cfg, TabPFNClassifier)
    eval_model.fit(X_train, y_train)
    probs = eval_model.predict_proba(X_test)[:, 1]
    return probs, steps


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    upstream_src = configure_import_path(args.prefer_upstream_src)

    target_spec = DATASETS[args.target_dataset]
    pool_specs = [spec for key, spec in DATASETS.items() if key != args.target_dataset]

    target_df = read_and_sample(target_spec, args.target_rows, args.seed)
    pool_frames = [read_and_sample(spec, args.pool_rows_per_dataset, args.seed) for spec in pool_specs]

    y_target = ensure_binary_int(target_df[target_spec.target_col])
    X_target_df = target_df.drop(columns=[target_spec.target_col])

    X_pool_frames = []
    y_pool_parts = []
    for spec, frame in zip(pool_specs, pool_frames):
        y_pool_parts.append(ensure_binary_int(frame[spec.target_col]))
        X_pool_frames.append(frame.drop(columns=[spec.target_col]))

    # Fit encoder on pooled domain + target train universe to keep feature alignment.
    joint_df = pd.concat([X_target_df] + X_pool_frames, axis=0, ignore_index=True)
    joint_encoded = pd.get_dummies(joint_df, drop_first=False)
    joint_encoded = joint_encoded.fillna(0)

    X_target_enc = joint_encoded.iloc[: len(X_target_df), :].to_numpy(dtype=np.float32)
    X_pool_enc = joint_encoded.iloc[len(X_target_df) :, :].to_numpy(dtype=np.float32)
    y_pool = np.concatenate(y_pool_parts, axis=0)

    X_train, X_test, y_train, y_test = safe_split(X_target_enc, y_target, args.test_size, args.seed)

    run_rows: list[dict[str, Any]] = []

    # GLM baseline
    t0 = time.perf_counter()
    glm = LogisticRegression(max_iter=1000, solver="liblinear", random_state=args.seed)
    glm.fit(X_train, y_train)
    glm_prob = glm.predict_proba(X_test)[:, 1]
    glm_metrics = evaluate_probs(y_test, glm_prob)
    run_rows.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "A",
            "target_dataset": target_spec.name,
            "seed": args.seed,
            "model_family": "glm",
            "model_variant": "logistic_regression",
            "target_rows": len(X_target_df),
            "pool_rows": len(X_pool_enc),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "tabpfn_device": args.tabpfn_device,
            "tabpfn_context_samples": args.tabpfn_context_samples,
            "tabpfn_max_finetune_steps": args.tabpfn_max_finetune_steps,
            "fine_tune_steps_executed": "",
            **glm_metrics,
            "fit_predict_wall_time_sec": f"{time.perf_counter() - t0:.6f}",
            "upstream_src": str(upstream_src) if upstream_src is not None else "",
            "notes": "",
        }
    )

    # RandomForest baseline
    t0 = time.perf_counter()
    rf = RandomForestClassifier(n_estimators=300, random_state=args.seed, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_prob = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate_probs(y_test, rf_prob)
    run_rows.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "A",
            "target_dataset": target_spec.name,
            "seed": args.seed,
            "model_family": "tree",
            "model_variant": "random_forest",
            "target_rows": len(X_target_df),
            "pool_rows": len(X_pool_enc),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "tabpfn_device": args.tabpfn_device,
            "tabpfn_context_samples": args.tabpfn_context_samples,
            "tabpfn_max_finetune_steps": args.tabpfn_max_finetune_steps,
            "fine_tune_steps_executed": "",
            **rf_metrics,
            "fit_predict_wall_time_sec": f"{time.perf_counter() - t0:.6f}",
            "upstream_src": str(upstream_src) if upstream_src is not None else "",
            "notes": "",
        }
    )

    # CatBoost baseline (optional)
    try:
        from catboost import CatBoostClassifier

        t0 = time.perf_counter()
        cb = CatBoostClassifier(
            random_seed=args.seed,
            loss_function="Logloss",
            verbose=False,
            iterations=300,
            depth=6,
        )
        cb.fit(X_train, y_train)
        cb_prob = cb.predict_proba(X_test)[:, 1]
        cb_metrics = evaluate_probs(y_test, cb_prob)
        run_rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": "A",
                "target_dataset": target_spec.name,
                "seed": args.seed,
                "model_family": "tree",
                "model_variant": "catboost",
                "target_rows": len(X_target_df),
                "pool_rows": len(X_pool_enc),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "tabpfn_device": args.tabpfn_device,
                "tabpfn_context_samples": args.tabpfn_context_samples,
                "tabpfn_max_finetune_steps": args.tabpfn_max_finetune_steps,
                "fine_tune_steps_executed": "",
                **cb_metrics,
                "fit_predict_wall_time_sec": f"{time.perf_counter() - t0:.6f}",
                "upstream_src": str(upstream_src) if upstream_src is not None else "",
                "notes": "",
            }
        )
    except Exception as exc:
        run_rows.append(
            {
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "stage": "A",
                "target_dataset": target_spec.name,
                "seed": args.seed,
                "model_family": "tree",
                "model_variant": "catboost",
                "target_rows": len(X_target_df),
                "pool_rows": len(X_pool_enc),
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "tabpfn_device": args.tabpfn_device,
                "tabpfn_context_samples": args.tabpfn_context_samples,
                "tabpfn_max_finetune_steps": args.tabpfn_max_finetune_steps,
                "fine_tune_steps_executed": "",
                "roc_auc": "",
                "pr_auc": "",
                "brier": "",
                "log_loss": "",
                "ece": "",
                "fit_predict_wall_time_sec": "",
                "upstream_src": str(upstream_src) if upstream_src is not None else "",
                "notes": f"catboost_not_available: {exc}",
            }
        )

    # Raw TabPFN
    t0 = time.perf_counter()
    raw_prob = run_tabpfn_raw(
        X_train,
        y_train,
        X_test,
        tabpfn_device=args.tabpfn_device,
        n_estimators=args.tabpfn_n_estimators,
        seed=args.seed,
    )
    raw_metrics = evaluate_probs(y_test, raw_prob)
    run_rows.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "A",
            "target_dataset": target_spec.name,
            "seed": args.seed,
            "model_family": "tabpfn",
            "model_variant": "raw",
            "target_rows": len(X_target_df),
            "pool_rows": len(X_pool_enc),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "tabpfn_device": args.tabpfn_device,
            "tabpfn_context_samples": args.tabpfn_context_samples,
            "tabpfn_max_finetune_steps": args.tabpfn_max_finetune_steps,
            "fine_tune_steps_executed": "",
            **raw_metrics,
            "fit_predict_wall_time_sec": f"{time.perf_counter() - t0:.6f}",
            "upstream_src": str(upstream_src) if upstream_src is not None else "",
            "notes": "",
        }
    )

    # Domain-fine-tuned TabPFN
    t0 = time.perf_counter()
    tuned_prob, steps = run_tabpfn_finetuned(
        X_pool_enc,
        y_pool,
        X_train,
        y_train,
        X_test,
        tabpfn_device=args.tabpfn_device,
        context_samples=args.tabpfn_context_samples,
        n_estimators=args.tabpfn_n_estimators,
        max_finetune_steps=args.tabpfn_max_finetune_steps,
        seed=args.seed,
    )
    tuned_metrics = evaluate_probs(y_test, tuned_prob)
    run_rows.append(
        {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "stage": "A",
            "target_dataset": target_spec.name,
            "seed": args.seed,
            "model_family": "tabpfn",
            "model_variant": "domain_finetuned",
            "target_rows": len(X_target_df),
            "pool_rows": len(X_pool_enc),
            "train_rows": len(X_train),
            "test_rows": len(X_test),
            "tabpfn_device": args.tabpfn_device,
            "tabpfn_context_samples": args.tabpfn_context_samples,
            "tabpfn_max_finetune_steps": args.tabpfn_max_finetune_steps,
            "fine_tune_steps_executed": steps,
            **tuned_metrics,
            "fit_predict_wall_time_sec": f"{time.perf_counter() - t0:.6f}",
            "upstream_src": str(upstream_src) if upstream_src is not None else "",
            "notes": "",
        }
    )

    max_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    for row in run_rows:
        row["max_rss_bytes"] = max_rss_bytes
        row["total_run_wall_time_sec"] = f"{time.perf_counter() - start:.6f}"

    log_path = resolve_path(args.log_path)
    append_rows(log_path, run_rows)
    append_logbook_entry(
        logbook_path=resolve_path(args.logbook_path),
        args=args,
        target_dataset=target_spec.name,
        run_rows=run_rows,
    )

    print("=== Domain Fine-Tuning Stage A Pilot ===")
    print(f"target_dataset={target_spec.name}")
    print(f"target_rows={len(X_target_df)} pool_rows={len(X_pool_enc)}")
    print(f"train_rows={len(X_train)} test_rows={len(X_test)}")
    print(f"tabpfn_device={args.tabpfn_device} context={args.tabpfn_context_samples} finetune_steps={args.tabpfn_max_finetune_steps}")
    print(f"log_path={log_path}")
    print(f"logbook_path={resolve_path(args.logbook_path)}")
    print("\nModel results (ROC AUC | PR AUC | Brier | LogLoss | ECE):")
    for row in run_rows:
        if row.get("roc_auc", "") == "":
            print(f"- {row['model_variant']}: skipped ({row['notes']})")
            continue
        print(
            f"- {row['model_variant']}: "
            f"{float(row['roc_auc']):.4f} | {float(row['pr_auc']):.4f} | "
            f"{float(row['brier']):.4f} | {float(row['log_loss']):.4f} | {float(row['ece']):.4f}"
        )


if __name__ == "__main__":
    main()
