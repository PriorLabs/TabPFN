"""Local evaluation helpers for baselining (renamed from evaluation.py).

Provides regression and classification metrics and a convenience function
`evaluate_models` that detects whether predictions are probabilistic or
deterministic and computes sensible metrics for each model.
"""
import numpy as np
import pandas as pd


def rmse(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean(np.abs(a - b)))


def accuracy(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.mean(a == b))


def f1_score_simple(a, b):
    # lightweight F1 for binary classification
    a = np.asarray(a)
    b = np.asarray(b)
    tp = int(((a == 1) & (b == 1)).sum())
    fp = int(((a == 1) & (b != 1)).sum())
    fn = int(((a != 1) & (b == 1)).sum())
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * (precision * recall) / (precision + recall)


def auc_simple(y_true, y_score):
    """Compute a simple ROC AUC using numpy (only for binary labels).

    This is a small, dependency-free implementation sufficient for quick baselines.
    """
    try:
        # delegate to sklearn if available for accuracy and speed
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y_true, y_score))
    except Exception:
        y_true = np.asarray(y_true)
        y_score = np.asarray(y_score)
        # sort by score descending
        desc = np.argsort(-y_score)
        y_true_sorted = y_true[desc]
        pos = (y_true == 1).sum()
        neg = (y_true != 1).sum()
        if pos == 0 or neg == 0:
            return 0.5
        # compute ranks method
        cum_pos = np.cumsum(y_true_sorted == 1)
        auc = ((cum_pos[y_true_sorted != 1]).sum() - (neg * (pos + 1) / 2.0)) / (pos * neg)
        return float(auc)


def evaluate_models(preds: pd.DataFrame, y_test) -> pd.DataFrame:
    """Evaluate each model column in preds against y_test.

    Behavior:
    - If a column name ends with `_prob` it is treated as a probability score for
      the positive class and evaluated with AUC (and also thresholded at 0.5
      to compute accuracy and F1).
    - Otherwise it's treated as predicted labels (classification) or numeric
      predictions (regression). We attempt to infer task from y_test values.
    """
    y_vals = getattr(y_test, "values", y_test)
    is_binary = set(np.unique(y_vals)) <= {0, 1}
    rows = []
    # group columns by model base name (e.g. 'logistic' and 'logistic_prob')
    cols = list(preds.columns)
    seen = set()
    for col in cols:
        if col in seen:
            continue
        seen.add(col)
        if col.endswith("_prob"):
            base = col[:-5]
            prob_col = col
            label_col = None
            if base in preds.columns:
                label_col = base
                seen.add(base)
        else:
            base = col
            label_col = col
            prob_col = f"{col}_prob" if f"{col}_prob" in preds.columns else None
            if prob_col:
                seen.add(prob_col)
        # helper: detect probability-like arrays even if not named *_prob
        def _looks_like_prob(arr):
            try:
                a = np.asarray(arr)
            except Exception:
                return False
            # floats in [0,1]
            if np.issubdtype(a.dtype, np.floating):
                # ignore NaNs when checking bounds
                if a.size == 0:
                    return False
                mn = np.nanmin(a)
                mx = np.nanmax(a)
                if mn >= 0.0 and mx <= 1.0:
                    # if more than two unique values or non-integer floats -> likely probabilities
                    uniq = np.unique(a[~np.isnan(a)])
                    if len(uniq) > 2 or not np.all(np.isin(uniq, [0.0, 1.0])):
                        return True
            return False

        # if an explicit prob column exists, prefer it
        if prob_col is not None and prob_col in preds.columns:
            # classification with probabilities
            probs = preds[prob_col].values
            auc = auc_simple(y_vals, probs) if is_binary else None
            # threshold at 0.5 for predicted labels
            pred_labels = (probs >= 0.5).astype(int)
            acc = accuracy(pred_labels, y_vals)
            f1 = f1_score_simple(pred_labels, y_vals)
            rows.append({
                "model": base,
                "task": "classification",
                "auc": auc,
                "accuracy": acc,
                "f1": f1,
            })
        elif label_col is not None:
            vals = preds[label_col].values
            # If the column looks like probability scores (float in [0,1]) treat as probs
            if is_binary and _looks_like_prob(vals):
                probs = vals
                auc = auc_simple(y_vals, probs)
                pred_labels = (probs >= 0.5).astype(int)
                acc = accuracy(pred_labels, y_vals)
                f1 = f1_score_simple(pred_labels, y_vals)
                rows.append({
                    "model": base,
                    "task": "classification",
                    "auc": auc,
                    "accuracy": acc,
                    "f1": f1,
                })
            elif is_binary:
                # classification labels
                acc = accuracy(vals, y_vals)
                f1 = f1_score_simple(vals, y_vals)
                rows.append({
                    "model": base,
                    "task": "classification",
                    "auc": None,
                    "accuracy": acc,
                    "f1": f1,
                })
            else:
                # regression style numeric predictions
                rows.append({
                    "model": base,
                    "task": "regression",
                    "rmse": rmse(vals, y_vals),
                    "mae": mae(vals, y_vals),
                })

    return pd.DataFrame(rows).sort_values(by=["task", "model"])
