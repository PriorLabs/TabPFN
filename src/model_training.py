"""Local modeling helpers for baselining (renamed from modeling.py).

Provides a small set of baseline models for regression and classification and a
convenience `train_and_predict` helper that returns a DataFrame of predictions.
For classification models we also add per-model probability columns with a
"_prob" suffix when predict_proba is available.
"""
from typing import Dict, Iterable, Tuple, Union, Optional
import logging
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

# explicitly export the public API
__all__ = ["get_models", "train_and_predict", "cross_validate_models", "evaluate_models", "save_model", "load_model"]

# module logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


def _validate_xy(X, y):
    """Basic validation for X and y shapes.

    Raises ValueError if incompatible lengths or empty.
    """
    if X is None or y is None:
        raise ValueError("X and y must not be None")
    x_len = len(getattr(X, 'index', X)) if hasattr(X, '__len__') else None
    y_len = len(getattr(y, 'index', y)) if hasattr(y, '__len__') else None
    if x_len is not None and y_len is not None and x_len != y_len:
        raise ValueError(f"X and y have different lengths: {x_len} != {y_len}")
    if y_len == 0:
        raise ValueError("y is empty")


def save_model(model: BaseEstimator, path: str) -> None:
    """Persist a fitted model to disk using joblib.

    Overwrites existing file.
    """
    dirname = os.path.dirname(path)
    if dirname and not os.path.exists(dirname):
        os.makedirs(dirname, exist_ok=True)
    joblib.dump(model, path)
    logger.info("Saved model to %s", path)


def load_model(path: str) -> BaseEstimator:
    """Load a model previously saved with `save_model`.
    """
    model = joblib.load(path)
    logger.info("Loaded model from %s", path)
    return model


def get_models(task: str = "regression", include_boosters: bool = False) -> Dict[str, object]:
    """Return a dict of baseline models for the requested task.

    Supported tasks: "regression", "classification".
    If `include_boosters` is True, attempt to include LightGBM/XGBoost models when
    those libraries are installed. This parameter is optional and defaults to False
    for fast, lightweight baselines.
    """
    if task == "regression":
        # Lazy import to keep lightweight
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import RandomForestRegressor

        models = {
            "linear": LinearRegression(),
            "random_forest": RandomForestRegressor(n_estimators=100, random_state=0),
        }
        return models

    if task == "classification":
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

        models = {
            "logistic": LogisticRegression(solver="liblinear", random_state=0),
            "random_forest": RandomForestClassifier(n_estimators=100, random_state=0),
            "grad_boost": GradientBoostingClassifier(random_state=0),
        }
        # optionally add faster gradient boosters if available
        if include_boosters:
            try:
                from lightgbm import LGBMClassifier

                models["lightgbm"] = LGBMClassifier(n_estimators=100, random_state=0)
            except Exception:
                # lightgbm not installed; skip
                pass
            try:
                from xgboost import XGBClassifier

                models["xgboost"] = XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=0)
            except Exception:
                # xgboost not installed; skip
                pass
        return models

    raise NotImplementedError(f"Unknown task '{task}'; supported: regression, classification")


def train_and_predict(models: Dict[str, object], X_train, y_train, X_test) -> pd.DataFrame:
    """Train each model and return DataFrame of predictions indexed like X_test.

    For regressors the column name is the model key and contains numeric predictions.
    For classifiers the function stores the predicted class in column <name> and,
    when available, the positive-class probability in column <name>_prob.

    The function accepts pandas DataFrames/Series or numpy arrays.
    """
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        # predicted labels / values
        try:
            pred = model.predict(X_test)
        except Exception:
            # fallback if model expects arrays
            pred = model.predict(getattr(X_test, "values", X_test))
        preds[name] = pred

        # if probability predictions are available, store positive-class proba
        if hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X_test)
            except Exception:
                proba = model.predict_proba(getattr(X_test, "values", X_test))
            # assume binary classification; keep probability of class 1 when possible
            try:
                probs = proba[:, 1]
            except Exception:
                # fallback to ravel if single-column or unexpected shape
                probs = proba.ravel()
            preds[f"{name}_prob"] = probs

    df = pd.DataFrame(preds)
    # if X_test has an index, preserve it
    try:
        df.index = X_test.index
    except Exception:
        pass
    return df


# ...existing code...

def cross_validate_models(models: Union[Dict[str, object], Iterable[object]],
                          X, y, cv: int = 5, scoring: Optional[Union[list, dict]] = None,
                          random_state: int = 42, n_jobs: int = 1) -> pd.DataFrame:
    """Run cross-validation for each model and return a DataFrame of mean/std scores.

    Parameters
    - models: dict of name->estimator or an iterable of estimators
    - X, y: full dataset (pandas or arrays)
    - cv: number of folds (uses StratifiedKFold when possible)
    - scoring: list or dict of sklearn scoring names (defaults to accuracy, f1, roc_auc)
    """
    import numpy as np
    from sklearn.model_selection import cross_validate, StratifiedKFold, KFold

    if scoring is None:
        scoring = ["accuracy", "f1", "roc_auc"]

    # use StratifiedKFold for classification-like targets when possible
    try:
        y_vals = getattr(y, "values", y)
        cv_split = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    except Exception:
        cv_split = KFold(n_splits=cv, shuffle=True, random_state=random_state)

    # normalize models input to (name, estimator) pairs
    if isinstance(models, dict):
        items: list[Tuple[str, object]] = list(models.items())
    else:
        items = []
        for i, est in enumerate(models):
            name = getattr(est, "name", None) or (est.__class__.__name__ if hasattr(est, "__class__") else f"model_{i}")
            items.append((name, est))

    rows = []
    for name, model in items:
        try:
            res = cross_validate(model, X, y, cv=cv_split, scoring=scoring, return_train_score=False, n_jobs=n_jobs)
        except Exception as e:
            # try without roc_auc if that caused problems (e.g., non-probabilistic model)
            if isinstance(scoring, (list, tuple)) and "roc_auc" in scoring:
                reduced = [s for s in scoring if s != "roc_auc"]
                try:
                    res = cross_validate(model, X, y, cv=cv_split, scoring=reduced, return_train_score=False, n_jobs=n_jobs)
                except Exception:
                    rows.append({"model": name, "error": str(e)})
                    continue
            else:
                rows.append({"model": name, "error": str(e)})
                continue

        stats = {"model": name}
        for k, v in res.items():
            if k.startswith("test_"):
                metric = k.replace("test_", "")
                stats[f"{metric}_mean"] = float(np.mean(v))
                stats[f"{metric}_std"] = float(np.std(v))
        rows.append(stats)

    return pd.DataFrame(rows).set_index("model")

# ...existing code...

def evaluate_models(models: Dict[str, object], X: pd.DataFrame, y: pd.Series, cv: int = 5, n_jobs: int = 1):
    """Evaluate models using cross-validation and (optionally) holdout metrics.

    Returns a DataFrame with CV summary and holdout metrics where available.
    """
    # local imports to keep top-level lightweight
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    # Cross-validate models
    cv_results = cross_validate_models(models, X, y, cv=cv, n_jobs=n_jobs)

    # Evaluate on the full dataset as a quick holdout-like check (user may pass a true holdout)
    holdout_results = {}
    for name, model in models.items():
        try:
            model.fit(X, y)
            preds = model.predict(X)
            row = {
                "accuracy": accuracy_score(y, preds),
                "f1": f1_score(y, preds, average="weighted")
            }
            # add roc_auc when predict_proba is present and binary
            if hasattr(model, "predict_proba"):
                try:
                    probs = model.predict_proba(X)
                    # if binary, take column 1
                    if probs.ndim == 2 and probs.shape[1] >= 2:
                        row["roc_auc"] = roc_auc_score(y, probs[:, 1])
                except Exception:
                    # skip roc_auc if it fails
                    pass
            holdout_results[name] = row
        except Exception as e:
            holdout_results[name] = {"error": str(e)}

    holdout_df = pd.DataFrame(holdout_results).T
    # combine with CV results; produce a MultiIndex columns: ('CV', metric) and ('Holdout', metric)
    return pd.concat([cv_results, holdout_df], axis=1, keys=["CV", "Holdout"])