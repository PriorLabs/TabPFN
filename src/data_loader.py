"""Local baselining data loading helpers (renamed from data_loading.py).
"""
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_data(path: str = None, sample_n: int = 1000, random_state: int = 0) -> pd.DataFrame:
    """Load dataset from CSV if available; otherwise generate a small synthetic DataFrame.

    Returns a DataFrame containing a numeric target column named 'ClaimFrequency'.
    """
    if path:
        p = Path(path)
        if p.exists() and p.suffix.lower() == ".csv":
            return pd.read_csv(p)

    # Synthetic fallback
    rng = np.random.RandomState(random_state)
    X1 = rng.normal(size=sample_n)
    X2 = rng.randint(0, 5, size=sample_n)
    claim_freq = np.maximum(0, 0.1 * X1 + 0.5 * X2 + rng.normal(scale=0.5, size=sample_n))
    df = pd.DataFrame({
        "feature1": X1,
        "feature2": X2,
        "ClaimNb": claim_freq,
    })
    return df


def preprocess_data(df: pd.DataFrame, target: str = "ClaimNb", test_size: float = 0.2, random_state: int = 0):
    """Simple preprocessing: split features/target and train/test split.

    Returns X_train, X_test, y_train, y_test (pandas objects).
    """
    if target not in df.columns:
        raise ValueError(f"target column '{target}' not found in DataFrame columns: {list(df.columns)}")

    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # keep pandas types / indices for easier alignment later
    return X_train, X_test, y_train, y_test
