import cProfile
import io
import pstats
import random

from sklearn.datasets import fetch_openml
import numpy as np

from itertools import product


RANDOM_SEED = 42


from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

from tabpfn import TabPFNClassifier

def get_data_parkinsons() -> tuple[np.ndarray, np.ndarray]:
    # Parkinson's Disease dataset: Predict Parkinson's disease presence
    # Features: Voice measurements (e.g., frequency, amplitude)
    # Samples: 195 cases
    df = fetch_openml('parkinsons')
    X, y = df.data, df.target

    # Encode target labels to classes
    le = LabelEncoder()
    y = le.fit_transform(y)

    # Convert all categorical columns to numeric
    for col in X.select_dtypes(['category']).columns:
        X[col] = X[col].cat.codes

    X = X.to_numpy()
    return X, y

def get_data_snythetic(num_samples: int, num_features: int, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    # Synthetic dataset: Predict class membership
    # Features: 10 continuous features
    # Samples: 1000 cases
    np.random.seed(RANDOM_SEED)
    # num_samples = 195
    # num_features = 22
    # num_classes = 2

    X = np.random.rand(num_samples, num_features)
    y = np.random.randint(0, num_classes, num_samples)
    return X, y

def run_tabpfn() -> None:
    num_sample_list = [100]
    num_feature_list = [2, 4, 8, 16, 32, 64, 128, 200, 300, 400, 500]
    num_class_list = [2]
    csv_string = "num_samples,num_features,num_classes,train_time,predict_time\n"
    for num_samples, num_features, num_classes in product(num_sample_list, num_feature_list, num_class_list):
        print("------------------------------------------")
        print(f"num_samples: {num_samples}, num_features: {num_features}, num_classes: {num_classes}")
        X, y = get_data_snythetic(num_samples, num_features, num_classes)
        print("dims of data", X.shape, y.shape)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=6)
        import time
        t_start = time.time()
        # Train and evaluate TabPFN
        tabpfn_model = TabPFNClassifier(random_state=RANDOM_SEED).fit(X_train, y_train)
        t_train = time.time()
        print(f"TabPFN Training Time: {t_train - t_start:.4f} seconds")
        y_pred = tabpfn_model.predict_proba(X_test)
        t_predict = time.time()
        print(f"TabPFN Prediction Time: {t_predict - t_train:.4f} seconds")

        csv_string += f"{num_samples},{num_features},{num_classes},{t_train - t_start},{t_predict - t_train}\n"

        # Calculate ROC AUC (handles both binary and multiclass)
        # keep below part in case y_pred optimized away (I dont expect in pyhton, but just in case)
        score = roc_auc_score(y_test, y_pred if len(np.unique(y)) > 2 else y_pred[:, 1])
        print(f"TabPFN ROC AUC: {score:.4f}")

    with open("tabpfn_speed.csv", "w") as f:
        f.write(csv_string)

def profile() -> None:
    """
    Prints top N methods, sorted by time.
    Equivalent to:
        python -m cProfile -o data/profile.txt main.py -n 100
    Options:
        time, cumulative, line, name, nfl, calls
    -----------
    ncalls - for the number of calls.
    time/tottime - for the total time spent in the given function
    (and excluding time made in calls to sub-functions)
    cumulative/cumtime - is the cumulative time spent in this and all subfunctions
    (from invocation till exit). This figure is accurate even for recursive functions.
    """
    random.seed(0)
    command = (
        "run_tabpfn()"
    )
    profile_file = "profile.txt"
    sort = "time"

    cProfile.run(command, filename=profile_file, sort=sort)
    s = io.StringIO()
    stats = pstats.Stats(profile_file, stream=s)
    stats.sort_stats(sort).print_stats(100)

    with open('profile-readable.txt', 'w+') as f:
        f.write(s.getvalue())


if __name__ == "__main__":
    # profile()
    run_tabpfn()