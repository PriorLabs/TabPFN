# use get_total_memory and compare it against result from psutils
# run it only if the it is windows os.name == "nt"
from __future__ import annotations

import os

import pytest

import numpy as np

def test_internal_windows_total_memory():
    if os.name != "nt":
        pytest.skip("Windows specific test")
    import psutil

    from tabpfn.utils import get_total_memory_windows

    utils_result = get_total_memory_windows()
    psutil_result = psutil.virtual_memory().total / 1e9
    assert utils_result == psutil_result


def test_internal_windows_total_memory_multithreaded():
    # collect results from multiple threads
    if os.name != "nt":
        pytest.skip("Windows specific test")
    import threading

    import psutil

    from tabpfn.utils import get_total_memory_windows

    results = []

    def get_memory() -> None:
        results.append(get_total_memory_windows())

    threads = []
    for _ in range(10):
        t = threading.Thread(target=get_memory)
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
    psutil_result = psutil.virtual_memory().total / 1e9
    assert all(result == psutil_result for result in results)

def test_infer_categorical_features():
    from tabpfn.utils import infer_categorical_features
    X = np.array([[np.nan, "NA"]], dtype=object).reshape(-1, 1)
    out = infer_categorical_features(X, provided=[0], min_samples_for_inference=0, max_unique_for_category=2, min_unique_for_numerical=5)
    assert out == [0]



