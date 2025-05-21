# use get_total_memory and compare it against result from psutils
# run it only if the it is windows os.name == "nt"
from __future__ import annotations

import os


def test_internal_windows_total_memory():
    if os.name == "nt":
        import psutil

        from tabpfn.utils import get_total_memory_windows

        utils_result = get_total_memory_windows()
        psutil_result = psutil.virtual_memory().total / 1e9
        assert utils_result == psutil_result


def test_internal_windows_total_memory_multithreaded():
    # collect results from multiple threads
    if os.name == "nt":
        import threading

        import psutil

        from tabpfn.utils import get_total_memory_windows

        results = []

        def get_memory():
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


def test_infer_device_auto_defaults_to_cpu_when_no_accelerator(monkeypatch):
    """`infer_device_and_type('auto')` returns CPU when no accelerator found."""
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)

    from tabpfn.utils import infer_device_and_type

    device = infer_device_and_type("auto")
    assert device.type == "cpu"


def test_infer_device_tpu_requires_torch_xla(monkeypatch):
    """Using TPU without torch_xla installed raises an error."""
    monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
    import pytest

    from tabpfn.utils import infer_device_and_type

    with pytest.raises(ValueError, match="torch_xla must be installed"):
        infer_device_and_type("tpu")
